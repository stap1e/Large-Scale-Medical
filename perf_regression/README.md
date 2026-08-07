# OCL GPU starvation 调查运行手册

这里的工具用于性能回归取证，不以“平均 step time 更小”或“把 batch 塞满显存”
代替 GPU 空泡定位。

## 1. 两种运行必须分开

### 分阶段诊断（侵入式）

`--diagnose_gpu_gaps` 使用 `perf_counter()` 和 CUDA Event，并在诊断窗口内
每个 training attempt 末尾同步一次，以获得
H2D/forward/loss/backward/optimizer phase span。
它会改变跨 step overlap，不能用该运行宣称生产 GPU utilization。

```bash
cd Self-supervised
python ocl_train.py \
  --dataset_mode ctrate_subset \
  --data_root /path/to/CT_RATE/train \
  --datalist_json /path/to/fixed_subset.json \
  --cache_dir /path/to/prewarmed_cache \
  --cache \
  --batch_size 4 \
  --workers 16 \
  --prefetch_factor 2 \
  --persistent_workers \
  --pin_memory \
  --num_steps 550 \
  --diagnose_gpu_gaps \
  --gap_profile_steps 550 \
  --disable_final_artifacts
```

`--slow_step_threshold_ms` 必须为正；若不传，则自动以最近 100 step 中位数的
1.5 倍判慢。默认输出：

```text
perf_regression/step_timing.csv
perf_regression/effective_config.json
```

DDP rank 0 使用上述文件名，其余 rank 自动写
`step_timing_rank<N>.csv`。主要列包括：

- 12 个要求的 CPU/CUDA phase timing（单位统一为 ms）
- global step、逻辑 epoch、batch index、是否 epoch 首 batch、step Unix 起止时间
- volume/sample ID、文件路径
- cache hit/miss/write
- worker ID 与每个样本完整 `__getitem__` 时间
- prefetch result queue depth
- batch shape、源 tensor 是否 pinned、non-blocking 是否有效
- log/checkpoint/validation 触发位
- attempt index、optimizer update 是否成功、是否 AMP overflow
- allocated/reserved GPU memory
- slow-step A–G 自动分类
- 上一 step 的 logging-side CUDA 尾部到当前 H2D-start 的 gap candidate，
  以及该 gap 前是否触发 checkpoint/validation

`gap_profile_steps` 按成功 optimizer update 计数；AMP overflow attempt 会保留
在 CSV 中但不占用该额度。达到窗口后，主进程会清除 worker diagnostic event，
worker 不再做 cache hash/前后两次 `stat`；仅由 diagnose 开启的 NVTX 也停止
（显式 `--emit_nvtx` 或尚未完成的 profiler schedule 仍会保留 ranges）。最多仍
会消费预取队列中已经插桩的少量 batch。完整窗口由后台线程写 CSV，训练退出时会
join 并检查写入错误。诊断期的 cache hash/stat 本身有额外开销，因此只应用其
分类结果定位，不能把该 pass 的吞吐当生产吞吐。

汇总：

```bash
python perf_regression/analyze_step_timing.py \
  perf_regression/step_timing.csv \
  --warmup 50 --measure 500
```

### 真实 utilization / throughput（非逐 step 同步）

该模式只在 warmup 边界和测量终点同步，500 个测量 step 内不插入诊断同步：

```bash
cd Self-supervised
python ocl_train.py \
  ...相同数据和模型参数... \
  --num_steps 550 \
  --throughput_warmup_steps 50 \
  --throughput_measure_steps 500 \
  --throughput_output ../perf_regression/throughput.json \
  --disable_final_artifacts
```

同时运行 `nvidia-smi dmon`。`throughput.json` 给出每 rank/全局 samples/s，
以及延迟读取 CUDA Event 得到的 production median/p95 step cycle；500-step
窗口内不逐步同步。这里 `raw_samples` 是输入 volume 数，
`input_query_crops` 是进入 OCL head 的 query crop 数；每个 query crop 会生成并
编码两个 masked view，因此 `effective_training_views` 和 `views/s` 按
`2 * input_query_crops` 计算。最终 artifacts 被排除，但训练窗口内正常的
log/checkpoint/cache 行为仍保留。

正式 CT-RATE 训练使用 `scripts/train_ctrate_subset_ocl.sh`；它默认
workers=16、cache on、log=1000、checkpoint=20000、sync timing off。原
`train_ctrate_subset_smoke.sh` 现在会在长于 50 step 时明确警告。

## 2. 自动分类解释

| 类别 | 判据 |
|---|---|
| A | iterator wait 相对滚动基线突增；检查 gzip、磁盘、transform、cache、epoch 首 batch |
| B | main-thread batch prepare 突增；检查 Python/collate/view/crop/metadata |
| C | H2D submit 或 CUDA memcpy 突增；检查 pin 与真正的 async copy |
| D | 触发日志且 logging time 突增 |
| E | 触发 checkpoint 且 checkpoint time 相对滚动基线突增 |
| F | 触发 validation 且 validation time 突增；当前 OCL 没有 validation，此列应恒为 0/False |
| G | forward/backward CUDA 时间正常而总 wall 有未归因洞；重点找 host sync/主线程阻塞 |

若某个 CUDA Event 覆盖的 phase 显著变长，会标为
`G_CUDA_PHASE_OR_PYTHON_LAUNCH`。Event elapsed time 包含该 phase 内的
Python kernel launch 间隙，必须结合 profiler/Nsight timeline 区分“kernel
本身变慢”和“主线程未持续提交 kernel”，不能仅凭该字段断言是 CUDA compute
variance。

## 3. torch.profiler 与 NVTX/Nsight

### torch.profiler

默认 schedule 为 `wait=20, warmup=10, active=200`：

```bash
python Self-supervised/ocl_train.py \
  ... \
  --num_steps 550 \
  --save_gap_trace \
  --profiler_wait_steps 20 \
  --profiler_warmup_steps 10 \
  --profiler_active_steps 200 \
  --gap_trace_dir perf_regression/traces
```

如果已知 gap 在 checkpoint 20,000 附近，可让 profiler 低开销等待到附近，而
不是 profile 前 20 个正常 step，例如：

```bash
--num_steps 20250 \
--save_gap_trace \
--profiler_wait_steps 19770 \
--profiler_warmup_steps 10 \
--profiler_active_steps 300
```

仅 `--save_gap_trace` 不会自动开启逐 step CUDA Event 同步；需要 CSV timing 时
再显式加 `--diagnose_gpu_gaps`。

### Nsight Systems（推荐用于最终空泡证据）

`--emit_nvtx` 只添加 range，不同步：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --force-overwrite=true \
  -o perf_regression/nsys/ocl_gap \
  python Self-supervised/ocl_train.py \
    ...相同训练参数... \
    --num_steps 550 \
    --emit_nvtx \
    --disable_final_artifacts
```

先执行 `mkdir -p perf_regression/nsys`。不同 Nsight Systems 版本的取值可能
不同，目标机先用 `nsys profile --help` 确认；常见版本接受
`process-tree|system-wide|none`，不接受 `cpu/true`。

timeline 中对齐：

```text
dataloader_wait
batch_prepare
h2d
forward
loss
backward
optimizer
logging
checkpoint
validation
```

另外有 `optimizer_zero_grad`、`forward_input_concat`、`view_generation`、
`backbone_view1/2`（或 `backbone_merged_views`）和 `scheduler` 子范围，用于避免把
主线程 launch 空段误归为无标签的 G 类。

重点看 GPU kernel 之间的白区，同时查看对应 CPU thread 是否：

- 在 `cudaDeviceSynchronize/cudaStreamSynchronize` 或 scalar D2H 上等待；
- 执行 Python mask/crop loop；
- 卡在 `read/pread/futex`；
- 执行 checkpoint 序列化/write/fsync；
- 提交大量小 kernel 而跟不上 GPU。

Exact “GPU idle gap 总时长”以 Nsight CUDA GPU row 的空白区为准。
CSV 的 inter-step gap 是 event-to-event 候选值。它从上一 step 的 logging-side
CUDA 尾部开始，checkpoint 位于该 event 之后，因此下一行的
`gap_preceded_by_checkpoint` 用于正确对齐；候选区间可能包含 D2H 或极小的
非训练 kernel，不能冒充精确 idle integral。

## 4. 系统级对齐

训练启动后取得 PID：

```bash
pgrep -n -f 'ocl_train.py'
```

一键记录：

```bash
OUTPUT_DIR=perf_regression/system_A1 \
  bash perf_regression/monitor_system.sh TRAIN_PID
```

用户要求的原始命令是：

```bash
nvidia-smi dmon -s pucvmet -d 1
pidstat -dru -p TRAIN_PID 1
iostat -xz 1
vmstat 1
pidstat -t -dru -p TRAIN_PID 1
```

脚本为跨文件按秒对齐增加 timestamp 选项：dmon `-o DT`、iostat `-t`、
vmstat `-t`。若目标机旧版本不支持，先运行各命令 `--help`，去掉不支持的
timestamp 选项并用外层采集器补统一时间戳。

GPU util 下降时检查：

- `pidstat`: 训练 PID 的 `iodelay`/read KB/s 是否升高，还是单一 thread 占满核；
- `iostat`: cache/data 盘 `await`、queue、`%util` 是否尖峰/100%；
- `vmstat`: `wa`、`b`、context switch 是否升高；
- 是否有其他 PID 同时读写相同共享盘；
- 网络存储还需由存储端查看 latency/IOPS/throttling。

整机 CPU 0.7% + 单进程约 100% 很可能恰好是一颗逻辑核，需用 thread 版
pidstat 区分 Python main thread 与 `pt_data_pin`。

## 5. A/B 回归矩阵

先完全预热并冻结同一 cache。运行：

```bash
cd Self-supervised
python cache_warmup.py \
  --data_root /path/to/CT_RATE/train \
  --datalist_json /path/to/fixed_subset.json \
  --cache_dir /path/to/prewarmed_cache \
  --workers 8
cd ..
```

warmup 的 `--workers` 现在会实际并行构建 cache；父进程丢弃 crop payload，避免
把大批无用 tensor 经 IPC 传回。

然后运行：

```bash
export DATA_ROOT=/path/to/CT_RATE/train
export DATALIST_JSON=/path/to/fixed_subset.json
export CACHE_DIR=/path/to/prewarmed_cache
export GPU_ID=0
export BATCH_SIZE=4
export WORKERS=16
export CHECKPOINT_INTERVAL=100

bash perf_regression/run_ab_matrix.sh
```

每个 case 自动跑两遍：

1. 50 warmup + 500 step 的侵入式 phase timing；
2. 50 warmup + 500 step 的非逐步同步 utilization/throughput。

`TRACE_CASE`（默认 A1）在上述 production measurement **之后**再跑第三遍
profiler-only capture，写入 `<case>/traces`。它不带
`--diagnose_gpu_gaps`/`--sync_timing`，因此 timeline 不含人为逐 step
`cudaDeviceSynchronize`；放在 production pass 后也避免只为 A1 预热 OS page
cache 再测 A1。

Cases：

- A1：在插桩版中显式恢复回归行为（scale polling、CPU concat、双重
  checkpoint serialization、每 step loss.item、短 epoch iterator、9 个无用 crop）
- A2：A1 + 关闭训练日志
- A3：A1 + 关闭 checkpoint
- A4：A1 + 禁止 cache write（主矩阵的 cache 已预热，因此这里只验证稳定 hit
  路径不因开关改变，不能单独证明冷 cache 写入无影响）
- A5：A1 + 恢复旧 loader 的 `persistent_workers=False` 和
  `drop_last=False`；数据仍是固定 CT subset，不能冒充完整 old mixed loader
- A6：修复后的默认路径

若测量窗口发现 cache miss，脚本会失败，防止把“首 epoch 建 cache”混进稳定
A/B。

PersistentDataset cache 文件固定不等于 Linux OS page cache 固定。正式结论至少
交错/反序重复 A1 与 A6（可用 `CASES` 控制顺序），报告重复间方差；不要只接受
固定 A1→…→A6 单次运行。共享存储还需避开其他作业的竞争窗口。

例如至少用不同输出目录执行两个顺序：

```bash
CASES="A1_current A6_fixed" \
OUTPUT_ROOT=perf_regression/runs/a1_first \
bash perf_regression/run_ab_matrix.sh

CASES="A6_fixed A1_current" \
OUTPUT_ROOT=perf_regression/runs/a6_first \
bash perf_regression/run_ab_matrix.sh
```

冷 cache 写入另用两个**专用空目录**运行。一次 CW0→CW1 不能“隔离”写入成本，
因为先运行者会预热原始数据的 OS page cache；脚本支持反序，但第二次必须使用
另一组空 cache 根目录和输出目录。每个根目录下会自动建立互不共享的
`timing/` 与 `production/` PersistentDataset cache。脚本先跑 production cases，
再跑 timing cases，避免侵入式 pass 在 utilization capture 之前预热 OS page
cache：

```bash
export WRITE_CACHE_DIR=/dedicated/empty/order1_write
export READ_ONLY_CACHE_DIR=/dedicated/empty/order1_read_only
export SUBSET_SIZE=256
export OUTPUT_ROOT=perf_regression/cache_write_ablation/write_first
CASE_ORDER=CW0_CW1 bash perf_regression/run_cache_write_ablation.sh

export WRITE_CACHE_DIR=/dedicated/empty/order2_write
export READ_ONLY_CACHE_DIR=/dedicated/empty/order2_read_only
export OUTPUT_ROOT=perf_regression/cache_write_ablation/read_only_first
CASE_ORDER=CW1_CW0 bash perf_regression/run_cache_write_ablation.sh
```

若要把第二次也视为“冷 OS cache”，需在等价的重启/受控缓存状态下运行；本脚本
不会以 root 清理全机 page cache。报告两个顺序的差值范围，并与 iostat/pidstat
对齐。仅当方向和量级跨顺序稳定时，才把它作为 cache-write 因果证据。

```text
CW0 = cold PersistentDataset cache，允许写入
CW1 = cold PersistentDataset cache，禁止写入
CW2 = CW0 生成的 cache 上的稳定 hit 路径
```

每组同时生成侵入式 `step_timing.csv` 和非逐 step 同步的
`throughput.json`/`nvidia_dmon.txt`。CW0/CW1 的 dmon 覆盖首个逻辑 epoch 和随后
300 step；其 throughput 窗口位于首 epoch 之后。CW0-vs-CW1 首 epoch 只是一项
有顺序敏感性的 cache-write 候选，CW2 才是纯 prewarmed hit 参考。首 epoch 结果
绝不能代表稳定训练性能。脚本遇到非空目录会退出，不会删除用户 cache。

A0 不会被脚本伪造。仓库的 `1e9e967` 没有 CT-RATE loader，不能在“不改代码”
条件下使用相同 CT 数据。应把用户实际已知正常的旧副本/checkout 作为 A0，
固定同一 GPU、数据、batch、seed、precision、model、optimizer、step 和 cache，
并保存其 commit/argv、dmon 与 Nsight 结果。

### 启动参数回归（与固定 batch 主矩阵分开）

smoke 默认本身就是强回归候选，不能只测试代码开关。以下矩阵始终固定
`batch_size=1`，逐步移除 `sync_timing`、每 step 日志、checkpoint，再切换为
workers=16 + prewarmed cache：

```bash
bash perf_regression/run_launch_ablation.sh
```

输出 L0–L4 的 `throughput.json` 与 timestamped dmon。该表回答“实际 argv 是否
导致回归”，但 batch=1，不能与 batch=4 的 A0–A6 samples/s 合并。

## 6. DataLoader sweep

```bash
export DATA_ROOT=/path/to/CT_RATE/train
export DATALIST_JSON=/path/to/fixed_subset.json
export CACHE_DIR=/path/to/prewarmed_cache
bash perf_regression/run_dataloader_sweep.sh
```

自动测试：

```text
workers = 4, 8, 12, 16
prefetch_factor = 2, 4
persistent_workers = True
pin_memory = True
```

每组同样分 timing/util 两遍。选择标准是低 iterator wait p95、少 idle gap、高
samples/s 和稳定 dmon 曲线，不是 worker 数最大。固定顺序会逐渐预热 OS page
cache；初筛后至少把前两名反序重复，再确定最终 worker/prefetch 配置。

## 7. 双 OCL view 合并验证

默认暂不打开 `--merge_ocl_views`。目标 GPU 上分别验证 FP32/AMP、checkpoint
开/关：

```bash
python perf_regression/check_ocl_view_merge.py
python perf_regression/check_ocl_view_merge.py --amp
python perf_regression/check_ocl_view_merge.py --use_checkpoint
python perf_regression/check_ocl_view_merge.py --amp --use_checkpoint
```

脚本使用固定、预生成的 x1/x2，比较 separate/merged 的两个 embedding、当前
loss、全部 parameter gradients，并分别记录 peak allocated memory。全部通过且
merged 显存不超限后，才在性能 A/B 中加 `--merge_ocl_views`。

## 8. 消融开关

以下开关只用于诊断，默认行为不变：

```text
--disable_training_logging
--disable_checkpoint
--disable_validation
--disable_cache_write
--disable_encoder_export
--disable_data_integrity_check
```

`--disable_validation` 当前是显式 no-op，因为 OCL 根本没有 validation；
`validation_time` 应恒为零附近的 Python range 开销。

回归复现专用、不得用于生产：

```text
--legacy_amp_scale_polling
--legacy_loss_item_each_step
--cpu_concat_before_h2d
--duplicate_checkpoint_serialization
```

周期 checkpoint 默认仍包含 optimizer/scheduler/scaler，以保留现有 resume
语义；修复只消除了同一步的第二次 `torch.save`，并用 hard link/copy 更新
`model_current_epoch.pt`。`--no-full-periodic-checkpoint` 可生成轻量
model+step 快照来做 E 类诊断，但从该快照恢复会重新初始化 optimizer，不能称为
optimizer-state resume。训练结束的 final/current checkpoint 始终保存上述
训练状态。

即使 full checkpoint 也没有保存 Python/NumPy/Torch/CUDA RNG、DataLoader
prefetch 队列或 sampler offset；resume 会从 deterministic sampler 的开头重新
取数，所以目前不是逐位可复现的 exact continuation。这是 checkpoint 正确性
限制，不是训练中周期 GPU 空泡的来源。

当前 checkpoint 仍在训练主线程同步执行；若 E 类仍主导，优先使用
`--disable_checkpoint` 做因果消融并恢复合理的长 interval。异步 checkpoint
需要一致的 GPU→CPU snapshot/状态版本保证，本次没有用不安全的后台
`torch.save(model.state_dict())` 伪装成修复。
