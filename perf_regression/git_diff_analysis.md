# OCL GPU 利用率性能回归：Git 与启动配置取证

调查对象：`Self-supervised/ocl_train.py` 及 CT-RATE/MONAI 数据路径。  
调查日期：2026-07-28。  
本报告区分“Git/静态代码已经证明的回归机制”和“必须在目标 96 GiB GPU、
同一数据/cache 状态下由 timeline 确认的运行时占比”。

## 结论先行

Git 中最可信的正常候选基线是 `1e9e967`（2026-07-13，首次引入 OCL），
最可信的回归边界是：

```text
484c8eed70bc14e0a9d32f4ad9167409c6c261b7
2026-07-16
feat: 接入 CT-RATE 小规模子集预训练与断点恢复流程...
```

`backup-8d576ad` 分支上的 `8d576ad` 含同一组相关代码变更，但它不是当前
`main`/`HEAD` 的祖先；两者在本调查涉及的文件上相同。因而对当前主线回答
“从哪个 commit 开始”时使用 `484c8ee`，而不是备份分支 hash。

该提交一次性改写了 OCL 训练循环、AMP strict step control、checkpoint payload、
CT-RATE loader 和 smoke 启动方式。根据代码位置可以明确分类：

- **G 类（首要代码回归）**：每个 AMP step 新增两次
  `GradScaler.get_scale()` 主机读取；第一次位于 forward 与 backward 之间，
  会排空 CUDA 队列并暂停 CPU 提交。若使用 smoke/README 命令，还会每 step
  两次显式 `torch.cuda.synchronize()`。
- **E 类（强周期回归）**：smoke/README 每 100 step 在主线程连续保存两份
  model + AdamW/AMSGrad optimizer + scheduler + scaler。旧默认是每 20,000 step，
  且旧周期 payload 只有 model。
- **A/B 类（强周期/单核候选）**：smoke 默认 `workers=0`、`cache=off`，
  NIfTI gzip、MONAI transforms、collate 和 11 个 crop 全部落在训练主线程；
  新 CT 子集很短，又把 iterator/prefetch 重新 prime 的周期从约 75,000 step
  缩短到常见 16–64 step。
- **D 类（启动参数回归）**：smoke 默认每 step 打印并经 `tee` 写日志。

因此，从静态证据看，当前回归不是单一“batch 太小”，而是 **G + E + A/B**
叠加；实际每个空泡应按 cadence 和 timing CSV 分别归类。没有目标机 trace
时，不能诚实宣称这些类别各占多少，也不能宣称修复后已经恢复 100%。

## Git 边界

### `1e9e967`：旧候选

旧 AMP 提交顺序是：

```python
loss = model(img)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

旧版也有 step 末尾 `loss.item()`，所以它是已有同步点，不是
`1e9e967..484c8ee` 的新增回归；但它仍会阻止跨 step 提前提交。

### `484c8ee`：首个高置信度回归提交

新增 strict step control：

```python
scale_before = scaler.get_scale()
scaler.scale(loss).backward()
...
scaler.step(optimizer)
scaler.update()
update_succeeded = scaler.get_scale() >= scale_before
```

`get_scale()` 在 CUDA scale 已初始化后通过 host scalar 读取获得值。第一处
发生在 forward 已提交、backward 尚未提交时：

```text
forward enqueue
host scalar read / stream wait
GPU 完成 forward 后无后续 backward 可执行
CPU 恢复并 enqueue backward
```

这正是 GPU starvation，而不只是 step 平均时间增加。

同一提交还把 checkpoint 从：

```python
{"global_step": global_step, "state_dict": model.state_dict()}
```

扩展为 model、optimizer、scheduler、scaler。当前 optimizer 为
`AdamW(amsgrad=True)`，稳定后通常有 `exp_avg`、`exp_avg_sq`、
`max_exp_avg_sq` 三份参数尺度的状态。单文件 tensor payload 粗略从 1 份模型
增长到约 4 份，且同一步仍连续序列化两次。

### 后续提交

- `2c75359..0009add` 只新增/修改 `cache_warmup.py`，没有修改训练 loop；
  但连续修订旁证了在线 cache miss/写盘是近期实际故障面。
- `2cb458f` 全局包装 `torch.load(weights_only=False)` 以适配 Torch 2.6+。
  wrapper 的 Python 开销不是首要空泡来源，但这说明运行环境可能偏离仓库规定
  的 Torch 2.1.1。A/B 必须固定 Torch/CUDA/MONAI 版本。
- `models/ocl_head.py`、legacy MONAI transforms 和原始 mixed loader 在
  `1e9e967..HEAD` 原本没有变化，所以逐样本 mask loop、双 backbone 调用和
  9 个无用 base crop 会放大问题，但不能单独解释这段 Git 区间的回归。

如果用户所说的正常版本来自未提交工作区或外部副本，Git 不能替代该副本；
应把它作为 A0 保存，而不能把 `1e9e967` 冒充为“同一 CT-RATE 数据”的 A0，
因为该 commit 尚无 CT-RATE loader。

## 启动参数对比

| 配置 | `1e9e967` 默认 | 当前直接运行默认（改动前） | CT smoke 默认 | README Stage C |
|---|---:|---:|---:|---:|
| batch size | 4 | 4 | 1 | 2 |
| num workers | 16 | 16 | 0 | 4 |
| prefetch factor | PyTorch 默认 2 | 默认 2 | 不适用 | 默认 2 |
| persistent workers | False | CT 模式 True | False | True |
| pin memory | True | CUDA 可用时 True | True | True |
| drop last | False | CT/OCL True | True | True |
| multiprocessing context | 默认 | 默认 | 默认 | 默认 |
| worker init function | MONAI default `list_dataset_worker_init_fn` | 同左 | workers=0，未调用（配置仍为 MONAI default） | MONAI default |
| cache | True | True | **False** | True |
| cache rate / cache num | N/A（PersistentDataset） | N/A | N/A | N/A |
| log interval | 1,000 | 1,000 | **1** | 20 |
| checkpoint interval | 20,000 | 20,000 | **100** | 100 |
| validation interval | 无 | 无 | 无 | 无 |
| sync timing | 无 | False | **True** | **True（未覆盖默认）** |

注意旧 parser 的 `--cache False` 是 truthy 字符串，可能实际仍开 cache；当前
布尔 parser 会真正关闭。仅看命令文本相同不足以证明运行配置相同。

修复后的诊断 run 会从 loader 实例读取实际 `worker_init_fn`，并把 `sys.argv`
和实际 loader 配置写入
`perf_regression/effective_config.json`，避免再次只比较 parser 默认值。
`run_launch_ablation.sh` 另以固定 batch=1 实测 smoke argv：L0 保留
workers=0/cache off/log=1/sync=on/checkpoint=100，L1–L3 逐项关闭同步、日志和
checkpoint，L4 再切换 production input pipeline；该表与 batch=4 的 A0–A6
主矩阵分开报告。

## 周期性操作审计

| 周期 | 操作 | 分类 | 说明 |
|---|---|---|---|
| 每 step（改动前） | 两次 `GradScaler.get_scale()` | G | 新增；第一次切在 forward/backward 中间 |
| 每 step（smoke） | 两次显式 CUDA synchronize | G | 诊断计时不能用于利用率基准 |
| 每 step（改动前后旧逻辑） | `loss.item()` | G/D | 已改成 GPU 累积，只在 log interval 读取 |
| 每 step | CPU `concat_image` 后 H2D | B/C | 新 CPU tensor 通常不再 pinned |
| 每 `log_every` | print + tee | D | 没有 TensorBoard/wandb |
| 每 `eval_num` | 两份全量 checkpoint | E | `eval_num` 名称误导；这里没有 validation |
| 每 `len(loader)` | iterator reset / queue re-prime | A | 短 CT subset 后变得高频 |
| cache 首次访问 | gzip/load/spacing/cache write | A | 应区分首 epoch 与稳定 cache |
| 训练结束 | final checkpoint/model/encoder | 非训练中空泡 | 不能解释中途周期下降 |

静态搜索还确认当前 OCL step/epoch 路径没有：

- TensorBoard `add_scalar/add_histogram/add_image`
- wandb
- validation/val loader
- `torch.cuda.empty_cache()`
- NIfTI、NumPy、JSON、pickle 周期写入
- datalist JSON 重读
- 每 epoch 数据完整性扫描
- 周期 encoder export

manifest JSON、患者唯一性和全路径 `is_file()` 都只在 loader 初始化时执行一次。
按患者确定性抽样、目录遍历和 NIfTI header 检查只在 JSON 生成工具中执行。
当前 full checkpoint 会恢复 model/optimizer/scheduler/scaler/global step，但不
保存 Python/NumPy/Torch/CUDA RNG、worker prefetch 状态或 sampler offset；
resume 会重放 deterministic sampler 开头的数据。因此它不是逐位 exact
continuation，不过该缺口发生在 resume 边界，不解释稳定训练中的周期空泡。

## DataLoader 与 cache

旧 mixed loader 的大小下界为：

```text
N = 8A + 187,555
A >= 14,376
N >= 302,563
```

旧默认单卡 batch 4 时，一个 iterator 至少约 75,640 step。CT 子集常见
`N=256`：

- 单卡 batch 4：约每 64 step 到边界。
- 8 rank、每 rank batch 2：约每 16 step 到边界。

`persistent_workers=True` 只避免重建进程，不能让已经耗尽的 iterator 在当前
epoch 尾部预取下一 epoch。若慢 step 总是 `batch_index=0` 或周期等于
`len(loader)`，就是 A 类。

胸部 transform 的 PersistentDataset 边界在首个 Randomizable transform
`RandShiftIntensityd(prob=0)` 之前。因此 miss 需要完成：

```text
NIfTI .nii.gz load/decompress
Orientation
Spacing/resample
intensity scale
foreground crop
pad
torch.save cache
```

即使概率为 0，该 transform 类型仍截断 deterministic cache。cache hit 仍需
从磁盘 `torch.load` 大 volume，后续 crop/view 每次执行。必须分开看：

1. 空 cache 的首 epoch；
2. cache 全部预热后的稳定 500 step；
3. 共享盘/网络盘上 cache hit 的竞争。

## OCL 主线程放大器（不是 484c8ee 新增）

1. `VoCoAugmentation` 每 volume 生成 2 个 query + 9 个 base crop，OCL 只用
   query。9 个 base crop 仍经过随机增强、collate、IPC 和 pin。
2. 旧 `concat_image()` 在主线程 CPU `torch.concatenate`，破坏 DataLoader
   已完成的 pin，随后 `non_blocking=True` 可能无效。
3. `_make_view()` 两次逐样本 Python loop；`patch_rand_drop()` 又用 NumPy/Python
   while loop 提交许多小 CUDA kernel。它可能使一个主线程 launch-bound，但
   该代码自 OCL 首次提交起未变。
4. 两个 view 分别调用 backbone。默认 OCL backbone 使用 LayerNorm 和
   InstanceNorm，不使用 `projection_head` 中的 BatchNorm；在
   `drop_path=0` 时按 batch 维合并在算法上可行，但必须在目标 GPU 验证输出、
   loss、全部梯度和显存。

现有 loss 的 `cat([e1,e2], dim=0).reshape(-1,2,D)` 还存在一个独立的正样本
配对正确性问题；它不是本次性能回归，性能 A/B 保持原 loss ordering，不能把
算法修复混入同一个对比。

## 本次修复如何对应回归

- strict step control 改用 optimizer step post-hook 判断 GradScaler 是否真正
  调用了 `optimizer.step()`；保留“overflow 不推进 scheduler/global_step”
  语义，不再调用 `get_scale()`。
- loss 在 GPU 上累计，只在 log interval 取一次 host scalar。
- CT-RATE 使用连续 deterministic batch sampler：每个逻辑 epoch 的
  `seed+epoch` shuffle、rank partition、full-batch drop 仍保留，但 DataLoader
  iterator 不结束，预取队列不再每几十 step 清空。
- pinned view 分别 `non_blocking` H2D，再在 GPU 拼接；CSV 同时记录源 tensor
  是否 pinned、异步复制是否实际成立。
- OCL 默认不再生成未消费的 9 个 base crop；兼容/A1 可用
  `--no-skip-unused-ocl-crops` 恢复。
- 周期 checkpoint 只序列化一次；同盘用 hard link 原子更新 `current`，不支持
  hard link 时退化为文件 copy，但仍避免第二次 GPU state 序列化。默认仍保存
  optimizer/scheduler/scaler 以保持 resume 语义，因此单次同步 checkpoint 仍是
  E 类候选，不能把“去掉双写”夸大成完全异步。
- 新增 `--legacy_amp_scale_polling`、`--cpu_concat_before_h2d`、
  `--duplicate_checkpoint_serialization`、`--legacy_loss_item_each_step`
  仅用于在带插桩版本中复现 A1，默认均关闭。
- 诊断窗口结束后清除 worker event，停止 cache hash/stat metadata 插桩与诊断
  NVTX；CSV 异步落盘，避免诊断工具在窗口后持续占用主线程。
- production throughput 的 CUDA Event 数由
  `--throughput_measure_steps`（默认 500）限制，不再随 200 万 step 长训线性增长。
- throughput 将原始 volume、输入 query crop 与实际编码的 masked view 分开计数；
  每个 query crop 生成两个 masked view，因此 `views/s` 使用
  `2 * input_query_crops`，不再把 query crop 数误报成训练 view 数。
- 冷 cache 的 CW0/CW1 差值受 OS page-cache 顺序影响，不能单次运行就称为
  “隔离写入成本”；脚本支持 `CASE_ORDER=CW0_CW1`/`CW1_CW0`，正式归因要求
  新目录反序重复并报告顺序敏感性。

## 当前可以明确回答的八个问题

1. **从哪个 commit 开始？** 首个高置信度边界是 `484c8ee`；精确 A0 若来自
   外部旧副本，仍需保留该副本的 hash/argv。
2. **空泡时 CPU 做什么？** 代码证明至少可能在等 AMP scalar sync；在 100-step
   周期做 torch serialization/磁盘写；在短 epoch 首 batch 等 worker；smoke
   下还在主线程做 NIfTI/MONAI/crop。
3. **是否 DataLoader 断粮？** 新短 dataset 存在明确的周期性 queue re-prime
   机制；是否命中当前每一个观测空泡由 `iterator_wait_time`/epoch boundary
   实测确认。
4. **是否日志/checkpoint？** 无 TensorBoard；print 取决于实际 `log_every`。
   checkpoint 是确定的 E 类周期阻塞候选，尤其 interval=100。
5. **是否 cache/I/O？** 冷 cache 是确定风险；稳定 cache 是否仍受共享盘影响需
   用 path/cache 状态与 iostat 对齐。
6. **是否主线程单核？** smoke 的 workers=0 和 OCL Python mask/crop/CPU cat
   能明确造成单主线程工作；timing/profiler 会区分 B 与 G。
7. **修复后是否持续接近 100%？** 当前环境没有 Torch/MONAI、目标 GPU 和数据，
   尚未运行，不能虚报。
8. **samples/s 提升多少？** 同上；由 `run_ab_matrix.sh` 的同机 50 warmup +
   500 measured step 结果填写。

最终运行判据：若 A6 的 utilization 序列仍有空泡，必须以 slow-step CSV 和
Profiler/Nsight timeline 继续归类，不能用增大 batch 或无意义 CUDA 运算掩盖。
