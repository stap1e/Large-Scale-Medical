# OCL/MiCL 3D 预训练接入说明与下游评测手册

> 审计基准：当前 `main` 的 `HEAD=2cb458f`，并包含 2026-08-06 工作区中尚未提交的 OCL 性能诊断改动。本文会明确区分“已提交代码”和“当前未提交代码”。本文件只说明现状和后续操作，不代表当前 OCL 实现已经通过算法正确性验收。

## 0. 先看结论

本仓库中的 OCL 是一个“参考 OCL/MiCL 思想的 3D 改编版本”：它复用 VoCo 的 3D 数据管线和 `SwinUNETR(use_v2=True)` 编码器，用两次 3D 随机噪声块替换产生视图，再计算 batch 内双向对比损失。上游 `external/OCL` 只用于阅读和比对，训练时不会 import 或执行该子模块中的代码。

正式做下游实验前有四个必须明确的事实：

1. **当前正样本配对代码存在 P0 级语义错误。** `torch.cat([e1, e2], 0).reshape(-1, 2, D)` 在 query 数大于 1 时不会把同一个 query 的两份 masked view 配在一起。必须先修复、增加单测并重新预训练；现有 checkpoint 不能表述为正确的 OCL/MiCL 结果。
2. **下游没有 `--name OCL` 分支。** 直接传 `--name OCL` 会走 `get_model()` 的兜底分支并从头初始化。应给每个选定的下游目录增加 OCL 路由/显式 checkpoint 参数，或暂时用独立 `pretrained_root` 和兼容文件名加载。
3. **推荐的迁移权重是 `encoder_final.pt`。** 它只包含 `raw_model.backbone.state_dict()`，键名是 `swinViT.*`、`encoder1.*` 等，最适合装入下游 MONAI SwinUNETR。
4. **先用 MONAI 分割任务验证。** 当前 OCL 编码器与 `Downstream/monai/*` 的 `SwinUNETR(use_v2=True)` 对齐；nnU-Net 分支是另一套网络和预处理，仓库没有提供 OCL→nnU-Net 的权重转换。

推荐的完整顺序是：

```text
修正正样本配对
  → 配对/梯度/短训单测
  → 固定无下游泄漏的预训练清单
  → OCL 重新预训练并导出 encoder_final.pt
  → 检查下游实际匹配的 key/参数量
  → scratch / 官方 VoCo / OCL 同配置、同 split、同 seed 微调
  → 用最佳验证 checkpoint 做一次最终测试
  → 多 seed 汇总 mean ± std，并报告逐类指标与训练成本
```

## 1. OCL 接入到底加入和修改了哪些文件

### 1.1 首次接入：提交 `1e9e967`

首次 OCL 提交新增了 6 项，没有修改当时已有的 VoCo 文件：

| 文件 | 类型 | 作用 |
|---|---|---|
| `.gitmodules` | 新增 | 声明 `external/OCL` 子模块，URL 为 `git@github.com:stap1e/OCL.git`。 |
| `external/OCL` | 新增 gitlink | 固定到上游提交 `fd2cbdbfa430edc3059a1f3555929bada172ac50`，只作只读算法参考。 |
| `Self-supervised/models/ocl_head.py` | 新增 | 新建 `OCLHead3D`：两视图生成、共享编码、tSP 变换和双向对比损失。 |
| `Self-supervised/ocl_train.py` | 新增 | 仿照 `voco_train.py` 建立独立 OCL 入口，替换 `VoCoHead`，保留数据加载、AMP、DDP、优化器和调度框架。 |
| `Self-supervised/ocl_train.sh` | 新增 | 最小启动器，默认 Base 模型和单进程 `torchrun`。 |
| `dataset_logic.md` | 新增 | 记录原始多数据集管线、采样、crop、cache 等数据逻辑；不参与运行。 |

首次版本的核心改法是：

```python
# ocl_train.py（首次版本的核心思想）
img, _, _ = batch                 # 仍接收 VoCo 的三元组
img = concat_image(img).cuda()    # 只使用 query 分支
loss = model(img)                 # model 从 VoCoHead 换成 OCLHead3D
```

也就是说，首次接入没有重写数据集，而是“保留 VoCo 视图生成，丢弃 base crop 和几何标签，只把 query 交给 OCL head”。

首次版本应视为算法接入原型，而不是当前可恢复训练的完成态：它会把 `--noamp` 再覆盖为 AMP 开启，resume 只恢复模型和 `global_step`，一个 loader pass 内可能越过目标 `num_steps`，最终保存还引用了未定义的 `args.epochs`。此外，早期布尔参数没有严格解析，字符串 `"False"` 仍可能被当成真；这些问题是后续重写训练入口和抽出 `pretrain_common.py` 的直接原因。

### 1.2 CT-RATE、断点续训和编码器导出：提交 `484c8ee` 及后续提交

提交 `484c8ee` 把首次版本从“能跑的独立入口”扩展成可检查、可中断恢复、可导出下游编码器的训练流程：

| 文件 | 变更 | 新增逻辑 |
|---|---|---|
| `Self-supervised/ocl_train.py` | 大幅重写 | 严格 `num_steps`、数据源选择、AMP、完整 resume、周期 checkpoint、最终 encoder 导出。 |
| `Self-supervised/utils/pretrain_common.py` | 新增 | VoCo/OCL 共用参数、布尔解析、loader 路由、checkpoint schema、resume 和 `encoder_final.pt` 导出。 |
| `Self-supervised/utils/data_utils_ctrate_subset.py` | 新增 | 与原始多数据集清单隔离的 CT-RATE subset loader、manifest 校验、MONAI cache 和 CPU 数据检查。 |
| `tools/build_ctrate_subset.py` | 新增 | 按患者确定性采样一个 volume，检查 NIfTI header，输出 datalist、患者清单和跳过报告。 |
| `scripts/train_ctrate_subset_smoke.sh` | 新增 | Stage A/B/C 数据检查、短训、cache、checkpoint 和 resume 冒烟入口。 |
| `Self-supervised/utils/__init__.py` | 修改 | 把数据 loader 改成惰性 import，CT-RATE-only 运行不再解析所有无关 JSON。 |
| `Self-supervised/utils/ops.py` | 修改 | `patch_rand_drop()` 的随机张量改用 `x.device`，避免 DDP 时写死错误 GPU。 |
| `Self-supervised/voco_train.py` | 修改 | 同步接入共享数据参数和 checkpoint/resume 逻辑。 |
| `Self-supervised/README.md` | 修改 | 增加 CT-RATE subset 构建、数据检查和分阶段 smoke 文档。 |

随后 `2c75359` 到 `0009add` 多次演进了 `Self-supervised/cache_warmup.py`；`2cb458f` 为可信本地 MONAI cache 增加了新版 PyTorch `torch.load(weights_only=False)` 兼容处理。

### 1.3 当前工作区尚未提交的性能与数据管线改动

当前工作区中，与 OCL 直接相关的下列内容仍未提交，复现实验时必须把 Git commit 和 dirty 状态一起记录：

| 文件 | 状态 | 当前逻辑 |
|---|---|---|
| `Self-supervised/ocl_train.py` | 修改 | GPU-gap 诊断、生产吞吐窗口、设备端拼接、严格 AMP step、低同步日志、checkpoint 单次序列化及 A/B 开关。 |
| `Self-supervised/models/ocl_head.py` | 修改 | 将编码和 loss 拆成 `encode_views()` / `contrastive_loss()`，增加 NVTX range 和可选双视图合并编码。 |
| `Self-supervised/utils/data_utils.py` | 修改 | 原始 loader 增加 prefetch/pin/persistent worker 和诊断包装。 |
| `Self-supervised/utils/data_utils_ctrate_subset.py` | 修改 | `ContinuousBatchSampler`、只读 cache、完整 batch、诊断 metadata。 |
| `Self-supervised/utils/pretrain_common.py` | 修改 | DataLoader 参数、诊断消融开关、一次序列化两个 checkpoint 名称、可选 encoder 导出。 |
| `Self-supervised/utils/voco_trans.py` | 修改 | OCL 默认跳过从未参与 loss 的 9 个 VoCo base crops。 |
| `Self-supervised/cache_warmup.py` | 修改 | 多 worker 并行预热，父进程丢弃大 tensor payload，只保留 cache 副作用。 |
| `scripts/train_ctrate_subset_smoke.sh` | 修改 | 长于 50 step 时提示不要把同步 smoke 参数当生产性能配置。 |
| `Self-supervised/utils/perf_diagnostics.py` | 未跟踪新增 | CUDA Event、NVTX、worker/cache metadata、slow-step 分类和 profiler trace。 |
| `scripts/train_ctrate_subset_ocl.sh` | 未跟踪新增 | CT-RATE OCL 生产启动器，默认 Base、cache on、16 workers、低频日志。 |
| `perf_regression/*` | 未跟踪新增 | timing 汇总、系统监控、A/B 矩阵、cache/loader sweep 和 view merge 等价性检查。 |

一个部署层面的硬约束是：当前 `ocl_train.py`、`ocl_head.py` 和 loader 已经 import `utils.perf_diagnostics`。如果只提交修改过的文件，却漏掉未跟踪的 `Self-supervised/utils/perf_diagnostics.py`，训练会直接报 `ModuleNotFoundError`。

当前 `perf_regression/` 内文件的用途如下：

| 文件 | 用途 |
|---|---|
| `README.md` | GPU starvation 调查手册。 |
| `analyze_step_timing.py` | 汇总 step CSV 的 median/p95 和 gap 分类。 |
| `aggregate_ab_results.py` | 聚合多个 A/B case。 |
| `monitor_system.sh` | 同步采集 `nvidia-smi dmon`、`pidstat`、`iostat`、`vmstat`。 |
| `run_ab_matrix.sh` | 运行旧行为复现与修复后默认路径的 A1–A6 对照。 |
| `run_cache_write_ablation.sh` | 比较冷 cache 写入、只读 miss 和稳定 hit。 |
| `run_dataloader_sweep.sh` | 搜索 workers/prefetch 组合。 |
| `run_launch_ablation.sh` | 分离 smoke 启动参数造成的性能回归。 |
| `check_ocl_view_merge.py` | 比较两次编码和合并编码的 embedding/loss/gradient/显存；**它不检查正样本配对是否正确**。 |

### 1.4 哪些是运行依赖，哪些只是参考

```text
真正运行：
  ocl_train.py
    ├── models/ocl_head.py
    │   ├── models/voco_head.py::Swin
    │   └── utils/ops.py::patch_rand_drop
    ├── utils/pretrain_common.py
    ├── utils/data_utils.py 或 utils/data_utils_ctrate_subset.py
    ├── utils/data_trans.py / utils/voco_trans.py
    └── utils/perf_diagnostics.py（当前工作区）

只作参考：
  external/OCL
  dataset_logic.md
  perf_regression/git_diff_analysis.md
```

## 2. 当前 OCL 数据流和训练逻辑

### 2.1 端到端调用链

```text
ocl_train.main()
  → build_parser() + args.pretraining_method="ocl"
  → 初始化单卡或 NCCL DDP
  → OCLHead3D(args)
  → get_pretraining_loader(args)
      ├── dataset_mode=original      → utils.data_utils.get_loader
      └── dataset_mode=ctrate_subset → utils.data_utils_ctrate_subset.get_loader
  → MONAI deterministic transforms
  → VoCoAugmentation：S 个 64³ query；OCL 默认不生成 9 个 base crop
  → DataLoader collate
  → query 分别 H2D，再在 GPU 上拼成 [B×S,1,64,64,64]
  → OCLHead3D.forward()
      → _make_view() × 2
      → 共享 Swin encoder × 2（或实验性的 merged call）
      → L2 normalize + tSP + 双向 CE
  → AMP backward / optimizer / scheduler
  → logging / checkpoint / encoder export
```

### 2.2 CT-RATE 的 transform

胸部 transform 位于 `Self-supervised/utils/data_trans.py:122-149`：

```text
LoadImaged("image")
→ EnsureChannelFirstd
→ Orientationd(RAS)
→ Spacingd(1.25, 1.25, 5.0)
→ HU [-1000, 500] 映射到 [0, 1]
→ CropForegroundd
→ SpatialPadd(roi_x, roi_y, roi_z)
→ RandShiftIntensityd(prob=0，当前实际不执行)
→ SpatialPadd(192, 192, 64)
→ RandSpatialCropd(192, 192, 64，固定大小、随机中心)
→ VoCoAugmentation(aug=True)
```

这里的 spacing 和 HU window 是胸部 transform 内的硬编码值，因此 `ocl_train.py` 暴露的 `--space_x/y/z`、`--a_min/a_max/b_min/b_max` 不会改变 CT-RATE chest transform。`--roi_x/y/z` 只影响前面的最小 padding，最终 query 仍由 `VoCoAugmentation` 固定裁成 `64³`。

### 2.3 `VoCoAugmentation` 对 OCL 实际返回什么

`Self-supervised/utils/voco_trans.py:17-50` 保留了 VoCo 的三元组返回契约：

```python
return imgs, labels, crops
```

设原始 volume batch 为 `Bv`，`sw_batch_size=S`（默认 2）：

| 阶段 | shape/内容 |
|---|---|
| 单个样本的 `imgs` | 长度为 `S` 的 list；每项 `image=[1,64,64,64]`。 |
| 单个样本的 `labels` | `[S,9]`，表示 query 和 3×3 base 网格的 XY 重叠；OCL 不使用。 |
| 单个样本的 `crops` | 当前 OCL 默认 `[]`；VoCo 或兼容开关下为 9 个 base crops。 |
| collate 后每个 query view | `[Bv,1,64,64,64]`。 |
| collate 后 labels | `[Bv,S,9]`。 |

query 的 XY 中心随机，Z 中心固定为 32；还会做随机 flip、90° rotate 和 intensity shift。当前优化通过 `skip_unused_ocl_crops=True` 避免生成、collate、IPC 和 pin OCL 从未使用的 9 个 base crops，但仍保留空 list，使训练入口不用改变三元组协议。

### 2.4 H2D 与 query 拼接

`Self-supervised/ocl_train.py:281-312` 先把 list 中每个 query tensor 取出，再分别做 non-blocking H2D，最后在设备端拼接：

```python
moved = [source.to(device, non_blocking=pinned) for source in sources]
output = torch.concatenate(moved, dim=1)
batch_size, view_count, x, y, z = output.shape
image = output.reshape(batch_size * view_count, 1, x, y, z)
```

因此进入 OCL head 的 query 数不是原始 volume batch，而是：

\[
B_q = B_v \times S
\]

默认 `batch_size=4`、`sw_batch_size=2` 时：

```text
list[2] × [4,1,64,64,64]
  → [4,2,64,64,64]
  → [8,1,64,64,64]
```

### 2.5 两个 3D masked view 如何生成

`Self-supervised/models/ocl_head.py:44-51` 对每个 query clone 两次，并分别调用 `patch_rand_drop()`：

```python
out = x.detach().clone()
for i in range(x.size(0)):
    out[i] = patch_rand_drop(self.args, out[i], max_drop=self.args.mask_drop)
```

`Self-supervised/utils/ops.py:17-45` 的具体行为不是“把 voxel 置零”：

1. 目标替换体素量为 `Uniform(0, mask_drop) × H × W × Z`。
2. 循环采样随机 3D cuboid，单边最大约为该维度的 25%。
3. 用随机高斯噪声生成 cuboid，再按该 cuboid 自身的 min/max 归一化到 `[0,1]`。
4. 把归一化噪声写回输入，直到累计 cuboid 体积达到目标。

所以默认 `mask_drop=0.3` 是随机替换量的上界，不是固定替换 30%；采样目标的期望约为 15%，而 cuboid 重叠时独立被替换的 voxel 比例还会更低。

### 2.6 共享 3D SwinUNETR 编码器

`OCLHead3D` 直接复用 `Self-supervised/models/voco_head.py:58-205` 的 `Swin`：

- MONAI `SwinTransformer`，`patch_size=2`、`window_size=7`、depths `[2,2,2,2]`、heads `[3,6,12,24]`、`use_v2=True`。
- 卷积 encoder 使用 InstanceNorm。
- 对五级 feature 分别做 `adaptive_avg_pool3d(1)` 后拼接。

若 `feature_size=F`，拼接维度为：

```text
encoder1:   F
encoder2:   F
encoder3:  2F
encoder4:  4F
encoder10: 16F
总 embedding D = 24F
```

| 模型规模 | `feature_size` | embedding `D` |
|---|---:|---:|
| Base | 48 | 1152 |
| Large | 96 | 2304 |
| Huge | 192 | 4608 |

默认对 `x1`、`x2` 调用同一个 backbone 两次。`--merge_ocl_views` 会先沿 batch 维拼成 `[2Bq,1,64,64,64]`，一次编码后再 `chunk(2)`；该开关默认关闭，只有在目标 GPU 上通过 FP32/AMP、checkpoint 开/关的输出与梯度等价性检查且显存可接受后才应启用。

### 2.7 tSP 和对比损失的设计意图

`Self-supervised/models/ocl_head.py:39-42` 的 tSP 为：

\[
\operatorname{tSP}(s)=
\frac{0.5(1+s)}{1+(1-s)\kappa}\cdot\frac{1}{\tau},
\quad \kappa=\frac{1}{64},\ \tau=0.07
\]

设计意图应是：

```python
e1 = F.normalize(e1, dim=-1)     # [Bq,D]
e2 = F.normalize(e2, dim=-1)     # [Bq,D]
m, n = e1, e2
sim_mn = compute_tSP(m @ n.T)    # [Bq,Bq]
sim_nm = compute_tSP(n @ m.T)    # [Bq,Bq]
target = torch.arange(Bq)
loss = (CE(sim_mn, target) + CE(sim_nm, target)) / 2
```

对角线应表示“同一个 query 的两次 mask”，其余位置是 rank 内负样本。当前没有跨 rank `all_gather`，所以 DDP 只同步梯度，不扩大负样本池；相似度矩阵始终是每个 rank 自己的 `[Bq,Bq]`。

### 2.8 当前训练 step

主循环在 `Self-supervised/ocl_train.py:663-1133`：

1. 保持 DataLoader iterator；连续 sampler 模式下逻辑 epoch 改变但 iterator 不结束。
2. 解包 `(image_views, labels, base_views[, metadata])`。
3. query H2D 并在 GPU 拼接，`optimizer.zero_grad(set_to_none=True)`。
4. AMP forward，得到一个标量 loss。
5. `GradScaler.scale(loss).backward()`，可选 unscale + grad clip。
6. 通过 optimizer post-hook 判断 `scaler.step()` 是否真的调用了 `optimizer.step()`，避免每 step `get_scale()` 造成 host sync。
7. 只有 optimizer update 成功时才推进 scheduler 和 `global_step`；AMP overflow 不占训练步数。
8. loss 默认在 GPU 累加，只在日志点 `.item()`，避免每步 D2H 同步。
9. 到 `eval_num` 时保存 checkpoint；OCL 当前没有 validation hook，`eval_num` 实际是“保存间隔”，不是评估间隔。

默认优化器是：

```python
optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True)
```

这里没有传 `weight_decay=args.decay`。因此默认 `--opt adamw` 时，`--decay 1e-3` 实际被忽略，PyTorch 使用 AdamW 默认 `weight_decay=1e-2`。正式实验要么修正代码显式传入 `args.decay`，要么在论文配置中如实记录实际的 `1e-2`；不能只记录 CLI 表面值。

### 2.9 checkpoint、resume 和最终产物

`Self-supervised/utils/pretrain_common.py:240-417` 定义了统一 schema。完整 checkpoint 默认包含：

```text
format_version
global_step
state_dict              # 完整 OCLHead3D，通常以 backbone.* 开头
optimizer
scheduler
scaler                  # AMP 开启时
pretraining_method
feature_size
num_steps
training_state_included
```

周期保存时先序列化 `model_step{N}.pt`，再通过 hard link（失败时 copy）原子更新 `model_current_epoch.pt`，避免把同一大 checkpoint 序列化两次。

| 产物 | 内容 | 用途 |
|---|---|---|
| `model_current_epoch.pt` | 最新完整训练态。 | `--resume` 默认来源。 |
| `model_step{N}.pt` | 第 N 步归档；默认完整训练态。 | 回滚、预训练 checkpoint sweep。 |
| `model_final_epoch.pt` | 终态完整训练态。 | 归档或继续训练。 |
| `final_model.pt` | 完整 OCL head 的裸 state dict。 | 兼容旧代码，不是首选下游权重。 |
| `encoder_final.pt` | `{"global_step": N, "state_dict": raw_model.backbone.state_dict()}`。 | **首选 MONAI 下游初始化。** |

resume 会恢复模型、optimizer、scheduler 和 scaler，并允许把短 smoke checkpoint 的调度终点改成更大的 `num_steps`。但它没有保存 Python/NumPy/Torch/CUDA RNG、sampler offset 或 DataLoader 预取队列，因此是“优化训练态恢复”，不是逐位一致的 exact continuation。

## 3. 当前实现与上游 OCL 的差异

上游 `external/OCL/models_mae.py` 的 MiCL 逻辑是：2D patch embedding 后随机保留 token，把同一样本保留的 token 序列拆成两半，以 `[N,2,...] → [2N,...]` 的顺序送入共享 ViT，并用两份 CLS token 对比。

当前 3D 改编则是：

| 上游 OCL | 本仓库 3D 改编 |
|---|---|
| 2D ImageNet 图像。 | 3D CT/MRI query crop。 |
| token masking，删除/保留 patch token。 | 完整 voxel 网格上用随机噪声 cuboid 替换。 |
| 一次 mask 后把保留 token 分成两个不相交半序列。 | 对完整 query 独立调用两次 block replacement，两视图可重叠。 |
| ViT CLS token。 | SwinUNETR 五级 feature 的全局池化拼接。 |
| 默认 mask ratio 与当前 3D `mask_drop` 语义不同。 | `mask_drop=0.3` 只是随机替换上限。 |

因此正确表述应是“受 OCL/MiCL 启发的 3D medical adaptation”，而不是“严格复现官方 OCL”。如果研究目标要求严格复现，需要重新设计 3D patch-token masking、视图拆分和表示读取，而不只是修正下述配对 bug。

## 4. 正式训练和下游实验前必须处理的问题

### 4.1 P0：当前正样本配对错误

当前 `Self-supervised/models/ocl_head.py:89-92`：

```python
feats = torch.cat([e1, e2], dim=0)
feats = feats.reshape(-1, 2, feats.shape[-1])
m, n = feats[:, 0], feats[:, 1]
```

若 `Bq=4`，`cat` 后顺序为：

```text
e1_0, e1_1, e1_2, e1_3, e2_0, e2_1, e2_2, e2_3
```

直接 reshape 后实际配成：

```text
(e1_0,e1_1), (e1_2,e1_3), (e2_0,e2_1), (e2_2,e2_3)
```

而正确正对应该是：

```text
(e1_0,e2_0), (e1_1,e2_1), (e1_2,e2_2), (e1_3,e2_3)
```

建议改成下面任一等价形式：

```python
# 最直接
m, n = e1, e2

# 或保留显式 view 维
feats = torch.stack((e1, e2), dim=1)  # [Bq,2,D]
m, n = feats[:, 0], feats[:, 1]
```

修复后至少增加以下测试：

- 用带样本 ID 的人工 embedding 断言 `m[i] == e1[i]` 且 `n[i] == e2[i]`。
- `Bq=1,2,3,4` 都测试，避免只测单样本掩盖错误。
- 对同一预生成 `x1/x2` 比较 separate/merged encoder 的输出、loss 和每个参数梯度。
- 跑 10–100 step，检查 loss 有限、梯度有限、AMP overflow 行为和 checkpoint resume。

**必须在修复后重新预训练。** 修复前产生的权重学到的是另一种错误配对目标，不能直接用于宣称 OCL/MiCL 下游结果。

### 4.2 DDP 负样本只在单卡内

当前没有 embedding `all_gather`。公平比较时必须固定：

- GPU 数；
- 每 rank 的原始 `batch_size`；
- `sw_batch_size`；
- `Bq=batch_size×sw_batch_size`；
- gradient accumulation 规则。

如果以后实现跨卡 gather，应把它作为新的算法版本单独做消融，不能与当前 local-negative 版本混用同一实验名。

### 4.3 同一 volume 的多个 query 会互为负样本

修复正对后，一个 query 的正样本只是它自己的另一份 mask。同一 CT volume 产生的另一个 query 仍在相似度矩阵中，会被当作负样本。CT-RATE manifest 的“一患者一 volume”只防止不同重建作为患者级独立样本，不能消除同 volume query 的 false negative。

应至少做一个消融：

- `sw_batch_size=1`；
- `sw_batch_size=2` 当前逻辑；
- 或在 loss 中屏蔽同 volume 的其他 query。

### 4.4 原始多数据集模式的下游泄漏风险

`dataset_mode=original` 会混合 PreCT-160K 中多种公开数据，部分清单还包含某些下游任务的 validation/test 图像。即使预训练不使用分割标签，严格的独立下游评测仍可能构成图像级 transductive leakage。

正式比较前应做集合交叉检查：用患者 ID、文件名、原始数据 UID（如果有）和必要时影像 hash，确认目标下游 val/test 不在 OCL 预训练清单中。若无法排除，结果必须标为 transductive，不能与完全独立预训练直接比较。CT-RATE-only 的患者去重清单更适合先做干净的工程和方法验证。

### 4.5 其他需要记录或修正的实现点

- L2 normalization 是手写除法，没有 `eps`；建议改用 `F.normalize(..., eps=...)`。
- parser 没有验证 `temperature>0`、`kappa` 合法范围和 `0<=mask_drop<=1`。
- `kappa` help 写成了 softplus，但实际是上述有理函数。
- OCL 没有预训练 validation；预训练 loss 只能看优化是否异常，不能证明表征质量。
- MONAI `PersistentDataset` 不应在 deterministic transform、spacing、HU window 或代码改变后盲目复用旧 cache。最稳妥做法是使用新的 cache 目录。
- 当前性能改动尚未在目标 GPU 上形成已提交、可复核的最终 benchmark；A/B 报告不能替代算法正确性测试。

## 5. 修复后如何运行 OCL 预训练

### 5.1 环境

按仓库要求使用 Python 3.10.13，并保持安装顺序：

训练脚本面向 Linux 服务器环境，使用 bash、`resource`、CUDA/NCCL 和 Linux 风格路径；Windows 原生 PowerShell 不能按这些命令直接启动训练，Windows 开发机应使用配置好 GPU/CUDA 的 WSL2 或远端 Linux 机器。

```bash
conda create -n voco-ocl python=3.10.13 -y
conda activate voco-ocl

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

若要查看上游参考实现，初始化子模块：

```bash
git submodule update --init external/OCL
```

`.gitmodules` 使用 SSH URL；机器没有 GitHub SSH key 时，需要自行把该子模块 URL 临时改成 HTTPS 或配置 SSH。OCL 训练本身不依赖子模块，可以不初始化它。

### 5.2 准备 CT-RATE subset

仓库没有提供 CT-RATE 原始数据的一键下载脚本；先按数据集许可从官方来源取得 NIfTI 文件，并组织为：

```text
/data/CT_RATE/train/
  train_1/**/*.nii.gz
  train_2/**/*.nii.gz
  ...
```

再生成确定性、一患者一 volume 的 datalist：

```bash
cd /path/to/Large-Scale-Medical
python tools/build_ctrate_subset.py \
  --data_root /data/CT_RATE/train \
  --output_json /data/CT_RATE/ct_rate_subset_256.json \
  --num_patients 256 \
  --seed 2026 \
  --selection_mode one_volume_per_patient \
  --validate_header
```

输出包括：

- `ct_rate_subset_256.json`：MONAI Decathlon 风格训练清单；
- `ct_rate_subset_256_patients.txt`：被选患者；
- `ct_rate_subset_256_skipped.json`：空文件、坏 header 等跳过原因。

工具默认拒绝覆盖已有输出；只有明确确认旧清单可替换时才使用 `--overwrite`。

### 5.3 先做 CPU 数据检查

```bash
cd Self-supervised
python ocl_train.py \
  --dataset_mode ctrate_subset \
  --data_root /data/CT_RATE/train \
  --datalist_json /data/CT_RATE/ct_rate_subset_256.json \
  --cache_dir /cache/ctrate_subset_256 \
  --batch_size 1 \
  --workers 0 \
  --no-cache \
  --data_check_only
```

当前 OCL 默认应看到：

```text
random crops: [B×S, 1, 64, 64, 64]
labels:       [B, S, 9]
base crops:   omitted (unused by OCL)
```

### 5.4 预热 cache

生产训练前用与训练完全相同的 `data_root`、datalist 和 `cache_dir`：

```bash
cd Self-supervised
python cache_warmup.py \
  --dataset_mode ctrate_subset \
  --data_root /data/CT_RATE/train \
  --datalist_json /data/CT_RATE/ct_rate_subset_256.json \
  --cache_dir /cache/ctrate_subset_256 \
  --workers 8
```

cache 中含 MetaTensor/NumPy 对象，只能加载自己生成并信任的 cache；不要对来源不明的 `.pt` 使用 `weights_only=False`。

### 5.5 冒烟测试

```bash
cd /path/to/Large-Scale-Medical
DATA_ROOT=/data/CT_RATE/train \
DATALIST_JSON=/data/CT_RATE/ct_rate_subset_256.json \
CACHE_DIR=/cache/ctrate_subset_256 \
GPU_ID=0 NUM_STEPS=10 BATCH_SIZE=1 WORKERS=0 CACHE=0 \
LOGDIR="$PWD/Self-supervised/runs/ocl_smoke" \
bash scripts/train_ctrate_subset_smoke.sh
```

该脚本默认 `sync_timing=1`、`workers=0`、高频日志，只用于验证路径/shape/forward/backward，不用于评估 GPU 利用率或正式吞吐。

### 5.6 Base 模型生产训练

当前未提交的生产脚本默认 Base (`feature_size=48`)：

```bash
cd /path/to/Large-Scale-Medical
export DATA_ROOT=/data/CT_RATE/train
export DATALIST_JSON=/data/CT_RATE/ct_rate_subset_256.json
export CACHE_DIR=/cache/ctrate_subset_256
export GPU_IDS=0,1
export NPROC_PER_NODE=2
export MASTER_PORT=28814
export BATCH_SIZE=4
export WORKERS=16
export PREFETCH_FACTOR=2
export NUM_STEPS=2000000
export LOGDIR="$PWD/Self-supervised/runs/ctrate_ocl_B"
bash scripts/train_ctrate_subset_ocl.sh
```

`GPU_IDS` 中可见 GPU 数必须与 `NPROC_PER_NODE` 一致。该脚本没有转发 `FEATURE_SIZE`，所以 Large/Huge 应修改脚本或直接执行 `torchrun` 并显式传 `--feature_size 96/192`。

### 5.7 恢复训练

`num_steps` 是目标总步数，不是“再训练多少步”：

```bash
cd Self-supervised
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=28814 \
  ocl_train.py \
  --dataset_mode ctrate_subset \
  --data_root /data/CT_RATE/train \
  --datalist_json /data/CT_RATE/ct_rate_subset_256.json \
  --cache_dir /cache/ctrate_subset_256 \
  --cache --batch_size 4 --workers 16 \
  --feature_size 48 \
  --num_steps 2000000 \
  --eval_num 20000 \
  --logdir runs/ctrate_ocl_B \
  --resume
```

裸 `--resume` 会读取 `<logdir>/model_current_epoch.pt`；也可传明确路径。恢复时必须保持模型规模、输入通道和算法结构一致。

## 6. 下游任务：从数据下载到最终对比

### 6.1 先选哪套下游实现

| 路线 | 是否推荐用于第一轮 OCL 评估 | 原因 |
|---|---|---|
| `Downstream/monai/<Dataset>` | **推荐** | 与 OCL 都使用 MONAI `SwinUNETR(use_v2=True)`；`encoder_final.pt` 的 key/shape 可直接做尺寸匹配加载。 |
| `Downstream/nnUNet` | 暂不推荐 | 使用 nnU-Net plans 驱动的 CNN/ResidualEncoderUNet 和独立预处理；仓库只提供 `VoComni_nnunet.pt` 路径，没有 OCL 转换。 |
| 分类/配准/VLP 特殊目录 | 第二阶段 | `CC-CCII`、`LUNA16`、`Registration`、`M2KT`、`CT_CLIP` 都有独立模型、数据 schema 和指标，不能照搬通用分割步骤。 |

建议先选 3–5 个代理任务：

- `3D-IRCADb`：小规模 CT 肝/肿瘤分割，适合先验证权重接入；固定 JSON 为 15 train / 5 validation，3 类。
- `BTCV`：CT 多器官分割，14 类，常用主指标。
- `Spleen`：二类 CT 分割，观察简单器官任务迁移。
- `Amos`：16 类 CT 多器官，更大规模；默认 1000 epoch。
- `ACDC`：4 类 MRI 心脏分割，用于观察 CT 预训练的跨模态迁移，不能替代 CT 主结果。

先在这些任务上做 checkpoint sweep；只有方法正确且趋势稳定后，再扩到 50+ 任务。

### 6.2 下载下游数据

仓库文档提供的预处理数据集合是：

```text
https://huggingface.co/datasets/Luffy503/VoCo_Downstream
```

安装/登录 Hugging Face CLI 后，可以下载整个仓库：

```bash
pip install -U huggingface_hub
hf auth login
hf download Luffy503/VoCo_Downstream \
  --repo-type dataset \
  --local-dir /data/VoCo_Downstream
```

正式实验应把下载时的 HF revision/commit 一并记录；仅写数据集名称而不固定版本，后续无法确认 split 和文件是否变化。

整个集合很大，建议先在 Hugging Face `Files and versions` 中确认实际目录/压缩包名称，再按 pattern 只下载目标数据。例如仓库仍以 `3Dircadb1_convert/` 目录发布时：

```bash
hf download Luffy503/VoCo_Downstream \
  --repo-type dataset \
  --include "3Dircadb1_convert/**" \
  --local-dir /data/VoCo_Downstream
```

下载后，普通单模态分割数据应类似：

```text
/data/VoCo_Downstream/3Dircadb1_convert/
  imagesTr/
    1_0000.nii.gz
    ...
  labelsTr/
    1.nii.gz
    ...
```

注意：

- 仓库不是这些医学数据的作者，必须遵守原始许可并引用原论文。
- WORD 等数据需要向作者申请，不能用脚本绕过审批。
- `Downstream/README.md` 没有给每个原始数据集的完整下载命令；若 HF 中没有目标数据，应按该任务官方来源下载并转换成目录内 JSON 要求的布局，不能猜 URL。
- 下载完成后要核对 `imagesTr/*_0000.nii.gz` 与 `labelsTr/*.nii.gz` 一一对应、label ID 范围等于 `0..out_channels-1`、JSON 中每条路径存在。

不同目录对 `json_list` 的拼接方式并不完全相同：有的直接打开 `args.json_list`，有的会执行 `os.path.join(data_dir, json_list)`。最稳妥的做法是先阅读该任务的 `utils/data_utils.py`，并给 `--json_list` 传它所期待位置的真实文件；不能假设所有 JSON 都放在代码目录。仓库中还存在默认文件名与实际文件名不一致的目录，例如 SegThor 默认写 `dataset_segthor.json`，代码目录中却是 `dataset.json`，运行前必须按下载包修正。

### 6.3 准备 OCL、VoCo 和 scratch 三组权重

下游普通分割目录没有 OCL 路由。以 `3D-IRCADb` 为例，`models/__init__.py` 只识别 `VoCo/suprem/swin/clip_driven/mg/unimiss/dodnet`；`models/models.py::VoCo` 又按规模硬编码：

```text
feature_size=48  → VoCo_B_SSL_head.pt
feature_size=96  → VoCo_L_SSL_head.pt
feature_size=192 → VoCo_H_SSL_head.pt
```

推荐的长期做法是在**每个选定数据集自己的目录**增加：

- `--pretrained_path` 参数；
- `name == "OCL"` 分支；
- 构建与 VoCo 分支相同的 `SwinUNETR(use_v2=True)`；
- 从 `pretrained_path` 加载 `encoder_final.pt`；
- 输出实际命中的 key 数和参数量。

因为这些数据集目录是彼此独立的，改一个目录不会自动影响其他目录。

第一轮不改模型路由时，可以使用兼容文件名，但必须用独立目录，绝不能覆盖官方 VoCo：

```bash
mkdir -p /pretrained/official_voco_B /pretrained/ocl_B

# 官方 VoCo checkpoint
cp /download/VoCo_B_SSL_head.pt \
   /pretrained/official_voco_B/VoCo_B_SSL_head.pt

# OCL 兼容别名；encoder_final.pt 内仍有可识别的 state_dict
cp /path/to/ocl_run/encoder_final.pt \
   /pretrained/ocl_B/VoCo_B_SSL_head.pt

sha256sum /pretrained/official_voco_B/VoCo_B_SSL_head.pt \
          /pretrained/ocl_B/VoCo_B_SSL_head.pt
```

三组实验对应：

| 实验 | `name` | `pretrained_root` | 说明 |
|---|---|---|---|
| Scratch | 任意未识别值，例如 `None` | 不使用 | `get_model()` 兜底创建随机 SwinUNETR。 |
| 官方 VoCo-B SSL | `VoCo` | `/pretrained/official_voco_B` | 加载官方 `VoCo_B_SSL_head.pt`。 |
| OCL-B | `VoCo`（仅兼容路由） | `/pretrained/ocl_B` | 文件名兼容，但实验名称和日志必须明确写 OCL；长期应改成显式 `name=OCL`。 |

**不要直接传 `--name OCL`，当前代码会变成 scratch。**

也不要把 OCL encoder 塞给下游的 `--checkpoint` 来代替预训练 loader。普通 `main.py` 中的 `--checkpoint` 语义是加载已经微调过的完整 SwinUNETR，并恢复 `epoch/best_acc`；它不是稳定统一的 encoder 初始化入口，而且现有实现通常不恢复 checkpoint 中的 optimizer/scheduler。预训练权重应走 `get_model()`/`models.py::load()`，下游训练 resume 则单独处理。

### 6.4 必须验证实际加载了多少权重

普通下游 `models/models.py::load()` 会：

1. 从 `state_dict/network_weights/net/student` 中取权重；
2. 去掉 `module.`、`backbone.`，把 `swin_vit` 改成 `swinViT`；
3. 只复制“同名且 shape 相等”的项；
4. 其余项用模型当前随机值补齐；
5. 对补齐后的完整 dict 调 `strict=True`。

所以日志中的 `strict=True` **不等于预训练权重全部命中**。正式实验应在每个下游 loader 中加入统计：

```python
current = model.state_dict()
matched = {
    k: state_dict[k]
    for k, value in current.items()
    if k in state_dict and state_dict[k].shape == value.shape
}
matched_numel = sum(current[k].numel() for k in matched)
total_numel = sum(value.numel() for value in current.values())
print(f"matched keys: {len(matched)}/{len(current)}")
print(f"matched params: {matched_numel}/{total_numel} ({matched_numel/total_numel:.2%})")
assert any(k.startswith("swinViT.") for k in matched)
assert any(k.startswith("encoder1.") for k in matched)
```

预期 decoder 和任务输出头随机初始化；`feature_size` 不一致会造成大量 backbone shape 不匹配，必须直接失败而不是静默继续。

### 6.5 以 3D-IRCADb 为例配置微调

该目录当前默认配置来自 `Downstream/monai/3D-IRCADb/main.py`：

| 配置 | 默认值 |
|---|---:|
| `in_channels/out_channels` | `1 / 3`（background、liver、liver tumor） |
| `feature_size` | 48 |
| spacing | `(1.5,1.5,1.5)` |
| HU window | `[-175,250] → [0,1]` |
| ROI | `(96,96,96)` |
| batch / sliding-window batch | `1 / 4` |
| optimizer / LR / weight decay | AdamW / `3e-4` / `1e-5` |
| epochs / validation interval | `500 / 10` |
| inference overlap | `0.75` |
| loss / metric | DiceCE / mean Dice（不含 background） |

修改 `Downstream/monai/3D-IRCADb/train.sh` 中的路径和端口：

```bash
name=VoCo
pretrained_root=/pretrained/ocl_B
logdir=runs/ocl_B_seed2026
feature_size=48
data_dir=/data/VoCo_Downstream/3Dircadb1_convert
cache_dir=/cache/downstream/3dircadb_v1
use_ssl_pretrained=True
use_persistent_dataset=True
```

当 scratch、VoCo 和 OCL 的 deterministic data transform 完全一致时，三组应使用同一份已经预热并冻结的 cache（例如统一改成 `/cache/downstream/3dircadb_v1`），或分别使用内容等价且都预热完成的 cache；不要让某一方法承担冷 cache 写入而其他方法只读热 cache。若 transform 有任何变化，则使用新的带版本目录，不能继续共享旧 cache。

然后：

```bash
cd Downstream/monai/3D-IRCADb
source activate voco-ocl
sh train.sh
```

每个并行作业必须使用不同 `master_port` 和 `logdir`。目录内 `main.py` 还硬编码了 `CUDA_VISIBLE_DEVICES`；应删除硬编码、让外部 launcher 控制，或改成目标 GPU。当前 `train.sh` 的 `torchrun` 没有指定 `nproc_per_node`，实际是单进程启动。

部分 `main.py --distributed` 又会在进程内部 `mp.spawn` 所有可见 GPU，因此不要同时用多进程 `torchrun` 和内部 spawn，避免重复启动。先按该任务现有入口选择一种 DDP 方式并做两卡 smoke。另有不少 `trainer.py` 用 `len(loader)//4` 作为打印间隔；极小 smoke 数据集若少于 4 个 batch 会发生取模除零，需要先把间隔改成 `max(1, len(loader)//4)`。

### 6.6 先修复下游布尔参数陷阱

许多普通分割 `main.py` 写的是：

```python
parser.add_argument("--use_ssl_pretrained", default=True)
parser.add_argument("--use_persistent_dataset", default=True)
```

而 `train.sh` 传 `--use_ssl_pretrained False`。argparse 会得到字符串 `"False"`，它在 Python `if` 中仍为真，因此脚本表面写 False 时可能仍加载 SSL，`use_persistent_dataset=False` 也有同样问题。

为保持当前 shell 写法，建议统一加入严格布尔解析：

```python
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected boolean")

parser.add_argument("--use_ssl_pretrained", type=str_to_bool, default=True)
parser.add_argument("--use_persistent_dataset", type=str_to_bool, default=True)
```

修复后再运行 VoComni、SSL、cache on/off 对照；否则实验标签可能与实际加载的权重不一致。

### 6.7 加入可复现 seed

普通 MONAI 分割入口目前普遍没有 `--seed`，且启用了 `cudnn.benchmark=True`。正式多 seed 对比前，在每个选定目录中统一加入：

```python
from monai.utils import set_determinism

parser.add_argument("--seed", type=int, default=2026)

# 必须在构建 dataset、model 和 DataLoader 之前执行
set_determinism(seed=args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

DDP 时应在 `main_worker` 已确定 rank 后设置可复现的 rank/worker seed（例如数据随机性使用 `args.seed + rank`），并保证每种方法采用同一规则；不能只在父进程设一次 seed 就假定所有 worker 可复现。模型参数会由 DDP 同步，但数据 augmentation、sampler 和 worker RNG 仍需单独核对。

建议 seeds 为 `2026, 2027, 2028`，所有方法使用同一组 seed。若为了速度选择非确定性 cuDNN，也要保持三组完全相同并在报告中声明；至少保存实际 command、commit、CUDA/cuDNN 版本和数据 split hash。

### 6.8 训练、选择 checkpoint 和验证

普通分割调用链是：

```text
main.py
  → get_loader：train/validation JSON + transform + Dataset/PersistentDataset
  → get_model：构建网络并加载预训练 encoder
  → DiceCELoss + DiceMetric
  → trainer.run_training
      → train_epoch（AMP）
      → 每 val_every 个 epoch 做整卷 sliding-window validation
      → 若 mean Dice 提升，保存/复制 best 为 model.pt
      → 最近一次验证保存为 model_final.pt
```

用于最终比较的是 `model.pt`（最佳验证 Dice），不是简单取最后一个 epoch。

仓库根部 `Downstream/val.py` 和 `test.py` 是需要复制/编辑的模板，不是所有数据集可直接运行的统一入口。必须把以下参数改成与训练一致：

- `test_data_path`；
- `test_label_path`（只有 val）；
- `trained_pth`；
- `feature_size`、`in_channels`、`out_channels`；
- HU window、spacing、ROI、sliding-window batch 和 overlap；
- GPU；
- image→label 文件名映射。

同理，根目录的 `Downstream/train.sh` 不是一个可以在 `Downstream/` 直接执行的统一训练器；普通任务必须进入自己的 `Downstream/monai/<Dataset>/`，让本地 `models`、`utils` 和相对 JSON import 正确解析。

`val.py` 默认用 `name[:-12] + '.nii.gz'` 把 `case_0000.nii.gz` 映射到 `case.nii.gz`。数据不遵循 `_0000.nii.gz` 命名时必须修改。它还会扫描整个 `test_data_path`；以 3D-IRCADb 为例，若直接指向包含 20 例的 `imagesTr`，就会把训练 15 例也算进去。应让它读取 JSON 的 `validation` 列表，或建立只含 15–19 的 validation-only 目录。

`val.py/test.py` 对微调后的完整模型使用 `strict=True`，因此评测时的 `feature_size/out_channels/use_v2` 必须与训练完全一致。

MRI 任务不能机械套用 CT 模板。ACDC、AMOS-MRI、ATLAS-MRI、BRATS、BRATS21、Heart、Hip、MM-WHS-MRI、Prostate 等目录的实际 loader 常使用 `NormalizeIntensityd(nonzero=True, channel_wise=True)`，有些也不执行 `Spacingd`；应以该目录的真实 train/validation transform 为准，并把相同逻辑复制到评测脚本。BRATS/Prostate 等多通道任务还需要按训练方式组装多模态输入，通用单通道 `val.py` 不能直接使用。

### 6.9 公平比较矩阵

最小可信矩阵：

| 方法 | encoder 初始化 | decoder/head 初始化 | 其余设置 |
|---|---|---|---|
| Scratch | 随机 | 随机 | 与其他组相同 |
| Official VoCo-B SSL | 官方 `VoCo_B_SSL_head.pt` | 随机 | 与其他组相同 |
| OCL-B | 修复后重训的 `encoder_final.pt` | 随机 | 与其他组相同 |

必须固定：

- 同一 train/validation/test split，且 test 不参与调参；
- 同一 `feature_size=48`、`use_v2=True`、输入/输出通道；
- 同一下游预处理、ROI、augmentation、loss、optimizer、LR schedule、epoch、batch、验证频率和 early-selection 规则；
- 同一 GPU 数和有效 batch；
- 同一 seed 集合；
- 同一数据 cache 语义；修改 deterministic transform 后换 cache 目录；
- 每种方法都用验证集选择最佳 checkpoint，并只在最终配置确定后跑一次 test。

SuPreM、Swin 等对比分支在部分 `models.py` 中使用 `use_v2=False`，而 VoCo/OCL 使用 `use_v2=True`；这不再是“只改变初始化权重”的纯对照。主因果表应优先放 scratch、同架构官方 VoCo 和 OCL；异架构方法可以作为外部基线单列，并明确架构差异。

不要比较预训练 loss 的绝对值：VoCo 和 OCL 的 loss 定义、负样本数和尺度都不同。预训练质量的主结论来自相同下游协议下的迁移指标。

### 6.10 指标与结果表

普通分割至少报告：

- 每个类别 Dice（不含 background）；
- macro mean Dice；
- 多 seed mean ± std；
- 如果任务关注小病灶，再报告 HD95/NSD 等任务官方指标；当前通用 trainer 默认只有 Dice，需要新增同一实现给所有方法；
- 参数量、峰值显存、训练时长或 GPU-hours；
- 实际匹配的预训练 key 数/参数比例。

推荐结果表：

| Dataset | Method | Seed | Best val Dice | Test mean Dice | Per-class Dice | Matched encoder params | GPU-hours |
|---|---|---:|---:|---:|---|---:|---:|
| 3D-IRCADb | Scratch | 2026 |  |  |  | 0 |  |
| 3D-IRCADb | VoCo-B SSL | 2026 |  |  |  |  |  |
| 3D-IRCADb | OCL-B | 2026 |  |  |  |  |  |

多 seed 汇总时同时保留逐 seed 行，不能只给最好的一次。若只有 validation split 没有独立 test，应把列明确写成 validation，不能称 test performance。

### 6.11 OCL 预训练 checkpoint sweep

因为 OCL 没有内置 validation，判断“预训练到哪一步最好”的可靠方法是下游代理任务：

1. 固定保存间隔，例如每 20k/100k step。
2. 选 3D-IRCADb、BTCV、Spleen 等 2–3 个代理任务。
3. 对 `model_step{N}.pt` 使用同一个下游短协议微调；该 checkpoint 的 `state_dict` 带 `backbone.` 前缀，现有 loader 会去前缀。
4. 画 `下游 Dice vs 预训练 step` 曲线。
5. 性能稳定进入 plateau 后再选择候选 step，并在完整任务/完整 epoch 上确认。

不能用 `data_check_only`、10-step smoke、GPU 利用率或预训练 loss 最低点替代下游表征评估。

### 6.12 推荐消融

在主对比完成后，逐项只改变一个因素：

| 消融 | 目的 |
|---|---|
| `mask_drop` | 比较 0.1/0.3/0.5，但要记录实际独立 voxel 替换比例。 |
| `temperature`、`kappa` | 检查 tSP 形状与 logit 尺度。 |
| `sw_batch_size=1/2` | 检查同 volume query false negative。 |
| local negatives / cross-rank gather | 检查负样本池；作为不同算法版本。 |
| separate / merged views | 只在数值与梯度等价通过后比较吞吐和显存。 |
| 预训练数据量 | 固定患者级清单，比较 256、2k、全量；避免数据量和算法一起变化。 |
| 冻结 encoder / 全量 fine-tune | 区分线性可分性和完全适配能力。 |
| 25%/50%/100% labels | 检查低标注场景收益。 |

## 7. nnU-Net 和特殊任务如何处理

`Downstream/nnUNet` 当前示例命令是：

```bash
nnUNetv2_plan_and_preprocess -d <DATASET_ID> -c 3d_fullres \
  --verify_dataset_integrity
nnUNetv2_train <DATASET_ID> 3d_fullres <FOLD> -tr nnUNetTrainer_pre
```

但 `nnUNetTrainer_pretrain.py` 加载的是 nnU-Net 架构对应的 `VoComni_nnunet.pt`，不是 MONAI SwinUNETR encoder。`encoder_final.pt` 不能因为都是 3D 模型就直接载入。若要做 OCL→nnU-Net，需要先明确：

- 下游网络是否改成 SwinUNETR，或设计 Swin→CNN 的权重映射；
- key、shape 和功能层的映射；
- nnU-Net 自己的 spacing/crop/normalization；
- 与原 nnU-Net baseline 完全相同的 plans 和 folds。

在仓库提供转换并通过权重命中测试前，不应把 nnU-Net 结果列为 OCL encoder 的直接迁移结果。

公平的 nnU-Net 泛化评估应运行固定 folds（通常 0–4）并汇总；仓库 README 中的 `fold=all` 示例会把全部病例用于训练/验证，适合特定全量训练用途，不应当作独立验证结果。最终可用 `--val --val_best` 验证每个 fold 的最佳 checkpoint。该 fork 的 `paths.py` 还硬编码了 `/data/nnUNet_raw`、`/data/nnUNet_preprocessed`、`/data/nnUNet_results`，运行前需与实际存储一致。

分类、配准、VLP 也不是通用脚本：

| 任务 | 目录 | 典型指标 | 额外工作 |
|---|---|---|---|
| 分类 | `CC-CCII`、`LUNA16` | accuracy、AUC | 给分类 backbone 增加 OCL 权重路由，固定 fold。 |
| 配准 | `Registration` | warped Dice、TRE | 当前 TransMorph 系与 OCL SwinUNETR 不是直接同构。 |
| 报告生成 | `M2KT` | BLEU、CIDEr 等 | 需定义 OCL volume encoder 如何接入语言模型。 |
| 检索/词汇分类 | `CT_CLIP` | Recall@K、分类指标 | 需保留 CT-文本配对 split，防止患者/报告泄漏。 |

这些特殊目录当前还有不少任务级配置需要先修：例如 CC-CCII 的训练代码使用了未在 parser 中定义的 `pretrained_path`，LUNA16/Registration 有作者机器的绝对路径，M2KT 启动脚本也需按实际 shell 语法和数据路径复核。普通分割也不是全部开箱即用：PENGWIN 的 `main.py` 默认 `out_channels=31`，而仓库 JSON 标签表只列 4 类，必须依据实际预处理 label 的 unique values 决定配置，不能直接猜一个值运行。

## 8. 实验记录和验收清单

每次预训练和下游运行至少保存：

- Git commit、`git status --short` 和 diff/patch；
- 完整 argv 与 shell 脚本；
- `effective_config.json`；
- Python、PyTorch、CUDA、cuDNN、MONAI 版本和 GPU 型号；
- 数据清单、患者列表和 SHA-256；
- 预训练 checkpoint SHA-256；
- 下游 JSON/fold、seed、实际权重匹配统计；
- TensorBoard/log、best checkpoint、逐病例/逐类指标；
- 训练时长、峰值显存和异常/重启记录。

正式发布 OCL 下游结果前逐项确认：

- [ ] 已修复 `cat + reshape` 正样本配对并加入 `Bq>1` 单测。
- [ ] 已重新预训练，没有复用修复前 checkpoint。
- [ ] 预训练 train 与下游 val/test 做过患者/影像去重。
- [ ] AdamW 实际 weight decay 与记录一致。
- [ ] 每 rank query 数和负样本策略固定。
- [ ] `encoder_final.pt` 的 `feature_size` 与下游一致。
- [ ] 下游显式识别 OCL，或兼容别名位于独立目录且 hash 已记录。
- [ ] 打印并断言了匹配 key/参数量，没有静默近似全随机初始化。
- [ ] 修复了下游字符串布尔参数问题。
- [ ] 删除/修正了硬编码 GPU，端口和 logdir 不冲突。
- [ ] scratch、VoCo、OCL 使用同一 split、同一配置和同一 seed 集合。
- [ ] 使用 best validation checkpoint；test 只在方案冻结后评估。
- [ ] 报告逐 seed、mean ± std、逐类指标和训练成本。
- [ ] 没有把 smoke、吞吐、预训练 loss 或 validation-only 结果误写成独立 test 结论。

## 9. 关键代码索引

| 内容 | 文件位置 |
|---|---|
| OCL 参数、训练入口 | `Self-supervised/ocl_train.py:63-244,453-1244` |
| query 解包、H2D、拼接 | `Self-supervised/ocl_train.py:272-312,709-778` |
| 严格 AMP step | `Self-supervised/ocl_train.py:780-840` |
| checkpoint 触发 | `Self-supervised/ocl_train.py:907-946` |
| OCL head、tSP、loss | `Self-supervised/models/ocl_head.py:25-131` |
| 3D 噪声块替换 | `Self-supervised/utils/ops.py:17-45` |
| Swin backbone 和 pooled embedding | `Self-supervised/models/voco_head.py:58-205` |
| loader 路由和共享数据参数 | `Self-supervised/utils/pretrain_common.py:80-223` |
| checkpoint schema、resume、encoder 导出 | `Self-supervised/utils/pretrain_common.py:229-417` |
| CT-RATE manifest、cache、loader | `Self-supervised/utils/data_utils_ctrate_subset.py:102-278` |
| CT-RATE 数据检查 | `Self-supervised/utils/data_utils_ctrate_subset.py:315-393` |
| 连续 batch sampler | `Self-supervised/utils/data_utils_ctrate_subset.py:48-100` |
| chest transform | `Self-supervised/utils/data_trans.py:122-149` |
| query/base crop 逻辑 | `Self-supervised/utils/voco_trans.py:17-151` |
| subset 构建器 | `tools/build_ctrate_subset.py` |
| smoke/生产启动 | `scripts/train_ctrate_subset_smoke.sh`、`scripts/train_ctrate_subset_ocl.sh` |
| 下游总说明 | `Downstream/README.md` |
| 3D-IRCADb 模型加载 | `Downstream/monai/3D-IRCADb/models/models.py:11-45,152-190` |
| 3D-IRCADb 训练配置 | `Downstream/monai/3D-IRCADb/main.py`、`train.sh`、`trainer.py` |
| 通用验证/测试模板 | `Downstream/val.py`、`Downstream/test.py` |
