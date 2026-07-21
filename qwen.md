# Large-Scale-Medical (VoCo) 代码文件作用汇总

本文按目录逐文件说明各代码文件的作用。仓库是 **VoCo**（大规模 3D 医学影像预训练 + 50+ 下游任务）的研究代码，五个顶层实验目录相互独立、各自自包含，没有共享包或 monorepo 工具。数据管线的细节（shape、清单契约、缓存、采样）详见同目录的 `dataset_logic.md`，本文侧重“每个文件干什么”。

---

## 1. 根目录

| 文件/目录 | 作用 |
|---|---|
| `README.md` | 项目主文档：方法介绍、预训练模型表（31M~1.2B）、加载预训练权重的示例代码（尺寸检查的 strict load）、安装与微调/预训练入口指引、引用 BibTeX。 |
| `AGENTS.md` | 给 AI 助手的仓库导航说明：仓库结构、环境、启动脚本、模型/检查点约定、数据存储注意事项。 |
| `dataset_logic.md` | 详尽的数据集加载与调度逻辑文档（五个目录的 manifest、预处理、缓存、采样、batch 结构、已知实现问题）。 |
| `qwen.md` | 本文件：代码文件作用汇总。 |
| `requirements.txt` | conda `--file` 导出（linux-64）的完整依赖清单。关键固定版本：`python==3.10.13`、`monai==1.3.0`、`numpy==1.26.2`、`transformers==4.24.0`、`batchgenerators`、`dynamic-network-architectures`。注意 torch 需按 README 单独用 cu118 源安装。 |
| `LICENSE` | 开源许可证。 |
| `.gitignore` | Git 忽略规则。 |
| `.gitmodules` | 声明子模块 `external/OCL`（指向 `git@github.com:stap1e/OCL.git`）。 |
| `download.sh` | 用 `hf` CLI 断点续传式下载 PreCT-160K 数据集（hf-mirror 镜像、失败后 300s 重试循环）。路径硬编码到 `/home/bld/data/...`。 |
| `assets/` | README 引用的图片（如 `data.svg`）。 |
| `.agents/`、`.claude/` | AI 工具配置/技能目录，与训练代码无关。 |

---

## 2. Self-supervised/ —— VoCo / OCL 自监督预训练

入口链路：`single_train.sh` / `dist_{B,L,H}.sh` → `voco_train.py` → `utils.data_utils.get_loader()` → 腹/头颈/胸三个区域 Dataset 合并成全局 `ConcatDataset` → VoCoHead。

### 启动脚本
| 文件 | 作用 |
|---|---|
| `single_train.sh` | 单进程启动（每 GPU volume batch=4，Base 默认 feature_size=48）。 |
| `dist_B.sh` / `dist_L.sh` / `dist_H.sh` | 8 进程分布式（`torchrun`），分别对应 Base(48)/Large(96)/Huge(192)，每 rank batch=2。 |
| `ocl_train.sh` | OCL（MiCL）方法的启动脚本。 |
| `README.md` | 本目录的预训练说明文档。 |

### 训练入口
| 文件 | 作用 |
|---|---|
| `voco_train.py` | **VoCo 预训练主入口**。解析参数、构建 `VoCoHead`、`WarmupCosineSchedule`、AMP、可选 DDP，循环遍历 loader 直到 `num_steps`（默认 2,000,000），定期存 checkpoint。 |
| `ocl_train.py` | **OCL（MiCL）预训练入口**。复用同一数据管线（`get_pretraining_loader`）和 backbone，仅把自监督算法逻辑换成 `OCLHead3D`（掩码图像对比学习）。 |

### models/
| 文件 | 作用 |
|---|---|
| `__init__.py` | 包标记。 |
| `voco_head.py` | **VoCo 核心模型**：3D SwinUNETR backbone（`Swin`）+ projection_head + 三类损失（query-base 位置预测、跨 volume inter-volume loss、base 去相关正则）。也提供 nnUNet 风格 PlainConvUNet 构建。 |
| `ocl_head.py` | **OCL/MiCL 预训练头**：复用 `voco_head.Swin` backbone，按 external/OCL 的 MiCL 算法（随机掩码生成两视图→同编码器→InfoNCE 风格对比）实现，不修改 VoCo 原文件。 |
| `PlainConvUNet_load.py` | 构建 plain nnUNet 风格 UNet，用于加载 nnUNet 格式权重。 |

### optimizers/
| 文件 | 作用 |
|---|---|
| `__init__.py` | 包标记。 |
| `lr_scheduler.py` | 学习率调度器，含 `WarmupCosineSchedule`（被入口使用）。 |

### utils/
| 文件 | 作用 |
|---|---|
| `__init__.py` | 包标记。 |
| `data_utils.py` | **总 loader**：`get_loader(args)` 把腹部×8、头颈×8、胸部×1 合并为全局 `ConcatDataset`，配置 Sampler/DataLoader。 |
| `data_utils_abdomen.py` | 腹部有标签数据（BTCV/FLARE22/AMOS/WORD 等）的 Dataset 构造，路径硬编码 `/data/...`。 |
| `data_utils_tumor.py` | 肿瘤/HealthyCT 数据（FLARE23/PANORAMA/LiTS/Pancreas/KiTS/HealthyCT）构造。 |
| `data_utils_headneck.py` | 头颈及其余 image-only 数据（HNSCC/QIN/TotalSegmentator 等）构造。 |
| `data_utils_chest.py` | 胸部数据（LUNA16/CT-RATE/NLST 等）构造。 |
| `data_utils_ctrate_subset.py` | **独立的 CT-RATE 子集 loader**，用于 VoCo/OCL 管线冒烟验证；不依赖上面的清单模块，仅复用胸部 transform。 |
| `dataloader_bdmap.py` | AbdomenAtlas TXT 病例列表加载（每行第一列拼 `ct.nii.gz`/`label.nii.gz`）。 |
| `data_trans.py` | 四类区域的预处理 transform 定义 + 分布式 `Sampler`。 |
| `voco_trans.py` | `VoCoAugmentation`：从 192×192×64 大块生成 S 个 query + 9 个 base crop 及几何重叠软标签。 |
| `ops.py` | `concat_image()` 等：把 query/base 列表沿 crop 维拼接、reshape 成 backbone 输入。 |
| `pretrain_common.py` | VoCo/OCL 入口共享的运行时工具：布尔参数解析、数据参数注册、loader 获取、resume 路径解析、checkpoint 保存/恢复、`--data_check_only` 等。 |
| `utils.py` | 通用工具：`AverageMeter` 等。 |
| `vis_voco.py` | 可视化脚本（依赖外部 `model.backbone.dinov2`，非主训练链路）。 |

---

## 3. VoComni/ —— 全监督预训练（伪标签数据）

入口：`main.py` → `trainer.run_training()`，数据 `utils.data_utils.get_loader()`，清单 `VoComni.json`（约 20,020 train + 23 val，20 类器官/肿瘤）。

| 文件 | 作用 |
|---|---|
| `main.py` | **全监督预训练入口**：构建 `SwinUNETR(use_v2=True)`、`DiceCELoss`+`DiceMetric`+sliding-window inferer、optimizer/scheduler，调用 trainer。 |
| `trainer.py` | 训练循环：`train_epoch`/`val_epoch`、定期验证、TensorBoard、best checkpoint。 |
| `VoComni.json` | Decathlon 风格清单（`imagesTr/*_0000.nii.gz` ↔ `labelsTr/*.nii.gz`）。 |
| `omni_B.sh` / `omni_L.sh` / `omni_H.sh` | Base/Large/Huge 三种规模的分布式启动脚本。 |
| `README.md` | 本目录说明文档。 |

### models/（backbone zoo）
| 文件 | 作用 |
|---|---|
| `__init__.py` | 包标记。 |
| `models.py` | 各预训练方法（VoCo/SuPrem/Swin/clip_driven/MG/UniMiss/DoDNet/STUNet）的构建函数 + 尺寸检查 strict `load()` 辅助。 |
| `unet.py` | `UNet3D`（Models Genesis backbone）。 |
| `dodnet.py` | `DoDNet_UNet3D`（DoDNet backbone）。 |
| `MiT.py` / `MiT_encoder.py` / `MiT_utils.py` | UniMiss 的 Transformer+CNN 分割网络、编码器及 DINO 风格工具。 |
| `STUNet.py` | STU-Net backbone（nnUNet 风格 SegmentationNetwork）。 |
| `neural_network.py` | `NeuralNetwork`/`SegmentationNetwork` 基类（滑窗推理基础设施）。 |
| `Patch_embeds.py` | 权重标准化卷积块 + patch embed 辅助。 |
| `PlainConvUNet_load.py` | 构建 plain nnUNet UNet 以加载 nnUNet 权重。 |

### optimizers/、utils/
| 文件 | 作用 |
|---|---|
| `optimizers/lr_scheduler.py` | `LinearWarmupCosineAnnealingLR` 等调度器。 |
| `utils/data_utils.py` | `get_loader(args)`：train/val transform 管线 + PersistentDataset/CacheDataset + 分布式 Sampler。 |
| `utils/utils.py` | `resample_3d`、`dice`、`AverageMeter`、`distributed_all_gather` 等共享工具。 |
| `nnUNet_preprocessed/*.json` | nnUNet 计划文件（dataset_fingerprint、nnUNetPlans）。 |

---

## 4. Semi-supervised/ —— 有标签 + 无标签预训练

设计上同时使用 VoComni（image+label）与 imagesUn（image-only）两个 loader；当前实现不完整（详见 `dataset_logic.md` §4）。

| 文件 | 作用 |
|---|---|
| `voco_train.py` | 半监督训练入口（前半程监督分割损失，后半程冻结 teacher 生成伪标签加半监督损失）。 |
| `gen_json.py` | 扫描 `imagesUn/*.nii.gz` 生成无标签清单 `dataset_unlabeled.json`（注意：当前模板路径误写为 `imagesTr`，需修正）。 |
| `VoComni.json` | 有标签清单。 |
| `dist_B/L/H.sh`、`single_train.sh` | 启动脚本。 |
| `optimizers/lr_scheduler.py` | 学习率调度器。 |
| `utils/data_utils.py` | labeled/unlabeled loader 构造。 |
| `utils/ops.py` | batch 拼接工具。 |
| `utils/utils.py` | 通用工具。 |

---

## 5. Omni-supervised/ —— 双数据流（自监督 + 监督）

入口 `voco_train.py` 同时创建 `train_loader`（PreCT/VoCo 自监督流）与 `omni_loader`（VoComni 监督流），同一 step 共同贡献 loss。

| 文件 | 作用 |
|---|---|
| `voco_train.py` | Omni 训练入口，同步消费两个 loader。 |
| `dist_B/L/H.sh`、`single_train.sh` | 启动脚本。 |
| `README.md` | 说明文档。 |
| `models/voco_head.py` | VoCo 模型（含分割头，复用 query 产生 semi_outputs）。 |
| `models/PlainConvUNet_load.py` | plain nnUNet UNet 构建。 |
| `optimizers/lr_scheduler.py` | 学习率调度器。 |
| `utils/data_utils.py` | PreCT 自监督流总 loader（腹×8/头颈×8/胸×1）。 |
| `utils/data_utils_abdomen/tumor/headneck/chest.py` | 各区域 Dataset 构造（复制自 Self-supervised）。 |
| `utils/data_utils_omni.py` | **VoComni 监督流** loader（image+label 正负 patch 采样，固定 batch=1）。 |
| `utils/data_trans.py` | 预处理 transform + Sampler。 |
| `utils/voco_trans.py` | VoCo query/base crop 与几何标签生成。 |
| `utils/dataloader_bdmap.py` | AbdomenAtlas TXT 加载。 |
| `utils/ops.py`、`utils/utils.py` | batch 拼接 / 通用工具。 |

---

## 6. Downstream/ —— 50+ 下游任务微调与评测

含两套独立实现：`monai/`（每数据集独立 main/trainer/utils）与 `nnUNet/`（上游 nnU-Net v2 + 自定义 VoCo trainer）。

### 顶层模板文件
| 文件 | 作用 |
|---|---|
| `README.md` | 下游任务权威文档：50+ 数据集表、下载布局、预训练模型库、`pretrained_root` 目录约定、微调步骤、可调参数、val/test 用法。 |
| `train.sh` | 通用单卡启动模板（以 3D-IRCADb 为例，`torchrun main.py`）。 |
| `val.py` | 验证模板：`SwinUNETR` + sliding-window，对带标签图像算 per-case Dice，反变换回原始空间并保存预测。需用户改路径/通道/预处理参数。 |
| `test.py` | 纯推理模板：同 val.py 但无标签/无指标，只输出 NIfTI 预测。 |

### 6.1 Downstream/monai/ —— 标准分割模板（以 3D-IRCADb 为例）

每个数据集目录通常包含：

| 文件 | 作用 |
|---|---|
| `main.py` | 任务入口：解析参数（`--name`/`--feature_size`/`--pretrained_root`/ROI/spacing/强度等），构建模型 `models.get_model(args)`、`DiceCELoss`+`DiceMetric`+inferer+optimizer，调用 `trainer.run_training()`。 |
| `trainer.py` | 训练循环：`train_epoch`/`val_epoch`/`save_checkpoint`/`run_training`，含 AMP、定期验证、TensorBoard、best checkpoint。 |
| `train.sh` | 启动脚本（设置 name/pretrained_root/data_dir/cache_dir 等）。 |
| `<Dataset>.json` | Decathlon 风格清单（training/validation split）。 |
| `README.md` | 该任务的结果表与 checkpoint 链接。 |
| `models/__init__.py` | `get_model(args)` 分派器：按 `--name` 选 backbone，`None`=从头训练。 |
| `models/models.py` | 各预训练方法构建函数 + 尺寸检查 strict `load()`（VoCo 按 feature_size 与 `use_ssl_pretrained` 选 `VoComni_*` 或 `VoCo_*_SSL_head`）。 |
| `models/unet.py`、`dodnet.py`、`MiT*.py`、`STUNet.py`、`neural_network.py`、`Patch_embeds.py`、`PlainConvUNet_load.py` | 各对比方法 backbone（MG/DoDNet/UniMiss/STU-Net 等）。 |
| `optimizers/lr_scheduler.py` | `LinearWarmupCosineAnnealingLR` 等。 |
| `utils/data_utils.py` | `get_loader(args)`：train/val transform + PersistentDataset/CacheDataset + 分布式 Sampler。 |
| `utils/utils.py` | `resample_3d`/`dice`/`AverageMeter`/`distributed_all_gather` 等。 |

15 个 CT 目录共用同一 `utils/data_utils.py` 模板（AIIB23、Aorta、ASOCA、CHAOS、Colon、FUMPE、Kipa、KiTs、LiTs、Panc、Parse22、Sliver07、StructSeg19、TCIA_Panc、Vessel）；其余标准目录接口相同但 split/标签转换/局部 transform 有差异。

**偏离标准模板的目录类别：**
- **分类（CSV/fold）**：`CC-CCII`（新冠分类，`csv/`+`--fold`+`model.py`+`eval.py`）、`LUNA16`（配置驱动 `luna_ncc_3d_config.py`，代码分散在 `datasets_3D/`、`networks/`、`trainers/`）。
- **视觉-语言（VLP）**：`CT_CLIP`（CT-RATE 词汇分类/报告-体积检索，多任务脚本 + `CTCLIPTrainer.py` + `ct_clip/`/`transformer_maskgit/`）、`M2KT`（CTRG 报告生成，多入口 + `config/`/`modules/`/`models/` + NLP 指标）。
- **配准（pkl，TransMorph 系）**：`Registration/IXI`、`Registration/OASIS`（从 `.pkl` 读 volume/seg 对，多方法 `train_*.py`/`infer_*.py`）。
- **多模态 + fold 分割**：`BRATS21`（4 模态 MRI，`brats21_folds.json` 按 fold 划分 + 自定义标签重映射）。
- **TXT 清单分割**：`AbdomenAtlas`（`dataset/dataset_list/*.txt` + `dataloader_bdmap.py` 硬编码 class_map）。
- MRI 模板（`ACDC`、`ATLAS-MRI`、`BRATS`、`Heart`、`Hip`、`Prostate`、`AMOS-MRI`）：不做 CT HU 窗，改为逐通道 `NormalizeIntensityd`。

### 6.2 Downstream/nnUNet/ —— nnU-Net v2 fork + VoCo trainer

| 文件 | 作用 |
|---|---|
| `README.md` | 说明这是上游 nnU-Net + 自定义 VoCo trainer；`Dataset503_VoComni` 工作流（`plan_and_preprocess` → `nnUNetv2_train ... -tr nnUNetTrainer_pre`）。 |
| `setup.py` / `pyproject.toml` | 包安装/打包元数据（提供 `nnUNetv2_train` 等控制台入口）。 |
| `LICENSE` | 上游 nnU-Net Apache-2.0 许可。 |
| `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_pretrain.py` | **VoCo 自定义 trainer** `nnUNetTrainer_pre`：构建 plans 驱动的 PlainConvUNet/ResidualEncoderUNet，加载 `VoComni_nnunet.pt`（尺寸检查 strict load）。需在此改硬编码 checkpoint 路径。 |
| `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_swin.py` | 备选 trainer `nnUNetTrainer_swin`：换用 Swin backbone。 |
| `nnunetv2/training/nnUNetTrainer/vit.py` | 供 swin trainer 使用的 Swin-UNETR 风格 backbone。 |
| `nnunetv2/run/run_training.py` | `nnUNetv2_train` 入口（解析 `-tr` trainer）。 |
| `nnunetv2/dataset_conversion/generate_dataset_json.py` | 生成 `dataset.json`（v2 schema：`channel_names`/`labels`/`numTraining`/`file_ending`）。 |
| `nnunetv2/dataset_conversion/Dataset*.py` | 各数据集转换为 nnUNet raw 布局的脚本。 |
| `nnunetv2/{experiment_planning,preprocessing,imageio,training/dataloading,training/loss,inference,evaluation,utilities}/` | 上游 nnU-Net v2 标准子系统（规划/预处理/读盘/2D-3D loader/Dice-CE-深监督损失/poly-LR/滑窗推理/指标）。 |

仅 `nnUNetTrainer_pretrain.py`、`nnUNetTrainer_swin.py`、`vit.py` 是 VoCo 新增，其余为上游原样代码。

---

## 7. external/OCL —— 子模块（只读参考）

`external/OCL`（git 子模块，`stap1e/OCL`）是 OCL/MiCL 的原始实现，被 `Self-supervised/models/ocl_head.py` 只读参考。含 `models_mae.py`（`MiCLAutoencoderViT`）、`models_vit.py`、pretrain/finetune/linprobe 入口与 `util/` 工具。本仓库不修改它。

---

## 8. tools/ 与 scripts/ —— 管线验证辅助

| 文件 | 作用 |
|---|---|
| `tools/build_ctrate_subset.py` | 构建确定性、按患者去重的 CT-RATE 子集清单（仅检查 NIfTI 头，不复制/移动源文件；输出 datalist JSON + patients.txt + skipped 报告）。 |
| `scripts/train_ctrate_subset_smoke.sh` | CT-RATE 子集冒烟测试脚本：必要时调用上面的工具生成清单，选 `ocl`/`voco` 入口，先 `--data_check_only` 再 GPU 短训（默认 10 step）。 |

---

## 9. 环境与启动约定（速查）

- Python 3.10.13，conda 环境；torch 2.1.1+cu118 单独安装，其余 `pip install -r requirements.txt`。
- 各 `.sh` 以 `source activate YOUR-CONDA-ENVIRONMENT` 开头，需替换为自己的环境名。
- `feature_size`：48=Base，96=Large，192=Huge。
- `--name` 选预训练族（VoCo/suprem/swin/clip_driven/mg/unimiss/dodnet），`None`=从头训练。
- `--use_ssl_pretrained`：true 加载 `VoCo_*_SSL_head`，否则加载 `VoComni_*`。
- 权重加载为尺寸检查 strict load：`in_channels!=1` 或 `out_channels!=21` 时跳过首/末层。
- 无测试套件/lint/CI；验证靠手动运行训练/评测脚本。
- 各目录 README 是该目录的权威文档，优先于根 README。

---

## 10. 模型数据流、模型应用与 Loss / 算法流程详解

本节基于源码逐行核对，补充各预训练/微调方法的**张量 shape 流转**、**模型结构应用**与**损失/算法逻辑**。符号约定：`C=feature_size`（48/96/192），`B'`=当前 rank 的 volume batch，`S=sw_batch_size`（query 数），9 个 base 为 3×3 XY 网格。

### 10.1 共享 backbone：3D SwinUNETR（`Swin`）

`Self-supervised/models/voco_head.py` 的 `Swin` 类是 VoCo/OCL 共用的编码器（无解码器）。`use_v2=True`，`depths=[2,2,2,2]`、`num_heads=[3,6,12,24]`、`window_size=7`、`patch_size=2`。

```text
输入 x_in: [N, 1, 64, 64, 64]
  └ swinViT(x_in) → hidden_states_out（多尺度特征列表 hs[0..4]）
  enc0 = encoder1(x_in)        → [N, C,   *, *, *]
  enc1 = encoder2(hs[0])       → [N, C,   ...]
  enc2 = encoder3(hs[1])       → [N, 2C,  ...]
  enc3 = encoder4(hs[2])       → [N, 4C,  ...]
  dec4 = encoder10(hs[4])      → [N, 16C, ...]
  forward_encs: 每个 enc → adaptive_avg_pool3d(1,1,1) → view[N, ch] 后 concat
输出: [N, C+C+2C+4C+16C] = [N, 24C]
```

`24C` 即投影头输入维度：`C=48→1152`、`C=96→2304`、`C=192→4608`（与 `VoCoHead` 中 `in_dim` 的分支一致）。

### 10.2 VoCo 自监督（`VoCoHead`，Self-supervised）

**模型结构**：`backbone=Swin` + `student`/`teacher` 两个 `projection_head`（`in_dim→1024→1024→1024`，含 BN+ReLU）。teacher 由 student 做 EMA 更新（`momentum=0.9`，`@torch.no_grad`）。

**入口数据流**（`voco_train.py` 训练循环）：

```text
DataLoader batch = (img, labels, crops)
  img:   list[S]，每项 {image:[B',1,64³]}
  labels: [B', S, 9]   （几何重叠软标签，float64）
  crops:  list[9]，每项 {image:[B',1,64³]}
concat_image() 后：
  img(queries) = [B'×S, 1, 64³]      # 默认单卡 B'=4,S=2 → [8,1,64³]
  crops(bases) = [B'×9, 1, 64³]      # → [36,1,64³]
  labels       = [B', S, 9]          # → [4,2,9]
```

**前向 `forward(img, crops, labels)`**：

```text
inputs = cat([img, crops], dim=0) → [B'×(S+9), 1, 64³]   # 默认 [44,1,64³]
embeddings = backbone(inputs)     → [B'×(S+9), 24C]
aug = Dropout1d(0.2)(embeddings)  # 特征级增强
student = projection_head(aug)    → [B'×(S+9), 1024]
EMA_update_teacher()
teacher = projection_head(embeddings) [no_grad] → [B'×(S+9), 1024]
切分：x_* = 前 B'×S（queries），bases_* = 后 B'×9
```

**三项损失**（对每个病例 `i` 循环，最后对 `B'` 取平均，`loss = intra + inter + b_loss`）：

1. **intra（体积内位置预测）**
   - `logits = online_assign(x_stu[S,1024], bases_tea[9,1024])`：逐 query 对 9 个 base 算余弦相似度并 `ReLU` → `[S,9]`。
   - `intra_loss = ce_loss(label[i][S,9], logits[S,9])`。
2. **inter（跨体积对比）**
   - 取本地 batch 中下一个病例 `j=(i+1)%B'` 的 bases。
   - `pred1 = online_assign(x_tea, inter_bases_tea)`，`pred2 = online_assign(x_stu, inter_bases_stu)`。
   - `inter_loss = ce_loss(pred1.detach(), pred2)`（teacher 软目标监督 student）。
3. **b_loss（base 去相关正则）**
   - `regularization_loss(bases_stu[9,1024])`：9 个 base 两两余弦相似度 → `ReLU` → 平方，按对数取平均，促使 base 表征解相关。

**关键子函数**：

- `online_assign(feats, bases)`：`F.cosine_similarity` 逐样本 → `[b,k]`，再 `F.relu`。
- `ce_loss(labels, logits)`（软标签交叉熵变体）：
  ```text
  pos_dis  = |labels - logits|
  pos_loss = -labels·log(1 - pos_dis + 1e-6)，按 labels.sum() 归一
  neg_loss = -(labels==0)·log(1 - logits + 1e-6)，按负样本数归一
  返回 pos_loss + neg_loss
  ```
- 跨病例配对仅在各 rank 内进行，**不跨 GPU all-gather**；DDP 只同步梯度。

### 10.3 OCL / MiCL 自监督（`OCLHead3D`，Self-supervised）

复用同一 `Swin` backbone 与数据管线，仅替换算法逻辑为掩码图像对比学习（参考只读子模块 `external/OCL`）。

```text
x: [B, 1, H, W, Z]
x1 = _make_view(x)   # patch_rand_drop 随机 3D 块丢弃，max_drop=0.3
x2 = _make_view(x)   # 第二次独立掩码 → 两个不同视图
e1 = backbone(x1) → [B, 24C]，L2 归一化
e2 = backbone(x2) → [B, 24C]，L2 归一化
feats = cat([e1,e2]) → [2B, D] → reshape [B, 2, D]
m = feats[:,0]，n = feats[:,1]
sim_mn = compute_tSP(m @ n.T)，sim_nm = compute_tSP(n @ m.T)   # [B, B]
labels = arange(B)
loss = (cross_entropy(sim_mn, labels) + cross_entropy(sim_nm, labels)) / 2
```

- `compute_tSP(x)`：正样本相似度变换 `0.5·(1+x)/(1+(1-x)·κ)/τ`，`κ=1/64`、`τ=temperature=0.07`。
- 本质是对称 InfoNCE；**无 EMA teacher、无几何 crop**（这些是 VoCo 特有，已移除）。OCL 入口只使用 query 部分。

### 10.4 VoComni 全监督预训练

**模型**：`SwinUNETR(use_v2=True)`（标准带解码器分割网络，`out_channels=21`）。

**训练数据流**（`main.py` + `trainer.train_epoch`）：

```text
manifest image,label → Load+ChannelFirst → [1,X,Y,Z]
RAS + 1.5³ mm spacing → CropForeground + pad(≥96) → RandCropByPosNegLabeld(K=4)
  每病例 list[4]，每项 image,label = [1,96,96,96]
MONAI collate(B=4): image,label = [16, 1, 96³]     # 16 = B×K
SwinUNETR logits  = [16, 21, 96³]
```

**损失与验证**：

- `DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)`（label 为 0–20 单通道类别 ID，损失内部转 21 类 one-hot）。
- `train_epoch`：AMP `autocast` 前向 → `loss_func(logits, target)` → `scaler`/`optimizer.step()` → `scheduler.step()`；分布式用 `distributed_all_gather` 汇总 loss。
- 验证 `val_epoch`：`sliding_window_inference`（roi 96³）拼回整卷 `[1,21,Hv,Wv,Dv]`；`post_label=AsDiscrete(to_onehot=21)`、`post_pred=AsDiscrete(argmax=True, to_onehot=21)`；`DiceMetric(include_background=False)` 逐病例聚合。

### 10.5 Omni-supervised 双数据流（`Omni-supervised/models/voco_head.py`）

**模型差异**：此处 `Swin` 在编码器之上**增加解码器**（`decoder5..decoder1` + `UnetOutBlock`），因此 backbone 同时输出 `embeddings`（对比用）和 `logits`（分割用）。

**前向 `forward(img, crops, labels, omni_inputs, omni_labels)`**：

```text
# 自监督流（PreCT）
inputs = cat([img, crops]) → [B'×(S+9), 1, 64³]
embeddings, semi_outputs = backbone(inputs, frozen=True)   # 编码器 detach，不回传对比梯度
semi_outputs = semi_outputs[:B'×S]                          # query 分割 logits [B'×S, 21, 64³]
# 监督流（VoComni，固定 batch=1，K=4）
_, omni_outputs = backbone(omni_inputs[4,1,64³], frozen=False) → [4, 21, 64³]

# 对比部分：与 Self-supervised 完全相同的 intra + inter + b_loss
aug/student/teacher/EMA → 三项 VoCo 损失

# 监督分割损失
seg_loss = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)(omni_outputs, omni_labels)
return (loss + seg_loss), semi_outputs
```

- `frozen=True` 时对 `enc0..dec4` 调 `.detach()`，使对比损失不更新编码器，仅监督流 `frozen=False` 时分割梯度回传编码器。
- 两流不按病例对应，只在同一 optimizer step 共同贡献 loss；`S=4`（与 Self-supervised 的 `S=2` 不同）。

### 10.6 下游微调（`Downstream/monai/<Dataset>`）

**模型选择 `get_model(args)`**（`models/__init__.py`）：按 `--name` 分派——`VoCo`/`suprem`/`swin`/`clip_driven`/`mg`/`unimiss`/`dodnet`，`None`/未知则从头构建 `SwinUNETR(use_v2=True)`。

- `VoCo(args)`：构建 `SwinUNETR(use_v2=True)`，按 `feature_size` 与 `use_ssl_pretrained` 选择权重——
  ```text
  feature_size 48/96/192 → VoComni_{B,L,H}.pt
  use_ssl_pretrained=True → VoCo_{B,L,H}_SSL_head.pt
  ```
  从 `args.pretrained_root` 拼接路径并 `load()`。
- 对比方法 backbone 各异：`SuPrem/Swin/Universal` 用 `SwinUNETR(use_v2=False)`，`MG` 用 `UNet3D`，`UniMiss` 用 `MiT`，`DoDNet` 用 `DoDNet_UNet3D`，`stunet` 用 `STUNet`。

**权重加载 `load(model, model_dict)`**（尺寸检查的 strict load）：

```text
1. 从 state_dict / network_weights / net / student 键中取出 state_dict
2. 去掉前缀 module. / backbone.，并把 swin_vit → swinViT
3. 逐键：仅当键存在且 shape 完全一致才采用预训练权重，否则保留当前模型权重
4. load_state_dict(strict=True)
```

> 因此当 `in_channels!=1` 或 `out_channels!=21` 时，首层/末层 shape 不匹配会自动跳过（保留随机初始化），其余层照常加载。

**训练/验证数据流与损失**（`main.py` + `trainer.py`，以 3D-IRCADb 为例）：

```text
train: RandCropByPosNegLabeld(num_samples=K) → collate 展平
  image [B×K, 1, Rx,Ry,Rz]，label [B×K, 1, Rx,Ry,Rz]（单通道类别 ID）
  model logits [B×K, out_channels, Rx,Ry,Rz]
val: 整卷 [1, C, Hv,Wv,Dv] → sliding_window_inference(roi=inf_size, overlap) → [1, out_channels, Hv,Wv,Dv]
```

- 损失：默认 `DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)`；`--squared_dice` 时改用 `squared_pred=True` 版本。
- 后处理：`post_label=AsDiscrete(to_onehot=out_channels)`、`post_pred=AsDiscrete(argmax=True, to_onehot=out_channels)`；指标 `DiceMetric(include_background=False, reduction=MEAN, get_not_nans=True)`。
- `train_epoch`：AMP 前向→`loss_func(logits,target)`→反向/步进，分布式 `distributed_all_gather` 汇总；`val_epoch`：滑窗推理 + `decollate_batch` + DiceMetric 聚合；`run_training` 负责 epoch 循环、`val_every` 定期验证、TensorBoard 与 best checkpoint（`model.pt`/`model_final.pt`）。
- `Rx/Ry/Rz`、spacing、HU 窗、`out_channels` 均由各目录 `main.py` 单独设定，MRI 任务改逐通道 `NormalizeIntensityd` 而非 CT HU 窗。

---

## 11. 数据加载与处理逻辑详解

本章按“磁盘文件 → manifest → Dataset transform → DataLoader batch → 训练消费”的顺序，说明数据如何被加载和处理。源码主要在 `Self-supervised/utils/`（`data_utils.py`、`data_trans.py`、`voco_trans.py`、`ops.py`、`data_utils_*.py`）与 `VoComni/utils/data_utils.py`。更完整的逐数据集清单契约见 `dataset_logic.md`。

### 11.1 总体加载架构（Self-supervised）

`utils/data_utils.py::get_loader(args)`：

```text
abdomen_ds  = get_ds_abdomen(args)    # 腹部（含肿瘤/Atlas/DeepLesion）
headneck_ds = get_ds_headneck(args)   # 头颈
chest_ds    = get_ds_chest(args)      # 胸部

abdomen_ds  = ConcatDataset([abdomen_ds] × 8)   # 索引空间放大 8 倍（不复制影像/缓存）
headneck_ds = ConcatDataset([headneck_ds] × 8)
train_ds    = ConcatDataset([abdomen_ds, headneck_ds, chest_ds])   # 8A + 8H + C

train_sampler = Sampler(train_ds) if distributed else None
DataLoader(train_ds, batch_size, shuffle=(sampler is None),
           num_workers, sampler, pin_memory=True)   # 默认 drop_last=False
```

- “×8”只扩展索引空间，使腹部/头颈被采样的概率提高；一次完整遍历会访问每个腹部/头颈病例 8 次、胸部 1 次，每次重新跑随机 crop/增强，故同一病例八次输入通常不同。
- **无** dataset-balanced batch、区域轮询或固定配额；一个 batch 内可能混合不同数据集、不同身体区域。

**manifest 解析时机**：各 `data_utils_*.py` 把 `load_decathlon_datalist(...)` 写在**模块顶层**，因此 `import utils.data_utils` 时就解析所有 JSON/TXT（不是等到 `get_loader`）。任一必需清单缺失会在训练初始化前直接失败。路径根（`/data/...`）与 cache 根大量硬编码在各 `data_utils_*.py`。

**区域 Dataset 组装**（以 `get_ds_abdomen` 为例）：每个子数据集分别包成 `PersistentDataset`（`args.cache=True`）或普通 `Dataset`，再 `ConcatDataset([tumor, BTCV, flare, Amos, WORD, flare23, Atlas, DeepLesion, PANORAMA])`。有标签数据用 `get_abdomen_trans`，DeepLesion（image-only）用 `get_abdomen_trans_without_label`。

### 11.2 分布式 Sampler（`data_trans.py::Sampler`）

```text
num_samples = ceil(N / world_size)
total_size  = num_samples × world_size      # 补齐到 world_size 整数倍
__iter__: shuffle 时 torch.randperm(generator=manual_seed(epoch))
          补齐索引后取 indices[rank::world_size]
valid_length = 本 rank 实际有效样本数（用于 distributed_all_gather 的 is_valid）
```

- `set_epoch(epoch)` 已定义；Self-supervised 入口每轮遍历会调用 `train_loader.sampler.set_epoch(loader_epoch)`（VoComni trainer 每 epoch 也调用）。非分布式时 `shuffle=True`。

### 11.3 四类预处理 transform 管线（`data_trans.py`）

四条管线共享 shape 生命周期，差异在 spacing、HU 窗与是否 resize。设磁盘为单通道 3D NIfTI：

```text
LoadImaged            → [X,Y,Z] float32 MetaTensor（label 同，或不加载）
EnsureChannelFirstd   → [1,X,Y,Z]
Orientationd(RAS)     → [1,Xr,Yr,Zr]（改轴序，不定大小）
Spacingd              → [1,Xs,Ys,Zs]（image bilinear / label nearest；大小取决于原 spacing）
ScaleIntensityRanged  → shape 不变，HU 窗 clip 到 [0,1]（只处理 image）
CropForegroundd       → [1,Xf,Yf,Zf]（bbox 由 image 非零区域决定）
SpatialPadd           → 各轴至少 [roi=64,64,64]
RandShiftIntensityd(prob=0)  ← 缓存边界（见 11.5）
[headneck/DeepLesion: Resized → 192×192×64]   # 3D 写 mode="bilinear"，应为 trilinear（已知阻塞）
SpatialPadd(192,192,64) + RandSpatialCropd(192,192,64)   # 普通腹部/胸部从更大 volume 随机裁
VoCoAugmentation(aug=True)   # 见 11.4
```

| 管线 | spacing (mm) | HU 窗 [a_min,a_max] | 大块构造方式 |
|---|---|---|---|
| 腹部有标签 `get_abdomen_trans` | `args.space`（默认 1.5³） | [-175, 250] | pad/random crop 192×192×64 |
| 腹部无标签 `..._without_label`（DeepLesion） | 同上 | [-175, 250] | `Resized`→192×192×64 后 crop |
| 胸部 `get_chest_trans` | (1.25, 1.25, 5.0) | [-1000, 500] | pad/random crop 192×192×64 |
| 头颈 `get_headneck_trans` | (1.5, 1.5, 1.5) | [-175, 250] | `Resized`→192×192×64 后 crop |

- 医学分割 label 只在部分腹部数据中与 image 同步做方向/spacing/padding；进入 `VoCoAugmentation` 后被删除，**不参与自监督损失**。胸部/头颈从开始就只读 image。

### 11.4 VoCo crop 与几何标签生成（`voco_trans.py::VoCoAugmentation`）

每个成功到达 `192×192×64` 的 volume：

```text
__call__(x_in):
  删除 x_in['label']
  num_crops=3, max_roi=192
  vanilla_trans, labels = get_vanilla_transform(num=S=sw_batch_size, num_crops=3, max_roi=192)
  crops_trans = get_crop_transform(num_crops=3)
  imgs  = [trans(x_in) for trans in vanilla_trans]   # S 个 query
  crops = [trans(x_in) for trans in crops_trans]      # 9 个 base
  return imgs, labels, crops
```

- **query（vanilla）**：`get_position_label` 在 XY 平面随机取中心 `center_x,center_y ∈ [32,159]`（`np.random.randint(low=half=32, high=max_roi-half=160)`，high 不含），Z 中心固定 `roi//2=32`，`SpatialCropd` 裁出 `64³`。
- **base（crop）**：`get_crop_transform` 固定 3×3 网格，中心 `((i+0.5)·64, (j+0.5)·64, 32)`，即 `(32/96/160, 32/96/160, 32)`，裁出 9 个 `64³`。
- **几何软标签**：`labels[S,9]`，每行是一个 query 对 9 个 base cell 的 **XY 面积重叠占比**——`area = (dx·dy)/roi²`（`dx/dy` 为 query 与 cell 在 X/Y 的重叠长度），非负、行和为 1、最多同时覆盖 4 个 base。
- **增强**：`aug=True` 时 query 与 base 各自独立做 `RandFlipd(prob=0.2, axis=0/1/2)`、`RandRotate90d(prob=0.2, max_k=3)`、`RandShiftIntensityd(offsets=0.1, prob=0.5)`。

**单病例 Dataset 输出**：

```text
(
  imgs:   list[S]，每项 {"image": float32 MetaTensor [1,64,64,64]},
  labels: NumPy float64 [S, 9],
  crops:  list[9]，每项 {"image": float32 MetaTensor [1,64,64,64]}
)
```

不含 case ID / dataset ID / 身体区域 ID / 医学 label。

### 11.5 PersistentDataset 缓存边界

`args.cache=True` 时每个子数据集用自己的 `PersistentDataset` 与 cache 目录。第一个随机变换 `RandShiftIntensityd(prob=0)`（虽概率为 0，但属 MONAI Randomizable）构成缓存边界：

```text
会缓存：Load / Orientation / Spacing / ScaleIntensityRange / CropForeground / 第一个 pad
不缓存：该边界之后的 resize、192 大块裁剪、VoCo query/base 生成与增强
```

因此启用缓存不会把固定 crop 永久保存，同一病例每次仍生成不同自监督视图。**修改 spacing/强度窗/确定性预处理后应清理或更换旧缓存**。注意已知问题：`data_utils_chest.py` 的 `chest_cache_root='data/cache/chest/'` 缺前导 `/`（写到相对目录）；MELA 的 cache 目录被设成原始数据目录 `/data/MELA/`。

### 11.6 batch 拼接（`ops.py::concat_image`）

DataLoader collate 后、送入模型前：

```text
collate 后：imgs list[S]，每项 {image:[B',1,64³]}；crops list[9]；labels [B',S,9]
concat_image(imgs):  取每项 ['image'] → cat(dim=1) → [B',S,64³] → view(-1,1,64³) = [B'×S,1,64³]
concat_image(crops): 同理 → [B'×9,1,64³]
```

默认单卡 `B'=4,S=2`：query `[8,1,64³]`、base `[36,1,64³]`、labels `[4,2,9]`。分布式每 rank 通常 `B'=2,S=2`：query `[4,1,64³]`、base `[18,1,64³]`、labels `[2,2,9]`。几何标签默认 collate 为 `torch.float64`（代码未显式 `.float()`）。

### 11.7 OCL 掩码视图（`ops.py::patch_rand_drop`）

OCL 入口复用 VoCo 管线得到 query volume，再由 `OCLHead3D._make_view` 调 `patch_rand_drop` 生成掩码视图：

```text
patch_rand_drop(x, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
  随机丢弃累计达 max_drop·H·W·Z 个体素的若干 3D 块
  丢弃区域填充归一化到 [0,1] 的高斯噪声（x_rep=None 时）
```

同一 volume 独立调用两次得到 `x1,x2` 两个视图送入对比学习。`ops.py` 另含 `rot_rand`（随机 90° 旋转 + 旋转标签）、`aug_rand`（块丢弃 + 跨样本替换）等辅助增强。

### 11.8 VoComni / 下游监督 loader（`VoComni/utils/data_utils.py`）

监督任务的 loader 与自监督不同，核心是 `RandCropByPosNegLabeld` 正负 patch 采样：

```text
train_transform:
  Load image+label → ChannelFirst/RAS → Spacing(1.5³, bilinear/nearest)
  → ScaleIntensityRange([a_min,a_max]→[0,1]) → CropForeground → SpatialPad(≥roi)
  → RandCropByPosNegLabeld(label_key, pos:neg, num_samples=sw_batch_size, image_threshold=0)
  → RandFlipd(axis0/1/2) + RandRotate90d + RandShiftIntensityd
val_transform: 同上但全确定性（无随机 crop/增强），整卷进滑窗推理
```

- `pos:neg`（如 9:1）是选 patch 中心时的正负权重，**不保证** batch 精确 90% 正样本；`image_threshold=0` 约束负样本中心位于有效 image 区域。
- 一个病例产生 `num_samples=K` 个固定 ROI patch；collate 后 train 为 `[B×K, 1, roi³]`、label `[B×K, 1, roi³]`。
- Dataset 选择：`use_persistent_dataset=True` → train/val 均 `PersistentDataset`（缓存边界为第一个随机变换 `RandCropByPosNegLabeld`）；否则 train 用 `CacheDataset(cache_num=24, cache_rate=1.0)`（仅缓存最多 24 例确定性前缀到 RAM）、val 用普通 `Dataset`。
- val loader 固定 `batch_size=1, shuffle=False`；分布式 val sampler 用 `valid_length` 排除补齐样本。下游各 `monai/<Dataset>/utils/data_utils.py` 基本复制此模板，仅 spacing/HU 窗/ROI/`out_channels` 与 split 取法不同（MRI 改逐通道 `NormalizeIntensityd`）。

### 11.9 数据正确性检查关卡（运行前）

建议把“能解析 manifest → 能读一个病例 → 能组成一个 batch”作为三个独立关卡，前两关不要启动 DDP 或遍历完整 160K：

| 关卡 | 应验证条件 |
|---|---|
| manifest | split 存在；必需 key 为非空路径；与 `base_dir` 拼接后文件存在；image/label case ID 配对。 |
| 单个 NIfTI | channel-first 后 image 为 `[C,X,Y,Z]`、值有限、C 与 `in_channels` 一致；label 空间 shape 与 image 一致。 |
| 分割 label | unique 值在 `[0,out_channels-1]` 或入模前有明确重映射；不要把 one-hot `[K,X,Y,Z]` 当单通道类别图。 |
| 标准分割 batch | train `[B×K,C,Rx,Ry,Rz]`、label `[B×K,1,Rx,Ry,Rz]`；val 的 B=1、空间可变。 |
| Self-supervised batch | tuple 恰有三项；concat 后 query `[B'×S,1,64³]`、base `[B'×9,1,64³]`、几何 label `[B',S,9]`。 |
| 双流任务 | 分别打印两 loader 的长度/键/shape；确认较短 iterator 的重建策略，不能只验证第一步。 |
| cache | 写入专用 cache 根；改 spacing/window/确定性 transform 后用新 cache 或清理旧 cache。 |

> Self-supervised 已知实现问题（详见 `dataset_logic.md` §2.12）：LiTS 用 `train+train`、HealthyCT 三列表都读 liver JSON、DeepLesion/headneck 的 3D `Resized(mode="bilinear")` 应为 `trilinear`、`--cache`/`--noamp`/`--resume` 的布尔与字符串解析不可靠等。修改后仍需先单卡抽样检查 shape 与 cache 位置。

---

## 12. 多数据中心 / 多数据源联合训练机制

> **重要前提（澄清）**：本仓库的“多中心联合训练”是**中心化（centralized）联合训练**——假设所有中心/来源的数据都已汇聚到同一集群的 `/data/...` 下，由**单个统一模型**在一次运行中同时学习全部数据。它**不是联邦学习（federated learning）**：没有跨中心通信协议、没有隐私保护/梯度聚合框架、数据不留在各中心。下文从“数据层如何混合”和“计算层如何并行”两个维度说明联合训练如何实现。

### 12.1 数据层：多中心/多源数据如何混合成一个训练流

**（1）三级数据结构**：`Self-supervised/utils/data_utils.py::get_loader`

```text
每个身体区域 = 多个中心/数据集 Dataset 的 ConcatDataset
  abdomen_ds  = ConcatDataset([tumor, BTCV, flare22, Amos, WORD, flare23, Atlas, DeepLesion, PANORAMA])
  headneck_ds = ConcatDataset([HNSCC, QIN, HeadNeckPET, TCGA-HNSC, TotalSegmentator, Colonography])
  chest_ds    = ConcatDataset([LUNA16, TCIA-COVID, STOIC21, LIDC, StonyBrook, MELA, CT-RATE, NLST])

区域级加权：
  abdomen_ds  = ConcatDataset([abdomen_ds] × 8)
  headneck_ds = ConcatDataset([headneck_ds] × 8)
  train_ds    = ConcatDataset([abdomen_ds, headneck_ds, chest_ds])   # 有效长度 N = 8A + 8H + C
```

**（2）区域采样概率**（由索引空间占比决定）：

```text
P(abdomen)  = 8A / N
P(headneck) = 8H / N
P(chest)    = C  / N      （C=139,219，含 NLST 84,830 + CT-RATE 47,149，故胸部样本最多）
```

- “×8”只为提高腹部/头颈被采样概率（其原始样本远少于胸部），仅扩展索引空间，**不复制影像、不建多份缓存**。
- **区域内无数据集级均衡**：区域内是裸 `ConcatDataset`，采样概率正比于各数据集样本数，没有 dataset-balanced batch、区域轮询或每数据集固定配额。因此胸部由 NLST/CT-RATE 主导，小数据集被淹没。

**（3）batch 与中心无关（center-agnostic）**：

- 全局 shuffle 后，**一个 batch 内可同时出现不同中心、不同身体区域的病例**；模型与损失不感知病例来自哪个中心（无 center ID、无 dataset ID 进入 sample）。
- 异构采集协议由**区域级 transform 归一化**：不同中心原始 spacing/强度窗各异，经各自 `Spacingd` + `ScaleIntensityRanged`（腹部/头颈 [-175,250]、胸部 [-1000,500]）统一映射到单通道 `[0,1]`，再裁成统一 `64³` crop，从而消除中心间物理差异。
- **自监督几何标签与中心无关**：VoCo 的位置软标签是 query 与 base 的纯空间 XY 面积重叠比，不依赖任何中心提供的器官/肿瘤标注，因此无标签中心数据（DeepLesion、头颈、胸部）可直接参与联合训练。

**（4）监督流与自监督流的联合（Omni-supervised）**：

- Omni 在同一 step 同步消费两个独立 loader：PreCT 自监督混合流（多中心）+ VoComni 监督流（伪标签），两流**不按病例/中心对应**，只在同一 optimizer step 共同贡献 `loss + seg_loss`。
- VoComni/Semi 的全监督/半监督流是单一来源（VoComni 20K），不涉及多中心混合。

### 12.2 计算层：分布式并行如何扩展联合训练

联合训练靠 DDP 在单机多卡上扩展（脚本默认单机 8 卡）。仓库存在**两种启动模式**：

**模式 A —— torchrun env-based DDP（Self-supervised / Omni-supervised）**

```text
dist_B.sh:  python -m torch.distributed.launch --nproc_per_node=8 --master_addr=localhost --master_port=25805 voco_train.py ...
single_train.sh:  torchrun --master_port=28804 voco_train.py ...   # 单卡
```

`voco_train.py::main()` 初始化：

```text
args.local_rank  = int(os.environ["LOCAL_RANK"])
args.distributed = int(os.environ["WORLD_SIZE"]) > 1
dist.init_process_group(backend="nccl", init_method="env://")   # 读 MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE/LOCAL_RANK
model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
```

- `find_unused_parameters=True`：因 VoCo 三分支损失并非所有参数每步都参与反传。
- 脚本 `master_addr=localhost` → **默认单机**；理论上多机可通过 `torchrun --nnodes/--node_rank` + 设置 `MASTER_ADDR/MASTER_PORT` 扩展（代码用 `env://` 已具备条件），但仓库未提供多机脚本。

**模式 B —— mp.spawn（VoComni / Downstream monai）**

```text
omni_B.sh:  torchrun --master_port=21503 main.py ...   # 仅用于拉起进程
VoComni/main.py::main():
  args.ngpus_per_node = torch.cuda.device_count()
  args.world_size     = ngpus_per_node × args.world_size   # world_size 参数语义为“节点数”，默认 1
  mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
main_worker(gpu, args):
  args.rank = args.rank × ngpus_per_node + gpu
  dist.init_process_group(backend=dist_backend, init_method=dist_url, world_size, rank)
```

- 该模式按本机 GPU 数 spawn 进程，rank 由 `节点rank×ngpus+gpu` 计算；本质面向**单机多卡**（多机需 `world_size>1` 且配合可达的 `dist_url`，spawn 本身只管理本机 GPU）。

**数据分片（自定义 `Sampler`，两模式通用）**：

```text
num_samples = ceil(N / world_size)；total_size 补齐到 world_size 整数倍（make_even）
每 rank 取 indices[rank::world_size]；valid_length 记录本 rank 有效样本数（用于 all_gather 的 is_valid）
```

- 各 rank 拿到**互不重叠**的病例子集，全局合起来覆盖（约）整个混合数据集；`set_epoch` 改变排列（Self-supervised 入口每轮调用）。

**梯度同步与对比学习的 locality（关键限制）**：

- DDP 仅对各 rank **梯度做 all-reduce** 同步；模型参数保持一致。
- VoCo 的跨体积 inter-volume 对比与 base 配对**只在各 rank 本地 batch 内进行，不跨 GPU all-gather**（无 SimCLR 式 all-gather、无 MoCo 队列）。因此每个 rank 看到的“负样本/其他病例”仅限本地 `B'` 个 volume。
- 这是大规模分布式对比训练的已知简化：扩大 GPU 数提升的是吞吐与全局 batch，但**不扩大对比的负样本池**（负样本池仍为单卡 batch）。

**全局有效 batch**：

```text
全局 volume batch = 每卡 batch_size × world_size
  dist_B.sh: 每卡 batch_size=2 × 8 卡 = 16 volume/step
  每 volume 再展开为 S 个 query + 9 个 base（Self-supervised S=2，Omni S=4）
```

**checkpoint 与 resume**：

- 仅 `rank==0` 保存 checkpoint（`model_current_epoch.pt` + `model_step{N}.pt`）。
- Self-supervised 当前入口经 `pretrain_common.restore_checkpoint` 做 resume，会恢复模型、`global_step`、optimizer、scheduler 与 AMP scaler，并校验 `scheduler.last_epoch==global_step`（注：`dataset_logic.md` §2.12 描述的“resume 不恢复 optimizer/scheduler”为重构前旧行为，现代码已恢复）；DDP 多 rank 写同一文件/路径拼接仍需长期训练前确认。

### 12.3 联合训练速查表

| 维度 | 机制 | 位置 |
|---|---|---|
| 多源混合 | 区域 ConcatDataset + 腹部/头颈×8、胸部×1 | `Self-supervised/utils/data_utils.py` |
| 区域内权重 | 裸拼接，按样本数比例（无均衡） | `data_utils_{abdomen,headneck,chest}.py` |
| 中心差异归一 | 区域级 spacing + HU 窗 → 单通道 [0,1] → 64³ | `data_trans.py` |
| 无标签中心可用 | 几何软标签纯空间、与中心标注无关 | `voco_trans.py` |
| 并行启动 | torchrun env-DDP（Self/Omni）或 mp.spawn（VoComni/下游） | `dist_*.sh`、`main.py` |
| 数据分片 | `Sampler` indices[rank::world_size] | `data_trans.py::Sampler` |
| 梯度同步 | DDP all-reduce 梯度；对比配对不跨卡 | `voco_train.py` |
| 全局 batch | 每卡 batch × world_size（×(S+9) crop 展开） | 启动脚本 |

---

## 13. 自监督预训练的评估手段

> **核心结论**：VoCo/OCL 自监督预训练**没有预训练阶段的验证集/验证循环**，预训练质量**不靠预训练自身的指标衡量，而是通过下游任务迁移效果间接评估**（50+ 任务 benchmark）。预训练期只做工程性 sanity check 与 loss 监控。

### 13.1 预训练期：无验证，仅监控 loss

`Self-supervised/voco_train.py` 的训练循环：

```text
while global_step < num_steps:
    for batch in train_loader:
        loss = model(img, crops, labels)   # VoCo 三项损失 / OCL 对比损失
        ... backward / step ...
        run_loss.update(loss.item())        # 仅累计训练 loss
        if global_step % eval_num == 0: save_checkpoint(...)   # eval_num 只控制存盘频率
```

- **`eval_num`（默认 20,000）只控制 checkpoint 保存频率，不执行任何验证**（无 val loader、无 Dice/acc 计算）。
- 唯一在线监控是每 `log_every` 步打印的 `run_loss.avg`（AverageMeter）、学习率、单步 data/compute 耗时与 `MaxMem`（峰值显存）。
- 因此**预训练 loss 下降只是收敛信号，不代表表征质量**； cannot 用它判断模型好坏。

### 13.2 训练期工程性验证（非科学评估）

用于确认管线正确、能跑通，**不评价模型性能**：

| 手段 | 入口/脚本 | 验证内容 |
|---|---|---|
| CPU 数据检查 | `voco_train.py/ocl_train.py --data_check_only`（需 `--dataset_mode ctrate_subset`） | 不建模型/不初始化 CUDA，仅跑 transform/collate，核对 query/base/label 的键与 shape |
| Stage A 冒烟 | `scripts/train_ctrate_subset_smoke.sh`（`NUM_STEPS=10`） | 10 步 path/shape/forward/backward 通路 |
| Stage B 冒烟 | 同上（`NUM_STEPS=500, CACHE=1`） | cache 落盘 + checkpoint 保存 |
| Stage C 稳定性/resume | 同上（`NUM_STEPS=5000` + `RESUME=1`） | 长训稳定性与断点续训 |
| resume 快检 | `NUM_STEPS=2`→`NUM_STEPS=4 RESUME=1` | checkpoint/resume 工程正确性 |

CT-RATE subset（`tools/build_ctrate_subset.py` 生成患者去重清单）只为在不动 160K 全量数据的前提下验证管线，**不是性能评测**。

### 13.3 预训练产物：评估与下游衔接的桥梁

`pretrain_common.save_final_artifacts` 在训练结束（rank0）保存：

| 产物 | 内容 | 用途 |
|---|---|---|
| `model_current_epoch.pt` / `model_step{N}.pt` | `format_version=2` payload：`state_dict`（完整 head）+ optimizer + scheduler + scaler + `global_step` | 断点续训 |
| `model_final_epoch.pt` | 同上（终态完整 payload） | 续训/归档 |
| `final_model.pt` | 原始完整模型 `state_dict`（legacy） | 兼容旧加载 |
| **`encoder_final.pt`** | `{"global_step", "state_dict": backbone.state_dict()}`，**plain backbone 键（`swinViT.*`、`encoder1.*`…）** | **下游 SwinUNETR 直接加载，无需去 `backbone.` 前缀** |

`encoder_final.pt` 是把自监督编码器迁移到下游的关键：其键名与下游 `SwinUNETR(use_v2=True)` 完全对应。发布的 `VoCo_{B,L,H}_SSL_head.pt` 即此类编码器权重。

### 13.4 科学评估：下游任务迁移评估（本质）

预训练表征质量用**下游任务表现**衡量，这是本仓库 benchmark 的设计本质：

```text
预训练编码器（encoder_final / VoCo_*_SSL_head.pt）
  → 下游 SwinUNETR（get_model + 尺寸检查 strict load，见 §14.4）
  → 在目标任务上 fine-tune（或 linear probe）
  → 用任务指标评测：分割 Dice / 分类 accuracy / 配准 Dice·TRE / VLP 报告·检索指标
  → 与 SuPreM、Swin、clip_driven、MG、UniMiss、DoDNet 等基线对比
```

- README 称“花费 10,000+ GPU 小时评估 50+ 下游任务”，下游 Dice/acc 即预训练方法的最终评分。
- 同一编码器在 `--name=VoCo` 下用 `--use_ssl_pretrained` 加载 SSL_head，与 `VoComni_*`（全监督）及其它基线在相同预处理/ROI 设置下公平对比。

### 13.5 可选：线性探针（linear probing）

`external/OCL/main_linprobe.py` 提供 OCL 的线性探针评估（冻结 backbone、只训练线性头）。VoCo 主线评估以下游全量微调为主，linear probe 多用于自监督表征的辅助诊断。

### 13.6 如何判断预训练“是否到位”

由于预训练无内置验证指标，“到位”无法由预训练自身某个分数判定，需组合以下手段（按可靠性从高到低）：

**（1）下游迁移评估（金标准）**
- 定期把 checkpoint（每 `eval_num` 步的 `model_step{N}.pt` / 终态 `encoder_final.pt`）迁移到下游微调，比较 Dice/acc。
- 全量 50+ 任务太贵（10,000+ GPU 小时），实操选**代表性代理子集**（如 BTCV/AMOS/Spleen 几个标准分割 + 一个分类）做 checkpoint sweep。
- 画“下游指标 vs 预训练步数”曲线，**性能不再随步数提升（饱和/plateau）即认为到位**，取最优 checkpoint。

**（2）Linear probe（更便宜的代理）**
- 冻结 backbone、只训线性头，在少量标注数据上评估；比全量微调便宜，可更频繁跑，作为表征质量快速代理。

**（3）训练 loss 曲线（弱信号，必要不充分）**
- 观察 `run_loss.avg` 是否 plateau。但 VoCo 的 `intra+inter+b_loss` 为复合损失，**绝对值不可解释、不可跨配置/跨方法比较**；loss 下降≠表征好，plateau 也不保证最优，仅作“还在不在学”的参考。

**（4）工程/预算信号**
- 默认 `num_steps=2,000,000` 是**预设算力预算，不是收敛判据**；`warmup_cosine` 调度跑完即停。可配合“相邻 checkpoint loss 差小于阈值”辅助判断。

**推荐流程**：固定算力下，每隔一段步数存 checkpoint → 在 3~5 个代理下游任务（或 linear probe）上评测 → 画饱和曲线 → 选 plateau 处权重。

**本仓库特有注意**：
- 对比负样本仅在各 rank 本地 batch（不跨卡 all-gather），loss 尺度随 per-rank batch 变化，比较时须固定 batch 配置。
- VoCo loss 与 OCL(InfoNCE) loss 量纲不同，不能互比。
- 仅验证管线是否跑通用 `--data_check_only` / CT-RATE subset 冒烟测试，那不是质量评估。

---

## 14. 下游任务完整流程

下游有两套独立实现：`Downstream/monai/`（每数据集自包含，`SwinUNETR` 为主）与 `Downstream/nnUNet/`（nnU-Net v2 + VoCo trainer）。

### 14.1 准备阶段

1. **下载预处理数据集**：从 HuggingFace（`Luffy503/VoCo_Downstream`）获取，多为 nnU-Net 风格布局：
   ```text
   YOUR/DOWNSTREAM/DATA/<Dataset>/
     ├── imagesTr/*_0000.nii.gz   # 单通道 CT，_0000 为模态后缀
     └── labelsTr/*.nii.gz        # 单通道类别 ID，不带 _0000
   ```
2. **组织预训练权重**：把 `VoComni_*.pt` / `VoCo_*_SSL_head.pt` 及对比方法 checkpoint 放入同一 `pretrained_root` 目录（见根 README 清单）。
3. 部分数据集（如 WORD）需向原作者申请，从原始链接下载。

### 14.2 MONAI 微调流程（以 `3D-IRCADb` 为例）

```bash
cd Downstream/monai/3D-IRCADb
source activate YOUR-CONDA-ENVIRONMENT
sh train.sh
```

`train.sh` 需修改的参数（**必改**）：

| 参数 | 含义 |
|---|---|
| `name` | 预训练族：`VoCo/suprem/swin/clip_driven/mg/unimiss/dodnet`；`None`=从头训练 |
| `pretrained_root` | 预训练权重目录 |
| `feature_size` | 48=B，96=L，192=H（须与权重一致） |
| `data_dir` | 数据集根目录 |
| `cache_dir` | 缓存目录（`use_persistent_dataset` 启用，需额外存储） |
| `master_port` | 每个进程用不同端口 |
| `logdir` | 结果保存目录 |
| `use_ssl_pretrained` | true→`VoCo_*_SSL_head`，否则 `VoComni_*` |
| `use_persistent_dataset` | true→`PersistentDataset` 缓存加速 |

执行链路：

```text
torchrun main.py
  → get_loader(args)：train/val transform + PersistentDataset/CacheDataset + Sampler
  → get_model(args)：按 name 构建网络并尺寸检查加载预训练权重（§14.4）
  → DiceCELoss + DiceMetric + sliding_window_inference(inferer) + optimizer/scheduler
  → trainer.run_training：train_epoch(AMP) / 定期 val_epoch / TensorBoard / best checkpoint(model.pt)
```

预处理设置（`a_min/a_max/roi/spacing`）由各目录 `main.py` 精心默认（借鉴 nnU-Net，10,000+ GPU 小时调得）；**可能与预训练不一致**（如预训练 roi=64、部分下游 roi=96），公平对比建议沿用仓库默认。

### 14.3 预训练 → 下游权重衔接（`models/models.py::load`）

```text
get_model(args) → 如 name=VoCo：
  按 feature_size + use_ssl_pretrained 选 VoComni_{B,L,H}.pt 或 VoCo_{B,L,H}_SSL_head.pt
  → load(model, model_dict)：
      取 state_dict（state_dict/network_weights/net/student 键）
      去 module./backbone. 前缀、swin_vit→swinViT
      逐键 shape 一致才采用，否则保留当前权重 → strict load
```

- 因 `encoder_final.pt` 已是 plain backbone 键，加载无需去 `backbone.` 前缀即可对齐 SwinUNETR。
- 当 `in_channels!=1` 或 `out_channels!=21` 时，首层/末层 shape 不匹配自动跳过（随机初始化），其余层照常加载——这就是预训练编码器适配任意下游通道数的机制。

### 14.4 验证与测试（`Downstream/val.py` / `test.py` 模板）

`val.py`（带标签，算指标 + 存预测）：

```text
get_test_loader：扫描 test_data_path，label 由 name[:-12]+'.nii.gz' 配对（去掉 "_0000.nii.gz" 12 字符后缀）
  test_transforms：Load→ChannelFirst→RAS→Spacing→ScaleIntensityRange→CropForeground→SpatialPad（全确定性）
SwinUNETR(use_v2=True) ← torch.load(trained_pth)["state_dict"]，strict=True（整模加载，非尺寸检查）
sliding_window_inference(roi=96, sw_batch_size=4, overlap=0.75)
DiceMetric(include_background=False, reduction=MEAN, get_not_nans)：
  post_label=AsDiscrete(to_onehot=out_channels)，post_pred=AsDiscrete(argmax, to_onehot)
  逐病例聚合 Dice，run_acc 累计
Invertd(nearest_interp) 反变换回原始空间 → SaveImaged → 重命名 name[:-7]+'_trans.nii.gz' → name[:-12]+'.nii.gz'
```

`test.py`：同结构但**无标签/无指标**，仅推理 + 反变换 + 保存预测到 `save_prediction_path`。

> 两文件都是**需逐任务编辑的模板**：`test_data_path/test_label_path/trained_pth`、`in_channels/out_channels`、`a_min/a_max/spacing/roi` 必须与训练一致。`val.py` 默认 `out_channels=14`、`CUDA_VISIBLE_DEVICES="5"` 等均为示例值。

### 14.5 nnU-Net 下游流程（`Downstream/nnUNet`）

```bash
# 1. raw 数据 + dataset.json(v2: channel_names/labels/numTraining/file_ending)
# 2. 规划与预处理
nnUNetv2_plan_and_preprocess -d 503 -c 3d_fullres --verify_dataset_integrity
# 3. 用 VoCo 自定义 trainer 训练（加载 VoComni_nnunet.pt）
nnUNetv2_train 503 3d_fullres all -tr nnUNetTrainer_pre
```

- `nnUNetTrainer_pre`（`nnUNetTrainer_pretrain.py`）构建 plans 驱动的 PlainConvUNet/ResidualEncoderUNet，再用尺寸检查 strict load 加载 `VoComni_nnunet.pt`（**checkpoint 路径硬编码在该文件，需修改**）。
- 数据走 nnU-Net 自己的 `.npy/.npz` 预处理管线，**不复用 MONAI loader/cache**；注意 MONAI 预训练与 nnU-Net 微调的预处理设置不一致。

### 14.6 特殊任务入口

| 任务类型 | 目录 | 说明 |
|---|---|---|
| 分类 | `monai/CC-CCII`、`monai/LUNA16` | CSV/fold 或配置驱动，非 Decathlon JSON；指标 accuracy |
| 配准 | `monai/Registration`（IXI/OASIS） | 从 `.pkl` 读 volume/seg 对，TransMorph 系多方法脚本；指标 Dice/TRE |
| 视觉-语言 | `monai/M2KT`（报告生成）、`monai/CT_CLIP`（词汇分类/报告-体积检索） | 独立框架与多入口；指标 BLEU/CIDEr、检索等 |

### 14.7 下游评估指标速查

| 任务 | 指标 | 计算位置 |
|---|---|---|
| 分割（CT/MRI） | Dice（`include_background=False`，滑窗推理 + 逐病例平均） | `trainer.val_epoch` / `val.py` |
| 分类 | accuracy | `CC-CCII/eval.py`、`LUNA16` |
| 配准 | Dice（形变后）/ TRE | `Registration` 各 infer 脚本 |
| VLP | 报告 BLEU/CIDEr 等、检索召回 | `M2KT/metrics.py`、`CT_CLIP` |

### 14.8 训练循环与验证 / checkpoint 机制（`trainer.py::run_training`）

```text
for epoch in range(start_epoch, max_epochs):
    if distributed: train_loader.sampler.set_epoch(epoch); barrier()   # 每轮重排
    train_loss = train_epoch(...)            # AMP 前向/反向，分布式 all_gather loss
    writer.add_scalar("train_loss", ...)     # TensorBoard（rank0）
    if (epoch+1) % val_every == 0:
        val_avg_acc = val_epoch(...)         # 滑窗推理 + DiceMetric，整卷验证
        val_avg_acc = mean(val_avg_acc)
        writer.add_scalar("val_acc", ...)
        if val_avg_acc > val_acc_max:        # 刷新最佳
            save_checkpoint(..., "model.pt") # 保存 best（含 optimizer/scheduler）
        save_checkpoint(..., "model_final.pt")          # 每次验证都存最新
        if b_new_best: copyfile(model_final.pt → model.pt)
return val_acc_max                            # 返回最佳 Dice
```

- **`model.pt` = 验证 Dice 最佳的权重**（下游评测/发布用它）；`model_final.pt` = 最近一次验证的权重。
- checkpoint 内含 `epoch`、`best_acc`、`state_dict`（DDP 时取 `model.module`）、可选 `optimizer/scheduler`。
- 验证用 `val_every` 控制频率；分布式验证用 `distributed_all_gather(..., is_valid=idx<sampler.valid_length)` 排除补齐样本。
- 训练结束打印 `Best Accuracy`（即最佳平均 Dice）。

### 14.9 优化器与学习率调度（`main.py`）

| 配置 | 实现 |
|---|---|
| `optim_name=adam` | `Adam(lr, weight_decay=reg_weight)` |
| `optim_name=adamw` | `AdamW(lr, weight_decay=reg_weight)` |
| `optim_name=sgd` | `SGD(lr, momentum, nesterov=True, weight_decay)` |
| `lrschedule=warmup_cosine` | `LinearWarmupCosineAnnealingLR(warmup_epochs=warmup_epochs×len(train_loader), max_epochs=max_epochs×len(train_loader))`（**按 step 计**） |
| `lrschedule=cosine_anneal` | `CosineAnnealingLR(T_max=max_epochs)`（按 epoch 计；resume 时 `scheduler.step(start_epoch)`） |

- 分布式且 `norm_name=batch` 时先 `SyncBatchNorm.convert_sync_batchnorm`，再包 `DistributedDataParallel`。
- `scheduler.step()` 在每个 optimizer step 后调用（warmup_cosine 是 step 级调度）。

### 14.10 端到端完整示例（3D-IRCADb，肝/肝肿瘤 3 类）

```bash
cd Downstream/monai/3D-IRCADb
# 编辑 train.sh：name=VoCo, feature_size=48, pretrained_root, data_dir=/data/3Dircadb1_convert/,
#               cache_dir, use_ssl_pretrained=False(→加载 VoComni_B.pt), use_persistent_dataset=True
sh train.sh
```

数据流与 shape（`out_channels=3`，roi=96）：

```text
训练：JSON training → Load+RAS+1.5³spacing+HU[-175,250]→CropForeground+pad
      → RandCropByPosNegLabeld(num_samples=K) → flip/rotate/shift
      collate: image [B×K, 1, 96³]，label [B×K, 1, 96³]（类别 ID）
      SwinUNETR logits [B×K, 3, 96³] → DiceCELoss(include_background=False, to_onehot_y, softmax)
验证：整卷 [1,1,Hv,Wv,Dv] → sliding_window(roi=96, overlap) → [1,3,Hv,Wv,Dv]
      → AsDiscrete(argmax/to_onehot) → DiceMetric(include_background=False)
微调产出：runs/.../model.pt（最佳 Dice）
评测：python val.py（编辑 test_data_path/test_label_path/trained_pth/out_channels=3）→ 逐病例 Dice + 存预测
```

### 14.11 预训练模型选择决策

```text
是否使用预训练？
  否 → name=None（从头训练 SwinUNETR use_v2=True）
  是 → 选族 name：
        VoCo（本文）/ suprem / swin / clip_driven / mg / unimiss / dodnet
      选规模 feature_size（须与权重一致）：48=B / 96=L / 192=H
      选权重类型（仅 name=VoCo 有效）：
        use_ssl_pretrained=True  → VoCo_{B,L,H}_SSL_head.pt（自监督预训练）
        use_ssl_pretrained=False → VoComni_{B,L,H}.pt（全监督预训练）
      其余 name 各自固定 checkpoint（如 suprem→supervised_suprem_swinunetr_2100.pth）
```

- 公平对比时所有方法沿用各目录 `main.py` 的默认预处理/ROI（10,000+ GPU 小时调得）。
- nnU-Net 下游单独用 `VoComni_nnunet.pt`（31M）。

### 14.12 下游常见问题与注意事项

- **`val.py`/`test.py` 是需逐任务编辑的模板**：默认 `out_channels=14`、`CUDA_VISIBLE_DEVICES="5"`、`a_min/a_max/spacing/roi` 等都是示例值，必须改成与训练一致，否则滑窗尺寸/类别数错误。
- **label 配对靠文件名后缀**：`val.py` 用 `name[:-12]+'.nii.gz'` 去掉 image 的 `_0000.nii.gz`（12 字符）得到 label 名；预测输出再经 `Invertd` 反变换回原始空间并重命名。数据不是 `_0000` 命名时需自行调整切片偏移。
- **`val.py` 整模 strict 加载**：`torch.load(trained_pth)["state_dict"]` 后 `strict=True`，要求模型结构与训练完全一致；这与 `get_model` 的尺寸检查 `load()` 不同（后者允许首末层 shape 不匹配）。
- **`use_persistent_dataset` 需额外存储**：缓存到 `cache_dir` 加速，但占空间；修改 spacing/HU 窗/确定性 transform 后应换 cache 目录。
- **预处理可能与预训练不一致**（预训练 roi=64，部分下游 roi=96）： fair 对比建议沿用仓库默认，不要照搬预训练参数。
- **多卡**：下游 `main.py` 支持 `--distributed`（torchrun 拉起），每进程用不同 `master_port`；单卡可直接 `python main.py`（视各目录脚本）。
- **特殊任务不走标准模板**：分类（CC-CCII/LUNA16）、配准（Registration，`.pkl` + TransMorph）、VLP（M2KT/CT_CLIP）各有独立入口、数据 schema 与指标，需按各自目录 README 操作。
