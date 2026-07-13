# 数据集加载与调度逻辑

本文记录仓库五个独立实验目录的数据集入口、清单格式、预处理、缓存、采样和训练消费方式。重点说明 `Self-supervised/` 的混合预训练逻辑。仓库不提供根级统一 Dataset；各实验目录必须从自己的工作目录启动。

本文按“磁盘文件 → manifest → Dataset transform → DataLoader batch → 训练循环”的顺序描述数据。即使不查看源码，也可以据此准备目录、编写兼容 manifest，并判断一个 batch 的键、维度和用途。除特别说明外，以下固定 shape 均来自源码；外部数据的原始空间大小只能写成变量，因为仓库不附带病例文件。

## 1. 仓库数据管线结构

```text
Large-Scale-Medical/
├── Self-supervised/                 # PreCT-160K，VoCo/OCL 自监督预训练
│   ├── single_train.sh              # 单进程启动
│   ├── dist_{B,L,H}.sh              # 8进程，不同模型规模
│   ├── voco_train.py                # VoCo 训练入口
│   ├── ocl_train.py                 # OCL 入口，复用同一数据管线
│   ├── models/voco_head.py          # VoCo backbone与三类损失
│   ├── jsons/                       # 仓库内 Decathlon 风格清单和 Atlas TXT
│   └── utils/
│       ├── data_utils.py            # 三大区域合并与最终 DataLoader
│       ├── data_utils_abdomen.py    # 腹部数据
│       ├── data_utils_tumor.py      # 肿瘤/HealthyCT 数据
│       ├── data_utils_headneck.py   # 头颈及其他数据
│       ├── data_utils_chest.py      # 胸部数据
│       ├── dataloader_bdmap.py      # AbdomenAtlas TXT 加载
│       ├── data_trans.py            # 预处理和分布式 Sampler
│       ├── ops.py                   # query/base batch展开
│       └── voco_trans.py            # VoCo query/base crop 生成
├── VoComni/                         # VoComni 全监督预训练
│   ├── main.py
│   ├── VoComni.json
│   └── utils/data_utils.py
├── Semi-supervised/                 # VoComni 有标签 + imagesUn 无标签
│   ├── voco_train.py
│   ├── VoComni.json
│   ├── gen_json.py
│   └── utils/data_utils.py
├── Omni-supervised/                 # PreCT 自监督流 + VoComni 监督流
│   ├── voco_train.py
│   ├── jsons/VoComni.json
│   └── utils/{data_utils*.py,data_utils_omni.py}
└── Downstream/
    ├── monai/<Dataset>/             # 每个任务独立 main/trainer/utils
    └── nnUNet/                      # 上游 nnU-Net v2 数据管线
```

常见数据位于仓库外部的 `/data/...`。代码一般先解析 JSON/TXT 得到路径字典，真正的 NIfTI 影像在 Dataset 被索引时才由 `LoadImaged` 读取。相对路径（例如 `./jsons/...`）相对于进程当前工作目录，因此应先 `cd` 到对应实验目录再运行脚本。

### 1.1 Shape、键和 dtype 约定

本文统一使用下列符号：

```text
X,Y,Z      原始或某一处理中间阶段的三个空间轴，大小随病例变化
C          图像通道数；本仓库绝大多数 CT 管线要求 C=1
B          DataLoader 配置的 volume batch size
B'         当前 rank 实际拿到的 volume 数；最后一个 batch 可能 B'<B
K          每个有标签 volume 采出的 patch 数，即 sw_batch_size
S          Self-supervised 每个 volume 的 query 数，也取 sw_batch_size
R=(Rx,Ry,Rz)  分割训练的 ROI；Self-supervised 的大块固定为 (192,192,64)
```

PyTorch/MONAI 3D 网络使用 channel-first：单个 volume 为 `[C,X,Y,Z]`，batch 为 `[B,C,X,Y,Z]`。磁盘中的标量 NIfTI 通常是 `[X,Y,Z]`；`EnsureChannelFirstd` 后才成为 `[1,X,Y,Z]`。`LoadImaged` 在 MONAI 1.3.0 默认把 NIfTI 数据读为 `float32 MetaTensor`，所以分割 label 虽具有整数语义，进入 transform 后通常也是 float32；最近邻插值用来保持类别编号。

不要从 JSON 的 `tensorImageSize: "4D"` 推断实际 shape。它只是 Decathlon 元数据，loader 并不用它 reshape 数据。真正的 channel 数由文件/manifest 与 `EnsureChannelFirstd` 决定，空间尺寸由文件 header 和 transforms 决定。

### 1.2 仓库实际读取的文件格式

| 格式 | 哪些目录读取 | 读取内容和约束 |
|---|---|---|
| `.nii.gz` NIfTI | 所有预训练目录、绝大多数 MONAI 下游、nnUNet raw | 3D image；分割任务还需同空间的 label。单模态 nnUNet image 名称使用 `_0000.nii.gz`，label 不带模态后缀。|
| Decathlon JSON | Self-supervised、VoComni、Semi/Omni、标准 MONAI 下游 | 顶层 split 数组；元素至少含 `image`，监督任务还含 `label`。值为相对或绝对路径。|
| TXT 病例列表 | Self-supervised/AbdomenAtlas、Downstream/monai/AbdomenAtlas | 每行第一个 tab 字段是 case ID，再拼出 `<case>/ct.nii.gz` 和 `<case>/label.nii.gz`。|
| CSV | `CC-CCII`、`M2KT`、部分 `CT_CLIP` | 分类 fold、临床元数据、报告文本或多标签；列名由各自 Dataset 硬编码。|
| `.npy/.npz` | `CC-CCII`、`CT_CLIP`、LUNA16 部分任务、nnUNet preprocessed | 已预处理数组。key 不统一：CT_CLIP 尝试 `arr_0`/`data`；nnUNet `.npz` 固定 `data`/`seg`。|
| pickle `.pkl` | Registration、nnUNet properties | Registration 直接读取 volume/seg 对；nnUNet 保存 spacing、crop bbox、前景坐标等病例属性。只应读取可信来源的 pickle。|
| PNG/JPEG | M2KT 的 2D 报告分支 | PIL 读取一张或两张图；当前 CTRG 3D 分支改为 NIfTI。|

很多医学数据集原始发布格式是 DICOM，但这些 loader 不扫描 DICOM series；Self-supervised、VoComni 和标准分割入口都假定数据已转换为 NIfTI。把 DICOM 文件夹直接填入 JSON 不会自动完成序列排序、HU 转换或 affine 构建。

### 1.3 Decathlon JSON 的最小契约

典型 Decathlon 风格清单为：

```json
{
  "training": [
    {
      "image": "./imagesTr/case_0000.nii.gz",
      "label": "./labelsTr/case.nii.gz"
    }
  ],
  "validation": []
}
```

MONAI `load_decathlon_datalist(..., base_dir=data_dir)` 会把清单中的相对路径解析到数据根目录。

无标签条目只需：

```json
{"image": "./imagesUn/case_0000.nii.gz"}
```

多模态任务可令 `image` 为路径数组，例如 BRATS21 的四个 MRI 模态：

```json
{"fold": 0, "image": ["case_t1.nii.gz", "case_t1ce.nii.gz", "case_t2.nii.gz", "case_flair.nii.gz"], "label": "case_seg.nii.gz"}
```

此时 `LoadImaged`/`EnsureChannelFirstd` 将图像组织成 `[4,X,Y,Z]`，而 label 仍为 `[1,X,Y,Z]`。普通 CT 条目应得到 `[1,X,Y,Z]`。image 和 label 必须描述同一病例、方向/物理空间可对齐；代码不会根据文件名替用户配对。

`load_decathlon_datalist` 只消费调用指定的 split；顶层 `description`、`labels`、`modality`、`numTraining` 等不会进入 sample。`base_dir` 仅拼接相对路径，绝对路径保持不变。解析结果最初是路径 dict，真正读盘发生在 Dataset 被索引时；Self-supervised 是一个例外点在于其多个 manifest 在模块导入阶段就被解析。

## 2. Self-supervised：统一混合训练

### 2.1 核心结论

`Self-supervised/` 默认不是为每个数据集分别训练模型，而是在一次运行中：

1. 构造腹部、头颈和胸部三个大 Dataset；
2. 把三个区域合并成一个全局 `ConcatDataset`；
3. 使用一个全局 DataLoader 随机取样；
4. 用同一个 VoCo 模型持续训练全部数据。

“统一训练”不表示把所有影像同时载入内存。启动时主要读取所有路径清单；每个病例被采样时才读取影像。启用 `PersistentDataset` 后，确定性预处理结果会在首次访问时落盘缓存。

这是组合与调度设计，不等于当前 checkout 可不修改地跑完全程。DeepLesion/headneck 的 3D resize mode、若干 manifest/cache 路径和训练收尾代码存在后文列出的阻塞或缺陷；修复后仍然是一个统一模型、一个混合 loader，而不是改成逐数据集训练。

默认启动链路：

```text
single_train.sh / dist_{B,L,H}.sh
  → voco_train.py
  → utils.data_utils.get_loader(args)
  → get_ds_abdomen + get_ds_headneck + get_ds_chest
  → 全局 ConcatDataset
  → Sampler/DataLoader
  → (query crops, geometric labels, base crops)
  → VoCoHead
```

`ocl_train.py` 复用相同的 `get_loader()` 和数据权重，只在模型侧改为 OCL；当前 loader 仍会生成 VoCo 的几何标签和 base crops，而 OCL 入口只使用 query 部分。

### 2.2 清单解析时机和路径

各 `data_utils_*.py` 的 `load_decathlon_datalist()` 位于模块顶层。因此 `voco_train.py` 导入 `utils.data_utils` 时就会解析所有清单，不是等到 `get_loader()` 才解析。任一必需 JSON/TXT 缺失都可能使程序在训练初始化前失败。

- 仓库清单：`Self-supervised/jsons/*.json`。
- 部分腹部清单直接从 `/data/<Dataset>/dataset*.json` 读取。
- Atlas 由 `jsons/dataset_list/AbdomenAtlas1.0.txt` 列出病例名。
- 影像根和 cache 根大量硬编码在 `utils/data_utils*.py`，迁移数据时需要逐文件修改。

#### Self-supervised manifest 契约

仓库内当前被主 loader 消费的影像路径均为 `.nii.gz`。允许的条目只有两种：

```text
有标签：{"image":"...nii.gz", "label":"...nii.gz"}
无标签：{"image":"...nii.gz"}
```

各组对字段的实际要求如下。这里的“忽略 label”是指 manifest 可以带 label 路径，但 transform 不读它。

| 分组/数据集 | loader 读取的 split | 条目必须提供 | 说明 |
|---|---|---|---|
| BTCV、FLARE22、AMOS、WORD | 见外部 JSON | `image,label` | 外部 JSON 不在仓库；WORD 的 train/val/test 被合并，因此 test 条目也必须有 label。|
| FLARE23、PANORAMA、LiTS、Pancreas、KiTS、HealthyCT | `training`，部分再加 `validation` | `image,label` | 走有标签腹部 transform；医学 label 最终会被删除。|
| AbdomenAtlas | TXT 全部行 | case ID | 每行第一列拼成 `<case>/ct.nii.gz` 和 `<case>/label.nii.gz`。|
| DeepLesion | `training` | `image` | 唯一走 image-only 腹部 resize 管线的当前数据集。|
| HNSCC、QIN、HeadNeckPET、TCGA-HNSC、TotalSegmentator、Colonography | `training` | `image` | 全部 image-only；原始空间 shape 可不同。|
| LUNA16 | `training` | `image`；当前 JSON 也带 `label` | chest transform 只加载 image，validation 不使用。|
| 其余胸部七个数据集 | `training` | `image` | TCIA-COVID、STOIC21、LIDC、StonyBrook、MELA、CT-RATE、NLST。|

Atlas TXT 示例：

```text
BDMAP_00000001
BDMAP_00000002<TAB>其他可选字段
```

代码执行 `line.strip().split('\t')[0]`，所以只有第一列参与路径生成。主链路硬编码根目录 `/data/AbdomenAtlasMini1.0/`；最终期望：

```text
/data/AbdomenAtlasMini1.0/
└── BDMAP_00000001/
    ├── ct.nii.gz       # 单通道 CT，磁盘通常 [X,Y,Z]
    └── label.nii.gz    # 与 CT 对齐的离散分割图
```

JSON 中可以存在未被使用的 split 或字段。例如 LUNA16 的 `label`、LiTS/Pancreas 的 `test` 都不会因此自动进入训练；以 `load_decathlon_datalist` 调用的 split 和后续 transform 的 `keys=` 为准。

### 2.3 数据集组成

#### 腹部组 `A`

由 `utils/data_utils_abdomen.py`、`utils/data_utils_tumor.py` 和 `utils/dataloader_bdmap.py` 构造：

| 数据集 | 当前使用范围 | manifest | `base_dir` / 原始根 |
|---|---|---|---|
| BTCV | training + validation | `/data/BTCV/dataset_0.json` | `/data/BTCV` |
| FLARE22 | training + validation | `/data/Flare22/dataset.json` | `/data/Flare22/` |
| AMOS2022 | training + validation | `/data/Amos2022/dataset_CT.json` | `/data/Amos2022/` |
| WORD | training + validation + test | `/data/WORD/dataset.json` | `/data/WORD/` |
| FLARE23 | training，4,000 例 | `jsons/flare23_new.json` | `/data/Flare23/` |
| DeepLesion | training，1,618 例，无 label | `jsons/DeepLesion.json` | `/data/DeepLesion/` |
| PANORAMA | training，2,229 例 | `jsons/PANORAMA.json` | `/data/PANORAMA/` |
| AbdomenAtlas | TXT 全部 5,195 例 | `jsons/dataset_list/AbdomenAtlas1.0.txt` | `/data/AbdomenAtlasMini1.0/` |
| LiTS | 当前代码为 training + training，共 216 个索引 | `jsons/dataset_lits.json` | `/data/Dataset003_Liver/` |
| Pancreas | training + validation，共 281 例 | `jsons/dataset_panc.json` | `/data/Dataset007_Pancreas/` |
| KiTS | training + validation，共 489 例 | `jsons/dataset_kits.json` | `/data/Dataset220_KiTS2023/` |
| HealthyCT | 当前实际将 liver 116 例读取三次，共 348 个索引 | `jsons/healthy_ct_*.json` | `/data/HealthyCT/healthy_ct/` |

完整腹部长度取决于四个 `/data/...` 外部清单：

```text
A = |BTCV train+val|
  + |FLARE22 train+val|
  + |AMOS train+val|
  + |WORD train+val+test|
  + 14,376
```

#### 头颈组 `H`

`utils/data_utils_headneck.py` 只读取各清单的 `training`；条目均只有 `image`：

| 数据集 | 数量 | JSON | `base_dir` |
|---|---:|---|---|
| HNSCC | 1,071 | `jsons/HNSCC.json` | `/data/HNSCC_convert_v1/` |
| QIN HeadNeck | 892 | `jsons/QIN_HeadNeck.json` | `/data/QIN_convert_v1/` |
| HeadNeckPET | 384 | `jsons/HeadNeckPet.json` | `/data/HNPC_convert_v1/` |
| TCGA-HNSC | 762 | `jsons/TCGA_HNSC.json` | `/data/TCGA-HNSC_convert_v1/` |
| TotalSegmentator | 1,203 | `jsons/Totalsegmentator_dataset.json` | `/data/Totalsegmentator_dataset/` |
| CT Colonography | 1,730 | `jsons/ColonographyTrials.json` | `/data/CT_COLONOGRAPHY_converted_v1/` |

因此 `H=6,042`。例如 JSON 中 `{"image":"00001.nii.gz"}` 会解析成对应 `base_dir/00001.nii.gz`；TotalSegmentator 则使用类似 `s1405/ct.nii.gz` 的二级路径。

#### 胸部组 `C`

`utils/data_utils_chest.py` 只读取各清单的 `training`：

| 数据集 | 数量 | JSON | `base_dir` / image 示例 |
|---|---:|---|---|
| LUNA16 | 843 | `jsons/dataset_LUNA16_0.json` | `/data/Luna16-jx/subset_0/<uid>.nii.gz` |
| TCIA-COVID | 722 | `jsons/dataset_TCIAcovid19_0.json` | `/data/TCIAcovid19/data/volume-*.nii.gz` |
| STOIC21 | 2,000 | `jsons/stoic21.json` | `/data/stoic21/nii_gz/*.nii.gz` |
| LIDC | 589 | `jsons/LIDC.json` | `/data/LIDC_convert_v1/*.nii.gz` |
| StonyBrook Chest CT | 2,316 | `jsons/StonyBrookChestCT.json` | `/data/StonyBrookChestCT_v1/*.nii.gz` |
| MELA | 770 | `jsons/MELA.json` | `/data/MELA/mela_*.nii.gz` |
| CT-RATE | 47,149 | `jsons/ct_rate.json` | `/data/CT-RATE/dataset/train/<patient>/<study>/*.nii.gz.nii.gz`（按清单字面值） |
| NLST | 84,830 | `jsons/NLST_convert_v1.json` | `/data/NLST_convert_v1/*.nii.gz` |

因此 `C=139,219`。

胸部组没有子数据集均衡采样，因而 NLST 与 CT-RATE 占胸部样本的大多数。

### 2.4 合并和采样权重

`utils/data_utils.py` 先把同一个腹部 Dataset 对象放入 `ConcatDataset` 8 次，再对头颈做同样处理；胸部只保留一次：

```text
train_ds = ConcatDataset([
  abdomen_ds × 8,
  headneck_ds × 8,
  chest_ds × 1
])
```

最终有效长度为：

```text
N = 8A + 8H + C
  = 8A + 187,555
```

区域采样概率为：

```text
P(abdomen) = 8A / N
P(headneck) = 8H / N
P(chest) = C / N
```

这里的“×8”只扩展索引空间，不复制原始影像，也不建立八份缓存。一次完整 DataLoader 遍历会访问每个腹部/头颈病例 8 次、胸部病例 1 次；全局 shuffle 会把这些访问打散。每次访问会重新运行随机 crop 和增强，因此同一病例的八次输入通常不同。

没有 dataset-balanced batch、区域轮询或每数据集固定配额。batch 内可能同时出现不同数据集、不同身体区域的病例。

### 2.5 区域预处理

预处理定义在 `utils/data_trans.py`。

四条管线共同遵循以下 shape 生命周期。设磁盘是单通道 3D NIfTI：

| 阶段 | image shape | label shape | 说明 |
|---|---|---|---|
| manifest | 路径字符串 | 路径字符串或不存在 | 此时没有读影像。|
| `LoadImaged` | `[X,Y,Z]` float32 MetaTensor | `[X,Y,Z]` float32，或不加载 | shape 来自 NIfTI header；不固定。|
| `EnsureChannelFirstd` | `[1,X,Y,Z]` | `[1,X,Y,Z]` | 模型要求单通道；真实 4D 文件需另行核验。|
| `Orientationd(RAS)` | `[1,Xr,Yr,Zr]` | 同步置换/翻转 | 改变轴顺序但不定义固定大小。|
| `Spacingd` | `[1,Xs,Ys,Zs]` | 有标签腹部同 shape | image 双线性；label 最近邻；新大小取决于原 spacing。|
| 强度映射 | shape 不变，值 clip 到 `[0,1]` | 不变 | HU 窗只处理 image。|
| `CropForegroundd` | `[1,Xf,Yf,Zf]` | 有标签腹部同步 | bbox 由 image 非零区域决定。|
| 第一次 pad | 各轴至少 `[64,64,64]` | 有标签腹部同步 | 默认 `roi_x/y/z=64`。|
| 大块构造 | 目标 `[1,192,192,64]` | 不再参与大块 crop | 普通腹部/chest 可精确 random crop；DeepLesion/headneck 的当前 resize mode 有阻塞，见下文。|
| `VoCoAugmentation` | S 个 query + 9 个 base，均 `[1,64,64,64]` | 删除 | Dataset 最终不返回医学 label。|

上述 `float32` 是 MONAI 1.3.0 默认和代码预期；仓库不含真实 volume，无法对每个外部文件的 header、原始 dtype 和原始 `[X,Y,Z]` 做运行时断言。

#### 腹部有标签数据

```text
Load image+label
→ EnsureChannelFirst
→ RAS Orientation
→ spacing=args.space_{x,y,z}，默认 1.5³ mm
→ image HU [-175,250] 映射至 [0,1]
→ 按 image CropForeground
→ pad 至至少 roi 大小
→ pad/random crop 为 192×192×64
→ VoCoAugmentation
```

DeepLesion 使用只加载 image 的变换，设计上在 VoCo 增强前 resize 到 `192×192×64`。但当前对 3D volume 写的是 `Resized(mode="bilinear")`；MONAI/PyTorch 的 5D 三维插值通常要求 `trilinear`，因此这条管线很可能在 resize 处报维度/mode 错误。

#### 胸部

```text
只加载 image
→ RAS
→ spacing=(1.25,1.25,5.0) mm
→ HU [-1000,500] 映射至 [0,1]
→ 前景裁剪/pad
→ 随机裁出 192×192×64
→ VoCoAugmentation
```

#### 头颈

```text
只加载 image
→ RAS
→ spacing=(1.5,1.5,1.5) mm
→ HU [-175,250] 映射至 [0,1]
→ 前景裁剪
→ resize 为 192×192×64
→ VoCoAugmentation
```

这里的 resize 同样写成 3D `mode="bilinear"`，存在与 DeepLesion 相同的运行阻塞。把两处 mode 修成 `trilinear` 后，四类管线才都会向 VoCo 增强提供精确 `[1,192,192,64]`：头颈和 DeepLesion 强制 resize，普通腹部和胸部从更大的规范化 volume 随机裁取。

仍需在运行前抽样检查，以发现损坏 NIfTI、错误 channel 轴、缺少 label 或无法由 foreground crop 处理的异常病例。

### 2.6 医学标签与几何标签

医学分割 label 只在部分腹部数据中与 image 同步执行方向、spacing 和 padding。`VoCoAugmentation` 随后会删除真实 `label`，它不参与自监督采样或损失。胸部和头颈变换从开始就只读取 image。

训练循环中的 `labels` 是在线生成的几何软标签：query crop 与 3×3 base 网格在 XY 平面上的面积重叠比例，不是器官/肿瘤分割标签。

### 2.7 VoCo crop 和 batch 结构

每个成功经过预处理的 `192×192×64` 体数据由 `VoCoAugmentation` 生成如下结构；DeepLesion/headneck 需先修复上一节的 3D resize mode 才能可靠到达此处：

- `S=args.sw_batch_size` 个随机 query，默认 `S=2`；
- `3×3=9` 个固定网格 base crops；
- 每个 query/base 大小均为 `64×64×64`；
- query 中心只在 XY 平面随机，Z 中心固定为 32；
- 一个 `[S,9]` query-base 几何重叠软标签矩阵；
- query 和 base 分别执行随机翻转、90 度旋转和强度平移。

Dataset 单病例输出：

```text
(
  imgs:   list[S]，每项 {"image": float32 MetaTensor [1,64,64,64]},
  labels: NumPy float64 [S,9],
  crops:  list[9]，每项 {"image": float32 MetaTensor [1,64,64,64]}
)
```

它不包含 case ID、dataset ID、身体区域 ID 或医学 label。9 个 base 的顺序是 3×3 XY 网格按外层 `i`、内层 `j` 展开；中心坐标为 `(32/96/160, 32/96/160, 32)`。query 的 XY 中心分别从整数 `[32,159]` 采样，Z 中心固定为 32；几何标签每行是 query 对 9 个 cell 的 XY 面积占比，非负、理论和为 1，最多同时覆盖 4 个 base。

DataLoader 默认 `batch_size=B`，没有 `drop_last=True`，所以应以当前本地 batch 大小 `B'` 描述。MONAI collate 后、调用 `concat_image()` 之前：

```text
imgs:   list[S]，每项 {image: [B',1,64,64,64]}
labels: torch.float64 [B',S,9]
crops:  list[9]，每项 {image: [B',1,64,64,64]}
```

随后 `utils.ops.concat_image()` 沿 crop 维拼接并 reshape：

```text
query tensor = [B'×S,1,64,64,64]
base tensor  = [B'×9,1,64,64,64]
labels       = [B',S,9]
```

默认单卡 `B=4,S=2`：

```text
queries = [8,1,64,64,64]
bases   = [36,1,64,64,64]
labels  = [4,2,9]
```

模型再把 query 和 base 沿第 0 维合并为 backbone 输入 `[B'×(S+9),1,64,64,64]`；默认单卡为 `[44,1,64,64,64]`。分布式脚本通常每 rank 为 `B'=2,S=2`，因此每 rank 的 query/base/label 分别是 `[4,1,64³]`、`[18,1,64³]`、`[2,2,9]`。8 卡只通过 DDP 同步梯度；crop 和跨病例配对都局限于各 rank，没有跨 GPU all-gather。

几何标签由 NumPy Python float 生成，默认 collate 后通常为 `torch.float64`，当前代码没有显式 `.float()`。如果重写损失或混合其他标签，应显式检查 dtype，不能只检查 shape。

模型对 query 与本病例 bases 计算位置预测损失；还把当前病例与本地 batch 中下一个病例的 bases 配对计算 inter-volume loss，并对 9 个 base embedding 加去相关正则。该跨病例配对不识别数据集来源，也不跨 GPU all-gather。

### 2.8 PersistentDataset 缓存

默认 `args.cache=True`，每个子数据集使用自己的 `PersistentDataset` 和 cache 目录。首次访问病例时执行并保存确定性预处理；后续从磁盘缓存恢复，再继续执行随机变换。

第一个随机变换是 `RandShiftIntensityd(prob=0)`。虽然实际概率为 0，它仍属于 MONAI Randomizable 变换，因此构成缓存边界：

```text
会缓存：Load/Orientation/Spacing/Scale/CropForeground/第一个 pad
不缓存：该边界之后的 resize、192 大块裁剪、VoCo query/base 生成和增强
```

因此启用 cache 不会把固定 crop 永久保存，同一病例每次仍能生成不同的自监督视图。修改 spacing、强度窗或确定性预处理后，应避免继续复用旧缓存。

### 2.9 DataLoader 与分布式 Sampler

非分布式运行：

```text
sampler=None
shuffle=True
batch_size=args.batch_size
num_workers=args.workers，默认16
pin_memory=True
drop_last=False（默认）
```

分布式运行使用 `utils.data_trans.Sampler`：

1. 生成全局索引随机排列；
2. 把总长度补齐到 world size 的整数倍；
3. 每个 rank 按 `indices[rank::world_size]` 取自己的样本；
4. 每个 rank 获得 `ceil(N/world_size)` 个索引。

当前 `Sampler.set_epoch()` 虽已定义，但训练代码没有调用。其 `epoch` 保持 0，分布式每次完整遍历都使用 `manual_seed(0)`，所以病例索引顺序固定；随机裁剪和增强仍会变化。非分布式 `shuffle=True` 通常会在新建 iterator 时重新排列。

### 2.10 step 调度和启动脚本

训练没有真正的 epoch 参数。`train()` 每次完整遍历一次 `train_loader`，外层循环不断重复：

```python
while global_step < args.num_steps:
    global_step, loss, best_val = train(..., train_loader, ...)
```

- 默认目标为 `num_steps=2,000,000`。
- `eval_num=20,000` 只控制 checkpoint 保存，不执行验证。
- batch 循环内没有在达到 `num_steps` 后立即 break，因此会跑完当前完整遍历，最终步数可能超出目标。
- 学习率 scheduler 每个 optimizer step 更新一次。
- resume 只恢复模型和 `global_step`，不恢复 optimizer/scheduler。

默认脚本配置：

| 脚本 | 进程数 | 每 GPU volume batch | 常规全局 volume batch | 模型规模 |
|---|---:|---:|---:|---|
| `single_train.sh` | 1 | 4 | 4 | Base 默认48 |
| `dist_B.sh` | 8 | 2 | 16 | Base 48 |
| `dist_L.sh` | 8 | 2 | 16 | Large 96 |
| `dist_H.sh` | 8 | 2 | 16 | Huge 192 |

每个 volume 还会展开为 `S` 个 query 和 9 个 bases；这与 volume batch 数不同。

### 2.11 Self-supervised 使用步骤

1. 按 README 准备 `/data/<Dataset>/...`，并确认所需 JSON/TXT 都存在。
2. 修改 `utils/data_utils_abdomen.py`、`data_utils_tumor.py`、`data_utils_headneck.py`、`data_utils_chest.py` 和 `dataloader_bdmap.py` 中的数据根与 cache 根。
3. 从 `Self-supervised/` 目录启动，保证 `./jsons/...` 能正确解析。
4. 先用少量 worker/单卡检查：各区域打印的数据量、一个 batch 的 crop shape、cache 是否写到预期位置。
5. 再选择 `single_train.sh` 或与 feature size 对应的 `dist_B/L/H.sh`。
6. 多卡时为每个作业设置独立 `master_port`；建议补上 `train_loader.sampler.set_epoch(pass_index)` 后再进行长期训练。
7. 确认存储预算。PreCT-160K 原始数据约 22.6 TB，持久缓存还可能需要约 30 TB。

若只想训练某一类数据，当前没有命令行选择器。需要在 `utils/data_utils.py` 中调整最终 `ConcatDataset`，或在对应 `get_ds_*()` 中删减子 Dataset；同时重新评估有效权重和缓存目录。

### 2.12 Self-supervised 当前实现问题

- LiTS 使用 `train + train`，读取的 validation 未加入。
- HealthyCT 的 liver/pancreas/kidney 三个列表实际都读取 `healthy_ct_liver.json`。
- DeepLesion 和 headneck 对 3D volume 使用 `Resized(mode="bilinear")`；5D 三维插值应使用 `trilinear`，统一混合 loader 采到这些数据时很可能直接报错。
- `data_utils_chest.py` 的 `chest_cache_root='data/cache/chest/'` 少了前导 `/`，会写到相对目录。
- MELA 的 cache 目录直接设置成 `/data/MELA/` 原始数据目录。
- `--cache` 没有使用可靠的布尔解析；`--cache False` 会成为 truthy 字符串，仍可能启用缓存。
- `--noamp` 虽然存在，但入口随后强制设置 `args.amp=True`，不能按参数直觉关闭 AMP。
- 分布式训练没有调用 `sampler.set_epoch()`，完整遍历之间病例顺序固定。
- `dist_H.sh` 传入字符串 `--resume False`，但非空字符串仍会触发 resume。
- resume 不恢复 optimizer/scheduler，学习率计划会重新开始。
- 训练结束引用未定义的 `args.epochs`，最终保存阶段会报错。
- DDP 最终保存还有多 rank 写同一文件及路径拼接问题，需要长期训练前修正。

## 3. VoComni：全监督预训练

入口是 `VoComni/main.py`，在初始化模型后调用 `utils.data_utils.get_loader(args)`。路径参数默认：

```text
data_dir=/data/VoComni
json_list=./VoComni.json
cache_dir=/data/cache/VoComni
use_persistent_dataset=True
```

`VoComni.json` 当前包含约 20,020 个 training 病例和 23 个 validation 病例。相对路径指向：

```text
VoComni/
├── imagesTr/*_0000.nii.gz
└── labelsTr/*.nii.gz
```

一个可直接被 loader 消费的条目是：

```json
{
  "image": "./imagesTr/VoComni_10000_0000.nii.gz",
  "label": "./labelsTr/VoComni_10000.nii.gz"
}
```

默认 `data_dir=/data/VoComni`，所以两条路径实际解析为 `/data/VoComni/imagesTr/...` 和 `/data/VoComni/labelsTr/...`。`_0000` 是 nnUNet/Decathlon 的第 0 个模态后缀，此处表示单通道 CT；不是 case ID 的一部分。image 与 label 都是一个 `.nii.gz`，磁盘通常分别为 `[X,Y,Z]`，加通道后均为 `[1,X,Y,Z]`。

label 是单通道类别 ID 图，不是 21 通道 one-hot。有效语义为 0–20：0 背景，1–13 为脾、双肾、胆囊、食管、肝、胃、主动脉、IVC、静脉、胰腺、右/左肾上腺，14–20 为肝/胰/肾肿瘤、COVID、结肠、结肠癌、肺癌。训练损失内部才把 label 转成 21 类目标；网络 logits 是 21 通道。

注意 JSON 的 `numTraining=20043` 实际把 train 和 validation 总数写在一起；loader 不信任这个字段，而是直接读取数组，因此训练长度仍为 20,020、验证长度为 23。

训练预处理：

```text
Load image+label
→ channel first/RAS
→ 默认1.5³ spacing（image bilinear，label nearest）
→ HU [-175,250] 映射到 [0,1]
→ CropForeground
→ pad至默认96³
→ RandCropByPosNegLabeld，默认pos:neg=9:1
→ 三轴flip/90度rotate/intensity shift
```

`num_samples=args.sw_batch_size`，因此一个病例会产生多个训练 patch；MONAI collate 后，每次优化的实际 patch 数通常约为 `batch_size × sw_batch_size`。验证仅执行确定性预处理，以 batch 1 整卷进入 sliding-window inference。

更精确的 shape 流是：

```text
manifest: image,label 都是路径字符串
Load + channel first: image,label = [1,X,Y,Z]，通常 float32
RAS + 1.5³ mm spacing: [1,Xs,Ys,Zs]
foreground crop + pad: [1,Xp,Yp,Zp]，每轴至少96
RandCropByPosNegLabeld(K=4): 每病例返回 list[4]，每项 image,label=[1,96,96,96]
MONAI collate(B=4): image,label=[16,1,96,96,96]
SwinUNETR logits: [16,21,96,96,96]
```

`pos:neg=9:1` 是选择 patch 中心时的正负采样权重，不保证每个 batch 精确有 90% 正 patch；`image_threshold=0` 还约束负样本中心位于有效 image 区域。随机翻转、90°旋转和强度平移不改变 patch shape。

验证不执行随机 crop，故一个病例输出 `image,label=[1,Hv,Wv,Dv]`，三轴至少 96 但仍随病例变化。DataLoader 再加 batch 维后为 `[1,1,Hv,Wv,Dv]`；sliding-window 以 `96³` ROI 分块，拼回整卷 logits `[1,21,Hv,Wv,Dv]`。

Dataset 选择：

- `use_persistent_dataset=True`：train/val 都用 `PersistentDataset`。
- 否则：train 用 `CacheDataset(cache_num=24, cache_rate=1.0)`，val 用普通 `Dataset`。

对 train 而言，缓存边界是第一个随机变换 `RandCropByPosNegLabeld`：Load、方向、spacing、强度映射、foreground crop 和 pad 可缓存；正负 patch 位置以及后续增强每次访问仍重新采样。val transform 全部确定性，Persistent 模式会缓存预处理整卷。`CacheDataset(cache_num=24)` 不是缓存全数据，只把最多 24 个病例的确定性前缀放在 RAM。

分布式时使用自定义 Sampler，训练器每 epoch 会调用 `train_loader.sampler.set_epoch(epoch)`；验证 sampler 不 shuffle，并利用 `valid_length` 排除补齐样本。`use_persistent_dataset` 同样不是可靠的布尔 CLI 参数，字符串 `False` 仍可能被判为真。

## 4. Semi-supervised：设计逻辑与当前状态

设计目标是同时使用：

```text
labeled_loader   = VoComni image+label
unlabeled_loader = imagesUn image-only
```

设计上共同数据根应为：

```text
<data_dir>/
├── imagesTr/*_0000.nii.gz   # 有标签单通道 CT
├── labelsTr/*.nii.gz        # 0..20 类 ID，和 imagesTr 对齐
└── imagesUn/*_0000.nii.gz   # 无标签单通道 CT
```

`VoComni.json` 的有标签条目与上一节完全相同。无标签 `dataset_unlabeled.json` 应至少是：

```json
{"training": [{"image": "./imagesUn/case_0000.nii.gz"}]}
```

当前 checkout 不包含生成后的 `dataset_unlabeled.json`。`gen_json.py` 通过扫描 `/data/imagesUn/*.nii.gz` 生成标识符，但当前模板错误地写成 `./imagesTr/<id>.nii.gz`，运行前必须改为 `imagesUn` 并确认文件名不会重复附加/丢失 `_0000`。

有标签 loader 基本复制 VoComni 的 JSON、transform、PersistentDataset/CacheDataset 和 DataLoader。无标签清单应由 `gen_json.py` 生成 `dataset_unlabeled.json`；训练前半程只计算监督分割损失，后半程由冻结 teacher 为无标签 image 生成伪标签，再增加半监督损失。

如果修复下面的问题，默认 `B=4,K=4,ROI=96³` 的预期接口是：

```text
labeled["image"] [16,1,96,96,96] float32
labeled["label"] [16,1,96,96,96]，类别ID
unlabeled["image"] [16,1,96,96,96] float32
teacher logits      [16,21,96,96,96]
pseudo label        [16,1,96,96,96]，argmax后的类别ID
student logits      [16,21,96,96,96]
```

这里的 16 是 `B×K`，不是病例 batch。teacher 只在训练中点复制一次 student 参数，之后代码没有 EMA 更新；“冻结”指 `eval()+no_grad()`。

当前目录不能按现状直接运行，主要问题：

1. `voco_train.py` parser 没有定义 `data_dir/json_list/cache_dir/use_persistent_dataset`，但 `utils/data_utils.py` 会访问这些参数。
2. parser 同样没有定义 loader 需要的 `pos/neg`；定义的 `cache` 反而没有被该 loader 使用。
3. 无标签 transform 使用 `RandCropByPosNegLabeld(label_key='label')`，输入字典却只有 `image`；image-only 应改用 `RandSpatialCropSamplesd` 或提供可靠伪标签后再做正负裁剪。
4. README 使用 `imagesUn/`，但 `gen_json.py` 当前生成的相对路径是 `./imagesTr/...`。
5. `get_loader()` 返回 `[train_loader,val_loader]`，入口却把整个列表赋给 `labeled_loader`；第一次 `next()` 得到的是 DataLoader 对象而不是 batch dict。
6. 训练函数只创建一次 labeled/unlabeled iterator，在 `num_steps` 循环中直接 `next()`，没有捕获 `StopIteration` 或重新创建 iterator。即便前述错误修复，有标签流单进程约 5,005 step 就耗尽，远早于默认训练中点。
7. `cache` 与 `use_persistent_dataset` 的参数命名/用法不一致。

因此此目录目前更接近实验性 baseline。使用前应先统一参数、无标签 crop 策略、JSON 目录名和 iterator 重启逻辑。

## 5. Omni-supervised：双数据流

入口 `Omni-supervised/voco_train.py` 同时创建：

```python
train_loader = get_loader(args)       # PreCT/VoCo 自监督流
omni_loader = get_loader_omni(args)   # VoComni image+label 监督流
```

### PreCT 流

`utils/data_utils.py` 及区域 `data_utils_*.py` 基本复制 `Self-supervised/`：腹部×8、头颈×8、胸部×1，产生 query、几何标签和 bases。路径与数据清单同样大量硬编码。

与独立 Self-supervised 的关键参数差异是 `S=sw_batch_size=4`。默认 `B=4` 时：

```text
collate后 imgs:  list[4]，每项 image [4,1,64,64,64]
collate后 bases: list[9]，每项 image [4,1,64,64,64]
几何 labels:    [4,4,9]
concat后 query: [16,1,64,64,64]
concat后 bases: [36,1,64,64,64]
```

这 16 个 query 还会被模型的分割头复用，产生 `[16,21,64,64,64]` 的 `semi_outputs`。因此 Omni 里的 PreCT 流同时提供 VoCo 对比输入和后半程伪标签分割输入。

### VoComni 流

`utils/data_utils_omni.py` 硬编码：

```text
data_dir=/data/VoComni
cache_dir=/data/cache/VoComni
json=./jsons/VoComni.json
```

它执行 image+label 的 spacing、强度缩放、前景裁剪、pad、正负 patch 采样和随机增强；其 DataLoader 固定 `batch_size=1`。`args.cache=True` 时使用 `PersistentDataset`，否则普通 `Dataset`。

该 manifest 有 20,043 个条目且全部在 `training`，没有 validation；文件名是原数据集风格（首项如 `3Dircadb1_10_0000.nii.gz`），并非独立 `VoComni/VoComni.json` 的统一 `VoComni_N_0000.nii.gz`。两份 JSON 只有在磁盘文件名匹配时才可互换。

监督流默认 `ROI=64³,K=4`，固定病例 batch 为 1，因此每 step 给出：

```text
omni["image"] [4,1,64,64,64]
omni["label"] [4,1,64,64,64]，0..20类别ID
监督 logits      [4,21,64,64,64]
```

两个流不是按病例一一对应：一个 PreCT batch 可以混合任意数据集，另一个监督 batch 独立随机取一个 VoComni 病例的 4 个 patch。它们只在同一 optimizer step 中共同贡献 loss。

训练每 step 同时 `next(train_loader)` 和 `next(omni_loader)`，把 VoCo 自监督输入与 VoComni 监督 patch 交给同一模型。当前代码没有可靠的 iterator 循环重启，任一较短 loader 耗尽都可能引发 `StopIteration`；Self-supervised 数据清单中的重复、路径问题和 3D `bilinear` resize 阻塞也会被这一复制版继承。

单进程监督流最多约 20,043 step，DDP 每 rank 更短，因而默认情况下会早于 1,000,000 step 的 teacher 启动点耗尽。两个 sampler 也没有 epoch/pass 级 `set_epoch()`。如果修复循环，建议分别维护两个 pass 计数，并在各自 iterator 重建前设置对应 sampler epoch；不要假设两个 loader 长度相同。

## 6. Downstream/monai

### 6.1 标准分割任务

多数下游分割目录是完全自包含的，例如：

```text
Downstream/monai/3D-IRCADb/
├── main.py
├── trainer.py
├── train.sh
├── val.py / test.py
├── 3D-IRCADb.json
└── utils/data_utils.py
```

`main.py` 定义本任务的 `data_dir/json_list/cache_dir`、spacing、ROI、强度范围和增强概率，然后调用本目录 `utils.data_utils.get_loader(args)`。不存在跨数据集的中央 loader。

典型流程：

```text
Decathlon JSON training/validation
→ Load image+label
→ channel first/RAS
→ task-specific spacing
→ CT强度窗或MRI归一化
→ CropForeground/pad
→ RandCropByPosNegLabeld
→ flip/rotate/intensity aug
→ PersistentDataset或CacheDataset
→ train/val DataLoader
```

标准 CT 分割输入契约与 VoComni 相同：JSON item 为 `image,label` 两个 NIfTI 路径，单病例加载后为 `[1,X,Y,Z]`。训练 transform 的 `RandCropByPosNegLabeld(num_samples=K)` 返回 K 个固定 ROI，默认 collate 会把病例维和 patch 列表展平：

```text
train image [B×K, C, Rx,Ry,Rz]，通常 C=1
train label [B×K, 1, Rx,Ry,Rz]，单通道类别ID
model logits [B×K, out_channels, Rx,Ry,Rz]
val image/label [1,C,Hv,Wv,Dv] / [1,1,Hv,Wv,Dv]
```

`Rx/Ry/Rz`、spacing、HU 窗和 `out_channels` 均由目标目录的 `main.py` 决定，不能从另一个任务复制。验证整卷 shape 是 transform 后的可变尺寸，网络用该目录配置的 ROI 做 sliding-window inference。

15 个目录的 `utils/data_utils.py` 是同一 CT 模板：`AIIB23`、`Aorta`、`ASOCA`、`CHAOS`、`Colon`、`FUMPE`、`Kipa`、`KiTs`、`LiTs`、`Panc`、`Parse22`、`Sliver07`、`StructSeg19`、`TCIA_Panc`、`Vessel`。`3D-IRCADb`、`Abdomen1k`、`Amos`、`BHSD`、`BTCV`、`COVID`、`Flare22`、`LNDb`、`Lung`、`MM-WHS`、`PANORAMA`、`PENGWIN`、`SegThor`、`Spleen`、`Totalsegmentator`、`Word` 的总体接口相同，但 split、标签转换或局部 transform 有差异。

MRI 模板用于 `ACDC`、`ATLAS-MRI`、`BRATS`、`Heart`、`Hip`、`Prostate`，`AMOS-MRI` 近似。它不做 CT HU window，改为对非零体素逐通道 `NormalizeIntensityd(channel_wise=True)`，当前模板也不做 `Spacingd`。因此多模态输入保持 `C>1`：Prostate JSON 的 `image` 是两个 NIfTI 路径，训练 patch 为 `[B×K,2,Rx,Ry,Rz]`；BRATS/BRATS21 为 4 通道。关闭 PersistentDataset 时，这组通常退化为普通 `Dataset`，不是 24 例 `CacheDataset`。

- train loader：参数化 batch、shuffle、workers、pin memory。
- val loader：通常 batch 1、无 shuffle，用 sliding-window inference。
- 分布式：每目录复制一份自定义 Sampler，按 rank 切片并补齐。
- PersistentDataset 关闭时，训练通常只用 `CacheDataset(cache_num=24)`，验证用普通 Dataset。

最后一条主要描述 CT 模板；MRI 模板和特殊任务应以各自实现为准。

MRI 任务仍常用 Decathlon JSON，但变换不同。例如 BRATS 对非零区域逐通道标准化，而不是使用 CT HU 窗。

### 6.2 split 差异

并非所有目录都有显式 validation 数组：

- Abdomen1k：从 training 列表尾部取 72 例作为验证。
- SegThor：尾部 8 例作为验证。
- PENGWIN：尾部 20 例作为验证。
- COVID、BRATS21：根据 JSON 中的 `fold` 字段划分。
- CC-CCII：从 fold CSV 读取训练/验证病例。
- WORD 还提供独立 test loader。

因此修改某个下游任务时，应先检查该目录自己的 `utils/data_utils.py`，不能假设都使用 JSON 的 `validation` 字段。

BRATS21 是重要的 fold + 多模态例外。它只读取 JSON 的 `training` 数组；每项为：

```json
{
  "fold": 0,
  "image": ["flair.nii.gz", "t1ce.nii.gz", "t1.nii.gz", "t2.nii.gz"],
  "label": "seg.nii.gz"
}
```

`fold == args.fold` 的病例作为验证，其余作为训练。image 载入后为 `[4,X,Y,Z]`，按通道非零标准化；label 为 `[1,X,Y,Z]`。原标签 `0,1,2,4` 被重映射为 `0,1,2,3`，仍是单通道离散 label，不是 TC/WT/ET 三通道 mask。默认 `B=1,K=1,ROI=128³`，训练 image/label 为 `[1,4,128³]` 和 `[1,1,128³]`。MM-WHS 的 CT、MRI 两版都会把 `205,420,500,550,600,820,850 → 1..7`，其他值归 0；CT 默认形成 4 个 `64³` patch，MRI 默认形成 4 个 `96³` patch，网络均输出 8 类。

### 6.3 非标准任务的文件 schema 与返回 shape

#### AbdomenAtlas

训练/验证分别读取 `dataset/dataset_list/AbdomenAtlas1.0_{train,val}.txt`。每行第一列是 case ID，拼成：

```text
<data_dir>/<case>/ct.nii.gz
<data_dir>/<case>/label.nii.gz
```

Dataset dict 还带字符串 `name`。设计上的训练接口为标准分割 patch，默认 `B=1,K=4,ROI=96³`，即 image/label `[4,1,96,96,96]`。但当前目录的 loader 导入 `utils.data_trans`，checkout 内没有该模块，也缺少入口所需的若干同级文件，不能据此文档直接宣称可运行；需先补齐原项目文件或修正 import。

#### CC-CCII

这不是分割任务。fold CSV 至少需要 `patient_id,scan_id,target`，仓库文件还含 `zip_file,label,n_slice,...`；metadata CSV 需要 `patient_id` 以及年龄、性别、危重、肝/肺功能、病程列，但当前 `__getitem__` 计算了临床特征后没有返回它。影像路径固定拼为：

```text
<data_dir>/p<patient_id>-s<scan_id>.npy
```

原数组预期为 `[D,512,512]`、值约在 `[0,255]`。归一化并随机/中心裁成 `[D,384,384]`，加通道后单例返回：

```text
{"image": float32 [1,D,384,384], "label": scalar int}
```

batch 为 `[B,1,D,384,384]` 和 `[B]`，默认训练 `B=4`；`D` 不在 loader 内 pad，因此同一 batch 的病例必须已有相同切片数，否则默认 collate 会失败。网络配置为三分类 `out_channels=3`。

#### CT_CLIP

主要 Dataset 同时依赖报告清单和 3D volume。CSV 分支硬编码列 `VolumeName,Findings_EN,Impressions_EN`；当前 `prepare_samples()` 实际又硬编码读取 CTRG JSON 的 `train`，每项至少要有 `id,report`。volume loader 当前走 NPZ，尝试 key `arr_0`，失败后尝试 `data`；注释中保留了 NIfTI `nibabel` 分支。

数组先转轴、乘 1000、clip 到 `[-1000,200]`，再做 `(x+400)/600`（之后没有再次 clip），中心 crop/pad 到 `[480,480,240]`，最后 resize。单例为 `image [1,96,96,96]` 和一个报告字符串；默认训练 batch 为 `[8,1,96,96,96]` 加字符串列表。该目录存在多套训练/推理脚本及大量绝对路径，不能把一个脚本的 JSON/NPZ 契约推广到全部 CT_CLIP 入口。

当前主训练的 CTViT 配置却断言空间尺寸为 `480×480`、patch 为 30、temporal patch 为 15，与 loader 的 `96³` 不兼容，静态上会失败；传入的 `data_folder` 也被当前 `prepare_samples()` 的作者绝对路径覆盖。使用前必须统一实际入口、路径和模型输入尺寸。

独立 inference Dataset 返回 `(image,text,onehot,name)`；`onehot` 的长度由 labels CSV 除 `VolumeName` 外的列数决定。某些 CLI help 写 Excel，但实现调用的是 `pd.read_csv`，应提供真正的 CSV。

#### M2KT

CTRG 3D 分支读取 annotation JSON 的 `train/val/test`，元素至少含 `id,report`；`<image_dir>/<id>` 应是 `LoadImaged` 可读的 volume。另一个 CSV `CTRG_finding_labels.csv` 以 `id` 为首列、后接 14 个 finding 标签。image 被强度映射并 resize 为 `[1,96,96,96]`；报告被 tokenizer 截至 `max_seq_length=150`。

自定义 collate 返回：

```text
image_ids: tuple[str] 长度B
images:    [B,1,96,96,96]
token_ids: [B,Lmax] int64
mask:      [B,Lmax] float32
labels:    [B,14] float32
```

其中 `Lmax` 是当前 batch 最长报告长度，不超过 150。14 个 finding 值只有 CSV 中恰好等于 `1.0` 时才映射为正类；包括 `2` 在内的其他值都变为 0。当前 CTRG 分支忽略 annotation 中的 `image_path` 和 `num_slices`，直接使用 `image_dir/id`；也不做 orientation、spacing 或真正的 CT HU window，而是假定像素在 0–255。目录还保留 IU-Xray/MIMIC/COVID 的 2D PIL 分支：IU-Xray 单例为两张 RGB `[2,3,S,S]`，MIMIC/COVID 为一张 `[3,S,S]`，其 `image_path` schema 与 CTRG 不同。

#### LUNA16

该目录有独立的 Dataset registry，不使用 MONAI 下游通用 loader。以 `ClassificationLUNASet` 为例，训练读取无 header 的 `train_0.csv`/`train_1.csv`，每行 `[label,npy_path]`；验证/测试读取 `<flag>.csv`。`.npy` 原数组 `[Z,Y,X]` 被转为 `[1,X,Y,Z]`，单例返回 `(float32 image, float label[1], image_name)`，batch 再在最前加 B。配置默认期望 `64³`，但 Dataset 不 resize，故 `.npy` 必须预先具有模型所需尺寸。

`datasets_3D/__init__.py` 会无条件导入 `AE`、`CL`、`PTP`，但当前 checkout 缺少这些目录，因此连只想使用现有 Classification/Seg loader 也可能先在 import 阶段失败。不同 registry 项（Seg、MG、PCRL、Rubik cube）读取格式和返回 tuple 不同，应单独查看对应实现。

LUNA 分割另一次性读取 `x_<split>_64x64x32.npy` 与 `m_<split>_64x64x32.npy`，约定 image/mask batch 为 `[B,1,64,64,32]`。该实现还导入 MONAI 1.3 已移除的 `AddChannel`，也是当前兼容性阻塞。

#### Registration

IXI 和 OASIS 都用 `glob('*.pkl')`，pickle 内容不是通用 manifest：

- IXI 病例 pickle 必须能解包为 `(volume, segmentation)`；训练把固定 `atlas.pkl` 作为 moving/fixed 一端，返回 `x,y`，各 `[1,X,Y,Z]`，batch size 固定 1。验证返回 `x,y,x_seg,y_seg`。
- OASIS 训练病例也解包为 `(volume, segmentation)`，每次给当前病例随机配另一个病例，返回四个 `[1,X,Y,Z]` tensor；OASIS inference pickle 则必须直接包含 `(x,y,x_seg,y_seg)` 四项。

两卷在 `torch.cat((x,y),dim=1)` 后成为网络输入 `[B,2,X,Y,Z]`。TransMorph 配置期望 `(160,192,224)`，Dataset 本身不 resize，所以 pickle 必须提前处理到一致尺寸。pickle 可执行任意反序列化代码，只应使用可信数据。

这些目录不遵循标准 `utils/data_utils.py + Decathlon JSON` 模板。并非每个下游目录都同时存在 `val.py` 和 `test.py`；存在时也常有独立硬编码的测试路径、checkpoint、通道数和后处理，未必复用训练 loader。

## 7. Downstream/nnUNet

nnUNet 不复用 MONAI loader，而遵循 nnU-Net v2：

```text
nnUNet_raw/Dataset503_VoComni/
├── imagesTr/*_0000.nii.gz
├── labelsTr/*.nii.gz
└── dataset.json
```

raw 文件名本身就是配对契约。单通道病例 `caseA` 必须是：

```text
imagesTr/caseA_0000.nii.gz   # channel 0
labelsTr/caseA.nii.gz        # 单通道类别ID，不带_0000
```

多模态继续增加 `caseA_0001.nii.gz`、`_0002` 等；所有模态和 label 必须具有兼容的几何信息。nnUNet v2 的最小 `dataset.json` 不是本文件前面展示的 Decathlon split 列表，而是：

```json
{
  "channel_names": {"0": "CT"},
  "labels": {"background": 0, "organ": 1},
  "numTraining": 100,
  "file_ending": ".nii.gz"
}
```

病例由 `imagesTr/labelsTr` 文件名扫描发现；`channel_names` 决定模态数和 CT/MRI 归一化策略，`labels` 的值应从 0 背景开始并通常连续。这里不能直接用 MONAI/旧 Decathlon 的 `modality: {"0":"CT"}` 与 `labels: {"0":"background"}` 形式，除非先转换为 v2 schema。

先运行：

```bash
nnUNetv2_plan_and_preprocess -d 503 -c 3d_fullres --verify_dataset_integrity
```

原始 NIfTI 会被规划和预处理到 `nnUNet_preprocessed`，形成 `.npz`、可解包的 `.npy` 和病例属性 `.pkl`。训练时：

1. `nnUNetDataset` 扫描预处理目录的病例标识；
2. `splits_final.json` 提供五折 train/val，缺失时固定 seed 12345 自动生成；
3. 根据配置维度选择 `nnUNetDataLoader2D` 或 `nnUNetDataLoader3D`；
4. loader 从 `.npy/.npz` 读取 image/seg，并按计划的 patch 和增强策略采样。

预处理的核心顺序是 transpose → crop → resample/normalize。每个病例通常保存：

```text
<case>.npz      keys: data, seg
<case>.pkl      spacing、shape、crop bbox、class_locations等属性
```

解包后则优先 mmap 读取 `<case>.npy` 和 `<case>_seg.npy`。`data` 为 float32 `[C,*spatial]`；`seg` 为 int8/int16 `[1,*spatial]`（级联时可多一层 previous-stage seg）。这些是 nnUNet 自己的预处理结果，不能用 MONAI `PersistentDataset` cache 替代。

3D loader 的预增强 batch 接口是：

```text
{
  "data": float32 [B,C,P0,P1,P2],
  "seg":  int16   [B,Sg,P0,P1,P2],
  "properties": list[dict] 长度B,
  "keys": list[case_id] 长度B
}
```

`P0/P1/P2`、B、前景过采样比例都来自 plan/configuration，不是仓库固定 96³。2D loader 会先选择一个 slice，再返回 `[B,C,P0,P1]` 和 `[B,Sg,P0,P1]`。越过 volume 边界的 image 以 0 pad，seg 以 -1 pad；后续 augmentation 再生成真正送入网络的 tensor。

`splits_final.json` 位于 preprocessed dataset 根，每个元素是 `{"train":[case ids],"val":[case ids]}`。缺失时按排序后的训练 case 用 seed 12345 生成固定五折；命令中的 fold `all` 则把全部病例用于训练/验证。它和 MONAI manifest 的 `validation` 数组没有关系。

仓库自定义 `nnUNetTrainer_pre` 主要修改网络和预训练权重加载，没有重写数据管线。README 明确提示 MONAI 预训练与 nnUNet 微调的预处理设置不一致。

## 8. 各目录对比

| 目录 | 数据源 | Dataset 组合 | 训练输入 | 缓存 | 验证 |
|---|---|---|---|---|---|
| Self-supervised | 多个JSON/TXT、硬编码 `/data` | `8A+8H+C` 全局混合 | query+bases+几何软标签 | 多个 PersistentDataset | 无；只定期存权重 |
| VoComni | `VoComni.json` | 单一有标签数据集 | image/label patches | Persistent或最多24例RAM cache | 整卷滑窗 |
| Semi-supervised | VoComni + unlabeled JSON | 两个独立 loader | labeled + pseudo-labeled image | 设计上Persistent/Cache | 当前实现不完整 |
| Omni-supervised | PreCT混合 + VoComni | 两个同步 loader | VoCo crops + segmentation patches | Persistent或普通Dataset | 无独立验证 |
| Downstream/monai | 每任务自己的JSON/CSV/TXT | 每任务独立 | 任务相关 | 每任务配置 | 通常整卷滑窗 |
| Downstream/nnUNet | raw dataset.json → preprocessed | nnUNetDataset和fold split | 计划生成的2D/3D patches | 预处理 `.npy/.npz` | fold验证 |

## 9. 修改或运行前检查表

建议把“能解析 manifest”“能读一个病例”“能组成一个 batch”作为三个独立关卡。前两关不要启动 DDP，也不要遍历完整 160K 数据集。

| 检查点 | 应验证的条件 |
|---|---|
| manifest | split 存在；每个必需 key 是非空路径；与 `base_dir` 拼接后文件存在；image/label case ID 配对。|
| 单个 NIfTI | image 经过 channel-first 后是 `[C,X,Y,Z]`；值有限；C 与 `in_channels` 一致；label 空间 shape 与 image 一致。|
| 分割 label | unique 值在 `[0,out_channels-1]` 或进入模型前有明确重映射；不能把 one-hot `[K,X,Y,Z]` 当单通道类别图。|
| 标准分割 batch | train 是 `[B×K,C,Rx,Ry,Rz]`，label 是 `[B×K,1,Rx,Ry,Rz]`；val 的 B 为 1、空间尺寸可变。|
| Self-supervised batch | tuple 恰有三项；concat 后 query `[B'×S,1,64³]`、base `[B'×9,1,64³]`、几何 label `[B',S,9]`。|
| 双流任务 | 分别打印两个 loader 的长度、键和 shape；确认较短 iterator 的重建策略，不能只验证第一步。|
| cache | 写入的是专用 cache 根；修改 spacing/window/确定性 transform 后使用新 cache 或清理旧 cache。|

- 从正确的实验目录启动，确认相对 JSON/TXT 路径。
- 检查启动脚本和 Python 文件中的 `data_dir`、`cache_dir`、`json_list`。
- 不要把字符串形式的 `False` 当作可靠布尔值；检查 argparse 定义。
- 确认 JSON 的 `training/validation/test` 实际被如何拼接。
- 检查 cache 是否有足够空间，且没有指向原始数据目录。
- 多卡任务使用独立 `master_port`。
- Self-supervised 中修改数据集后重新计算 `8A+8H+C` 权重。
- 检查 DDP sampler 是否每轮调用 `set_epoch()`。
- 先读取一个病例、再跑一个 batch，核对 image/label/crop shape 后再启动长期训练。
- 下游任务始终以目标数据集目录内的 README、`main.py` 和 `utils/data_utils.py` 为准。
