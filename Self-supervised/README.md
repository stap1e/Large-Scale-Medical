<div align="center">
<h1>Large-Scale 3D Medical Image Pre-training with Geometric Context Priors</h1>

<a href="https://github.com/Luffy03/Large-Scale-Medical"><img src='https://img.shields.io/badge/arXiv-Preprint-red' alt='Paper PDF'></a>
<a href="https://openaccess.thecvf.com/content/CVPR2024/html/Wu_VoCo_A_Simple-yet-Effective_Volume_Contrastive_Learning_Framework_for_3D_Medical_CVPR_2024_paper.html"><img src='https://img.shields.io/badge/CVPR-Conference-red' alt='Paper PDF'></a>
<a href='https://huggingface.co/Luffy503/VoCo/tree/main'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/datasets/Luffy503/PreCT-160K'><img src='https://img.shields.io/badge/Dataset-PreCT--160K-green' alt='Dataset'></a>
</div>

<div align="center">
    <img src="assets/positions.png" alt="positions" width="500"/>
</div>

We observe that 3D medical images contain consistent geometric context, *i.e.*, consistent geometric relations between different organs, which leads to a promising way for learning consistent representations.
Motivated by this, we propose a simple-yet-effective **Vo**lume **Co**ntrast (**VoCo**) framework to leverage geometric context priors for self-supervision. 
Given an input volume, we extract base crops from different regions to construct positive and negative pairs for contrastive learning. Then we predict the contextual position of a random crop by contrasting its similarity to the base crops.
In this way, VoCo implicitly encodes the inherent geometric context into model representations, facilitating high-level semantic learning without annotations.

![framework](assets/framework.png)

## Pre-trained Models

| Model           | Params |                                           Checkpoint                                           |
|:----------------|-------:|:----------------------------------------------------------------------------------------------:|
| VoCo_B_SSL_head |    53M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_B_SSL_head.pt?download=true) |
| VoCo_L_SSL_head |   206M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_L_SSL_head.pt?download=true) |
| VoCo_H_SSL_head |   818M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_H_SSL_head.pt?download=true) |


## Pre-training

### Download Pre-training Dataset

Please refer to [Acknowledgment](#Acknowledgment). Download our  [PreCT-160K](https://huggingface.co/datasets/Luffy503/PreCT-160K) for pre-training.

The path of PreCT-160K should be organized as:
```
# or you can modify it in 'utils/data_utils*.py'
├── data
    ├── BTCV
    ├── TCIAcovid19
    ├── Luna16-jx
    ├── ...
    └── cache
```
**WARNING**: 
- It requires **22.6 TB** space to store the original datasets. For pre-training, it requires **30 TB** more space to cache the data, otherwise the pre-training will be very slow. And please store them in SSD.

**Some cases:**
![prior](assets/prior.png)

**Descriptions:**
![table1](assets/table1.png)



### Usage

```bash
cd Self-supervised
source activate YOUR-CONDA-ENVIRONMENT
# single GPU, if you don't have enough gpu resource
sh single_train
# multi-gpu
sh dist_B.sh
sh dist_L.sh
sh dist_H.sh
```

### CT-RATE subset pipeline smoke test

This repository also contains an OCL/MiCL entry point (`ocl_train.py`).  The
CT-RATE subset path below changes only the data source: it reuses the existing
chest preprocessing, VoCo view transform, backbone, and the selected entry
point's loss/training logic.  `--dataset_mode original` remains the default and
keeps the original multi-dataset loader.

The original `jsons/ct_rate.json` is deliberately not used or modified.  Build
a patient-disjoint manifest from the files that are actually present:

```bash
cd /path/to/Large-Scale-Medical
python tools/build_ctrate_subset.py \
  --data_root /home/bld/data/dataset/CT_RATE/train \
  --output_json /home/bld/data/dataset/CT_RATE/ct_rate_subset_256.json \
  --num_patients 256 \
  --seed 2026 \
  --selection_mode one_volume_per_patient \
  --validate_header
```

The command writes three files and refuses to replace any of them unless
`--overwrite` is explicitly supplied:

- `ct_rate_subset_256.json`: MONAI Decathlon datalist with relative POSIX paths.
- `ct_rate_subset_256_patients.txt`: selected patients in manifest order.
- `ct_rate_subset_256_skipped.json`: empty, zero-byte, or corrupt input reasons.

Run the CPU-safe data/transform/collate check before allocating a model or
initializing CUDA/DDP:

```bash
cd /path/to/Large-Scale-Medical/Self-supervised
python ocl_train.py \
  --dataset_mode ctrate_subset \
  --data_root /home/bld/data/dataset/CT_RATE/train \
  --datalist_json /home/bld/data/dataset/CT_RATE/ct_rate_subset_256.json \
  --cache_dir /home/bld/data/cache/ctrate_subset_256 \
  --batch_size 1 --workers 0 --no-cache \
  --data_check_only
```

The reused chest transform is `LoadImaged` -> channel first -> RAS -> spacing
`(1.25, 1.25, 5.0)` -> HU `[-1000, 500]` mapped to `[0, 1]` -> foreground crop
-> ROI pad -> fixed pad and random crop to `(192, 192, 64)` -> the existing
`VoCoAugmentation`.  The extra ROI pad and random (not center) fixed-size crop
are existing repository behavior.  With the defaults, the collated training
contract is:

```text
random crops: [2B, 1, 64, 64, 64]
base crops:   [9B, 1, 64, 64, 64]
labels:       [B, 2, 9]
```

OCL consumes the random-crop branch and creates its two masked views in the
existing `OCLHead3D`; VoCo consumes all three branches.  The smoke script uses
OCL by default because it exists in this working tree.  Set
`PRETRAIN_METHOD=voco` to exercise the original VoCo loss instead.

#### Stage A: 10-step path/shape/forward/backward smoke

```bash
cd /path/to/Large-Scale-Medical
GPU_ID=0 NUM_STEPS=10 BATCH_SIZE=1 WORKERS=0 NUM_PATIENTS=256 \
FEATURE_SIZE=48 CACHE=0 LOGDIR="$PWD/Self-supervised/runs/ctrate_stage_a" \
bash scripts/train_ctrate_subset_smoke.sh
```

The script first runs `data_check_only`, then runs at most 10 GPU steps.  If
CUDA is unavailable it exits after the CPU data check.

#### Stage B: 500-step cache/checkpoint smoke

```bash
cd /path/to/Large-Scale-Medical
GPU_ID=0 NUM_STEPS=500 BATCH_SIZE=2 WORKERS=4 NUM_PATIENTS=256 \
CACHE=1 SAVE_EVERY=100 LOG_EVERY=10 \
LOGDIR="$PWD/Self-supervised/runs/ctrate_stage_b" \
bash scripts/train_ctrate_subset_smoke.sh
```

#### Stage C: 5000-step stability and resume run

Start the run (a full 5000-step run is intentionally not part of the smoke
test), then rerun the same command with `RESUME=1` after an interruption:

```bash
cd /path/to/Large-Scale-Medical
GPU_ID=0 NUM_STEPS=5000 BATCH_SIZE=2 WORKERS=4 NUM_PATIENTS=256 \
CACHE=1 SAVE_EVERY=100 LOG_EVERY=20 \
LOGDIR="$PWD/Self-supervised/runs/ctrate_stage_c" \
bash scripts/train_ctrate_subset_smoke.sh

GPU_ID=0 NUM_STEPS=5000 BATCH_SIZE=2 WORKERS=4 NUM_PATIENTS=256 \
CACHE=1 SAVE_EVERY=100 LOG_EVERY=20 RESUME=1 RUN_DATA_CHECK=0 \
LOGDIR="$PWD/Self-supervised/runs/ctrate_stage_c" \
bash scripts/train_ctrate_subset_smoke.sh
```

`NUM_STEPS` is the target `global_step`, not an additional-step count.  A fast
checkpoint/resume engineering check can therefore use targets 2 and 4:

```bash
cd /path/to/Large-Scale-Medical
NUM_STEPS=2 RUN_DATA_CHECK=0 LOGDIR="$PWD/Self-supervised/runs/ctrate_resume_test" \
bash scripts/train_ctrate_subset_smoke.sh
NUM_STEPS=4 RESUME=1 RUN_DATA_CHECK=0 LOGDIR="$PWD/Self-supervised/runs/ctrate_resume_test" \
bash scripts/train_ctrate_subset_smoke.sh
```

`model_current_epoch.pt` and `model_final_epoch.pt` contain `global_step`, full
head weights, optimizer, scheduler, and AMP scaler state.  `final_model.pt`
keeps the legacy full-head state dict.  `encoder_final.pt` contains plain
`backbone.state_dict()` keys (`swinViT.*`, `encoder*`) inside its `state_dict`
field for downstream SwinUNETR loading.

For example, a two-GPU OCL launch is:

```bash
cd /path/to/Large-Scale-Medical/Self-supervised
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=28814 \
  ocl_train.py \
  --dataset_mode ctrate_subset \
  --data_root /home/bld/data/dataset/CT_RATE/train \
  --datalist_json /home/bld/data/dataset/CT_RATE/ct_rate_subset_256.json \
  --cache_dir /home/bld/data/cache/ctrate_subset_256 \
  --batch_size 2 --workers 4 --cache \
  --num_steps 500 --eval_num 100 \
  --feature_size 48 --logdir runs/ctrate_ddp
```

The subset loader uses `DistributedSampler`; OCL additionally drops incomplete
per-rank batches.  Each manifest contains one volume per patient, so `_1` and
`_2` reconstructions of a scan cannot become independent patient negatives.
For a future full training run, one could instead select one reconstruction per
scan or explicitly model `_1`/`_2` as positives; that is outside this subset
pipeline task.


## Acknowledgement <a name="Acknowledgment"></a>

 **NOTE THAT** we are not the authors of these datasets. Although all these datasets are publicly available for academic research, you need to cite the original works as shown in our paper. For certain datasets (e.g., [WORD](https://github.com/HiLab-git/WORD)) that necessitate approval from the authors, you need to download it from the original link.

## Citation

If you find this repo useful for your research, please consider citing the paper as follows:

```bibtex
@article{wu2024large,
  title={Large-Scale 3D Medical Image Pre-training with Geometric Context Priors},
  author={Wu, Linshan and Zhuang, Jiaxin and Chen, Hao},
  journal={arXiv preprint arXiv:2410.09890},
  year={2024}
}
@InProceedings{voco-v1,
    author    = {Wu, Linshan and Zhuang, Jiaxin and Chen, Hao},
    title     = {VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis},
    booktitle = {CVPR},
    month     = {June},
    year      = {2024},
    pages     = {22873-22882}
}
```
