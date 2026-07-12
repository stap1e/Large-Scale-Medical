# AGENTS.md — Large-Scale-Medical (VoCo)

Research codebase for **VoCo**, large-scale 3D medical image pre-training and 50+ downstream tasks. It is experimental ML code, not a packaged library.

## Repo shape (important)

- Five **independent** top-level experiment directories, each self-contained (own `main.py`/`trainer.py` + launch `.sh`). There is **no shared package, root `setup.py`/`pyproject.toml`, or monorepo tooling**. Do not assume cross-directory imports.
  - `Self-supervised/` — VoCo contrastive pre-training (`voco_train.py`).
  - `VoComni/` — fully-supervised pre-training on pseudo-labeled data (`main.py`, `trainer.py`).
  - `Semi-supervised/`, `Omni-supervised/` — pre-training with labeled + unlabeled data (`voco_train.py`, `VoComni.json`).
  - `Downstream/` — fine-tuning/eval, split into **`monai/`** (per-dataset `main.py`) and **`nnUNet/`** (follows upstream nnU-Net; see `Downstream/nnUNet/README.md`).
- Each downstream dataset lives under `Downstream/monai/<Dataset>/` with its own `main.py`, `trainer.py`, `train.sh`, `val.py`, `test.py`, `utils/`. Edit **that folder's** scripts; nothing is centrally configured.

## Environment

- Python **3.10.13** (see `requirements.txt`), run inside a **conda env** (the `.sh` scripts start with `source activate YOUR-CONDA-ENVIRONMENT` — replace with your env name).
- Install order matters:
  ```bash
  pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
  pip install -r requirements.txt
  ```
- Key pinned deps: `monai==1.3.0`, `numpy==1.26.2`, `transformers==4.24.0`, `batchgenerators`, `dynamic-network-architectures`. `requirements.txt` is a conda `--file` export (linux-64).

## Launching training

- Use the provided `.sh` scripts; they use `torchrun` or `torch.distributed.launch` with `nproc_per_node=8`. Multi-GPU scripts are `dist_B.sh` / `dist_L.sh` / `dist_H.sh`; single-GPU is `single_train.sh`.
- Pick a script by model size via `feature_size`: **48 = Base (B), 96 = Large (L), 192 = Huge (H)**.
- Example (`Downstream/monai/3D-IRCADb`):
  ```bash
  cd 3D-IRCADb
  source activate YOUR-CONDA-ENVIRONMENT
  sh train.sh
  ```
- Always edit the path variables in the `.sh` before running: `pretrained_root`, `data_dir`, `cache_dir`, `logdir`, and `master_port` (use a distinct port per process).

## Model / checkpoint conventions

- Backbone is MONAI **SwinUNETR** (`use_v2=True`). Default `out_channels=21` (20 organ/tumor classes + background) for VoComni-derived models.
- `--name` selects the pretrained family: `[VoCo, suprem, swin, clip_driven, mg, unimiss, dodnet]`; `None` = train from scratch (no pre-training).
- `--use_ssl_pretrained`: if true, loads `VoCo_*_SSL_head`; else loads `VoComni_*`.
- Checkpoint loading does a **size-checked strict load** (`README` "Load Pre-trained models"): the first/last layer is skipped if `in_channels != 1` or `out_channels != 21`.
- Pre-trained weights live in one `pretrained_root` dir holding all `.pt`/`.pth` files (see `README` "The path of pre-trained models").

## Data / storage (easy to get wrong)

- Pre-training dataset layout (PreCT-160K) is `data/<Dataset>/...` with a `cache/` dir; override in `utils/data_utils*.py` if needed.
- **Storage is huge**: PreCT-160K needs ~22.6 TB + ~30 TB cache (SSD recommended); VoComni ~10 TB. Without the cache, training is extremely slow.
- `--use_persistent_dataset` caches to `cache_dir` for speed — it requires the extra storage, don't enable blindly.

## Validation / testing

- `python val.py` and `python test.py` per downstream dataset; edit their in-file parameters (`test_data_path`, `test_label_path`, `trained_pth`, input/output channels, processing params) to match training.

## No test / lint / CI

There is **no test suite, linter, typechecker, formatter, or CI** in this repo. Verification is by running the training/eval scripts manually. Don't look for `pytest`, `ruff`, GitHub Actions, or pre-commit — they don't exist.

## Docs to trust

Per-directory `README.md` files are the canonical docs (root, `Downstream/`, each pre-training dir). Prefer them over the root `README.md` for directory-specific launch steps.
