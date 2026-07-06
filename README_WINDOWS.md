<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ArtiFixer for Windows

This fork adapts [ArtiFixer](https://github.com/nv-tlabs/ArtiFixer) to run on Windows
machines with consumer NVIDIA GPUs, following the approach of
[gaussian-splatting-Windows](https://github.com/jonstephens85/gaussian-splatting-Windows).

**Primary target hardware:** NVIDIA GeForce RTX 5060 Ti — 16 GB VRAM (Blackwell, `sm_120`),
64 GB system RAM, Windows 11 + Docker Desktop (WSL2) or native conda.

## What this fork changes

| Change | Why |
| --- | --- |
| `.gitattributes` forcing LF on `*.sh` / `Dockerfile*` | Windows checkouts with `core.autocrlf=true` broke `install_slangc.sh` inside the Linux build container (`$'\r': command not found`) |
| Dockerfiles normalize CRLF in submodule scripts before running them | The 3DGRUT submodule is not covered by this repo's `.gitattributes` |
| `model_eval.run_inference --cpu_offload` | The 14B checkpoint (~28 GB bf16 weights) cannot fit in 16 GB VRAM; this flag streams transformer weights from CPU RAM layer by layer |
| Captioning model auto-selection (`--captioning_model_id auto`, now the default) | The default Qwen3-VL-30B MoE captioner thrashes disk offload on 16 GB GPUs (7 % GPU utilization); on GPUs with < 40 GiB the dense Qwen3-VL-8B model is selected instead |
| Captioning loads via `AutoModelForImageTextToText` + `offload_folder` | Fixes a crash (`offload_folder` required) and supports both dense and MoE Qwen3-VL variants |
| Fixed `reverse=reverse` → `reverse_frames=reverse` in `data_processing/run_captioning.py` | Upstream bug: keyword mismatch crashed captioning with `--include_reverse` variants |
| `TORCH_EXTENSIONS_DIR` uses `tempfile.gettempdir()` instead of `/tmp` | `/tmp` does not exist on Windows |
| `install_env_windows.ps1` | Native conda setup script (PyTorch 2.11 cu128, SDPA attention, no flash-attn) |

## Choosing an installation path

| | Path A: Docker Desktop + WSL2 (**recommended**) | Path B: native Windows conda (experimental) |
| --- | --- | --- |
| ArtiFixer inference | ✅ | ✅ (`--cpu_offload` for 16 GB GPUs) |
| Captioning (data prep) | ✅ | ✅ |
| 3DGRUT sparse reconstruction / ArtiFixer3D | ✅ | ⚠️ experimental (JIT CUDA build via MSVC) |
| Training | ✅ (hardware permitting) | ❌ (needs Triton/NCCL; not supported) |
| flash-attention | FA3/FA4 built in image (unused on sm_120) | not installed — PyTorch SDPA fallback |

On Blackwell GPUs (`sm_120`, RTX 50-series) the code auto-selects PyTorch SDPA (cuDNN)
attention; flash-attention 3 is Hopper-only and is never used on this GPU, so the native
path loses no inference performance by skipping it.

---

## Path A — Docker Desktop + WSL2 (recommended)

### A.0 Prerequisites

- Windows 11, NVIDIA driver ≥ 570 (CUDA 12.8 support; check with `nvidia-smi`)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with the WSL2 backend
- Git for Windows

> **WSL2 memory:** by default WSL2 gets ~50 % of host RAM. Building flash-attention with
> many parallel jobs can trigger the Linux OOM killer *inside the WSL2 VM* and kill Docker
> itself (the build fails with `rpc error: ... EOF`). Either build with fewer jobs (below)
> or raise the VM budget in `%UserProfile%\.wslconfig`:
>
> ```ini
> [wsl2]
> memory=48GB
> ```

### A.1 Clone and build

```powershell
git config --global core.autocrlf false   # or rely on this repo's .gitattributes
git clone --recurse-submodules https://github.com/<your-fork>/ArtiFixer-Windows.git
cd ArtiFixer-Windows

# FLASH_ATTN_MAX_JOBS=4 keeps peak build memory within a ~32 GB WSL2 VM.
docker build --build-arg FLASH_ATTN_MAX_JOBS=4 -f Dockerfile.cuda12 -t artifixer:cuda12 .
```

### A.2 Run the container

PowerShell (note `${PWD}`; in cmd.exe use `%cd%` instead — `$PWD` does not work there):

```powershell
docker run --gpus all --ipc=host --rm -it `
    -v "${PWD}:/workspace/artifixer" `
    -v "D:\path\to\ArtiFixer-data:/data" `
    artifixer:cuda12
```

Inside the container the mount is lowercase `/workspace/artifixer` (Linux is case-sensitive).

### A.3 Download the checkpoint and log in to Hugging Face

```bash
cd /workspace/artifixer
mkdir -p /data/artifixer-checkpoints
hf download nvidia/ArtiFixer artifixer-14b.pt --local-dir /data/artifixer-checkpoints
export CHECKPOINT_PT=/data/artifixer-checkpoints/artifixer-14b.pt

# DL3DV is a gated dataset: accept its terms on huggingface.co first, create a
# Read token at Settings -> Tokens, then:
hf auth login
```

(The image ships the new `hf` CLI; the old `huggingface-cli` name is deprecated.)

### A.4 Demo scene (DL3DV) pipeline

The DL3DV zips contain `images_4/` + `transforms.json` (NeRFStudio format), **not** COLMAP
binaries — use this pipeline, not `prepare_colmap_artifixer_inputs` (that one is for your
own COLMAP scenes). The scripts read the `.zip` directly; no extraction needed.

```bash
export DL3DV_ROOT=/data/DL3DV-ALL-960P
export SCENE_ID=15ff83e2531668d27c92091c97d31401ce323e24ee7c844cb32d5109ab9335f7

# 1. Download one scene
python scripts/download_dl3dv_scene.py --local-dir "$DL3DV_ROOT" --scene-id "$SCENE_ID" --subdir 8K

# 2. Caption HDF5 (auto-selects Qwen3-VL-8B on 16 GB GPUs)
python -m data_processing.run_captioning \
    --dl3dv_dir "$DL3DV_ROOT" \
    --output_dir /data/artifixer-data/DL3DV-ALL-960P-captions \
    --scene_id "$SCENE_ID" \
    --hf_cache_dir /data/hf-cache

# 3. Sparse reconstruction HDF5 (3DGRUT)
python -m data_processing.run_sparse_reconstruction \
    --dl3dv_dir "$DL3DV_ROOT" \
    --output_root /data/artifixer-data/reconstructions \
    --work_root /data/artifixer-work/reconstructions \
    --scene_id "$SCENE_ID" \
    --num_selected_indices 2 3 6 12

# 4. Train/test split
python -m data_processing.trainval_test_split \
    --data_path /data/artifixer-data/reconstructions \
    --dl3dv_dir "$DL3DV_ROOT" \
    --output_root /data/artifixer-data

# 5. Inference — see the 16 GB flags below
export SPLIT_PATH=/data/artifixer-data/trainval_test_split.json
export PROMPT_ROOT=/data/artifixer-data/DL3DV-ALL-960P-captions
export SAVE_DIR=/data/artifixer-eval

python -m model_eval.run_inference \
    --evalset 3dgrut_dl3dv_ours \
    --checkpoint_pt "$CHECKPOINT_PT" \
    --save_dir "$SAVE_DIR" \
    --split_path "$SPLIT_PATH" \
    --dl3dv_dir "$DL3DV_ROOT" \
    --prompt_dir "$PROMPT_ROOT" \
    --cpu_offload \
    --max_neighbors_per_encode 1 \
    --save_frame_outputs_only
```

---

## Path B — Native Windows (experimental)

### B.0 Prerequisites

- Windows 11, NVIDIA driver ≥ 570
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Git for Windows
- Only for 3DGRUT (`-With3DGRUT`): Visual Studio 2022 Build Tools with the
  "Desktop development with C++" workload

### B.1 Install

```powershell
git clone --recurse-submodules https://github.com/<your-fork>/ArtiFixer-Windows.git
cd ArtiFixer-Windows
.\install_env_windows.ps1                # inference + captioning
# .\install_env_windows.ps1 -With3DGRUT  # additionally: sparse reconstruction / ArtiFixer3D
conda activate artifixer
```

The script installs PyTorch 2.11.0 + cu128 (Blackwell `sm_120` wheels), the pinned
diffusion stack from `Dockerfile.cuda12`, and skips flash-attention entirely — the model
falls back to PyTorch SDPA automatically (you will see
`Attention config: sm_120 (Blackwell) — auto SDPA` at startup).

### B.2 Run inference

```powershell
conda activate artifixer
$env:CHECKPOINT_PT = "D:\ArtiFixer-data\artifixer-checkpoints\artifixer-14b.pt"

python -m model_eval.run_inference `
    --evalset reconstructed_colmap `
    --checkpoint_pt "$env:CHECKPOINT_PT" `
    --save_dir "D:\ArtiFixer-data\artifixer-corrected" `
    --split_path "D:\ArtiFixer-data\artifixer-prep\my_scene\split.json" `
    --render_trajectory all_frames `
    --cpu_offload `
    --max_neighbors_per_encode 1 `
    --save_frame_outputs_only
```

Data preparation for your own COLMAP scenes (`prepare_colmap_artifixer_inputs`) and the
ArtiFixer3D stage require 3DGRUT (`-With3DGRUT` install, or run those stages in Docker
and share the data directory).

---

## Running on 16 GB VRAM (RTX 5060 Ti)

The released checkpoint is a 14B-parameter transformer: ~28 GB of bf16 weights, which do
not fit in 16 GB of VRAM. These flags make it run:

| Flag | Effect | Cost |
| --- | --- | --- |
| `--cpu_offload` | Transformer weights stay in CPU RAM and stream to the GPU per layer (accelerate hooks). Requires ~32 GB free system RAM. | Inference slows down (PCIe-bound weight streaming per denoise step) |
| `--max_neighbors_per_encode 1` | VAE-encodes neighbor frames one at a time | Slightly slower encode |
| `--save_frame_outputs_only` | Skips comparison/diagnostic video rendering | Fewer outputs |
| `--num_inference_steps 4` (default) | Distilled checkpoint needs only 4 steps | — |
| Captioning `--captioning_model_id auto` (default) | Picks Qwen3-VL-8B below 40 GiB VRAM (mostly on-GPU, no disk thrash) | Slightly weaker captions than 30B |

Additional tips:

- Close other GPU consumers (browsers with hardware acceleration, games) — the model uses
  every MiB available.
- On Windows, "Hardware-accelerated GPU scheduling" plus the WDDM display stack reserve
  some VRAM; a headless/secondary GPU has more available.
- Keep at least ~35 GB of free system RAM when using `--cpu_offload` with the 14B model
  (weights + activations + OS).

## Training on consumer GPUs (RTX 5060 Ti / RTX 5090)

Training always runs inside Docker/WSL2: the trainer is hard-wired to FSDP2, which needs
NCCL (Linux-only), even on a single GPU. Stage-1 SFT does not use flex_attention, so no
Triton/flash-attn is required. What fits where:

| Scenario | Approx. VRAM | RTX 5060 Ti 16 GB | RTX 5090 32 GB |
| --- | --- | --- | --- |
| 14B full fine-tune | ~140 GB | ❌ | ❌ (cluster only) |
| 14B LoRA from released checkpoint (`--lora_rank`) | ~29–31 GB | ❌ | ⚠️ tight — see below |
| 1.3B full fine-tune (`--model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers`) | ~11–15 GB | ✅ with the flags below | ✅ comfortable |

This fork adds these `model_training.train` flags for consumer-GPU fine-tuning:

- `--init_checkpoint_pt <path>` — start from a released single-file checkpoint
  (e.g. `artifixer-14b.pt`) instead of the Wan base weights.
- `--lora_rank N` (with `--lora_alpha`, `--lora_dropout`, `--lora_target_modules`) —
  freeze the base weights and train peft LoRA adapters on the attention projections only.
- `--optimizer adamw8bit` — bitsandbytes 8-bit AdamW, quarters optimizer-state memory
  (`pip install bitsandbytes` in the container).

### 14B LoRA fine-tune on RTX 5090 (32 GB)

```bash
accelerate launch --num_processes 1 --module model_training.train \
    --project_dir /data/runs/lora-14b \
    --init_checkpoint_pt /data/artifixer-checkpoints/artifixer-14b.pt \
    --lora_rank 32 \
    --split_path "$SPLIT_PATH" --dl3dv_dir "$DL3DV_ROOT" --prompt_dir "$PROMPT_ROOT" \
    --num_frames 41 \
    --gradient_accumulation_steps 128 \
    --dataloader_num_workers 2 \
    --max_iterations 2000 --save_steps 500 --validation_steps 500 \
    --log_with tensorboard
```

The frozen bf16 base is ~28 GB, so the remaining ~3 GB must cover activations: keep
`--num_frames` at 41 or below (drop to 21 if you OOM) and run the GPU headless if
possible. Training checkpoints store base + adapters; export a merged, single-file
checkpoint that works directly with `model_eval.run_inference --checkpoint_pt`
(pass the **same** `--lora_*` values used for training):

```bash
python -m model_eval.export_checkpoint_pt \
    --checkpoint_dir /data/runs/lora-14b/checkpoints/checkpoint_2000/pytorch_model_fsdp_0 \
    --output_pt /data/artifixer-checkpoints/artifixer-14b-mylora.pt \
    --lora_rank 32 \
    --run_id lora-14b --checkpoint 2000 --slot manual
```

### 1.3B full fine-tune on 16 GB

```bash
accelerate launch --num_processes 1 --module model_training.train \
    --project_dir /data/runs/my-finetune-1.3b \
    --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --split_path "$SPLIT_PATH" --dl3dv_dir "$DL3DV_ROOT" --prompt_dir "$PROMPT_ROOT" \
    --optimizer adamw8bit \
    --num_frames 41 \
    --gradient_accumulation_steps 128 \
    --dataloader_num_workers 2 \
    --log_with tensorboard
```

Notes: reduce `--num_frames` (81 → 41 or 21) if activations still OOM; keep
`--gradient_accumulation_steps` high so the effective batch stays near the paper's 128;
the 1.3B path trains from the Wan 1.3B base (the released ArtiFixer checkpoint is 14B and
cannot initialize a 1.3B model). `bitsandbytes` is required only for `--optimizer adamw8bit`
(`pip install bitsandbytes` in the container).

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `install_slangc.sh: line 4: $'\r': command not found` during `docker build` | CRLF line endings from `core.autocrlf=true`. Fixed in this fork (`.gitattributes` + Dockerfile normalization). For old checkouts: `sed -i 's/\r$//' thirdparty/3DGRUT-ArtiFixer/scripts/install_slangc.sh` |
| `docker build` dies with `rpc error: ... EOF` during flash-attention | WSL2 OOM killer killed Docker. Build with `--build-arg FLASH_ATTN_MAX_JOBS=4` and/or raise `memory=` in `.wslconfig` |
| `docker run` fails: `create $PWD: invalid characters` | You are in cmd.exe. Use `${PWD}` in PowerShell or `%cd%` in cmd.exe |
| `huggingface-cli` says it is deprecated | Use the `hf` CLI (`hf download`, `hf auth login`) |
| `GatedRepoError: 401` downloading DL3DV | Accept the dataset terms on huggingface.co, create a Read token, `hf auth login` |
| `generate_caption_hdf5() got an unexpected keyword argument 'reverse'` | Upstream bug, fixed in this fork |
| `ValueError: ... Please provide an offload_folder` during captioning | Fixed in this fork (offload folder is created automatically) |
| Captioning crawls at ~7 % GPU utilization | The 30B captioner is disk-offloading. Fixed by default (`auto` picks the 8B model on 16 GB GPUs) |
| CUDA OOM loading the 14B checkpoint | Add `--cpu_offload` (and `--max_neighbors_per_encode 1`) |
| `cd /workspace/artiFixer: No such file or directory` in the container | Linux paths are case-sensitive; the mount is `/workspace/artifixer` |
| `torch.cuda.is_available()` is `False` natively | Reinstall torch from the cu128 index (`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`); CPU wheels from PyPI have no CUDA |

A full log of the issues hit while bringing this up on the target machine is kept in
[SESSION_TROUBLESHOOTING.md](SESSION_TROUBLESHOOTING.md).

## Upstream documentation

Everything else (training, evaluation protocols, ArtiFixer3D/ArtiFixer3D+, dataset
layout) is unchanged from upstream — see [README.md](README.md).
