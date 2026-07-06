# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Native Windows environment setup for ArtiFixer (inference + captioning).
#
# Tested target: Windows 11, NVIDIA GeForce RTX 5060 Ti 16 GB (Blackwell, sm_120),
# driver >= 570 (CUDA 12.8 runtime support).
#
# Usage (from the repository root, in PowerShell):
#   .\install_env_windows.ps1                     # core env: ArtiFixer inference + captioning
#   .\install_env_windows.ps1 -With3DGRUT         # also install 3DGRUT (sparse reconstruction, ArtiFixer3D)
#   .\install_env_windows.ps1 -CondaEnv myenv     # custom conda env name
#
# Notes:
# - flash-attention 3/4 are NOT installed: they do not build on native Windows and
#   are not required. The code automatically falls back to PyTorch SDPA (cuDNN),
#   which supports Blackwell (sm_120).
# - The default KV-cache inference pipeline does not use flex_attention, so Triton
#   is not required for inference. Training paths that use block-causal masks need
#   Triton; on Windows install the community "triton-windows" package manually.
# - 3DGRUT compiles CUDA kernels at first use (JIT via slangtorch + torch cpp_extension),
#   which requires Visual Studio 2022 Build Tools (C++ workload) and the CUDA 12.8 toolkit.

param (
    [string]$CondaEnv = "artifixer",
    [switch]$With3DGRUT
)

$TorchVersion = "2.11.0"
$TorchIndexUrl = "https://download.pytorch.org/whl/cu128"

function Check-LastCommand {
    param($StepName)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: $StepName failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "$StepName completed successfully" -ForegroundColor Green
}

function Find-VisualStudioCompiler {
    Write-Host "Searching for Visual Studio C++ compiler (needed for 3DGRUT JIT builds)..." -ForegroundColor Yellow
    $searchPaths = @(
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\*\bin\Hostx64\x64",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\*\bin\Hostx64\x64",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\*\bin\Hostx64\x64"
    )
    foreach ($path in $searchPaths) {
        $resolvedPaths = Get-ChildItem -Path $path -ErrorAction SilentlyContinue | Sort-Object Name -Descending
        foreach ($resolvedPath in $resolvedPaths) {
            $clExe = Join-Path $resolvedPath.FullName "cl.exe"
            if (Test-Path $clExe) {
                Write-Host "Found Visual Studio compiler at: $($resolvedPath.FullName)" -ForegroundColor Green
                return $resolvedPath.FullName
            }
        }
    }
    return $null
}

Write-Host "`n=== ArtiFixer native Windows setup (conda env: $CondaEnv) ===" -ForegroundColor Cyan

# --- Preflight checks -------------------------------------------------------

Write-Host "`nChecking NVIDIA driver..." -ForegroundColor Yellow
& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
Check-LastCommand "NVIDIA driver check (install/update the NVIDIA driver if this failed)"

Write-Host "Checking conda..." -ForegroundColor Yellow
& conda --version
Check-LastCommand "Conda check (install Miniconda/Anaconda if this failed)"

Write-Host "Checking git..." -ForegroundColor Yellow
& git --version
Check-LastCommand "Git check"

# Prevent CRLF corruption of the 3DGRUT submodule shell scripts on future checkouts.
Write-Host "Setting core.autocrlf=false for this repository..." -ForegroundColor Yellow
& git config core.autocrlf false
& git -C thirdparty/3DGRUT-ArtiFixer config core.autocrlf false 2>$null

Write-Host "Initializing git submodules..." -ForegroundColor Yellow
& git submodule update --init --recursive
Check-LastCommand "Git submodule initialization"

# --- Conda environment ------------------------------------------------------

Write-Host "`nCreating conda environment '$CondaEnv' (Python 3.12)..." -ForegroundColor Yellow
& conda create -n $CondaEnv python=3.12 -y
Check-LastCommand "Conda environment creation"

# `conda run` avoids relying on shell activation inside the script.
function Invoke-EnvPip {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PipArgs)
    & conda run --no-capture-output -n $CondaEnv python -m pip @PipArgs
}

Write-Host "`nInstalling PyTorch $TorchVersion + CUDA 12.8 (supports Blackwell sm_120)..." -ForegroundColor Yellow
Invoke-EnvPip install "torch==$TorchVersion" torchvision --index-url $TorchIndexUrl
Check-LastCommand "PyTorch installation"

Write-Host "Verifying PyTorch CUDA support..." -ForegroundColor Yellow
& conda run --no-capture-output -n $CondaEnv python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'); print('arch list:', torch.cuda.get_arch_list())"
Check-LastCommand "PyTorch verification"

Write-Host "`nInstalling ArtiFixer Python dependencies (pinned to match Dockerfile.cuda12)..." -ForegroundColor Yellow
Invoke-EnvPip install "accelerate==1.13.0" "diffusers==0.37.1" "transformers==5.5.0" ftfy peft
Check-LastCommand "Diffusion stack installation"

Invoke-EnvPip install einops scipy wandb tqdm Pillow matplotlib opencv-python pyyaml torchmetrics imageio-ffmpeg h5py av torch-fidelity "huggingface_hub[cli]"
Check-LastCommand "Common dependencies installation"

Write-Host "Installing MoGe (metric-scale monocular depth)..." -ForegroundColor Yellow
Invoke-EnvPip install "git+https://github.com/microsoft/MoGe.git"
Check-LastCommand "MoGe installation"

# --- Optional: 3DGRUT (sparse reconstruction / ArtiFixer3D) ------------------

if ($With3DGRUT) {
    Write-Host "`n=== Installing 3DGRUT (experimental on native Windows) ===" -ForegroundColor Cyan

    $vsCompilerPath = Find-VisualStudioCompiler
    if (-not $vsCompilerPath) {
        Write-Host "Warning: Visual Studio 2022 C++ compiler (cl.exe) not found." -ForegroundColor Yellow
        Write-Host "3DGRUT CUDA kernels are JIT-compiled at first use and will fail without it." -ForegroundColor Yellow
        Write-Host "Install 'Visual Studio 2022 Build Tools' with the 'Desktop development with C++' workload." -ForegroundColor Yellow
    }

    Write-Host "Installing CUDA 12.8 toolkit (nvcc) into the conda environment..." -ForegroundColor Yellow
    & conda install -y -n $CondaEnv -c "nvidia/label/cuda-12.8.0" cuda-toolkit cmake ninja
    Check-LastCommand "CUDA toolkit installation"

    # Restrict JIT builds to the RTX 5060 Ti architecture for faster compiles.
    & conda env config vars set -n $CondaEnv "TORCH_CUDA_ARCH_LIST=12.0"
    if ($vsCompilerPath) {
        & conda env config vars set -n $CondaEnv "PATH=$vsCompilerPath;$env:PATH"
    }

    Write-Host "Installing 3DGRUT requirements..." -ForegroundColor Yellow
    Invoke-EnvPip install -r thirdparty/3DGRUT-ArtiFixer/requirements.txt
    Check-LastCommand "3DGRUT requirements installation"

    Invoke-EnvPip install -e thirdparty/3DGRUT-ArtiFixer
    Check-LastCommand "3DGRUT editable install"
} else {
    # The ArtiFixer inference import chain uses pure-Python camera-model helpers
    # from threedgrut; install the package without its heavy CUDA prerequisites.
    Write-Host "`nInstalling threedgrut package (pure-Python parts used by inference)..." -ForegroundColor Yellow
    Invoke-EnvPip install -r thirdparty/3DGRUT-ArtiFixer/requirements.txt
    Check-LastCommand "3DGRUT requirements installation"
    Invoke-EnvPip install -e thirdparty/3DGRUT-ArtiFixer
    Check-LastCommand "threedgrut package installation"
}

# --- Final verification ------------------------------------------------------

Write-Host "`nRunning import sanity checks..." -ForegroundColor Yellow
& conda run --no-capture-output -n $CondaEnv python -c "import diffusers, transformers, accelerate; print('diffusers', diffusers.__version__); print('transformers', transformers.__version__); print('accelerate', accelerate.__version__)"
Check-LastCommand "Diffusion stack import check"

& conda run --no-capture-output -n $CondaEnv python -c "from threedgrut.datasets.camera_models import OpenCVPinholeCameraModelParameters; print('threedgrut camera models ok')"
Check-LastCommand "threedgrut import check"

Write-Host "`n=================================================" -ForegroundColor Green
Write-Host "    INSTALLATION COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
Write-Host "Activate the environment with:  conda activate $CondaEnv" -ForegroundColor Cyan
Write-Host "Then see README_WINDOWS.md for inference commands tuned for 16 GB GPUs." -ForegroundColor Cyan
