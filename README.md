# SPLAT AND DISTILL: Augmenting Teachers with Feed-Forward 3D Reconstruction for 3D-Aware Distillation

## Overview

This project implements 3D-aware distillation using feed-forward 3D reconstruction. The dataset structure and setup are inspired by Fit3D and MVSplat, with modifications for segmentation and feature rendering.

---

# TODO: Add git install part + cd splat-and-distill


---

## 🏋️ Weights

SnD weights are available on [Hugging Face](https://huggingface.co/david-shavin/SnD).

### Using Pre-trained Weights

Load the weights directly using PyTorch Hub:

```python
import torch
import timm

# Download weights from Hugging Face
url = "https://huggingface.co/david-shavin/SnD/resolve/main/dinov2_small_snd.pth"
state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')

# Load into timm model
model = timm.create_model(
    "vit_small_patch14_dinov2.lvd142m",
    pretrained=True,
    num_classes=0,
    dynamic_img_size=True,
    dynamic_img_pad=False,
)
model.load_state_dict(state_dict, strict=False)
```

### Available Models

| Model | URL |
|-------|-----|
| DINOv2-Small + SnD | `https://huggingface.co/david-shavin/SnD/resolve/main/dinov2_small_snd.pth` |
| DINOv2-Base + SnD | `https://huggingface.co/david-shavin/SnD/resolve/main/dinov2_base_snd.pth` |

For the base model, use the same code but replace:
- URL: `dinov2_small_snd.pth` → `dinov2_base_snd.pth`
- Model name: `vit_small_patch14_dinov2.lvd142m` → `vit_base_patch14_dinov2.lvd142m`

---
## Dataset Structure

Create the following directory structure under your working directory:

```
datasets/
└── scannetpp/
    ├── metadata/
    │   ├── nvs_sem_train.txt
    │   ├── nvs_sem_val.txt
    │   ├── pretrained_feat_gaussians_train.pth
    │   ├── pretrained_feat_gaussians_val.pth
    │   ├── train_samples.txt
    │   ├── train_view_info.npy
    │   ├── val_samples.txt
    │   └── val_view_info.npy
    └── scenes/
        ├── {scene_id_0}/
        │   ├── images/
        │   ├── instance_segmentation/
        │   ├── points3d.ply
        │   ├── points3D.txt
        │   └── transforms_train.json
        ├── {scene_id_1}/
        └── ...
```

- Segmentation masks should match image filenames, extracted with [SAM](#) or downloaded from [ScanNet++](#).

---

## Setup Instructions

1. **Environment Setup**
   - Follow the environment setup instructions from [MVSplat](#), **but do not download the rasterizer from MVSplat**.
   - Instead, download and install the Ludvig rasterizer, which supports feature rendering:
     - [Ludvig Rasterizer](https://github.com/naver/ludvig/tree/main/gaussiansplatting/submodules)
   - **Important:** Before installing, modify the number of embeddings to rasterize (e.g., 384 for DINOv2-Small) in [`apply_weights.cu`](https://github.com/naver/ludvig/blob/main/gaussiansplatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/apply_weights.cu).

2. **Pretrained Models**
   - Download `re10k.ckpt` from [MVSplat](#) and save it to `checkpoints/`.
   - Download the backbone pretrained weight from [Unimatch](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth) and save to `checkpoints/`:
     ```sh
     wget 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth' -P checkpoints
     ```

---

## Training

Example command to start training:

```sh
python -m src.main +experiment=scannetpp.yaml data_loader.train.batch_size=1 checkpointing.load=checkpoints/re10k.ckpt checkpointing.resume=false model/vit=dinov2s
```

---

# Evaluation
We provide two evaluation setups for different downstream tasks. 
## Part 1: Semantic Segmentation & Depth Estimation
This section focuses on semantic segmentation and depth estimation evaluation.

### Setup

We follow the environment setup from [FiT3D](https://github.com/Yue-0/FiT3D). Install the required dependencies:

```bash
# Create conda environment
conda create -n fit3d python=3.10
conda activate fit3d
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
cd evaluation1
pip install -r requirements.txt
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```


### Install mmcv and mmsegmentation
```bash
cd mmcv
MMCV_WITH_OPS=1 pip install . --no-build-isolation -v
cd ../mmsegmentation
pip install -e . -v
```
### Environment Variables

Set the following environment variables before running evaluations:

```bash
# Set library path for CUDA libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib

# Set Python path for mmcv and mmsegmentation
export PYTHONPATH=$(pwd)/mmcv:$(pwd)/mmsegmentation:$PYTHONPATH
```

### Running Evaluations

```bash
# Move back to evaluation1 directory
cd ..
```
#### Semantic Segmentation (ScanNet++)
```bash
python linear_evaluate_segmentation.py \
    --backbone-type dinov2_small_snd \
    evaluation/baseline_configs/vits_scannetpp_sem_linear_config.py \
    --work-dir work_dirs/baseline_segmentation_eval/scannetpp/dinov2s \
    --eval_baseline
```
#### Depth Estimation (ScanNet++)
```bash
python linear_evaluate_depth.py \
    --backbone-type dinov2_small_snd \
    evaluation/baseline_configs/vits_scannetpp_depth_linear_config.py \
    --work-dir work_dirs/baseline_depth_eval/scannetpp/dinov2s \
    --eval_baseline
```
---
## Part 2: Surface Normal Estimation & Multiview Correspondence

We follow the environment setup from [Probe3D](https://github.com/mbanani/probe3d/tree/main). 
```bash
# Move to evaluation2 directory
cd ../evaluation2
```
Install the required dependencies:
```Base
conda create -n probe3d python=3.9 --yes
conda activate probe3d

# 2. Install NumPy 1.x FIRST to "pin" it in the conda solver
conda install numpy=1.26.4 -c conda-forge --yes

conda install pytorch=2.2.1 torchvision=0.17.1 pytorch-cuda=12.1 -c pytorch -c nvidia 
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
conda install -c conda-forge nb_conda_kernels=2.3.1

pip install -r requirements.txt
python setup.py develop

pip install protobuf==3.20.3    # weird dependency with datasets and google's api
pre-commit install              # install pre-commit
```
<!-- ```Base
# 1. Create and activate the environment
conda create -n probe3d python=3.9 --yes
conda activate probe3d

# 2. Install PyTorch, Torchvision, and CUDA 11.8 specifically
# We use version 2.1.2/0.16.2 as they are the most stable pair for 11.8
conda install pytorch=2.1.2 torchvision=0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia --yes

# 3. Install GPU-enabled Faiss for 11.8
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 --yes

# 4. Install Jupyter kernels and remaining dependencies
conda install -c conda-forge nb_conda_kernels=2.3.1 --yes

# 5. Install requirements and link the local 'evals' package
pip install -r requirements.txt
python setup.py develop

pip install protobuf==3.20.3    # weird dependency with datasets and google's api
pre-commit install              # install pre-commit
``` -->


### Running Evaluations

#### To run Multiview Correspondence (ScanNet) download the scannet_test_1500 from [Probe3D](https://github.com/mbanani/probe3d/tree/main). 
#### Run:

```bash
python evaluate_scannet_correspondence.py backbone.load_snd=True
```

#### Surface Normal Estimation (Coming Soon)
<!-- 
```bash
python train_snorm.py backbone=dino_b16 +backbone.return_multilayer=True

``` -->
---



## Useful Links
- [Fit3D](https://github.com/ywyue/FiT3D/tree/main)
- [Probe3D](https://github.com/mbanani/probe3d/tree/main)
- [MVSplat](https://github.com/donydchen/mvsplat)
- [Ludvig Rasterizer](https://github.com/naver/ludvig/tree/main/gaussiansplatting/submodules)
- [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/)
- [SAM](https://github.com/facebookresearch/segment-anything)
- [Unimatch Pretrained Weights](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth)