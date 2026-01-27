# SPLAT AND DISTILL: Augmenting Teachers with Feed-Forward 3D Reconstruction for 3D-Aware Distillation

## Overview

This project implements 3D-aware distillation using feed-forward 3D reconstruction. The dataset structure and setup are inspired by Fit3D and MVSplat, with modifications for segmentation and feature rendering.

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

## Running the Project

Example command to start training/evaluation:

```sh
python -m src.main +experiment=scannetpp.yaml data_loader.train.batch_size=1 checkpointing.load=checkpoints/re10k.ckpt checkpointing.resume=false model/vit=dinov2s
```

---

## Useful Links
- [Fit3D](https://github.com/ywyue/FiT3D/tree/main)
- [MVSplat](https://github.com/donydchen/mvsplat)
- [Ludvig Rasterizer](https://github.com/naver/ludvig/tree/main/gaussiansplatting/submodules)
- [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/)
- [SAM](https://github.com/facebookresearch/segment-anything)
- [Unimatch Pretrained Weights](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth)