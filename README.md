# SPLAT AND DISTILL: Augmenting Teachers with Feed-Forward 3D Reconstruction for 3D-Aware Distillation

## ICLR 2026

[**David Shavin**](https://davidshavin4.github.io/)<sup>1</sup>, [**Sagie Benaim**](https://sagiebenaim.github.io/)<sup>1</sup>

<sup>1</sup>The Hebrew University of Jerusalem

## [Project Page](https://davidshavin4.github.io/Splat-and-Distill/) | [Paper](https://arxiv.org/abs/2602.06032) | [Medium](#)

---

<p align="center">
  <img src="assets/teaser.jpg" alt="Splat and Distill Teaser" width="100%">
</p>

---

## 📌 Overview

**Splat and Distill (SnD)** is a fine-tuning pipeline that enhances 3D awareness in Vision Foundation Models (VFMs).

---

> 🚧 **Work in Progress:** This repository is under active development. The current codebase is a partial release. We are working on cleaning and documenting the remaining modules, which will be uploaded shortly.

---

## 🛠️ Development Status

- [x] Adding installation documentation
- [x] Releasing the full training pipeline
- [ ] Provide instruction for data download
- [ ] Providing pre-trained checkpoints for DINOv2-based aligners
- [ ] Adding evaluation code

---

## 📁 Dataset Structure

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

- Segmentation masks should match image filenames, extracted with [SAM](https://github.com/facebookresearch/segment-anything) or downloaded from [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/).

---

## ⚙️ Setup Instructions

1. **Environment Setup**

   - Follow the environment setup instructions from [MVSplat](https://github.com/donydchen/mvsplat), **but do not download the rasterizer from MVSplat**.
   - Instead, download and install the Ludvig rasterizer, which supports feature rendering:
     - [Ludvig Rasterizer](https://github.com/naver/ludvig/tree/main/gaussiansplatting/submodules)
   - **Important:** Before installing, modify the number of embeddings to rasterize (e.g., 384 for DINOv2-Small) in [`apply_weights.cu`](https://github.com/naver/ludvig/blob/main/gaussiansplatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/apply_weights.cu).

2. **Pretrained Models**
   - Download `re10k.ckpt` from [MVSplat](https://github.com/donydchen/mvsplat) and save it to `checkpoints/`.
   - Download the backbone pretrained weight from Unimatch and save to `checkpoints/`:
     ```sh
     wget 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth' -P checkpoints
     ```

---

## 🚀 Running the Project

Example command to start training/evaluation:

```sh
python -m src.main +experiment=scannetpp.yaml data_loader.train.batch_size=1 checkpointing.load=checkpoints/re10k.ckpt checkpointing.resume=false model/vit=dinov2s
```

---

## 🔗 Useful Links

- [Fit3D](https://github.com/ywyue/FiT3D/tree/main)
- [MVSplat](https://github.com/donydchen/mvsplat)
- [Ludvig Rasterizer](https://github.com/naver/ludvig/tree/main/gaussiansplatting/submodules)
- [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/)
- [SAM](https://github.com/facebookresearch/segment-anything)
- [Unimatch Pretrained Weights](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth)

---

## 🙌 Acknowledgement

This repository is based on [Fit3D](https://github.com/ywyue/FiT3D/tree/main) and [MVSplat](https://github.com/donydchen/mvsplat). We would like to thank the authors of these works for publicly releasing their code.

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{shavin2026splat,
  title={Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction For 3D-Aware Distillation},
  author={Shavin, David and Benaim, Sagie},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://arxiv.org/abs/2602.06032},
  eprint={2602.06032},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
