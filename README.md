## Introduction
This repository is for Development of a Bayesian Deep Learning Model for Microcalcification BI-RADS Classification Using Uncertainty Quantification.

## Preprocessing
- **YOLOv5** is used to pre-process images for self-supervised learning. It performs object detection on mammograms to isolate the breast region.
- The **annotation tool** is used to pre-process images for downstream task (microcalcification BI-RADS classification). It crops the microcalcification regions from mammograms based on bounding-box annotations.

## Self-Supervised Learning (SSL) & Downstream Tasks
- We adopt **MoCo v3** with **Vision Transformers (ViT)** for self-supervised learning and downstream tasks.
- Following the official MoCo v3 guidelines, we use **DeiT** as the framework for downstream fine-tuning.
- **MoCo v3 (official Github)** : https://github.com/facebookresearch/moco-v3
- **DeiT (official Github)** : https://github.com/facebookresearch/deit
