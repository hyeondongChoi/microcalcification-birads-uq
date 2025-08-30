## Introduction
This repository is for Development of a Bayesian Deep Learning Model for Microcalcification BI-RADS Classification Using Uncertainty Quantification.

## Preprocessing
- **YOLOv5** is used to pre-process images for self-supervised learning. It performs object detection on mammograms to isolate the breast region.
- The **annotation tool** is used to pre-process images for downstream task (microcalcification BI-RADS classification). It crops the microcalcification regions from mammograms based on bounding-box annotations.

## SSL & Downstream Tasks
- We adopt **MoCo v3** with **Vision Transformers (ViT)** for self-supervised learning and downstream tasks.
- Following the official MoCo v3 guidelines, we use **DeiT** as the framework for downstream fine-tuning.
- **MoCo v3 (official Github)** : https://github.com/facebookresearch/moco-v3
- **DeiT (official Github)** : https://github.com/facebookresearch/deit

## Usage (highlight)
### Self-Supervised Pre-Training (ViT-Base)
```bash
python /hdchoi00/SSL/moco/moco-v3/main_moco.py \
  -a vit_base \
  --resume /hdchoi00/SSL/moco/weights/vit-b-300ep.pth.tar \
  --batch-size 512 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=600 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.1 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  /hdchoi00/SSL/moco/data
```

### Downstream Fine-Tuning (ViT-Base)
- Training
```bash
python main-mammo.py \
  --model deit_base_patch16_224 \
  --data-path /hdchoi00/data/CNUH_data \
  --resume /hdchoi00/weights/ssl/ssl_checkpoint_599_converted.pth.tar \
  --output_dir /hdchoi00/results/vit_base_ssl \
  --epoch 100
```
- Inference
```bash
python main-mammo.py \
  --model deit_base_patch16_224 \
  --data-path /hdchoi00/data/CNUH_data \
  --resume /hdchoi00/results/vit_base_ssl/best_acc_checkpoint.pth \
  --inference ssl
```
