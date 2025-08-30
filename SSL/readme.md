## MoCo v3 for Self-Supervised Learning

### Introduction
This is a PyTorch implementation of MoCo v3 for self-supervised learning (SSL) with Vision Transformers (ViT) on mammography images.

We follow the official resources:
- **MoCo v3 (official GitHub)**: [https://github.com/facebookresearch/moco-v3](https://github.com/facebookresearch/moco-v3)
- **Paper**: [An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.02057) (arXiv:2104.02057)

and also Pre-trained models and configs files used in our experiments can be found here: [CONFIG.md](https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md) on **MoCo v3 Github**.

## Usage

According to the MoCo-v3 documentation, It is recommended to use **timm==0.4.9**.

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
