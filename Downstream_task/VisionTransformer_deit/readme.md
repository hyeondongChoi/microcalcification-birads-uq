# Fine-tuning with MoCo-v3 and DeiT

In the official [MoCo-v3 repository](https://github.com/facebookresearch/moco-v3), end-to-end fine-tuning is performed using [DeiT](https://github.com/facebookresearch/deit).  
Following this guideline, our fine-tuning is also based on DeiT.

---

## Main Files
- `main-mammo.py`
- `datasets.py`
- `engine.py`

---

## Usage

According to the MoCo-v3 documentation, it is recommended to use **timm==0.4.9**.

### Training

Parameter settings for each model can be found in the `config` folder of this repository.

- **ViT-Small (ImageNet pre-trained)**
```bash
python main-mammo.py \
  --model deit_small_patch16_224 \
  --data-path /hdchoi00/data/CNUH_data \
  --resume /hdchoi00/weights/imagenet/vit-s-300ep_converted.pth.tar \
  --output_dir /hdchoi00/results/vit_small_imagenet \
  --epoch 100
```
- **ViT-Base (ImageNet pre-trained)**
```bash
python main-mammo.py \
  --model deit_base_patch16_224 \
  --data-path /hdchoi00/data/CNUH_data \
  --resume /hdchoi00/weights/imagenet/vit-b-300ep_converted.pth.tar \
  --output_dir /hdchoi00/results/vit_base_imagenet \
  --epoch 100
```
- **ViT-Base (SSL pre-trained)**
```bash
python main-mammo.py \
  --model deit_base_patch16_224 \
  --data-path /hdchoi00/data/CNUH_data \
  --resume /hdchoi00/weights/ssl/ssl_checkpoint_599_converted.pth.tar \
  --output_dir /hdchoi00/results/vit_base_ssl \
  --epoch 100
```

### Inference

- **ViT-Small (ImageNet pre-trained)**
```bash
python main-mammo.py \
  --model deit_small_patch16_224 \
  --data-path /hdchoi00/data/CNUH_data \
  --resume /hdchoi00/results/vit_small_imagenet/best_acc_checkpoint.pth \
  --inference imagenet
```

- **ViT-Base (Scratch)**
```bash
python main-mammo.py \
  --model deit_base_patch16_224 \
  --data-path /hdchoi00/data/CNUH_data \
  --resume /hdchoi00/results/vit_base_scratch/best_acc_checkpoint.pth \
  --inference scratch
```

- **ViT-Base (ImageNet pre-trained)**
```bash
python main-mammo.py \
  --model deit_base_patch16_224 \
  --data-path /hdchoi00/data/CNUH_data \
  --resume /hdchoi00/results/vit_base_imagenet/best_acc_checkpoint.pth \
  --inference imagenet
```

- **ViT-base (SSL pre-trained)**
```bash
python main-mammo.py \
  --model deit_base_patch16_224 \
  --data-path /hdchoi00/data/CNUH_data \
  --resume /hdchoi00/results/vit_base_ssl/best_acc_checkpoint.pth \
  --inference ssl
