# Fine-tuning with CNN Models

## Usage
### Training

- **ResNet50 (Scratch)**
```bash
python main.py --mode train --model resnet50 --lr 1e-3 --wd 0.01 --epochs 100 --batch_size 256
```

- **ResNet50 (ImageNet pre-trained)**
```bash
python main.py --mode train --model resnet50 --lr 1e-5 --wd 0.01 --epochs 100 --batch_size 256 --pretrained
```

- **EfficientNet-B0 (Scratch)**
```bash
python main.py --mode train --model efficientnet_b0 --lr 1e-3 --wd 0.01 --epochs 300 --batch_size 256
```

- **EfficientNet-B0 (ImageNet pre-trained)**
```bash
python main.py --mode train --model efficientnet_b0 --lr 1e-4 --wd 0.01 --epochs 100 --batch_size 256 --pretrained
```

### Inference

- **ResNet50**
```bash
python main.py --mode inference --model resnet50 --checkpoint /hdchoi00/results/resnet50/best_acc_checkpoint.pth
```

- **EfficientNet-B0**
```bash
python main.py --mode inference --model efficientnet_b0 --checkpoint /hdchoi00/results/efficientnet_b0/best_acc_checkpoint.pth
```


We use ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for both the ImageNet-pretrained and scratch training regimes for consistency. In supplementary experiments, we also evaluate dataset-specific normalization and observe no performance benefits over ImageNet normalization; therefore, we report ImageNet-normalized results as our main setting.
