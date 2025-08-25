import os
import timm
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from inference import inference

data_transforms = {
    'train': transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),

    transforms.ToTensor(),

    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
    transforms.RandomApply([transforms.ElasticTransform(alpha=10.0)], p=0.3),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),

    # transforms.Normalize(mean=[0.3882, 0.3882, 0.3882], std=[0.1461, 0.1461, 0.1461]), # normalize with downstream dataset (scratch)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize with imagenet dataset (imagenet)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        # transforms.Normalize(mean=[0.3882, 0.3882, 0.3882], std=[0.1461, 0.1461, 0.1461]), # normalize with downstream dataset (scratch)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize with imagenet dataset (imagenet)
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        # transforms.Normalize(mean=[0.3882, 0.3882, 0.3882], std=[0.1461, 0.1461, 0.1461]), # normalize with downstream dataset (scratch)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize with imagenet dataset (imagenet)
    ]),
}

def create_exp_dir(base_dir='exp'):
    exp_dir = base_dir
    counter = 1
    while os.path.exists(exp_dir):
        exp_dir = f"{base_dir}_{counter}"
        counter += 1
    os.makedirs(exp_dir)
    return exp_dir

def train_model(model, criterion, optimizer, scheduler, model_name, num_epochs=25, exp_dir='exp', threshold=0.5):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    log_file = open(os.path.join(exp_dir, 'training_log.txt'), 'w')
    writer = SummaryWriter(log_dir=exp_dir)

    best_acc = 0.0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        log_file.write(f'Epoch {epoch}/{num_epochs - 1}\n')
        print('-' * 10)
        log_file.write('-' * 10 + '\n')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                if phase == 'train':
                    optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    probs = torch.softmax(outputs, dim=1)
                    preds = (probs[:, 1] >= threshold).long()
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                writer.add_scalar('Loss/val', epoch_loss, epoch)
                writer.add_scalar('Accuracy/val', epoch_acc, epoch)

                if epoch_acc >= best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), os.path.join(exp_dir, f'best_acc_{model_name}_2class.pth'))

                if epoch_loss <= best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), os.path.join(exp_dir, f'best_loss_{model_name}_2class.pth'))

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            log_file.write(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

        scheduler.step()
        print()
        log_file.write('\n')

    log_file.close()
    writer.close()
    return model

def evaluate_model(model, dataloader, criterion, device, mode, threshold=0.5):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            pos_probs = probs[:, 1]
            preds = (pos_probs >= threshold).long()
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(pos_probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    if mode == "evaluate":
        print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
        cm = confusion_matrix(all_labels, all_preds)
        print("\n📊 Confusion Matrix:")
        print(cm)
        
        print("\n📊 Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1"]))

        auroc = roc_auc_score(all_labels, all_probs)
        print(f"\n📊 AUROC: {auroc:.4f}")

    return epoch_loss, epoch_acc

def get_model(model_name, num_classes=2, pretrained=False):
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=0.1
    )
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or Inference model')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference', 'evaluate'], help='Mode: train, inference, or evaluate')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--wd', default=0.01, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--checkpoint', default='', help='get checkpoint from the path')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model on the validation set')
    parser.add_argument('--pretrained', action='store_true')
    args = parser.parse_args()

    data_dir = '/hdchoi00/data/CNUH_data'
    image_datasets = {x: datasets.ImageFolder(root=f'{data_dir}/{x}', transform=data_transforms[x])
                      for x in ['train', 'val', 'test']}
    
    threshold = 0.5
    train_batch_size = args.batch_size
    val_batch_size = int(train_batch_size * 4)
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    train_dataset = image_datasets['train']

    # Step 1: Calculate class counts
    class_counts = np.array([0] * len(class_names))
    for target in train_dataset.targets:
        class_counts[target] += 1

    # Step 2: Calculate weights for each class
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()  # Normalize weights

    # Step 3: Create sample weights for each instance in the dataset
    sample_weights = np.array([class_weights[target] for target in train_dataset.targets])

    # Step 4: Create a WeightedRandomSampler
    weighted_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Step 5: Create DataLoader with the weighted sampler
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=train_batch_size, sampler=weighted_sampler, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=val_batch_size, shuffle=False, num_workers=4),
        'test': DataLoader(image_datasets['test'], batch_size=val_batch_size, shuffle=False, num_workers=4)
    }

    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the selected model and modify the final layer to include dropout
    model_ft = get_model(args.model, num_classes=2, pretrained=args.pretrained)
    model_ft = model_ft.to(device)

    if args.mode == 'train':
        # Define loss function and optimizer with weight decay
        class_weights = torch.tensor([1.0, 1.0], device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer_ft = optim.AdamW(model_ft.parameters(), lr=args.lr, weight_decay=args.wd)

        # Define the cosine annealing scheduler
        scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=100)

        # Create experiment directory
        exp_dir = create_exp_dir()

        # Train the model
        model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler_ft, model_name=args.model, num_epochs=args.epochs, exp_dir=exp_dir,  threshold=threshold)

        # Save the trained model
        torch.save(model_ft.state_dict(), os.path.join(exp_dir, f'{args.model}_2class.pth'))

    elif args.mode == 'evaluate':
        if args.checkpoint:
            model_ft.load_state_dict(torch.load(args.checkpoint))
            print(f'Loaded model from {args.checkpoint}')
        class_weights = torch.tensor([1.0, 1.0], device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        evaluate_model(model_ft, dataloaders['test'], criterion, device, args.mode, threshold=threshold)

    elif args.mode == 'inference':
        if args.checkpoint:
            model_ft.load_state_dict(torch.load(args.checkpoint))
            print(f'Loaded model from {args.checkpoint}')

        inference(model_ft, device, threshold=threshold)
