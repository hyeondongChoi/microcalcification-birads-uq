# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms
import ast
from collections import Counter

class InferenceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform=transform

        self.label_map = {"grade 3": 0, "grade 4": 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        patient_id = row["patient_id"]
        laterality = row["laterality"]
        grade = row["grade"]
        image_paths = ast.literal_eval(row["files"])

        patch_dict = {}
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            patch_num = int(filename.split("_patch_")[1].split(".")[0])

            if patch_num not in patch_dict:
                patch_dict[patch_num] = []
            patch_dict[patch_num].append(img_path)

        grouped_images = []
        for patch_num in sorted(patch_dict.keys()):
            images = []
            for img_path in sorted(patch_dict[patch_num]):
                image = Image.open(img_path).convert('RGB')

                if self.transform:
                    image = self.transform(image)

                images.append(image)

            grouped_images.append(torch.stack(images))

        grouped_images = torch.stack(grouped_images)

        label = self.label_map[grade]

        return grouped_images, label

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    all_preds = []
    all_targets = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        threshold=0.5
        if threshold is not None and output.shape[1] == 2:
            prob = torch.softmax(output, dim=1)[:, 1]
            preds = (prob > threshold).long()
        else:
            preds = torch.argmax(output, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        acc1 = accuracy(output, target, topk=(1,))[0]

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_targets, all_preds)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, digits=4))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def inference(model, device, mode='scratch'):
    NORMS = {
        "scratch": dict(mean=[0.388, 0.388, 0.388],    std=[0.146, 0.146, 0.146]),
        "ssl":     dict(mean=[0.249, 0.249, 0.249],    std=[0.192, 0.192, 0.192]),
        "imagenet":dict(mean=[0.485, 0.456, 0.406],    std=[0.229, 0.224, 0.225]),
    }

    if mode not in NORMS:
        raise ValueError(f"Unknown inference mode: {mode} (choose from {list(NORMS.keys())})")

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMS[mode]["mean"], std=NORMS[mode]["std"]),
    ])

    csv_file = "F:/code/CNUH_data/test_inference.csv"
    dataset = InferenceDataset(csv_file, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)

    model.eval()

    correct = 0
    total = 0
    final_outputs = []
    true_labels = []
    all_probs = []

    for images, targets in metric_logger.log_every(data_loader, 10, header):
        # images.shape = [batch_size, num_patches, num_views, C, H, W]
        batch_size, num_patches, num_views, C, H, W = images.shape

        for b_idx in range(batch_size):
            patch_probs_list = []

            for p_idx in range(num_patches):
                view_probs = []
                for v_idx in range(num_views):
                    img_tensor = images[b_idx, p_idx, v_idx].unsqueeze(0).to(device)
                    with torch.cuda.amp.autocast():
                        output = model(img_tensor)

                    prob = torch.softmax(output, dim=1)[:, 1].item()
                    view_probs.append(prob)

                patch_avg_prob = sum(view_probs) / len(view_probs)
                patch_probs_list.append(patch_avg_prob)

            # final_prob = sum(patch_probs_list) / len(patch_probs_list)
            # final_pred = 1 if final_prob > 0.5 else 0

            max_prob = max(patch_probs_list)
            final_pred = 1 if max_prob > 0.5 else 0

            final_outputs.append(final_pred)
            true_label = targets[b_idx].item()
            true_labels.append(true_label)
            all_probs.append(max_prob)

            if final_pred == true_label:
                correct += 1
            total += 1

    accuracy = correct / total * 100
    print(f"\n✅ Accuracy: {accuracy:.2f}%")

    cm = confusion_matrix(true_labels, final_outputs)
    print("\n📊 Confusion Matrix:")
    print(cm)

    print("\n📊 Classification Report:")
    print(classification_report(true_labels, final_outputs, target_names=["Class 0", "Class 1"]))

    auroc = roc_auc_score(true_labels, all_probs)
    print(f"\n📊 AUROC: {auroc:.4f}")