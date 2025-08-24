# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

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
        laterality = row["laterality"]  # "Left" 또는 "Right"
        grade = row["grade"]
        image_paths = ast.literal_eval(row["files"])  # 문자열을 리스트로 변환

        # 🔹 패치 번호별로 이미지 그룹화
        patch_dict = {}
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            patch_num = int(filename.split("_patch_")[1].split(".")[0])  # 패치 번호 추출

            if patch_num not in patch_dict:
                patch_dict[patch_num] = []
            patch_dict[patch_num].append(img_path)

        # 🔹 패치 번호 순서대로 2D 리스트 생성
        grouped_images = []
        for patch_num in sorted(patch_dict.keys()):
            images = []
            for img_path in sorted(patch_dict[patch_num]):  # LCC & LMLO 순서 정렬
                image = Image.open(img_path).convert('RGB')  # RGB 변환

                if self.transform:
                    image = self.transform(image)

                images.append(image)

            grouped_images.append(torch.stack(images))

        grouped_images = torch.stack(grouped_images)

        label = self.label_map[grade]

        return grouped_images, label

class InferenceDataset_Open(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.label_map = {"grade 3": 0, "grade 4": 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        patient_id = row["patient_id"]
        laterality = row["laterality"]  # "Left" 또는 "Right"
        grade = row["grade"]
        image_paths = ast.literal_eval(row["files"])  # 문자열을 리스트로 변환

        # 🔹 패치 번호별로 이미지 그룹화 (패치 정보가 파일 이름 끝에 있음)
        patch_dict = {}
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            patch_num = int(filename.split("_")[-1].split(".")[0])  # 패치 번호 추출

            if patch_num not in patch_dict:
                patch_dict[patch_num] = []
            patch_dict[patch_num].append(img_path)

        # 🔹 패치 번호 순서대로 2D 리스트 생성
        grouped_images = []
        for patch_num in sorted(patch_dict.keys()):
            images = []
            for img_path in sorted(patch_dict[patch_num]):  # CC & MLO 순서 정렬
                image = Image.open(img_path).convert('RGB')  # RGB 변환

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

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # 결정 임계값 적용 여부에 따른 예측 방식 분기
        threshold=0.5
        if threshold is not None and output.shape[1] == 2:
            # 이진 분류라고 가정: 소프트맥스를 통해 클래스 1(positive)의 확률을 계산
            prob = torch.softmax(output, dim=1)[:, 1]
            preds = (prob > threshold).long()
        else:
            # 임계값이 지정되지 않았거나 다중 클래스인 경우: argmax로 예측
            preds = torch.argmax(output, dim=1)

        # 예측값 저장
        # preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        acc1 = accuracy(output, target, topk=(1,))[0]

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

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

"""
def inference(model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3882, 0.3882, 0.3882],
                                    std=[0.1461, 0.1461, 0.1461]),
        # transforms.Normalize(mean=[0.2444, 0.2444, 0.2444],
        #                             std=[0.1775, 0.1775, 0.1775]),


    ])

    csv_file = "/home/jovyan/ctm98/CNUH_data/test_inference.csv"
    dataset = InferenceDataset(csv_file, transform=transform)

    data_loader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)

    model.eval()

    correct = 0
    total = 0
    final_outputs = []
    true_labels = []
    all_patch_outputs = []

    for images, targets in metric_logger.log_every(data_loader, 10, header):

        batch_size, num_patches, num_views, C, H, W = images.shape

        for batch_idx in range(batch_size):
            patch_final_outputs = []  # 🔹 한 배치에서 모든 패치의 최종 output 저장

            for patch_idx in range(num_patches):  # 🔹 Patch_0, Patch_1, Patch_2 개별 처리
                patch_outputs = []  # 🔹 각 patch에 대한 view별 모델 예측 결과 저장

                for view_idx in range(num_views):  # 🔹 LCC & LMLO or RCC & RMLO
                    img_tensor = images[batch_idx, patch_idx, view_idx]  # (C, H, W)
                    img_tensor = img_tensor.to(device, non_blocking=True)

                    with torch.cuda.amp.autocast():
                        output = model(img_tensor.unsqueeze(0))  # (1, C, H, W)

                    prob = torch.softmax(output, dim=1)[:, 1].item()
                    pred = 1 if prob > 0.6 else 0
                    patch_outputs.append(pred)

                # 🔹 Patch별 최종 output 결정
                if patch_outputs == [0, 0]:  
                    patch_final = 0
                else:  # [0,1], [1,0], [1,1] -> 1
                    patch_final = 1

                patch_final_outputs.append(patch_final)  # 🔹 패치별 최종 output 저장

            all_patch_outputs.append(patch_final_outputs)  # 🔹 배치별 patch outputs 저장

            # 🔹 Voting을 이용한 최종 결과 결정
            counter = Counter(patch_final_outputs)  # 🔹 개수 세기
            if counter[0] > counter[1] or counter[0] == counter[1]:  # 0이 더 많으면 0
                final_decision = 0
            else:  # 1이 더 많거나 동점이면 1
                final_decision = 1

            final_outputs.append(final_decision)  # 🔹 배치별 최종 결정 저장
            true_labels.append(targets[batch_idx].item())  # 🔹 실제 정답 저장

            # 🔹 Accuracy 계산
            if final_decision == targets[batch_idx].item():  # 예측이 정답과 일치하면
                correct += 1
            total += 1

    # 🔹 최종 출력 결과 (Patch별 Output, 예측값 & 실제 target 값 함께 출력)
    for batch_idx, (patch_outputs, final_output, target) in enumerate(zip(all_patch_outputs, final_outputs, true_labels)):
        print(f"\n✅ Final Output for Batch {batch_idx}: {patch_outputs}")  # Patch 0,1,2의 결과 출력
        print(f"✅ Final Decision for Batch {batch_idx}: {final_output}")  # 최종 결정 출력
        print(f"🎯 True Label (Target): {target}")  # 실제 정답 출력

    # 🔹 Accuracy 출력
    accuracy = correct / total * 100
    print(f"\n✅ Accuracy: {accuracy:.2f}%")

    # 🔹 Confusion Matrix 계산
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(true_labels, final_outputs)
    print(cm)

    # 🔹 Classification Report 출력
    print("\n📊 Classification Report:")
    print(classification_report(true_labels, final_outputs, target_names=["Class 0", "Class 1"]))
"""

def inference(model, device):
    """
    Patch × View 구조를 확률 기반으로 결합하여 최종 예측을 수행.
    - patch, view별로 이진 예측 대신 softmax 확률을 추출.
    - 뷰별 평균 → patch별 확률
    - patch별 확률을 다시 평균하여 최종 확률
    """

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.3882, 0.3882, 0.3882], std=[0.1461, 0.1461, 0.1461]), # downstream
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagenet
        transforms.Normalize(mean=[0.2492, 0.2492, 0.2492], std=[0.1920, 0.1920, 0.1920]) # pretrained
    ])

    csv_file = "F:/code/CNUH_data/test_inference.csv"
    dataset = InferenceDataset(csv_file, transform=transform)
    # 원하는 batch_size 설정 (예: 4, 8, 16 등)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)

    model.eval()

    correct = 0
    total = 0
    final_outputs = []
    true_labels = []
    all_probs = []  # ROC/PR 계산용 확률

    for images, targets in metric_logger.log_every(data_loader, 10, header):
        # images.shape = [batch_size, num_patches, num_views, C, H, W]
        batch_size, num_patches, num_views, C, H, W = images.shape

        for b_idx in range(batch_size):
            patch_probs_list = []  # patch별 확률(뷰 평균)

            for p_idx in range(num_patches):
                view_probs = []
                for v_idx in range(num_views):
                    img_tensor = images[b_idx, p_idx, v_idx].unsqueeze(0).to(device)
                    with torch.cuda.amp.autocast():
                        output = model(img_tensor)  # (1, num_classes)

                    # softmax로 양성 클래스(1) 확률
                    prob = torch.softmax(output, dim=1)[:, 1].item()
                    view_probs.append(prob)

                # 뷰별 확률 평균 → patch 확률
                patch_avg_prob = sum(view_probs) / len(view_probs)
                patch_probs_list.append(patch_avg_prob)

            # patch 확률들을 다시 평균(또는 다른 방식) → 최종 확률
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

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
    cm = confusion_matrix(true_labels, final_outputs)
    print("\n📊 Confusion Matrix:")
    print(cm)

    print("\n📊 Classification Report:")
    print(classification_report(true_labels, final_outputs, target_names=["Class 0", "Class 1"]))

    auroc = roc_auc_score(true_labels, all_probs)
    print(f"\n📊 AUROC: {auroc:.4f}")