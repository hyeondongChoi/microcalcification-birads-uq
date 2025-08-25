import os
import ast
import torch
import utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve

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

def inference(model, device, threshold=0.5):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        # transforms.Normalize(mean=[0.3882, 0.3882, 0.3882], std=[0.1461, 0.1461, 0.1461]), # downstream dataset
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagenet
    ])

    csv_file = "/hdchoi00/data/CNUH_data/test_inference.csv"
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

    # print(true_labels)
    # print(all_probs)
