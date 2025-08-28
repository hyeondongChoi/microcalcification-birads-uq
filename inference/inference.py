import os
import ast
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from timm.models import create_model
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, f1_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize with imagenet dataset (imagenet)
    transforms.Normalize(mean=[0.2492, 0.2492, 0.2492], std=[0.1920, 0.1920, 0.1920]) # normalize with pretrain dataset (ssl)
])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = create_model(
    "deit_base_patch16_224",
    pretrained=False,
    num_classes=2,
    drop_rate=0.1,
    drop_path_rate=0.2,
    drop_block_rate=None,
    img_size=224
)
model = model.to(device)

# checkpoint_path = r"/hdchoi00/results/vit_base_imagenet/best_acc_checkpoint.pth"
checkpoint_path = r"/hdchoi00/results/vit-base_ssl/best_acc_checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

model.load_state_dict(checkpoint['model'])

def enable_dropout(m):
    if isinstance(m, torch.nn.Dropout):
        m.train()

def enable_droppath(m):
    if m.__class__.__name__ == "DropPath":
        m.train()

model.train()
model.apply(enable_dropout)
model.apply(enable_droppath)

optimal_T = 0.626 # Optimal temperature scaling factor (ImageNet : 0.460, SSL : 0.626)
optimal_threshold = 0.81 # Uncertainty theshold
T = 100  # Number of MC-Dropout forward passes
N_REPEATS = 100  # Number of overall inference repetitions

csv_file = r"/hdchoi00/data/CNUH_data/test_inference.csv"
test_dataset = InferenceDataset(csv_file, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)

label_map = {0: "grade 3", 1: "grade 4"}

results = []

for repeat_idx in tqdm(range(N_REPEATS)):
    correct = 0
    total = 0
    uncertain_count = 0

    all_preds = []
    all_true = []

    label_map = {0: "grade 3", 1: "grade 4"}

    for i, (images, targets) in enumerate(test_loader):
        # print("sample " + str(i))
        batch_size, num_patches, num_views, C, H, W = images.shape

        for b_idx in range(batch_size):
            patch_probs_list = []
            for p_idx in range(num_patches):
                # print("\tpatch " + str(p_idx))
                view_probs = []
                for v_idx in range(num_views):
                    img_tensor = images[b_idx, p_idx, v_idx].unsqueeze(0).to(device)
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        prob_list = []
                        for _ in range(T):
                            logits = model(img_tensor)
                            # Calibrate logits using the optimal temperature
                            calibrated_logits = logits / optimal_T
                            prob = torch.softmax(calibrated_logits, dim=1)[:, 1].item()
                            prob_list.append(prob)

                    mean_prob = np.mean(prob_list)
                    variance = np.var(prob_list)

                    if mean_prob >= 0.5:
                        confidence = mean_prob
                    else:
                        confidence = 1 - mean_prob

                    if confidence >= optimal_threshold:
                        view_probs.append(mean_prob)
                    else:
                        view_probs.append(None)
                
                # print("view_probs:", view_probs)
                valid_view_probs = [vp for vp in view_probs if vp is not None]
                if valid_view_probs:
                    patch_avg_prob = np.mean(valid_view_probs)
                else:
                    patch_avg_prob = None
                
                patch_probs_list.append(patch_avg_prob)

            # print("patch_probs:", patch_probs_list)

            valid_patch_probs = [vp for vp in patch_probs_list if vp is not None]

            if any(vp is None for vp in patch_probs_list):
                if any(vp is not None and vp >= 0.5 for vp in patch_probs_list):
                    result = "grade 4"
                else:
                    result = "uncertain"
            else:
                if any(vp >= 0.5 for vp in patch_probs_list):
                    result = "grade 4"
                else:
                    result = "grade 3"

            # print("\nResult:", result)
            # print("label:", targets[b_idx].item(), "\n")

            true_label = label_map.get(targets[b_idx].item(), "unknown")

            all_preds.append(result)
            all_true.append(true_label)

            if result == true_label:
                correct += 1
            
            if result == "uncertain":
                uncertain_count += 1
                
            total += 1

            # print("\nResult:", result)
            # print("label:", true_label, "\n")

    filtered_preds = []
    filtered_true = []
    for pred, true in zip(all_preds, all_true):
        if pred != "uncertain":
            filtered_preds.append(pred)
            filtered_true.append(true)

    if filtered_preds:
        cm = confusion_matrix(filtered_true, filtered_preds, labels=["grade 3", "grade 4"])
        if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    results.append({
        "repeat": repeat_idx + 1,
        "accuracy": round(accuracy * 100, 2),
        "sensitivity": round(sensitivity, 3),
        "specificity": round(specificity, 3),
        "f1_score": round(f1_score, 3),
        "uncertain_case": uncertain_count,
        "certain_case": len(filtered_preds),
        "data_coverage": round((len(filtered_preds) / len(all_preds)) * 100, 2)
    })

df = pd.DataFrame(results)
output_path = r"/hdchoi00/inference/ssl_mean-entropy_internal_Q1_81.xlsx"
df.to_excel(output_path, index=False)