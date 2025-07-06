# File: CNN/testModel.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 1. Define your Stage 1 CNN model
# -------------------------
class Stage1CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(Stage1CNN, self).__init__()

        self.batch_norm = nn.BatchNorm2d(3)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(5,7), padding=(2,3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,3))

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,3))

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,3))

        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.batch_norm(x)

        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# -------------------------
# 2. Load the Test Dataset
# -------------------------
def load_test_data(base_path, batch_size=32):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = datasets.ImageFolder(os.path.join(base_path, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return test_loader, test_dataset.classes

# -------------------------
# 3. Evaluate Model and Compute Metrics
# -------------------------
def calculate_specificity(cm):
    # Sensitivity = TP / (TP + FN)
    # Specificity = TN / (TN + FP)
    specificity_per_class = []
    for i in range(len(cm)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specificity_per_class.append(specificity)
    return specificity_per_class

def test_model(model, test_loader, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    specificity = calculate_specificity(cm)

    print("\n✅ Test Accuracy: {:.2f}%".format(acc * 100))
    print("🎯 Precision (macro avg): {:.2f}".format(precision))
    print("📢 Recall / Sensitivity (macro avg): {:.2f}".format(recall))
    print("💡 F1 Score (macro avg): {:.2f}".format(f1))
    print("🔍 Specificity (per class):")
    for i, spec in enumerate(specificity):
        print(f"   {class_names[i]}: {spec:.2f}")

    print("\n📊 Confusion Matrix:")
    print(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")  # save as image
    plt.show()  

    print("\n📋 Full Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=2))

# -------------------------
# 4. Main Execution
# -------------------------
if __name__ == '__main__':
    base_dataset_path = "/Users/shrey/Documents/Mini_Project_Dataset/Final_Split_3Class"
    model_path = "best_stage1_model_sdgm.pth"

    test_loader, class_names = load_test_data(base_dataset_path)
    model = Stage1CNN(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("✅ Loaded best model successfully.")

    test_model(model, test_loader, class_names)
