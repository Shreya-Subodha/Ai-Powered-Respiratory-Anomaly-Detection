# File: CNN/testModel_1D_CNN_LSTM.py

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------
# 1. Define 1D CNN + LSTM model (updated to match training)
# -------------------------
class CNN1D_LSTM_Model(nn.Module):
    def __init__(self, num_classes=3, lstm_hidden=128):
        super(CNN1D_LSTM_Model, self).__init__()

        self.cnn1d = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        x = x.mean(dim=2)          # (B, 3, 256, 256) -> (B, 3, 256)
        x = self.cnn1d(x)          # (B, 128, seq_len)
        x = x.permute(0, 2, 1)     # (B, seq_len, 128)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return self.fc(out[:, -1, :])

# -------------------------
# 2. Load test dataset
# -------------------------
def load_test_data(base_path, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(os.path.join(base_path, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return test_loader, test_dataset.classes

# -------------------------
# 3. Specificity Calculation
# -------------------------
def calculate_specificity(cm):
    specificity_per_class = []
    for i in range(len(cm)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificity_per_class.append(spec)
    return specificity_per_class

# -------------------------
# 4. Test and Evaluate
# -------------------------
def test_model(model, test_loader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    spec = calculate_specificity(cm)

    print(f"\n✅ Accuracy: {acc*100:.2f}%")
    print(f"🎯 Precision (macro avg): {prec:.2f}")
    print(f"📢 Recall / Sensitivity (macro avg): {rec:.2f}")
    print(f"💡 F1 Score (macro avg): {f1:.2f}")
    print("🔍 Specificity (per class):")
    for i, s in enumerate(spec):
        print(f"   {class_names[i]}: {s:.2f}")

    print("\n📊 Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - 1D CNN + LSTM (Updated)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_1dcnn_lstm_updated.png")
    plt.show()

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=2))

# -------------------------
# 5. Main
# -------------------------
if __name__ == '__main__':
    base_path = "/Users/shrey/Documents/Mini_Project_Dataset/Final_Split_3Class"
    model_path = "best_1dcnn_lstm_model.pth"

    test_loader, class_names = load_test_data(base_path)
    model = CNN1D_LSTM_Model(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("✅ Loaded best 1D CNN + LSTM model.")

    test_model(model, test_loader, class_names)
