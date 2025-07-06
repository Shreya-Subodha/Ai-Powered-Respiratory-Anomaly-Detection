# File: CNN/trainModel_1D_CNN_LSTM.py

import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# -------------------------
# 0. Use existing split folder (no re-splitting)
# -------------------------
DATA_DIR = "/Users/shrey/Documents/Mini_Project_Dataset/Final_Split_3Class"
CLASS_NAMES = ['high', 'moderate', 'low']

# -------------------------
# 1. 1D CNN + LSTM Model
# -------------------------
class CNN1D_LSTM_Model(nn.Module):
    def __init__(self, num_classes=3, lstm_hidden=128):
        super(CNN1D_LSTM_Model, self).__init__()

        self.cnn1d = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # x: (B, 3, 256, 256) → treat as 1D time-series by reshaping
        x = x.mean(dim=2)          # avg over height → (B, 3, 256)
        x = self.cnn1d(x)          # → (B, 128, seq_len)
        x = x.permute(0, 2, 1)     # → (B, seq_len, features) for LSTM

        out, _ = self.lstm(x)
        last = out[:, -1, :]       # last time step
        return self.fc(last)

# -------------------------
# 2. Dataloader
# -------------------------
def create_dataloaders(base_path, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train = datasets.ImageFolder(os.path.join(base_path, "train"), transform=transform)
    val = datasets.ImageFolder(os.path.join(base_path, "val"), transform=transform)
    test = datasets.ImageFolder(os.path.join(base_path, "test"), transform=transform)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0),
        train.classes
    )

# -------------------------
# 3. Train Function
# -------------------------
def train(model, train_loader, val_loader, epochs=15, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    save_path = "best_1dcnn_lstm_model.pth"

    for epoch in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            _, preds = torch.max(out, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total
        train_loss = loss_sum / len(train_loader)

        # Validation
        model.eval()
        val_total, val_correct, val_loss_sum = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss = criterion(out, y)
                val_loss_sum += val_loss.item()
                _, preds = torch.max(out, 1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss_sum / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} → Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model with val acc {val_acc:.2f}%")

# -------------------------
# 4. Main
# -------------------------
if __name__ == '__main__':
    train_loader, val_loader, test_loader, class_names = create_dataloaders(DATA_DIR)
    model = CNN1D_LSTM_Model(num_classes=3)
    train(model, train_loader, val_loader)