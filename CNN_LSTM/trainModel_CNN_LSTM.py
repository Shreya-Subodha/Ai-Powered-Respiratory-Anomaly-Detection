# File: CNN/trainModel_CNN_LSTM.py

import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

# -------------------------
# 0. Auto Split Dataset (80-10-10)
# -------------------------
def split_dataset(source_dir, target_base_dir, classes, split_ratio=(0.8, 0.1, 0.1)):
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(target_base_dir, split, cls), exist_ok=True)

    for cls in classes:
        files = [f for f in os.listdir(os.path.join(source_dir, cls)) if f.endswith('.png')]
        random.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])
        n_test = n_total - n_train - n_val

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        for split, file_list in splits.items():
            for file in file_list:
                src_path = os.path.join(source_dir, cls, file)
                dst_path = os.path.join(target_base_dir, split, cls, file)
                shutil.copy2(src_path, dst_path)

# -------------------------
# 1. CNN + LSTM Model
# -------------------------
class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_classes=3, lstm_hidden_size=128):
        super(CNN_LSTM_Model, self).__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, kernel_size=(5,7), padding=(2,3)),  # (3,128,128) → (64,H,W)
            nn.ReLU(),
            nn.MaxPool2d((2,3)),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,3)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,3)),
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden_size, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        x = self.conv(x)  # shape: (B, C=128, H, W)
        x = x.permute(0, 3, 1, 2)  # → (B, W, C, H)
        x = x.mean(dim=3)          # → (B, W, C)

        output, _ = self.lstm(x)   # → (B, W, hidden)
        last = output[:, -1, :]    # → (B, hidden)

        out = self.fc(last)
        return out

# -------------------------
# 2. Create Dataloaders
# -------------------------
def create_dataloaders(base_path, batch_size=32):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = datasets.ImageFolder(os.path.join(base_path, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(base_path, 'val'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(base_path, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train set size: {len(train_dataset)}")
    print(f"Val set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    return train_loader, val_loader, test_loader

# -------------------------
# 3. Train Model
# -------------------------
def train_model(model, train_loader, val_loader, num_epochs=15, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    save_path = 'best_cnn_lstm_model.pth'

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model at epoch {epoch+1} with val acc {val_acc:.2f}%")

    print(f"\nTraining complete. Best Val Accuracy: {best_val_accuracy:.2f}%")

# -------------------------
# 4. Main
# -------------------------
if __name__ == '__main__':
    source_mel_folder = "/Users/shrey/Documents/Mini_Project_Dataset/mel spectrograms"
    target_split_folder = "/Users/shrey/Documents/Mini_Project_Dataset/Final_Split_3Class"
    class_folders = ['high', 'moderate', 'low']

    #split_dataset(source_mel_folder, target_split_folder, class_folders)
    train_loader, val_loader, test_loader = create_dataloaders(target_split_folder)

    model = CNN_LSTM_Model(num_classes=3)
    train_model(model, train_loader, val_loader)
