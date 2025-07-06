# File: CNN/trainModel.py

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
# 0. Split dataset into train/val/test (80-10-10)
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
# 2. Define Dataloaders
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

    print(f"Train set size: {len(train_dataset)} images")
    print(f"Validation set size: {len(val_dataset)} images")
    print(f"Test set size: {len(test_dataset)} images")
    print(f"Classes: {train_dataset.classes}")

    return train_loader, val_loader, test_loader

# -------------------------
# 3. Training Loop
# -------------------------
def train_model(model, train_loader, val_loader, num_epochs=15, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    best_val_accuracy = 0.0
    save_path = 'best_stage1_model_sdgm.pth'

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model at epoch {epoch+1} with val acc {val_accuracy:.2f}%")

    print("\nTraining Completed.")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")

# -------------------------
# 4. MAIN Execution
# -------------------------
if __name__ == '__main__':
    source_mel_folder = "/Users/shrey/Documents/Mini_Project_Dataset/mel spectrograms"  # ← original dataset folder with 3 subfolders: high, moderate, low
    target_split_folder = "/Users/shrey/Documents/Mini_Project_Dataset/Final_Split_3Class"   # ← where the split will be created
    class_folders = ['high', 'moderate', 'low']

    # Step 1: Split the dataset
   # split_dataset(source_mel_folder, target_split_folder, class_folders)

    # Step 2: Load Data
    train_loader, val_loader, test_loader = create_dataloaders(target_split_folder)

    # Step 3: Train
    model = Stage1CNN(num_classes=3)
    train_model(model, train_loader, val_loader, num_epochs=15)
