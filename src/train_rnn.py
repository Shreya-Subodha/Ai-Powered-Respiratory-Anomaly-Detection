import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from mfcc_dataset import MFCCTimeseriesDataset

# === LSTM Model ===
class MFCCLSTMClassifier(nn.Module):
    def __init__(self, input_size=13, hidden_size=64, num_layers=1, num_classes=3):
        super(MFCCLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# === Load Dataset and Split ===
dataset = MFCCTimeseriesDataset("data/rnn_mfcc_metadata.csv")
all_indices = np.arange(len(dataset))

train_idx, test_idx = train_test_split(all_indices, test_size=0.25, random_state=42)
pd.read_csv("data/rnn_mfcc_metadata.csv").iloc[train_idx].to_csv("data/train_split.csv", index=False)
pd.read_csv("data/rnn_mfcc_metadata.csv").iloc[test_idx].to_csv("data/test_split.csv", index=False)

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)

# === Training ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MFCCLSTMClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🧠 Training started...")
for epoch in range(10):
    model.train()
    total_loss = 0
    for mfccs, labels in train_loader:
        mfccs, labels = mfccs.to(device), labels.to(device)
        outputs = model(mfccs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# === Save Model ===
model_path = "data/trained_rnn_model.pt"
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved at: {model_path}")
