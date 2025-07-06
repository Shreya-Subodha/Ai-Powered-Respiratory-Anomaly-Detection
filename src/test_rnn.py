import torch
from torch.utils.data import DataLoader
from mfcc_dataset import MFCCTimeseriesDataset
from train_rnn import MFCCLSTMClassifier  # Use same model class

# === Load Test Dataset ===
test_dataset = MFCCTimeseriesDataset("data/test_split.csv")
test_loader = DataLoader(test_dataset, batch_size=32)

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MFCCLSTMClassifier().to(device)
model.load_state_dict(torch.load("data/trained_rnn_model.pt", map_location=device))
model.eval()

# === Evaluate ===
correct = 0
total = 0

with torch.no_grad():
    for mfccs, labels in test_loader:
        mfccs, labels = mfccs.to(device), labels.to(device)
        outputs = model(mfccs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"📊 Test Accuracy (from saved model): {100 * correct / total:.2f}%")
