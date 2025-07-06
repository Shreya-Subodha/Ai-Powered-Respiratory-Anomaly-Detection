import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np

# -------------------------
# 1. Define CNN → LSTM model (same as training)
# -------------------------
class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_classes=3, lstm_hidden_size=128):
        super(CNN_LSTM_Model, self).__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, kernel_size=(5,7), padding=(2,3)),
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

        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv(x)                 # (B, C, H, W)
        x = x.permute(0, 3, 1, 2)        # (B, W, C, H)
        x = x.mean(dim=3)               # (B, W, C)
        output, _ = self.lstm(x)        # (B, W, hidden)
        last = output[:, -1, :]         # last time step
        return self.fc(last)

# -------------------------
# 2. Predict single image
# -------------------------
def predict_single_image(image_path, model_path, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = CNN_LSTM_Model(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Define same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # your spectrogram size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)  # shape: (1, 3, 256, 256)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    print(f"\n✅ Prediction: {class_names[predicted_class]}")
    print(f"🔢 Probabilities: {dict(zip(class_names, probs.squeeze().cpu().numpy().round(3)))}")

# -------------------------
# 3. Run
# -------------------------
if __name__ == '__main__':
    image_path = "/Users/shrey/Downloads/rinki5-filtered.png"  # Replace with your image file
    model_path = "best_cnn_lstm_model.pth"
    class_names = ['high', 'moderate', 'low']  # Change if needed

    predict_single_image(image_path, model_path, class_names)
