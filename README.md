# AI-Powered Respiratory Anomaly Risk Detection

> ⚕️ Smart triaging from lung sound. Deep learning meets respiratory diagnostics.

## 🔍 Overview
Designed an AI-based screening tool to predict **respiratory risk levels (High, Moderate, Low)** using lung sounds converted into **Mel spectrograms**. The model supports clinicians by providing quick, objective assessments in resource-constrained environments.

## ❗ Problem Statement

Traditional respiratory diagnostics rely on multiple time-consuming tests and subjective clinical interpretation, often delaying timely care. This project addresses the need for a faster, non-invasive, and objective risk assessment tool using AI-based analysis of lung sounds.

## 🚀 Highlights

- **Non-invasive & real-time ready**
- **CNN–BiLSTM** hybrid deep learning model
- Achieved **86.7% accuracy**, outperforming MobileNetV2 and standard CNNs
- Audio preprocessing includes **resampling, filtering, and WAV-level augmentation**
- Dataset: **ICBHI 2017 Respiratory Sound Database**

## 🛠️ Tech Brief

| Component           | Details                         |
|---------------------|----------------------------------|
| Feature Input       | 256×256 Mel spectrograms         |
| Preprocessing       | Resample (4kHz), bandpass filtering (80–2000 Hz)|
| Models Tested       | CNN, CNN–LSTM, 1D CNN–LSTM, MobileNetV2, CNN-BiLSTM |
| Best Model          | **CNN–BiLSTM** (ReLU, Adam, CrossEntropyLoss) |
| Dataset Split       | 80% train / 10% val / 10% test   |

## 📈 Performance

| Model           | Accuracy | Notes                      |
|-----------------|----------|----------------------------|
| CNN–BiLSTM      | **86.7%**  | Best overall, robust F1    |
| MobileNetV2     | 82.22%   | Lightweight baseline       |
| CNN             | 76.67%   | Effective spatial learning |
| 1D CNN–LSTM     | 71.11%   | Lower due to 1D input loss |

## 🧠 Clinical Relevance
Supports faster and more accurate triaging by reducing reliance on manual auscultation and slow diagnostics. Especially valuable in **primary care, rural health setups, and mobile clinics**.

Built by:
Kruthika S, Nishevithaa Gayatri T S, Shreya Subodha,
Department of Medical Electronics, Dayananda Sagar College of Engineering
