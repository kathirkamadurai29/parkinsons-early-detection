# ParkiScan — CNN–BiLSTM Framework for Early Parkinson's Disease Detection

> **Francis Xavier Engineering College — Dept. of Computer Science and Business Systems**  
> Guide: Mrs. Evangeline Sneha Michaeli (AP/CSBS)  
> Team: Kumaran L (95072318024) | Kathirkamadurai S (95072318021) | Karsten Jmittel K J (95072318020)

---

## 🧠 Project Overview

A deep learning system for **early detection of Parkinson's Disease** from voice recordings using a hybrid **2D Temporal CNN + Bidirectional LSTM** architecture.

The model analyzes vocal biomarkers — irregularities in pitch, rhythm, and spectral patterns caused by Parkinson's-related motor dysfunction — to produce a risk probability score.

---

## 🏗️ Architecture

```
Voice (.wav)
    ↓
Feature Extraction
  ├─ Mel-Spectrogram (128 bands)
  ├─ MFCC + Delta + Delta² (120 features)
  ├─ Chroma STFT (12 features)
  └─ Spectral Contrast + ZCR (9 features)
    ↓ Stack → (269, 130, 1) 2D feature map
    ↓
2D Temporal CNN
  ├─ Block 1: Conv2D(32) → BN → ReLU → MaxPool → Dropout
  ├─ Block 2: Conv2D(64) → BN → ReLU → MaxPool → Dropout
  └─ Block 3: Conv2D(128) → BN → ReLU → MaxPool → Dropout
    ↓ Reshape: (batch, time, features)
    ↓
Bidirectional LSTM
  ├─ BiLSTM(256, return_sequences=True) → Dropout
  ├─ BiLSTM(128, return_sequences=True) → Dropout
  └─ BiLSTM(64, return_sequences=False) → Dropout
    ↓
Dense Classifier
  ├─ Dense(128) → BN → ReLU → Dropout
  ├─ Dense(64) → ReLU → Dropout
  └─ Dense(1) → Sigmoid
    ↓
Risk Score (0–1)
```

---

## 📂 Project Structure

```
parkinsons_cnn_bilstm/
├── models/
│   └── cnn_bilstm.py         # Model architecture
├── utils/
│   └── feature_extractor.py  # Audio feature extraction
├── data/
│   ├── healthy/              # .wav files of healthy subjects
│   └── parkinsons/           # .wav files of PD patients
├── templates/
│   └── index.html            # Web application frontend
├── train.py                  # Training pipeline
├── predict.py                # Inference module
├── app.py                    # Flask web server
└── requirements.txt
```

---

## 🚀 Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

**Option A — Oxford Parkinson's Voice Dataset (recommended)**
- Download: [Oxford PD Dataset](https://archive.physionet.org/physiobank/database/pcgdb/)
- Place .wav files in `data/healthy/` and `data/parkinsons/`

**Option B — UCI Parkinson's Dataset (pre-extracted features)**
- Download: `parkinsons.data` from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- Place in `data/parkinsons.data`

### 3. Train the Model

```bash
python train.py
```

This will:
- Extract features from all .wav files
- Train the CNN-BiLSTM model
- Save best model to `models/best_model.h5`
- Generate training plots and evaluation metrics

**For K-Fold cross-validation:**
```python
# In train.py, set:
train_model(X, y, use_kfold=True, n_folds=5)
```

### 4. Run Web Application

```bash
python app.py
```

Open browser at: http://localhost:5000

---

## 📊 Expected Performance (Literature)

| Metric       | Score  |
|-------------|--------|
| Accuracy     | ~95.2% |
| AUC-ROC      | ~0.974 |
| Sensitivity  | ~97.1% |
| Specificity  | ~93.8% |
| F1 Score     | ~0.951 |

---

## 🎯 Key Features

### Voice Feature Extraction
- **Mel-Spectrogram**: 128 mel bands capturing frequency distribution
- **MFCC**: 40 coefficients + 1st and 2nd order deltas
- **Chroma STFT**: 12 pitch class features
- **Spectral Contrast**: 7-band spectral contrast
- **ZCR**: Zero-crossing rate
- **HNR / Jitter**: Voice quality metrics via pyin

### Model Training
- Stratified K-Fold cross-validation
- Class weighting for imbalanced datasets
- Early stopping + ReduceLROnPlateau
- L2 regularization + Dropout
- Binary cross-entropy loss

### Web Application
- Live microphone recording
- .wav file upload
- Risk score visualization
- Clinical interpretation

---

## 📋 API Endpoints

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/` | GET | Web interface |
| `/api/predict/file` | POST | Predict from .wav upload |
| `/api/predict/audio` | POST | Predict from base64 PCM |
| `/api/health` | GET | Server health check |
| `/api/demo` | GET | Demo prediction |

---

## 📚 References

1. Little, M.A., et al. (2009). Suitability of dysphonia measurements for telemonitoring of Parkinson's disease. *IEEE Trans. Biomed. Eng.*
2. Grosz, T., et al. (2021). CNN-based Parkinson's Disease Detection from Voice Signals. *IEEE Access*
3. UCI Machine Learning Repository — Parkinson's Dataset
4. Oxford Parkinson's Disease Detection Dataset (PhysioNet)

---

## ⚠️ Disclaimer

This project is for **research and academic purposes only**. It is NOT a certified medical device and should not be used as a substitute for professional neurological diagnosis.

---

*CNN-BiLSTM Framework for Early Parkinson's Disease Detection*  
*Francis Xavier Engineering College, 2024*
