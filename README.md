# 🗣️ Telugu Speech Emotion Detection using CRNN

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Emotion recognition from Telugu speech using a Convolutional Recurrent Neural Network (CRNN) with MFCC features. Achieves **>80% accuracy** on a 5‑emotion classification task (angry, happy, neutral, sad, surprised).

---

## 📌 Overview

Telugu is a low‑resource Dravidian language spoken by over 80 million people, yet very few AI systems exist for its affective analysis. This project builds a lightweight **CRNN** (CNN + GRU) that classifies emotions from Telugu speech clips. It is designed for real‑time or edge deployment and can be easily adapted to other Indian or international languages.

### Key Features
- **5 emotion classes** – Angry, Happy, Neutral, Sad, Surprised
- **Feature extraction** – 40 MFCCs + delta & delta‑delta coefficients
- **Data augmentation** – Noise addition, pitch shift, time stretch, volume change
- **Model** – CRNN (Conv1D layers + Bidirectional GRU + Dense classifiers)
- **Regularisation** – Dropout, Batch Normalisation, Early Stopping, AdamW optimizer
- **Performance** – 80%+ overall accuracy; Angry and Surprised classes best recognised

---

## 📂 Project Structure
Telugu-Emotion-Detection/
├── data/ # Raw audio and processed features (not uploaded)
│ ├── raw/ # Original .wav files
│ ├── augmented/ # Augmented copies
│ └── features/ # Extracted MFCC numpy arrays
├── notebooks/ # Exploratory analysis
├── src/
│ ├── preprocess.py # Resampling, trimming, augmentation
│ ├── features.py # MFCC, delta, delta-delta extraction
│ ├── model.py # CRNN architecture
│ ├── train.py # Training loop with callbacks
│ └── predict.py # Inference on new audio
├── models/ # Saved model weights (.h5)
├── results/ # Confusion matrix, classification report
├── requirements.txt # Dependencies
├── README.md # This file
└── emotion_detection_telugu.pptx # Project presentation (attached)

text

---

## 📊 Dataset

The Telugu Emotion Speech Dataset is a private collection of ~216 utterances (after cleaning) distributed across 5 emotions.  
Due to the small size, we applied **data augmentation** to balance classes and improve generalisation.

| Emotion   | Original | After Augmentation |
|-----------|----------|--------------------|
| Angry     | 55       | ~110               |
| Happy     | 52       | ~104               |
| Neutral   | 61       | ~122               |
| Sad       | 48       | ~96                |
| Surprised | ~40      | ~80                |

**Preprocessing**:
- Resampled to 16 kHz mono
- Silence trimmed (30 dB threshold)
- Fixed length 3 seconds (padding/truncation)
- Speaker‑aware stratified 80/20 train‑test split

> **Note**: The dataset is not publicly released due to privacy constraints. If you wish to replicate, you can use any Telugu emotional speech corpus or record your own.

---

## 🧠 Model Architecture (CRNN)
Input: (time_steps, 40 MFCCs)
↓
Conv1D(filters=64, kernel=5) + BN + ReLU + MaxPool
Conv1D(filters=128, kernel=3) + BN + ReLU + MaxPool
Conv1D(filters=256, kernel=3) + BN + ReLU + MaxPool
↓
Bidirectional GRU(units=128, return_sequences=False)
↓
Dense(128, ReLU) + Dropout(0.4)
Dense(64, ReLU) + Dropout(0.3)
Dense(5, Softmax) # 5 emotion classes

text

**Regularisation**:
- Batch Normalisation after each Conv1D
- Dropout (0.3–0.4) in dense layers
- L2 weight decay (1e‑4)
- Early stopping (patience=15 epochs)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/shivakrishna117/Telugu-Emotion-Detection.git
cd Telugu-Emotion-Detection
2. Install dependencies
bash
pip install -r requirements.txt
requirements.txt:

text
numpy==1.23.5
librosa==0.10.0
tensorflow==2.13.0
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
soundfile==0.12.1
3. Prepare your data
Place your .wav files in data/raw/ organised by emotion folders:

text
data/raw/
├── angry/
├── happy/
├── neutral/
├── sad/
└── surprised/
Then run:

bash
python src/preprocess.py
python src/features.py
4. Train the model
bash
python src/train.py
Training logs, best model, and plots are saved under models/ and results/.

5. Run inference on a new Telugu audio file
bash
python src/predict.py --audio path/to/your_file.wav
Output example:

text
Predicted emotion: Angry (confidence: 0.87)
📈 Results
Emotion	Precision	Recall	F1-score
Angry	0.85	0.88	0.86
Happy	0.78	0.75	0.76
Neutral	0.82	0.84	0.83
Sad	0.74	0.71	0.72
Surprised	0.81	0.83	0.82
Overall Accuracy: 81.5%

Unweighted Average Recall (UAR): 80.2%

Confusion matrix shows that Happy ↔ Angry and Sad ↔ Neutral are the most frequent misclassifications – a common challenge in speech emotion recognition due to overlapping prosodic features.

🔬 Augmentation & Ablation
We observed that:

SpecAugment (time & frequency masking) gave +3% accuracy over basic augmentation.

Pitch shifting (+2 semitones) helped distinguish high‑arousal emotions (Angry, Happy, Surprised).

Noise addition (σ=0.005) improved generalisation to real‑world background noise.

🧩 Extending to Other Languages
The pipeline is language‑agnostic. To adapt to Hindi, Tamil, English, or German:

Replace the audio dataset.

Change the number of output classes (if needed).

Retrain with the same src/train.py.


