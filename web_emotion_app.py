import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import joblib
import librosa
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---- Model definitions ----
class TunedMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hs1, hs2, d1, d2):
        super(TunedMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hs1)
        self.bn1 = nn.BatchNorm1d(hs1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(d1)
        self.fc2 = nn.Linear(hs1, hs2)
        self.bn2 = nn.BatchNorm1d(hs2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(d2)
        self.fc3 = nn.Linear(hs2, num_classes)
    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        return self.fc3(x)

# ---- Load objects from disk ----
scaler = joblib.load("mfcc_scaler.joblib")
le = joblib.load("label_encoder.joblib")
input_dim = len(scaler.mean_)
num_classes = len(le.classes_)
hs1, hs2, d1, d2 = 256, 128, 0.3, 0.3
mlp = TunedMLP(input_dim, num_classes, hs1, hs2, d1, d2)
mlp.load_state_dict(torch.load('best_tuned_telugu_nn.pt', map_location=torch.device('cpu')))
mlp.eval()

# ---- SVM and RF setup ----
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
svm = joblib.load("svm_model.joblib")
rf = joblib.load("rf_model.joblib")

# ---- Audio preprocessing and prediction ---
def extract_mfcc_features(audio_path, target_dim):
    y, sr = librosa.load(audio_path, sr=None)
    # Extract MFCCs; n_mfcc matches your training set (commonly 13, 20, etc.)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=target_dim)
    mfcc_mean = np.mean(mfcc, axis=1)  # Use mean across time frames for fixed input
    # If necessary, pad/truncate to target_dim
    if mfcc_mean.shape[0] < target_dim:
        mfcc_mean = np.pad(mfcc_mean, (0, target_dim - mfcc_mean.shape[0]), 'constant')
    elif mfcc_mean.shape[0] > target_dim:
        mfcc_mean = mfcc_mean[:target_dim]
    return mfcc_mean.reshape(1, -1)  # 2D array for sklearn/pytorch

def predict_from_audio(audio_file):
    # audio_file is a temp file from Gradio upload
    try:
        mfcc_features = extract_mfcc_features(audio_file, input_dim)
        # scale
        X_scaled = scaler.transform(mfcc_features)
        # NN Inference
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = mlp(X_tensor)
            nn_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            nn_pred_idx = np.argmax(nn_probs)
            nn_pred = le.inverse_transform([nn_pred_idx])[0]
            nn_conf = float(nn_probs[nn_pred_idx])
        # SVM
        svm_pred_idx = svm.predict(X_scaled)[0]
        svm_pred = le.inverse_transform([svm_pred_idx])[0]
        # RF
        rf_pred_idx = rf.predict(X_scaled)[0]
        rf_pred = le.inverse_transform([rf_pred_idx])[0]

        pred_table = pd.DataFrame({
            "Model": ["NeuralNet_MLP", "SVM", "Random Forest"],
            "Prediction": [nn_pred, svm_pred, rf_pred],
            "Confidence": [f"{nn_conf:.4f}", "N/A", "N/A"]
        })
        return pred_table
    except Exception as e:
        return f"Error processing audio: {str(e)}"

iface = gr.Interface(
    fn=predict_from_audio,
    inputs=[gr.Audio(source="upload", type="filepath", label="Upload Audio (.wav, .mp3, etc.)")],
    outputs=[gr.Dataframe(label="Predictions Table")],
    title="Emotion Detection from Telugu Speech Audio",
    description="Upload an audio clip to predict emotion using NeuralNet MLP, SVM, and Random Forest. (MFCC features are extracted internally.)"
)

if __name__ == "__main__":
    iface.launch()
