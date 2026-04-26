import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# ---- Hyperparams: set these to your best model values! ----
hs1, hs2, d1, d2 = 512, 256, 0.5, 0.3

# ---- Load artifacts ----
scaler = joblib.load("mfcc_scaler.joblib")
le = joblib.load("label_encoder.joblib")
model_state = torch.load("best_tuned_telugu_nn.pt", map_location=torch.device('cpu'))

# ---- Load data ----
df = pd.read_csv('mfcc_features_telugu.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y = le.transform(y)

X = scaler.transform(X)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# ---- Model matching training config ----
class TunedMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hs1, hs2, dropout1, dropout2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hs1)
        self.bn1 = nn.BatchNorm1d(hs1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout1)
        self.fc2 = nn.Linear(hs1, hs2)
        self.bn2 = nn.BatchNorm1d(hs2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout2)
        self.fc3 = nn.Linear(hs2, num_classes)
    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        return self.fc3(x)

mlp = TunedMLP(X.shape[1], len(le.classes_), hs1, hs2, d1, d2)
mlp.load_state_dict(model_state)
mlp.eval()

with torch.no_grad():
    logits = mlp(X)
    preds = torch.argmax(logits, dim=1)
    print("Test Classification Report:")
    print(classification_report(y.numpy(), preds.numpy()))
    print("Confusion Matrix:")
    print(confusion_matrix(y.numpy(), preds.numpy()))
