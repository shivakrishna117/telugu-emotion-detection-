import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Load matrix MFCC features
X = np.load("X_mfcc_cnn.npy")         # [samples, n_mfcc, n_frames]
y = np.load("y_mfcc_cnn.npy")         # [samples,]

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Add channel dimension for CNN input
X = X[:, np.newaxis, :, :]      # [samples, 1, n_mfcc, n_frames]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

class MFCCCNN(nn.Module):
    def __init__(self, n_mfcc, n_frames, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        flatten_dim = (n_mfcc//4)*(n_frames//4)*32
        self.fc1 = nn.Linear(flatten_dim, 64)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        return self.fc2(x)

n_mfcc, n_frames = X_train.shape[2], X_train.shape[3]
n_classes = len(le.classes_)
model = MFCCCNN(n_mfcc, n_frames, n_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 40
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1)%5==0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
    acc = np.mean(preds == y_test.numpy())
    print(f"Test Accuracy: {acc*100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test.numpy(), preds, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test.numpy(), preds))
