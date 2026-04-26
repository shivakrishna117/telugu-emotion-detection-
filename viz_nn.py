import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, hinge_loss
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

# --------------- Load, Encode, Split ---------------
df = pd.read_csv('mfcc_features_telugu.csv')
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Save scaler and label encoder for deployment/demo reproducibility
joblib.dump(scaler, "mfcc_scaler.joblib")
joblib.dump(le, "label_encoder.joblib")

# --------------- Neural Network ---------------
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

# Best config hyperparams (can use grid search results)
input_dim = X_train.shape[1]
num_classes = len(le.classes_)
hs1, hs2 = 256, 128
d1, d2 = 0.3, 0.3
lr = 0.001
epochs = 50

model = TunedMLP(input_dim, num_classes, hs1, hs2, d1, d2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Convert to tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_losses, test_losses, test_accs = [], [], []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    # Validation
    model.eval()
    with torch.no_grad():
        val_out = model(X_test_t)
        val_loss = criterion(val_out, y_test_t).item()
        val_pred = torch.argmax(val_out, dim=1)
        acc = (val_pred == y_test_t).float().mean().item()
    train_losses.append(loss.item())
    test_losses.append(val_loss)
    test_accs.append(acc)
    print(f"Epoch {epoch+1}: TrainLoss={loss.item():.4f} TestLoss={val_loss:.4f} TestAcc={acc*100:.2f}%")

# Final classification report and confusion matrix
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    preds = torch.argmax(logits, dim=1)
    print("\nMLP Classification Report")
    print(classification_report(y_test, preds.cpu().numpy()))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds.cpu().numpy()))

# ---- Loss and Accuracy curves ----
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Loss curves")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(test_accs, label='Test Accuracy', color='g')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MLP Accuracy curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------- SVM ---------------
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)
print("\nSVM Classification Report")
print(classification_report(y_test, svm_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_preds))

# --- SVM hinge loss curve vs margin (visual explanation) ---
margins = np.linspace(-2, 2, 100)
hinge_losses = np.maximum(0, 1 - margins)
plt.figure()
plt.plot(margins, hinge_losses, label="Hinge Loss Curve")
plt.xlabel("Margin (y*f(x))")
plt.ylabel("Hinge Loss")
plt.title("SVM Hinge Loss Curve vs Margin")
plt.legend()
plt.grid(True)
plt.show()

# --- SVM Decision Boundary, reduces to 2D with PCA ---
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
svm2d = SVC(kernel='rbf', C=1.0, gamma='scale')
svm2d.fit(X_2d, y)
def plot_decision_boundary(model, X, y):
    h = .02
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('SVM Decision Boundary (PCA-reduced data)')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()
plot_decision_boundary(svm2d, X_2d, y)

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("\nRandom Forest Classification Report")
print(classification_report(y_test, rf_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

# Feature importances
plt.figure()
plt.bar(range(X.shape[1]), rf.feature_importances_)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importances")
plt.grid(True)
plt.show()
