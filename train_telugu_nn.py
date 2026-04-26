import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import joblib

# Load, encode and split data
df = pd.read_csv('mfcc_features_telugu.csv')
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
y_train = y_train.astype(int)
y_test = y_test.astype(int)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Hyperparameter grid
hidden_sizes = [(256, 128), (512, 256), (128, 64)]
dropouts = [0.3, 0.4, 0.5]
lrs = [0.001, 0.0005]
optimizers = ['Adam', 'SGD']
best_acc = 0
best_params = {}
best_model = None

class TunedMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hs1, hs2, dropout1, dropout2):
        super(TunedMLP, self).__init__()
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

# Grid search for best configuration
for hs1, hs2 in hidden_sizes:
    for d1 in dropouts:
        for d2 in dropouts:
            for lr in lrs:
                for opt_name in optimizers:
                    model = TunedMLP(X_train.shape[1], len(le.classes_), hs1, hs2, d1, d2)
                    if opt_name == "Adam":
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                    elif opt_name == "SGD":
                        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                    criterion = nn.CrossEntropyLoss()
                    best_loss = float('inf')
                    patience = 10
                    counter = 0
                    epochs = 60
                    for epoch in range(epochs):
                        model.train()
                        optimizer.zero_grad()
                        outputs = model(X_train)
                        loss = criterion(outputs, y_train)
                        loss.backward()
                        optimizer.step()
                        # Early stopping on validation loss
                        val_outputs = model(X_test)
                        val_loss = criterion(val_outputs, y_test).item()
                        if val_loss < best_loss:
                            best_loss = val_loss
                            counter = 0
                        else:
                            counter += 1
                            if counter >= patience:
                                break
                    # Evaluate
                    model.eval()
                    with torch.no_grad():
                        logits = model(X_test)
                        preds = torch.argmax(logits, dim=1)
                        acc = (preds == y_test).float().mean().item()
                    # Save if best
                    if acc > best_acc:
                        best_acc = acc
                        best_params = {'hs1': hs1, 'hs2': hs2, 'dropout1': d1, 'dropout2': d2, 'lr': lr, 'optimizer': opt_name}
                        best_model = TunedMLP(X_train.shape[1], len(le.classes_), hs1, hs2, d1, d2)
                        best_model.load_state_dict(model.state_dict())
                        torch.save(best_model.state_dict(), 'best_tuned_telugu_nn.pt')
                    print(f"Params: hs1={hs1}, hs2={hs2}, drop1={d1}, drop2={d2}, lr={lr}, opt={opt_name} => Accuracy: {acc*100:.2f}%")

print(f"\nBest accuracy: {best_acc*100:.2f}% with params {best_params}")

# Classification report and confusion matrix
print("\nClassification Report and Confusion Matrix for Best Model:")
best_mlp = TunedMLP(X_train.shape[1], len(le.classes_),
                    best_params['hs1'], best_params['hs2'],
                    best_params['dropout1'], best_params['dropout2'])
best_mlp.load_state_dict(torch.load('best_tuned_telugu_nn.pt'))
best_mlp.eval()
with torch.no_grad():
    logits = best_mlp(X_test)
    preds = torch.argmax(logits, dim=1)
    print(classification_report(y_test.numpy(), preds.numpy()))
    print(confusion_matrix(y_test.numpy(), preds.numpy()))

# --- Save the scaler and label encoder after grid search ---
joblib.dump(scaler, "mfcc_scaler.joblib")
joblib.dump(le, "label_encoder.joblib")
