import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import hinge_loss

# Load and preprocess data
df = pd.read_csv("mfcc_features_telugu.csv")
if df['label'].dtype == object or not pd.api.types.is_integer_dtype(df['label']):
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Standardize features for SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 1. SVM: train, predict, classification report
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, random_state=42)
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, svm_preds))
print("SVM Confusion Matrix:")
print(confusion_matrix(y_test, svm_preds))

# 2. Plot SVM hinge loss curve vs margin for synthetic binary sample (visual explanation)
margins = np.linspace(-2, 2, 100)
hinge_losses = np.maximum(0, 1 - margins)
plt.figure()
plt.plot(margins, hinge_losses, label='Hinge Loss')
plt.xlabel("Margin (y*f(x))")
plt.ylabel("Hinge Loss")
plt.title("SVM Hinge Loss Curve vs Margin")
plt.legend()
plt.grid(True)
plt.show()

# 3. Calculate hinge loss on your test data (if binary)
if len(np.unique(y_test)) == 2:
    y_test_binary = np.where(y_test == svm.classes_[0], -1, 1)
    decision_scores = svm.decision_function(X_test)
    hl = hinge_loss(y_test_binary, decision_scores)
    print(f"SVM hinge loss on test data: {hl:.4f}")

# 4. Random Forest: train, feature importance plot
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_preds))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

# Plot Random Forest feature importances
plt.figure()
plt.bar(range(X.shape[1]), rf.feature_importances_)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importances")
plt.show()
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(18, 8))  # Wider aspect ratio, more space!
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.axis('off')

block_width = 0.12
block_height = 0.30
block_kwargs = dict(boxstyle='round,pad=0.45', fc='#d8e6fa', ec='navy', lw=2, alpha=0.97)

blocks = [
    ("Input Audio", "Telugu Speech", 0.05, 0.7),
    ("Preprocessing", "Noise Reduction\nSilence Trimming\nNormalization", 0.22, 0.7),
    ("Feature Extraction", "MFCC Computation", 0.39, 0.7),
    ("Model Architectures", "SVM Baseline\nCNN, RNN, CRNN,\nBiGRU, Attention, ...", 0.56, 0.7),
    ("Evaluation & Comparison", "Accuracy\nF1-score\nConfusion Matrix", 0.73, 0.7),
    ("UI Application", "Gradio/\nStreamlit Demo", 0.56, 0.3)
]

for (title, annotation, x, y) in blocks:
    ax.add_patch(FancyBboxPatch((x, y), block_width, block_height, **block_kwargs))
    ax.text(x + block_width / 2, y + block_height - 0.06, title,
            fontsize=15, fontweight='bold', ha='center', va='top')
    ax.text(x + block_width / 2, y + block_height / 2 - 0.03, annotation,
            fontsize=13, ha='center', va='center')

arrow_y = 0.85
arrow_flow = [
    ((0.05 + block_width, arrow_y), (0.22, arrow_y)),
    ((0.22 + block_width, arrow_y), (0.39, arrow_y)),
    ((0.39 + block_width, arrow_y), (0.56, arrow_y)),
    ((0.56 + block_width, arrow_y), (0.73, arrow_y)),
]
for start, end in arrow_flow:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=32, color='navy', lw=3))

# Down arrow (Model Architectures -> UI Application)
ax.add_patch(FancyArrowPatch((0.56 + block_width / 2, 0.70), (0.56 + block_width / 2, 0.44), arrowstyle='-|>', mutation_scale=32, color='navy', lw=3))

# Diagonal arrow (UI Application -> Evaluation)
ax.add_patch(FancyArrowPatch((0.56 + block_width / 2, 0.30 + block_height), (0.73 + block_width / 2, 0.70), arrowstyle='-|>', mutation_scale=32, color='navy', lw=3))

plt.title('End-to-End System Architecture for Telugu Speech Emotion Recognition', fontsize=18, fontweight='bold', pad=28)
plt.tight_layout()
plt.show()
