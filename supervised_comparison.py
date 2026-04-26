import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load and preprocess data
df = pd.read_csv('mfcc_features_telugu.csv')
if df['label'].dtype == object or not pd.api.types.is_integer_dtype(df['label']):
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Standardize features for the SVM
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_preds))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

# 4. Support Vector Machine (SVM) Classifier
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, random_state=42)
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)
print("\nSVM Classification Report:")
print(classification_report(y_test, svm_preds))
print("SVM Confusion Matrix:")
print(confusion_matrix(y_test, svm_preds))
