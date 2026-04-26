import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your MFCC features CSV
df = pd.read_csv('mfcc_features_telugu.csv')

# Show first few rows to verify
print("\nSample rows from dataset:")
print(df.head())

# 1. Distribution of emotion classes
print("\nClass distribution (counts):")
print(df['label'].value_counts())

plt.figure(figsize=(8,5))
sns.countplot(x='label', data=df, palette='Set2', order=df['label'].value_counts().index)
plt.title('Emotion Class Distribution')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.show()

# 2. Check for null values
print("\nNull value count per column:")
print(df.isnull().sum())

# 3. Summary statistics for features (excluding label column)
print("\nFeature summary statistics:")
print(df.describe().T)

# 4. Boxplot of features for each emotion (first 5 coefficients as sample)
plt.figure(figsize=(14, 8))
for i, col in enumerate(df.columns[:-1][:5]):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='label', y=col, data=df, palette='Set2')
    plt.title(f'Distribution of {col} by Emotion')
plt.tight_layout()
plt.show()

# 5. Pairplot (sample only first N rows for speed)
sample_df = df.sample(n=200) if len(df) > 200 else df
sns.pairplot(sample_df, hue='label', vars=df.columns[:3], palette='Set2')  # show MFCC_0, MFCC_1, MFCC_2
plt.suptitle("Pairplot of First 3 MFCC Features by Emotion")
plt.show()

# 6. Correlation heatmap
corr = df.iloc[:, :-1].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('MFCC Feature Correlation Heatmap')
plt.show()

print("\nEDA complete! Review visualizations for insights.")
