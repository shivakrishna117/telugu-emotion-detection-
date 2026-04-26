import pandas as pd
import librosa
import numpy as np

# Load the CSV file with all filepaths and labels
df = pd.read_csv(r"C:\Users\shiva\Desktop\ED TaI\trail set up\filelist_telugu.csv")  # Use the correct path!
filepaths = df['filepath'].tolist()
labels = df['label'].tolist()

features = []
labels_out = []

for filepath, label in zip(filepaths, labels):
    try:
        # Load audio and standardize sampling rate
        audio, sr = librosa.load(filepath, sr=16000)
        # Extract 13 MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        # Take mean across time frames to get fixed length vector
        mfcc_mean = np.mean(mfcc, axis=1)
        features.append(mfcc_mean)
        labels_out.append(label)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

# Compose features DataFrame and save to CSV
features_df = pd.DataFrame(features)
features_df['label'] = labels_out

features_df.to_csv('mfcc_features_telugu.csv', index=False)

print("MFCC features extracted and saved to mfcc_features_telugu.csv")
print(f'Files processed: {len(filepaths)}')
print(f'Features found: {len(features)}')
print(f'Labels found: {len(labels_out)}')
