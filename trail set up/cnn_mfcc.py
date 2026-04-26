import pandas as pd
import librosa
import numpy as np

df = pd.read_csv("filelist_telugu.csv")
filepaths = df['filepath'].tolist()
labels = df['label'].tolist()
n_mfcc, n_frames = 13, 40

X, y = [], []
for filepath, label in zip(filepaths, labels):
    try:
        audio, sr = librosa.load(filepath, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] < n_frames:
            mfcc = np.pad(mfcc, ((0,0),(0, n_frames-mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:,:n_frames]
        X.append(mfcc)
        y.append(label)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

X = np.stack(X)  # shape: [samples, n_mfcc, n_frames]
y = np.array(y)
np.save("X_mfcc_cnn.npy", X)
np.save("y_mfcc_cnn.npy", y)
print("Saved CNN input: X_mfcc_cnn.npy and y_mfcc_cnn.npy")
