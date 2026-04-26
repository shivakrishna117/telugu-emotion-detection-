import os
import pandas as pd

DATASET_DIR = r"C:\Users\shiva\Desktop\ED TaI\telugu"
filepaths = []
labels = []

for label in os.listdir(DATASET_DIR):
    class_dir = os.path.join(DATASET_DIR, label)
    if os.path.isdir(class_dir):
        for filename in os.listdir(class_dir):
            if filename.lower().endswith('.wav'):
                filepath = os.path.join(class_dir, filename)
                filepaths.append(filepath)
                labels.append(label)

df = pd.DataFrame({'filepath': filepaths, 'label': labels})
df.to_csv("filelist_telugu.csv", index=False)
print("File list created: filelist_telugu.csv")
