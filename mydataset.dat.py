import os
import pickle
import numpy as np
import librosa
from python_speech_features import mfcc

# Path to the genres folder
directory = r"C:\Users\hp\Downloads\Telegram Desktop\genres_original\genres_original"

# Generate and save the dataset
with open("mydataset.dat", "wb") as f:
    i = 0
    for folder in os.listdir(directory):
        i += 1
        if i == 11:  # Limit to first 10 folders
            break
        count = 0
        for file in os.listdir(os.path.join(directory, folder)):
            if count >= 5:  # Limit to 5 files per genre for testing
                break
            try:
                # Load audio file using librosa
                sig, rate = librosa.load(os.path.join(directory, folder, file), sr=None)
                mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
                covariance = np.cov(np.matrix.transpose(mfcc_feat))
                mean_matrix = mfcc_feat.mean(0)
                feature = (mean_matrix, covariance, i)
                pickle.dump(feature, f)
                count += 1
            except Exception as e:
                print(f"Got an exception: {e} in folder: {folder} filename: {file}")

print("Dataset generated and saved as 'mydataset.dat'")
