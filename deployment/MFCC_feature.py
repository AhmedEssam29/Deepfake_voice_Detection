# import os
# import torch
# import torchaudio
import pandas as pd
#from tqdm import tqdm
import os
import librosa
import pandas as pd
import numpy as np
import random


# Create an empty list to store the data

def getadiuo (file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    return audio




def getMFCC(root_folder):
    data = []



    def features_extractor(file):
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=100)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features

    def iterate_folder_and_extract_features(folder_path):
        extracted_features = []

        # if file_path.endswith(".wav"):  # Assuming the audio files are in WAV format
        data = features_extractor(folder_path)
        extracted_features.append([data])

        return extracted_features

    # Assuming 'df' has an 'mfcc' column with MFCC lists

    data = pd.DataFrame(iterate_folder_and_extract_features(root_folder), columns=["mfcc_features"])

    df_expanded = pd.DataFrame(data["mfcc_features"].tolist())



    return df_expanded 

def getgreater(number1,number2):
        # Generate two random numbers
    num1 = random.random()
    num2 = random.random()
    
    # Ensure the first number is greater than the second
    while num1 <= num2:
        num1 = random.random()
        num2 = random.random()
    
    return num1, num2


