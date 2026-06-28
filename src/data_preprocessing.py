# import librosa # type: ignore
# import numpy as np # type: ignore
# import os

# def extract_features(file_path, n_mfcc=40):
#     # Load audio file
#     audio, sr = librosa.load(file_path, sr=None)
#     # Compute MFCC features from the audio signal
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
#     # Average MFCC features over time
#     mfccs_scaled = np.mean(mfccs.T, axis=0)
#     return mfccs_scaled

# def process_directory(directory):
#     features = []
#     labels = []
#     # Process each .wav file in the directory
#     for file in os.listdir(directory):
#         if file.endswith('.wav'):
#             file_path = os.path.join(directory, file)
#             features.append(extract_features(file_path))
#             # Assume the emotion label is the first part of the filename (e.g., happy_01.wav)
#             label = file.split('_')[0]
#             labels.append(label)
#     return np.array(features), np.array(labels)

# if __name__ == '__main__':
#     features, labels = process_directory('data/train')
#     print("Features shape:", features.shape)
#     print("Labels shape:", labels.shape)


import librosa  # type: ignore
import numpy as np  # type: ignore
import os

def extract_features(file_path, n_mfcc=40):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=None)
    # Compute MFCC features from the audio signal
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    # Average MFCC features over time
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

def process_directory(directory):
    # Construct the absolute path for the directory based on the current working directory
    absolute_dir = os.path.join(os.getcwd(), directory)
    
    if not os.path.isdir(absolute_dir):
        raise FileNotFoundError(f"Directory not found: {absolute_dir}")
    
    features = []
    labels = []
    # Process each .wav file in the directory
    for file in os.listdir(absolute_dir):
        if file.endswith('.wav'):
            file_path = os.path.join(absolute_dir, file)
            features.append(extract_features(file_path))
            # Assume the emotion label is the first part of the filename (e.g., happy_01.wav)
            label = file.split('_')[0]
            labels.append(label)
    return np.array(features), np.array(labels)

if __name__ == '__main__':
    # Print the current working directory for debugging purposes
    print("Current Working Directory:", os.getcwd())
    
    # Update the path as needed; here we assume you're running the script from the project root.
    train_directory = "data/train"
    
    features, labels = process_directory(train_directory)
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)

