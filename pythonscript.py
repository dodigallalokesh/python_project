import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# Function to extract audio features
def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

# Function to train emotion detection model
def train_emotion_detection_model(features, labels):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(features, labels)
    return clf

# Function to predict emotion
def predict_emotion(model, audio_file):
    features = extract_audio_features(audio_file)
    emotion_label = model.predict([features])[0]
    return "happy" if emotion_label == 1 else "sad"

# Function to load Common Voice dataset
def load_common_voice_dataset(language_code, split='train'):
    common_voice_dataset = load_dataset('common_voice', language_code, split=split)
    return common_voice_dataset

# Function to print results
def print_results(dataset, emotion_detection_model):
    emotion_predictions = []

    for item in dataset:
        audio_file = item['path']
        emotion = predict_emotion(emotion_detection_model, audio_file)
        emotion_predictions.append((audio_file, emotion))

    # Print predicted emotions
    print("Emotion Predictions:")
    print(pd.DataFrame(emotion_predictions, columns=['Audio File', 'Emotion']))
    
    
    

	# Predict talk speed
    talk_speeds = calculate_talk_speed([entry[0] for entry in emotion_predictions])
    talk_speed_df = pd.DataFrame({
        'Audio File': [entry[0] for entry in emotion_predictions],
        'Talk Speed (s)': talk_speeds
    })

    # Print predicted talk speed as a table
    print("\nPredicted Talk Speed:")
    print(talk_speed_df)


    
    
    # Generate the line chart
    timestamps = range(len(emotion_predictions))  # Replace with actual timestamps
    emotions = [entry[1] for entry in emotion_predictions]

    emotion_values = [1 if emotion == "happy" else -1 for emotion in emotions]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, emotion_values, marker='o', linestyle='-')
    plt.xlabel("Time")
    plt.ylabel("Emotion")
    plt.title("Predicted Emotions Over Time")
    plt.yticks([-1, 1], ["sad", "happy"])
    plt.show()

# Language code for Common Voice (e.g., 'cy' for Welsh)
language_code = 'cy'

# Load Common Voice dataset
common_voice_train = load_common_voice_dataset(language_code, split='train')

# Extract features and labels
audio_paths = common_voice_train['path']
labels = common_voice_train['sentence']

# Features (audio files) and Labels (emotions) are used as placeholders.
# Replace this with your actual emotion labels, or any other attribute that indicates emotion in the Common Voice dataset.

# Example: Map labels to binary labels for emotion detection (for illustration purposes)
binary_labels = [1 if "happy" in label else 0 for label in labels]

# Train emotion detection model
emotion_detection_model = train_emotion_detection_model(audio_paths, binary_labels)

# Call the print_results function
print_results(common_voice_train, emotion_detection_model)











