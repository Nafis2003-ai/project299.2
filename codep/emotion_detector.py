import argparse # Import argparse for command-line arguments
import torch
import librosa
import soundfile as sf
import numpy as np
import os
import io # Needed for handling audio data from microphone if it were used with pipeline's raw input
from transformers import pipeline, AutoFeatureExtractor, AutoModelForSequenceClassification

# --- Configuration ---
# Define the pre-trained model to use for emotion detection
MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
# Required sample rate for the model
TARGET_SAMPLE_RATE = 16000 

# --- Model Loading ---
# Initialize the feature extractor and the model
# Using pipeline for simplicity, it handles feature extraction and classification
print(f"Loading emotion detection model: {MODEL_NAME}...")
# It's good practice to specify the device if you have a GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the pipeline. This will download the model and its components.
classifier = pipeline(
    "audio-classification", 
    model=MODEL_NAME, 
    device=device # Use the determined device
)
print("Model loaded successfully.")

# Get the emotion labels from the model's configuration
# The model's config.id2label maps numerical IDs to emotion names
emotion_labels = classifier.model.config.id2label
print(f"Detected emotion labels: {emotion_labels}")

# --- Emotion Detection Function ---
def detect_emotion(audio_file_path):
    """
    Detects emotion from an audio file using the pre-trained model.

    Args:
        audio_file_path (str): Path to the audio file.

    Returns:
        tuple: A tuple containing:
            - str: The detected emotion label and confidence.
            - dict: A dictionary of emotion probabilities.
    """
    if not audio_file_path or not os.path.exists(audio_file_path):
        return "Error: Audio file not found or path is empty.", {}

    # Ensure the audio file is at the target sample rate
    try:
        audio_data, current_sr = librosa.load(audio_file_path, sr=None, mono=True)
        if current_sr != TARGET_SAMPLE_RATE:
            print(f"Resampling audio from {current_sr}Hz to {TARGET_SAMPLE_RATE}Hz...")
            audio_data = librosa.resample(audio_data, orig_sr=current_sr, target_sr=TARGET_SAMPLE_RATE)
        
        # Prepare audio for the pipeline
        audio_for_pipeline = {
            'raw': audio_data,
            'sampling_rate': TARGET_SAMPLE_RATE
        }
    except Exception as e:
        return f"Error loading or processing audio file: {e}", {}

    print(f"Processing audio from {audio_file_path}...")
    # Perform inference
    # The pipeline returns a list of dictionaries, e.g., [{'score': 0.9, 'label': 'happiness'}, ...]
    prediction = classifier(audio_for_pipeline)

    # Sort predictions by score in descending order
    prediction = sorted(prediction, key=lambda x: x['score'], reverse=True)

    # Get the top emotion
    top_emotion = prediction[0]['label']
    top_score = prediction[0]['score']

    # Format all scores into a dictionary for display
    emotion_scores = {p['label']: f"{p['score']:.4f}" for p in prediction}

    result_text = f"Detected Emotion: {top_emotion} (Confidence: {top_score:.2%})"
    
    return result_text, emotion_scores

# --- Command-Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Emotion Detection System")
    parser.add_argument(
        "-a", "--audio_path", 
        type=str, 
        required=True, 
        help="Path to the input audio file (e.g., 'path/to/your/audio.wav')"
    )
    args = parser.parse_args()

    # Call the detection function
    result_text, emotion_scores = detect_emotion(args.audio_path)

    # --- Output to Console ---
    print("\n--- Speech Emotion Detection Results ---")
    print(result_text)
    print("\nEmotion Probabilities:")
    for label, score in emotion_scores.items():
        print(f"- {label}: {score}")
    print("--------------------------------------")

    # --- Save Output to File ---
    output_folder = "E:\project\\results"
    
    # Get the directory of the current script (emotion_detector.py)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_script_dir, output_folder)

    # Create the 'results' folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the output filename based on the input audio file
    # Get the base name of the input audio file (without path or extension)
    audio_base_name = os.path.splitext(os.path.basename(args.audio_path))[0]
    output_filename = f"{audio_base_name}_emotion_results.txt"
    output_filepath = os.path.join(output_dir, output_filename)

    # Write the results to the text file
    with open(output_filepath, "w") as f:
        f.write("--- Speech Emotion Detection Results ---\n")
        f.write(result_text + "\n\n")
        f.write("Emotion Probabilities:\n")
        for label, score in emotion_scores.items():
            f.write(f"- {label}: {score}\n")
        f.write("--------------------------------------\n")
    
    print(f"\nResults saved to: {output_filepath}")