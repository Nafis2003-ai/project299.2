import librosa
import numpy as np
import soundfile as sf
import argparse
import os

# --- Command-Line Argument Parsing ---
parser = argparse.ArgumentParser(description="Pitch Shift Audio File")
parser.add_argument(
    "-i", "--input_name",
    type=str,
    required=True,
    help="Name of the input audio file (e.g., 'my_voice.wav'). Must be in data\\pitch_shifter_inputs\\"
)
parser.add_argument(
    "-n", "--n_steps",
    type=int,
    default=5, # Default shift of 5 semitones
    help="Number of semitones to shift the pitch. Positive for higher, negative for lower. Default is 5."
)
parser.add_argument(
    "-o", "--output_name",
    type=str,
    default=None, # If not provided, a name will be generated
    help="Optional: Name for the output pitch-shifted audio file (e.g., 'shifted_voice.wav'). If not provided, a name will be generated."
)
args = parser.parse_args()

# --- Define Project Root and Paths ---
# This script is in E:\New Volume\project\codep\
# So, PROJECT_ROOT is E:\New Volume\project\
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the full path to the input audio file
input_audio_dir = os.path.join(project_root, "data", "pitch_shifter_inputs")
input_path = os.path.join(input_audio_dir, args.input_name)

# Construct the output directory
output_audio_dir = os.path.join(project_root, "results", "pitch_shifted_audio")
os.makedirs(output_audio_dir, exist_ok=True) # Create the output folder if it doesn't exist

# Determine the output file name
if args.output_name:
    output_filename = args.output_name
else:
    # Generate a default output name based on input and shift
    input_base_name = os.path.splitext(os.path.basename(args.input_name))[0]
    output_filename = f"{input_base_name}_shifted_{args.n_steps}st.wav"

output_path = os.path.join(output_audio_dir, output_filename)


# === Load Input Audio ===
try:
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    print(f"✅ Successfully loaded audio from {input_path} with sample rate {sr}.")
except Exception as e:
    print(f"❌ Failed to load audio from {input_path}: {e}")
    exit()

# === Perform Pitch Shifting ===
try:
    pitch_shifted_audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=args.n_steps)
    print(f"✅ Pitch shifting completed by {args.n_steps} semitones.")
except Exception as e:
    print(f"❌ Pitch shifting failed: {e}")
    exit()

# === Save Output ===
try:
    sf.write(output_path, pitch_shifted_audio, sr)
    print(f"✅ Pitch shifting complete! Saved to {output_path}")
except Exception as e:
    print(f"❌ Failed to save output to {output_path}: {e}")