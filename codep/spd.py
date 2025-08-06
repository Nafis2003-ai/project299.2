import argparse
import os
import gc
import numpy as np
import torch # pyannote.audio and whisper depend on torch

# Import pyannote.audio and pydub
try:
    from pyannote.audio import Pipeline
    from pydub import AudioSegment
    import whisper
except ImportError as e:
    print(f"❌ Error: Missing required library. Please ensure all dependencies are installed.")
    print(f"Specific error: {e}")
    print("Try running: pip install pyannote.audio pydub openai-whisper")
    print("Also ensure you have ffmpeg installed and in your system's PATH.")
    exit()

# --- Command-Line Argument Parsing ---
parser = argparse.ArgumentParser(description="Perform Speaker Diarization and Transcription on an Audio File")
parser.add_argument(
    "-i", "--input_name",
    type=str,
    required=True,
    help="Name of the input audio file (e.g., 'meeting.wav'). Must be in data\\spd_inputs\\"
)
args = parser.parse_args()

# --- Define Project Root and Paths ---
# This script is in E:\New Volume\project\codep\
# So, PROJECT_ROOT is E:\New Volume\project\
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the full path to the input audio file
input_audio_dir = os.path.join(project_root, "data", "training")
input_path = os.path.join(input_audio_dir, args.input_name)

# Construct the output directory and hardcoded file path
output_text_dir = os.path.join(project_root, "results", "spd_outputs")
os.makedirs(output_text_dir, exist_ok=True) # Create the output folder if it doesn't exist

# Hardcoded output file path
output_path = os.path.join(output_text_dir, "diarized_transcript.txt") # The output will always be saved as 'diarized_transcript.txt'

# --- Hugging Face Token ---

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("❌ Error: Hugging Face token (HF_TOKEN) environment variable not set.")
    print("Please set it before running the script. Get your token from huggingface.co/settings/tokens")
    exit()

# === Main Diarization and Transcription Logic ===
if __name__ == "__main__":
    if not os.path.exists(input_path):
        print(f"❌ Error: Input audio file not found at '{input_path}'. Please check the path.")
        exit()
    if not os.path.isfile(input_path):
        print(f"❌ Error: Provided path '{input_path}' is not a file.")
        exit()

    print(f"Processing audio from {input_path} for speaker diarization and transcription...")

    try:
        # Initialize diarization pipeline
        print("Loading diarization pipeline (this may take a while the first time)...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=HF_TOKEN
        )
        print("Diarization pipeline loaded.")

        # Run diarization
        print("Running speaker diarization...")
        diarization = pipeline(input_path)
        print("Diarization complete.")

        # Load audio using pydub
        print("Loading audio for transcription...")
        audio = AudioSegment.from_file(input_path) # Use from_file to handle various formats
        audio = audio.set_frame_rate(16000) # Whisper requires 16000 Hz sample rate
        print("Audio loaded and resampled to 16kHz.")

        # Load Whisper model
        print("Loading Whisper model (small.en)...")
        model = whisper.load_model("small.en")
        print("Whisper model loaded.")

        # Helper: convert pydub segment to numpy float32 array
        def to_float_array(segment):
            arr = np.array(segment.get_array_of_samples())
            # Convert to float32 and normalize to -1.0 to 1.0 range
            return arr.astype(np.float32) / 32768.0

        # Process segments and write to file
        print(f"Starting transcription and writing results to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Speaker Diarization and Transcription for: {os.path.basename(input_path)}\n\n")
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start = int(turn.start * 1000) # Convert to milliseconds
                end = int(turn.end * 1000)     # Convert to milliseconds
                
                segment = audio[start:end]
                audio_array = to_float_array(segment)
                
                # Transcribe the segment
                result = model.transcribe(audio_array, fp16=False) # fp16=False if not using GPU
                
                # Write to file
                f.write(f"[ {turn.start:.2f} -- {turn.end:.2f} ] {speaker} : {result['text']}\n")
                
                # Clean up memory
                del result, segment, audio_array
                gc.collect()
        print(f"✅ Speaker diarization and transcription complete! Results saved to {output_path}")

    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")
        print("Please ensure:")
        print("1. Your Hugging Face token is correctly set as an environment variable (HF_TOKEN).")
        print("2. All required Python libraries are installed in your 'sed-env'.")
        print("3. FFmpeg is installed on your system and its path is added to environment variables.")
        print("4. The input audio file is valid and accessible.")