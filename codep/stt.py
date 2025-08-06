import speech_recognition as sr
import argparse
import os

# --- Command-Line Argument Parsing ---
parser = argparse.ArgumentParser(description="Convert Speech to Text from an Audio File")
parser.add_argument(
    "-i", "--input_name",
    type=str,
    required=True,
    help="Name of the input audio file (e.g., 'my_speech.wav'). Must be in data\\stt_inputs\\"
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
output_text_dir = os.path.join(project_root, "results", "stt_outputs")
os.makedirs(output_text_dir, exist_ok=True) # Create the output folder if it doesn't exist

# Hardcoded output file path
output_path = os.path.join(output_text_dir, "transcript.txt") # The output will always be saved as 'transcript.txt'

# === Speech Recognition Logic ===
def convert_audio_to_text(audio_file_path):
    """
    Converts speech from an audio file to text using Google Speech Recognition.
    """
    r = sr.Recognizer()
    transcribed_text = ""
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Error: Input audio file not found at '{audio_file_path}'. Please check the path.")
        return None
    if not os.path.isfile(audio_file_path):
        print(f"❌ Error: Provided path '{audio_file_path}' is not a file.")
        return None

    print(f"Processing audio from {audio_file_path}...")
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = r.record(source)  # read the entire audio file
            transcribed_text = r.recognize_google(audio) # using Google Speech Recognition
            transcribed_text = transcribed_text.lower()
            print(f"✅ Transcription complete: {transcribed_text}")
    except sr.UnknownValueError:
        print("❌ Google Speech Recognition could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"❌ Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred during transcription: {e}")
        return None
    
    return transcribed_text

# === Main Execution ===
if __name__ == "__main__":
    transcription = convert_audio_to_text(input_path)

    if transcription:
        # === Save Output ===
        try:
            with open(output_path, "w") as file:
                file.write(transcription)
            print(f"✅ Transcription saved to {output_path}")
        except Exception as e:
            print(f"❌ Failed to save transcription to {output_path}: {e}")