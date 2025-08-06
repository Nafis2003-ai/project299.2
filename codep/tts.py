from gtts import gTTS
import argparse
import os

# --- Command-Line Argument Parsing ---
parser = argparse.ArgumentParser(description="Convert Text to Speech from a Text File")
parser.add_argument(
    "-i", "--input_name",
    type=str,
    required=True,
    help="Name of the input text file (e.g., 'my_text.txt'). Must be in data\\tts_inputs\\"
)
args = parser.parse_args()

# --- Define Project Root and Paths ---
# This script is in E:\New Volume\project\codep\
# So, PROJECT_ROOT is E:\New Volume\project\
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the full path to the input text file
input_text_dir = os.path.join(project_root, "data", "training")
input_path = os.path.join(input_text_dir, args.input_name)

# Construct the output directory and hardcoded file path
output_audio_dir = os.path.join(project_root, "results", "tts_outputs")
os.makedirs(output_audio_dir, exist_ok=True) # Create the output folder if it doesn't exist

# Hardcoded output file path
output_path = os.path.join(output_audio_dir, "texttospeech.mp3") # The output will always be saved as 'texttospeech.mp3'

# === Text-to-Speech Logic ===
def convert_text_to_speech(text_file_path):
    """
    Converts text from a file to speech using Google Text-to-Speech (gTTS).
    """
    if not os.path.exists(text_file_path):
        print(f"❌ Error: Input text file not found at '{text_file_path}'. Please check the path.")
        return None
    if not os.path.isfile(text_file_path):
        print(f"❌ Error: Provided path '{text_file_path}' is not a file.")
        return None

    print(f"Processing text from {text_file_path}...")
    try:
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        language = "en" # Hardcoded language to English
        # You can choose different tld (top-level domain) for different accents:
        # 'com.au' for Australian, 'co.uk' for British, 'com' for American, etc.
        speech = gTTS(text=text, lang=language, slow=False, tld='com') 
        
        print(f"✅ Text-to-Speech conversion complete.")
        return speech

    except Exception as e:
        print(f"❌ An unexpected error occurred during Text-to-Speech conversion: {e}")
        return None

# === Main Execution ===
if __name__ == "__main__":
    tts_object = convert_text_to_speech(input_path)

    if tts_object:
        # === Save Output ===
        try:
            tts_object.save(output_path)
            print(f"✅ Speech saved to {output_path}")
        except Exception as e:
            print(f"❌ Failed to save speech to {output_path}: {e}")