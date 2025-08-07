import argparse
import os
import soundfile as sf # Import soundfile for saving audio

# Assuming infer_tool is in the 'inference' directory
from inference.infer_tool import Svc 

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source", type=str, required=True, help="Input audio name without .wav")
parser.add_argument("-spk", "--speaker", type=str, required=True, help="Target speaker name")
args = parser.parse_args()

model = Svc("logs/pretrained_vc/G_0.pth", "configs/config.json")

audio_path = f"raw/{args.source}.wav"

# Call the infer method, which now *returns* the audio data and its sample rate
# The infer method returns `audio, audio.shape[-1]` where audio is a torch.Tensor
# So, `out_audio` will be the Tensor and `_` will be its length.
out_audio_tensor, _ = model.infer(args.speaker, tran=0, raw_path=audio_path)

# Convert the PyTorch tensor to a NumPy array for saving
out_audio_np = out_audio_tensor.cpu().numpy()

# --- Saving Logic ---
output_folder = "results"
# Construct the path to the 'results' folder relative to svc_infer.py
# Get the directory of the current script (svc_infer.py)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_script_dir, output_folder)

# Create the 'results' folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the output filename
# We'll use the source name and target speaker name for clarity
source_base_name = os.path.splitext(args.source)[0] # e.g., "my_audio" from "my_audio.wav"
output_filename = f"{source_base_name}_to_{args.speaker}_tran{0}.wav" # tran is 0 in your call

# Combine folder and filename
output_filepath = os.path.join(output_dir, output_filename)

# Save the audio file
# The Svc class has self.target_sample, which is the model's output sample rate.
# We can access it via `model.target_sample`.
sf.write(output_filepath, out_audio_np, model.target_sample)

print(f"Generated audio saved to: {output_filepath}")

