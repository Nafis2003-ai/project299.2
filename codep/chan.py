import librosa
import soundfile as sf
from so_vits_svc_fork.inference.core import Svc

# === Paths ===
model_path = "nardic/G_50000.pth"
config_path = "nardic/config.json"
input_path = "E:/python/data/training/source_fixed.wav"
output_path = "E:/python/results/change.wav"

# === Load input audio
audio, sr = librosa.load(input_path, sr=44100, mono=True)

# === Initialize SVC model
svc_model = Svc(
    net_g_path=model_path,
    config_path=config_path,
    device="cpu"
)

# === Inference (speaker name must match the one in config.json)
converted_audio, sr_out = svc_model.infer(
    speaker="Obama",              # <- Check your config.json if it's different!
    audio=audio,
    transpose=0,
    cluster_infer_ratio=0,
    auto_predict_f0=True,
    f0_method="crepe",
    noise_scale=0.4
)

# === Save output
sf.write(output_path, converted_audio, sr_out)
print(f"✅ Done! Saved to {output_path}")
