import librosa
import numpy as np
import joblib
from spafe.features.rplp import plp

# ==========================================================
# LOAD TRAINED MODEL AND SCALER
# ==========================================================
model = joblib.load("svm_speaker_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================================================
# LOAD AUDIO FILE
# ==========================================================
audio_path = "test.wav"      # Change this to your test audio

audio, sr = librosa.load(audio_path, sr=16000, res_type="kaiser_fast")
# Add small noise like in training (for stability)
audio = audio + np.random.normal(0, 1e-6, size=audio.shape)

print("Audio Loaded Successfully")
print("Sampling Rate:", sr)

# ==========================================================
# MFCC
# ==========================================================
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=13
)

mfcc_mean = np.mean(mfcc, axis=1)

# ==========================================================
# LPC
# ==========================================================
frame_length = 2048
hop_length = 1024
lpc_order = 12

lpc_features = []

for i in range(0, len(audio) - frame_length, hop_length):

    frame = audio[i:i + frame_length]

    try:
        coeffs = librosa.lpc(frame, order=lpc_order)
        lpc_features.append(coeffs)
    except Exception:
        continue

if len(lpc_features) == 0:
    lpc_mean = np.zeros(lpc_order + 1)
else:
    lpc_features = np.array(lpc_features)
    lpc_mean = np.mean(lpc_features, axis=0)

# ==========================================================
# PLP
# ==========================================================
try:
    plp_features = plp(audio, fs=sr, order=13)
    plp_mean = np.mean(plp_features, axis=0)
except Exception as e:
    print("PLP extraction failed:", e)
    plp_mean = np.zeros(13)

# ==========================================================
# COMBINE FEATURES
# ==========================================================
combined = np.concatenate([
    mfcc_mean,
    lpc_mean,
    plp_mean
])

combined = combined.reshape(1, -1)

# ==========================================================
# SCALE FEATURES
# ==========================================================
combined = scaler.transform(combined)

# ==========================================================
# PREDICT
# ==========================================================
prediction = model.predict(combined)

# Extract numeric speaker ID from prediction
speaker_str = str(prediction[0]).replace('Speaker', '').replace('_', '')
speaker_id = int(speaker_str)

print("\n===================================")
print(f"Speaker {speaker_id}")
print("===================================")