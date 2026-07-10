import librosa

audio, sr = librosa.load("dataset/sample.wav", sr=16000)

print("Sampling Rate:", sr)
print("Number of Samples:", len(audio))