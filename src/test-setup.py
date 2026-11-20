import librosa, torch, pretty_midi, music21
print("librosa", librosa.__version__)
print("torch", torch.__version__)
print("GPU available:", torch.cuda.is_available())