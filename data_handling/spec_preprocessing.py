import librosa
import numpy as np
from PIL import Image

def classifier_representation(y, window, step, sr, n_mels, fmin=0, fmax=12000, ref=np.max, top_db=80, mode='img'):
    n_fft = int(window * sr)  # Window size
    hop_length = int(step * sr)  # Step size

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, window="hann", n_mels=n_mels, fmin=fmin, fmax=fmax)
    spec = librosa.power_to_db(S, ref=ref, top_db=80.0)

    if mode == 'img':
        bytedata = (((spec + top_db) * 255 / top_db).clip(0, 255) + 0.5).astype(np.uint8)
        spec = Image.fromarray(bytedata)

    return spec
