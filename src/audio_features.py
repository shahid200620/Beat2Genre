import numpy as np
import librosa

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=30)
    length = len(y)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

    harmony, perceptr = librosa.effects.hpss(y)

    tempo = librosa.feature.tempo(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features = {
        "length": length,
        "chroma_stft_mean": np.mean(chroma),
        "chroma_stft_var": np.var(chroma),
        "rms_mean": np.mean(rms),
        "rms_var": np.var(rms),
        "spectral_centroid_mean": np.mean(spectral_centroid),
        "spectral_centroid_var": np.var(spectral_centroid),
        "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
        "spectral_bandwidth_var": np.var(spectral_bandwidth),
        "rolloff_mean": np.mean(rolloff),
        "rolloff_var": np.var(rolloff),
        "zero_crossing_rate_mean": np.mean(zero_crossing_rate),
        "zero_crossing_rate_var": np.var(zero_crossing_rate),
        "harmony_mean": np.mean(harmony),
        "harmony_var": np.var(harmony),
        "perceptr_mean": np.mean(perceptr),
        "perceptr_var": np.var(perceptr),
        "tempo": np.mean(tempo)
    }

    for i in range(20):
        features[f"mfcc{i + 1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc{i + 1}_var"] = np.var(mfcc[i])

    return features