import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from audio_features import extract_features

csv_data = pd.read_csv("data/features_30_sec.csv")
audio_name = "blues.00024.wav"

csv_row = csv_data[csv_data["filename"] == audio_name].iloc[0]
audio_path = r"D:\GTZAN\genres_original\blues\blues.00024.wav"

audio_features = extract_features(audio_path)

print("Feature comparison for:", audio_name)
print()

differences = []

for feature in audio_features:
    original = float(csv_row[feature])
    extracted = float(audio_features[feature])
    difference = abs(original - extracted)

    differences.append(difference)

    print(
        f"{feature}: "
        f"CSV={original:.6f} "
        f"Audio={extracted:.6f} "
        f"Difference={difference:.6f}"
    )

print()
print("Mean absolute difference:", np.mean(differences))
print("Maximum absolute difference:", np.max(differences))