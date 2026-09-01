import sys
import joblib
import pandas as pd

sys.path.insert(0, "src")

from audio_features import extract_features

model = joblib.load("output/best_model.joblib")
scaler = joblib.load("output/scaler.joblib")
label_encoder = joblib.load("output/label_encoder.joblib")

audio_path = sys.argv[1]

features = extract_features(audio_path)
X = pd.DataFrame([features])

X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

prediction = model.predict(X_scaled)[0]
probabilities = model.predict_proba(X_scaled)[0]

genres = label_encoder.inverse_transform(model.classes_)

print("Audio:", audio_path)
print("Predicted genre:", label_encoder.inverse_transform([prediction])[0])
print("Probabilities:")

for genre, probability in sorted(
    zip(genres, probabilities),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{genre}: {probability:.2%}")