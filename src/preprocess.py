import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("data/features_30_sec.csv")

X = df.drop(columns=["filename", "label"])
y = df["label"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "output/scaler.joblib")
joblib.dump(label_encoder, "output/label_encoder.joblib")

pd.DataFrame(X_train, columns=X.columns).to_csv("output/X_train.csv", index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv("output/X_test.csv", index=False)
pd.DataFrame(y_train, columns=["label"]).to_csv("output/y_train.csv", index=False)
pd.DataFrame(y_test, columns=["label"]).to_csv("output/y_test.csv", index=False)

print("Preprocessing completed successfully")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Features:", X_train.shape[1])
print("Classes:", len(label_encoder.classes_))
print("Genres:", list(label_encoder.classes_))