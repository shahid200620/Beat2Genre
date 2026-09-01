import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_train = pd.read_csv("output/X_train.csv")
X_test = pd.read_csv("output/X_test.csv")
y_train = pd.read_csv("output/y_train.csv")["label"]
y_test = pd.read_csv("output/y_test.csv")["label"]

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "model_name": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="macro"),
        "recall": recall_score(y_test, predictions, average="macro"),
        "f1_score": f1_score(y_test, predictions, average="macro")
    })

    trained_models[name] = model

results_df = pd.DataFrame(results)
results_df.to_csv("output/benchmark_results.csv", index=False)

plt.figure(figsize=(9, 5))
plt.bar(results_df["model_name"], results_df["f1_score"])
plt.title("Model Comparison Using Macro F1-Score")
plt.xlabel("Model")
plt.ylabel("Macro F1-Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("output/benchmark_comparison.png", dpi=300)
plt.close()

best_row = results_df.loc[results_df["f1_score"].idxmax()]
best_model_name = best_row["model_name"]

joblib.dump(trained_models[best_model_name], "output/best_model.joblib")

best_metrics = {
    "model_name": str(best_row["model_name"]),
    "accuracy": float(best_row["accuracy"]),
    "precision": float(best_row["precision"]),
    "recall": float(best_row["recall"]),
    "f1_score": float(best_row["f1_score"])
}

with open("output/best_model_metrics.json", "w") as file:
    json.dump(best_metrics, file, indent=4)

print(results_df.to_string(index=False))
print("\nBest model:", best_model_name)
print("Best macro F1-score:", round(best_metrics["f1_score"], 4))
print("\nTraining and benchmarking completed successfully")