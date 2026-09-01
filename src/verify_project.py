import json
import os
import joblib
import pandas as pd

required_files = [
    "output/benchmark_results.csv",
    "output/benchmark_comparison.png",
    "output/best_model_metrics.json",
    "output/best_model.joblib"
]

for file in required_files:
    if os.path.exists(file):
        print("PASS:", file)
    else:
        print("FAIL:", file)

benchmark = pd.read_csv("output/benchmark_results.csv")
required_columns = ["model_name", "accuracy", "precision", "recall", "f1_score"]

print("\nBenchmark columns:", list(benchmark.columns))
print("Benchmark models:", len(benchmark))

if list(benchmark.columns) == required_columns:
    print("PASS: Benchmark columns are correct")
else:
    print("FAIL: Benchmark columns are incorrect")

if len(benchmark) >= 3:
    print("PASS: At least three models are present")
else:
    print("FAIL: Fewer than three models are present")

if benchmark[required_columns[1:]].apply(lambda column: pd.api.types.is_numeric_dtype(column)).all():
    print("PASS: Benchmark metrics are numeric")
else:
    print("FAIL: Benchmark metrics are not numeric")

with open("output/best_model_metrics.json", "r") as file:
    metrics = json.load(file)

required_keys = ["model_name", "accuracy", "precision", "recall", "f1_score"]

print("\nJSON keys:", list(metrics.keys()))

if list(metrics.keys()) == required_keys:
    print("PASS: JSON keys are correct")
else:
    print("FAIL: JSON keys are incorrect")

if isinstance(metrics["model_name"], str):
    print("PASS: Model name is a string")
else:
    print("FAIL: Model name is not a string")

if all(isinstance(metrics[key], float) for key in required_keys[1:]):
    print("PASS: JSON metrics are floating-point values")
else:
    print("FAIL: JSON metrics are not floating-point values")

model = joblib.load("output/best_model.joblib")

print("\nSaved model:", type(model).__name__)
print("Expected features:", model.n_features_in_)

if model.n_features_in_ == 58:
    print("PASS: Model expects 58 features")
else:
    print("FAIL: Unexpected feature count")

print("\nProject verification completed")