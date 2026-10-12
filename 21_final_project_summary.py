
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS = PROJECT_DIR / "results"

metrics_file = RESULTS / "model_comparison_metrics.csv"
importance_file = RESULTS / "feature_importance.csv"
predictions_file = RESULTS / "model_comparison_predictions.csv"

print("=" * 60)
print("INDIAN RAILWAYS TRAIN DELAY PREDICTION")
print("FINAL PROJECT SUMMARY")
print("=" * 60)

# Model comparison
if metrics_file.exists():
    print("\n1. MODEL COMPARISON")
    print(pd.read_csv(metrics_file).to_string(index=False))
else:
    print("\nModel comparison metrics file not found.")

# Feature importance
if importance_file.exists():
    print("\n2. FEATURE IMPORTANCE")
    print(pd.read_csv(importance_file).to_string(index=False))
else:
    print("\nFeature importance file not found.")

# Prediction summary
if predictions_file.exists():
    df = pd.read_csv(predictions_file)
    print("\n3. PREDICTION DATA")
    print(f"Test records: {len(df):,}")
    print(f"Mean actual delay: {df['actual_delay'].mean():.2f} minutes")
    print(f"Median actual delay: {df['actual_delay'].median():.2f} minutes")
    print(f"Mean baseline prediction: {df['baseline_prediction'].mean():.2f}")
    print(f"Mean enhanced prediction: {df['enhanced_prediction'].mean():.2f}")
else:
    print("\nPrediction file not found.")

print("\n4. GRAPHS")
for filename in [
    "delay_distribution.png",
    "feature_importance.png",
    "actual_vs_predicted.png",
    "prediction_error_distribution.png"
]:
    status = "Available" if (RESULTS / filename).exists() else "Missing"
    print(f"{filename}: {status}")

print("\nSummary check complete.")
