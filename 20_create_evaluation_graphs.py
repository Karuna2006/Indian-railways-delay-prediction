
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
FILE = RESULTS_DIR / "model_comparison_predictions.csv"

print("Loading prediction results...", flush=True)

df = pd.read_csv(FILE)

print("Columns found:", df.columns.tolist())

# Use the enhanced model's predictions.
actual = df["actual_delay"]
predicted = df["enhanced_prediction"]
errors = actual - predicted

# Graph 1: Actual versus predicted delays.
plt.figure(figsize=(8, 6))
plt.scatter(actual, predicted, alpha=0.15, s=8)
limit = max(actual.max(), predicted.max())
plt.plot([0, limit], [0, limit], linestyle="--")
plt.xlabel("Actual Delay (minutes)")
plt.ylabel("Predicted Delay (minutes)")
plt.title("Actual vs Predicted Train Delays")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=300)
plt.close()

# Graph 2: Distribution of prediction errors.
plt.figure(figsize=(8, 6))
plt.hist(errors, bins=60)
plt.axvline(0, linestyle="--")
plt.xlabel("Prediction Error (Actual - Predicted, minutes)")
plt.ylabel("Number of Records")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "prediction_error_distribution.png", dpi=300)
plt.close()

print("\nGraphs saved successfully:")
print(RESULTS_DIR / "actual_vs_predicted.png")
print(RESULTS_DIR / "prediction_error_distribution.png")
