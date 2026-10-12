
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = Path(__file__).resolve().parent
FILE = PROJECT_DIR / "results" / "model_predictions.csv"

df = pd.read_csv(FILE, parse_dates=["date"])

actual = df["delay"]
predicted = df["predicted_delay"]
errors = actual - predicted

print("--- Prediction Error Analysis ---")
print("Test records:", f"{len(df):,}")
print("Actual mean delay:", round(actual.mean(), 2), "minutes")
print("Actual median delay:", round(actual.median(), 2), "minutes")
print("Predicted mean delay:", round(predicted.mean(), 2), "minutes")
print("MAE:", round(mean_absolute_error(actual, predicted), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(actual, predicted)), 2))
print("R²:", round(r2_score(actual, predicted), 4))

print("\nAbsolute error distribution:")
print(
    errors.abs().describe(
        percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
    ).round(2).to_string()
)

print("\n10 predictions with the largest errors:")
df["absolute_error"] = errors.abs()
print(
    df.nlargest(10, "absolute_error")[
        ["date", "train_no", "station_no", "delay",
         "predicted_delay", "absolute_error"]
    ].to_string(index=False)
)

print("\nEvaluation complete.")
