from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

FILE = RESULTS_DIR / "model_comparison_predictions.csv"

df = pd.read_csv(FILE, parse_dates=["date"])

df["baseline_error"] = (
    df["actual_delay"] - df["baseline_prediction"]
)
df["enhanced_error"] = (
    df["actual_delay"] - df["enhanced_prediction"]
)

df["baseline_abs_error"] = df["baseline_error"].abs()
df["enhanced_abs_error"] = df["enhanced_error"].abs()

# Group records by actual delay.
df["delay_group"] = pd.cut(
    df["actual_delay"],
    bins=[-float("inf"), 0, 15, 30, 60, 120, 1440],
    labels=[
        "Early arrival",
        "0-15 minutes",
        "15-30 minutes",
        "30-60 minutes",
        "60-120 minutes",
        "120+ minutes"
    ],
    include_lowest=True
)

print("\n--- Error by Actual Delay Group ---")
summary = df.groupby(
    "delay_group", observed=False
).agg(
    records=("actual_delay", "size"),
    baseline_MAE=("baseline_abs_error", "mean"),
    enhanced_MAE=("enhanced_abs_error", "mean")
)

print(summary.round(2).to_string())

print("\n--- Overall Improvement ---")
print(
    "Baseline MAE:",
    round(df["baseline_abs_error"].mean(), 2)
)
print(
    "Enhanced MAE:",
    round(df["enhanced_abs_error"].mean(), 2)
)

improved = (
    df["enhanced_abs_error"] < df["baseline_abs_error"]
).sum()

worse = (
    df["enhanced_abs_error"] > df["baseline_abs_error"]
).sum()

equal = len(df) - improved - worse

print("Records where enhanced model improved:", f"{improved:,}")
print("Records where enhanced model worsened:", f"{worse:,}")
print("Records with equal absolute error:", f"{equal:,}")

print("\n--- Largest Enhanced-Model Errors ---")
df["enhanced_abs_error"] = df["enhanced_abs_error"].round(2)

print(
    df.nlargest(10, "enhanced_abs_error")[
        [
            "date",
            "train_no",
            "station_no",
            "station_name",
            "actual_delay",
            "enhanced_prediction",
            "enhanced_abs_error"
        ]
    ].to_string(index=False)
)

print("\nAnalysis complete.")