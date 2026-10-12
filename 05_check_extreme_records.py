
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
FILE = PROJECT_DIR / "results" / "extreme_delay_records.csv"

print("Checking saved extreme-delay records...\n")

df = pd.read_csv(FILE)

print("Records in saved file:", len(df))

print("\n1. Largest delays:")
print(
    df.nlargest(15, "delay")[
        ["date", "station_name", "train_no", "delay"]
    ].to_string(index=False)
)

print("\n2. Extreme records grouped by date:")
print(
    df.groupby("date")
    .agg(
        records=("delay", "size"),
        maximum_delay=("delay", "max"),
        average_delay=("delay", "mean")
    )
    .sort_values("maximum_delay", ascending=False)
    .head(15)
    .to_string()
)

print("\n3. Trains appearing most often in these records:")
print(
    df["train_no"]
    .value_counts()
    .head(15)
    .to_string()
)

print("\nInvestigation complete.")
