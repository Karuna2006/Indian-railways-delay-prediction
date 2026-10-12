
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
FILE = DATA_DIR / "combined_delay.csv"

print("Checking suspicious dates...\n")

target_dates = ["2025-12-31", "2025-11-03", "2025-03-27"]
results = []

for chunk in pd.read_csv(
    FILE,
    usecols=["date", "station_name", "delay", "train_no"],
    chunksize=500_000
):
    selected = chunk[
        chunk["date"].isin(target_dates)
        & (chunk["delay"] > 1440)
    ]

    if not selected.empty:
        results.append(selected)

if results:
    df = pd.concat(results, ignore_index=True)

    print("Extreme delays by date:")
    print(
        df.groupby("date")
        .agg(
            records=("delay", "size"),
            unique_trains=("train_no", "nunique"),
            minimum_delay=("delay", "min"),
            maximum_delay=("delay", "max")
        )
        .to_string()
    )

    print("\nSample records:")
    print(df.head(20).to_string(index=False))
else:
    print("No matching extreme records found.")

print("\nCheck complete.")
