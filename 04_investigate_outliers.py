
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
DELAY_FILE = DATA_DIR / "combined_delay.csv"

total_rows = 0
extreme_count = 0
top_records = []

print("Starting outlier investigation...", flush=True)

for chunk in pd.read_csv(
    DELAY_FILE,
    usecols=["date", "station_no", "station_name", "delay", "train_no"],
    chunksize=500_000
):
    total_rows += len(chunk)

    extreme = chunk[chunk["delay"] > 1440]
    extreme_count += len(extreme)

    if not extreme.empty:
        top_records.append(extreme.nlargest(20, "delay"))

    print(
        f"Processed {total_rows:,} rows | "
        f"Extreme delays found: {extreme_count:,}",
        flush=True
    )

print("\nTotal records processed:", f"{total_rows:,}")
print("Delays over 24 hours:", f"{extreme_count:,}")

if top_records:
    top = pd.concat(top_records, ignore_index=True)
    top = top.nlargest(100, "delay")

    output_file = DATA_DIR.parent / "results" / "extreme_delay_records.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(output_file, index=False)

    print("\nLargest delays:")
    print(top.head(20).to_string(index=False))
    print("\nSaved results to:", output_file)
else:
    print("No delays over 24 hours found.")