
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
FILE = DATA_DIR / "combined_delay.csv"

PROJECT_DIR = DATA_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "cleaned_delay_data.csv"

total_rows = 0
missing_delay = 0
extreme_delay = 0
kept_rows = 0

print("Preparing the modelling dataset...", flush=True)

for i, chunk in enumerate(
    pd.read_csv(FILE, chunksize=500_000),
    start=1
):
    total_rows += len(chunk)

    missing_delay += int(chunk["delay"].isna().sum())

    valid = chunk.dropna(subset=["delay"]).copy()
    extreme_delay += int((valid["delay"] > 1440).sum())

    # Provisional rule for the first model
    valid = valid[valid["delay"] <= 1440]

    # Keep negative delays; do not assume they are errors.
    valid["date"] = pd.to_datetime(valid["date"], errors="coerce")
    valid = valid.dropna(subset=["date"])

    valid.to_csv(
        OUTPUT_FILE,
        mode="w" if i == 1 else "a",
        header=(i == 1),
        index=False
    )

    kept_rows += len(valid)

    print(
        f"Processed {total_rows:,} rows | "
        f"Kept {kept_rows:,} rows",
        flush=True
    )

print("\n--- Preparation Summary ---")
print("Total original records:", f"{total_rows:,}")
print("Missing delays:", f"{missing_delay:,}")
print("Delays above 1440 minutes:", f"{extreme_delay:,}")
print("Records retained:", f"{kept_rows:,}")
print("Saved to:", OUTPUT_FILE)
