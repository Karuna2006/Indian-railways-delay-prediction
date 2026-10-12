from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

delay_file = RESULTS_DIR / "cleaned_delay_data.csv"
schedule_file = DATA_DIR / "combined_schedule.csv"

print("Loading schedule data...")

schedule = pd.read_csv(schedule_file)

# Keep only the columns needed for checking matches.
schedule = schedule[["train_no", "station_no"]].drop_duplicates()

# Normalize the join keys.
schedule["train_no"] = pd.to_numeric(
    schedule["train_no"], errors="coerce"
)
schedule["station_no"] = pd.to_numeric(
    schedule["station_no"], errors="coerce"
)

schedule = schedule.dropna(
    subset=["train_no", "station_no"]
)

schedule_keys = pd.MultiIndex.from_frame(
    schedule[["train_no", "station_no"]]
)

matched = 0
total = 0

print("Checking cleaned delay records in chunks...")

for chunk in pd.read_csv(
    delay_file,
    chunksize=500_000
):
    chunk["train_no"] = pd.to_numeric(
        chunk["train_no"], errors="coerce"
    )
    chunk["station_no"] = pd.to_numeric(
        chunk["station_no"], errors="coerce"
    )

    valid = chunk.dropna(
        subset=["train_no", "station_no"]
    )

    keys = pd.MultiIndex.from_frame(
        valid[["train_no", "station_no"]]
    )

    matched += int(keys.isin(schedule_keys).sum())
    total += len(valid)

    print(f"Checked {total:,} records...")

print("\n--- Schedule Match Results ---")
print(f"Records checked: {total:,}")
print(f"Records matching schedule: {matched:,}")

if total:
    print(f"Match percentage: {matched / total * 100:.2f}%")
    print(f"Unmatched records: {total - matched:,}")
    print(
        f"Unmatched percentage: "
        f"{(total - matched) / total * 100:.2f}%"
    )

