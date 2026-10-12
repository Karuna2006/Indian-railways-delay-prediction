from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

delay_file = RESULTS_DIR / "cleaned_delay_data.csv"
schedule_file = DATA_DIR / "combined_schedule.csv"

print("Loading schedule station codes...")

schedule = pd.read_csv(
    schedule_file,
    usecols=["train_no", "station_name"]
)

for col in ["train_no", "station_name"]:
    schedule[col] = schedule[col].astype(str).str.strip().str.upper()

schedule = schedule.drop_duplicates(
    subset=["train_no", "station_name"]
)

schedule_keys = pd.MultiIndex.from_frame(
    schedule[["train_no", "station_name"]]
)

matched = 0
total = 0

print("Checking delay records in chunks...")

for chunk in pd.read_csv(
    delay_file,
    chunksize=500_000
):
    for col in ["train_no", "station_name"]:
        chunk[col] = chunk[col].astype(str).str.strip().str.upper()

    valid = chunk.dropna(
        subset=["train_no", "station_name"]
    )

    keys = pd.MultiIndex.from_frame(
        valid[["train_no", "station_name"]]
    )

    matched += int(keys.isin(schedule_keys).sum())
    total += len(valid)

    print(f"Checked {total:,} records...")

print("\n--- Station Code Match Results ---")
print(f"Records checked: {total:,}")
print(f"Records matching by train and station code: {matched:,}")

if total:
    print(f"Match percentage: {matched / total * 100:.2f}%")
    print(f"Unmatched records: {total - matched:,}")

print("\nCheck complete.")