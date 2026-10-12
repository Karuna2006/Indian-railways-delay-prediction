
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

schedule_file = DATA_DIR / "combined_schedule.csv"
details_file = DATA_DIR / "train_details.csv"

schedule = pd.read_csv(schedule_file)
details = pd.read_csv(details_file)

print("--- Schedule information ---")
print("Shape:", schedule.shape)
print("Actual columns:")
for col in schedule.columns:
    print(repr(col))

print("\nMissing values:")
print(schedule.isna().sum().to_string())

print("\nDuplicate train/station combinations:")
print(
    schedule.duplicated(["train_no", "station_no"]).sum()
)

print("\n--- Train details information ---")
print("Shape:", details.shape)
print("Duplicate train numbers:", details["train_no"].duplicated().sum())

print("\nTrain types:")
print(details["type_code"].value_counts(dropna=False).head(20))

print("\nSchedule inspection complete.")
