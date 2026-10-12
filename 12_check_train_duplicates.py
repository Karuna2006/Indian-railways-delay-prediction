
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

details = pd.read_csv(DATA_DIR / "train_details.csv")
schedule = pd.read_csv(DATA_DIR / "combined_schedule.csv")

print("--- Train detail duplicate analysis ---")

counts = details.groupby("train_no").agg(
    rows=("train_name", "size"),
    unique_names=("train_name", "nunique"),
    unique_types=("type_code", "nunique")
)

duplicate_numbers = counts[counts["rows"] > 1]

print("Train-detail rows:", len(details))
print("Unique train numbers:", details["train_no"].nunique())
print("Train numbers with multiple rows:", len(duplicate_numbers))
print(
    "Duplicate train numbers with conflicting types:",
    int((duplicate_numbers["unique_types"] > 1).sum())
)

print("\nExamples of duplicate train entries:")
examples = details[
    details["train_no"].isin(duplicate_numbers.head(10).index)
]
print(examples.to_string(index=False))

print("\n--- Schedule key check ---")
print("Schedule rows:", len(schedule))
print("Unique train/station pairs:",
      len(schedule.drop_duplicates(["train_no", "station_no"])))

print("\nCheck complete.")
