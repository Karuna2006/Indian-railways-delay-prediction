
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

files = [
    "combined_schedule.csv",
    "train_details.csv",
    "station_full_names.csv"
]

for filename in files:
    filepath = DATA_DIR / filename

    print("\n" + "=" * 60)
    print("FILE:", filename)

    if not filepath.exists():
        print("File not found:", filepath)
        continue

    try:
        df = pd.read_csv(filepath, nrows=5)
        print("\nColumns:")
        print(df.columns.tolist())

        print("\nFirst 5 rows:")
        print(df.to_string(index=False))
    except Exception as error:
        print("Could not inspect file:", error)

print("\nFeature inspection complete.")
