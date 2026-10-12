
from pathlib import Path
import pandas as pd

# The Python script is inside the data folder
DATA_DIR = Path(__file__).resolve().parent

print("Indian Railways Delay Prediction Project")
print("-" * 45)
print("Data folder:", DATA_DIR)

# Find CSV files
csv_files = list(DATA_DIR.glob("*.csv"))

print("\nCSV files found:")
for file in csv_files:
    size_mb = file.stat().st_size / (1024 * 1024)
    print(f"- {file.name}: {size_mb:.2f} MB")

# Preview the main delay dataset
delay_file = DATA_DIR / "combined_delay.csv"

if delay_file.exists():
    print("\nFirst five rows of the delay dataset:")

    sample = pd.read_csv(
        delay_file,
        nrows=5
    )

    print(sample.to_string(index=False))
    print("\nColumn names:")
    print(sample.columns.tolist())
else:
    print("\nERROR: combined_delay.csv was not found.")