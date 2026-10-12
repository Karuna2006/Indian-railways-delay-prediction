
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
DELAY_FILE = DATA_DIR / "combined_delay.csv"

print("INDIAN RAILWAYS - DATA QUALITY CHECK")
print("=" * 45)

# Read the file in chunks to control memory usage
chunk_size = 500_000

total_rows = 0
missing_delays = 0
valid_delays = 0
delay_min = float("inf")
delay_max = float("-inf")
delay_sum = 0.0
delay_count = 0

print("Reading the dataset in chunks...")

for chunk in pd.read_csv(
    DELAY_FILE,
    chunksize=chunk_size
):
    total_rows += len(chunk)

    # Count missing and non-missing delay values
    missing_delays += chunk["delay"].isna().sum()

    delays = chunk["delay"].dropna()

    valid_delays += len(delays)

    if not delays.empty:
        delay_min = min(delay_min, delays.min())
        delay_max = max(delay_max, delays.max())
        delay_sum += delays.sum()
        delay_count += len(delays)

print("\nDATA QUALITY SUMMARY")
print("-" * 30)
print(f"Total records: {total_rows:,}")
print(f"Missing delays: {missing_delays:,}")
print(f"Non-missing delays: {valid_delays:,}")

if total_rows > 0:
    print(
        f"Missing delay percentage: "
        f"{missing_delays / total_rows * 100:.2f}%"
    )

if delay_count > 0:
    print(f"Minimum delay: {delay_min}")
    print(f"Maximum delay: {delay_max}")
    print(f"Mean delay: {delay_sum / delay_count:.2f} minutes")

print("\nData quality check completed.")