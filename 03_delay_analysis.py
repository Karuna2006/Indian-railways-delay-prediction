
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent
DELAY_FILE = DATA_DIR / "combined_delay.csv"

chunk_size = 500_000
sample_parts = []

total_valid = 0
over_24_hours = 0
over_7_days = 0
over_30_days = 0
below_minus_120 = 0
zero_delays = 0
negative_delays = 0

print("Analysing delay values in chunks...")

for chunk in pd.read_csv(DELAY_FILE, chunksize=chunk_size):
    delays = chunk["delay"].dropna()

    total_valid += len(delays)
    over_24_hours += (delays > 1440).sum()
    over_7_days += (delays > 10080).sum()
    over_30_days += (delays > 43200).sum()
    below_minus_120 += (delays < -120).sum()
    zero_delays += (delays == 0).sum()
    negative_delays += (delays < 0).sum()

    # Keep a small sample for quantiles and a graph
    if len(delays) > 0:
        sample_parts.append(
            delays.sample(
                n=min(1000, len(delays)),
                random_state=42
            )
        )

sample = pd.concat(sample_parts, ignore_index=True)

print("\nCOUNTS OF EXTREME OR SPECIAL VALUES")
print("-" * 40)
print(f"Valid delays: {total_valid:,}")
print(f"Zero delays: {zero_delays:,}")
print(f"Negative delays: {negative_delays:,}")
print(f"Delays over 24 hours: {over_24_hours:,}")
print(f"Delays over 7 days: {over_7_days:,}")
print(f"Delays over 30 days: {over_30_days:,}")
print(f"Delays below -120 minutes: {below_minus_120:,}")

print("\nSAMPLE DELAY DISTRIBUTION")
print("-" * 40)
print(sample.describe(
    percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
))

# Plot the sample, limiting the x-axis to make
# the normal-range distribution easier to inspect.
normal_sample = sample[
    (sample >= -120) & (sample <= 720)
]

plt.figure(figsize=(10, 5))
plt.hist(normal_sample, bins=60, edgecolor="black")
plt.title("Sample Distribution of Train Delays")
plt.xlabel("Delay in minutes")
plt.ylabel("Frequency in sample")
plt.tight_layout()

output_file = DATA_DIR.parent / "results" / "delay_distribution.png"
output_file.parent.mkdir(exist_ok=True)
plt.savefig(output_file, dpi=150)
plt.show()

print("\nGraph saved to:", output_file)