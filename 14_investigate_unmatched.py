from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

delay_file = RESULTS_DIR / "cleaned_delay_data.csv"
schedule_file = DATA_DIR / "combined_schedule.csv"

schedule = pd.read_csv(
    schedule_file,
    usecols=["train_no", "station_no"]
).drop_duplicates()

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

unmatched_samples = []
total_unmatched = 0

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
    ).copy()

    keys = pd.MultiIndex.from_frame(
        valid[["train_no", "station_no"]]
    )

    unmatched = valid.loc[~keys.isin(schedule_keys)]

    total_unmatched += len(unmatched)

    if len(unmatched) > 0 and len(unmatched_samples) < 5:
        unmatched_samples.append(
            unmatched.head(5)[
                ["date", "train_no", "station_no", "station_name", "delay"]
            ]
        )

print("Total unmatched records:", f"{total_unmatched:,}")
print("\nSample unmatched records:")

if unmatched_samples:
    print(
        pd.concat(unmatched_samples, ignore_index=True)
        .to_string(index=False)
    )
else:
    print("No unmatched records found.")

print("\nInvestigation complete.")

