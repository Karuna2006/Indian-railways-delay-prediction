from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

schedule = pd.read_csv(
    DATA_DIR / "combined_schedule.csv",
    usecols=["train_no", "station_no", "station_name"]
)

details = pd.read_csv(
    DATA_DIR / "train_details.csv"
)

check_trains = [1021, 7012, 11033, 11274, 11906]

for train in check_trains:
    print(f"\n{'=' * 50}")
    print(f"TRAIN NUMBER: {train}")

    train_schedule = schedule[
        pd.to_numeric(schedule["train_no"], errors="coerce") == train
    ]

    train_details = details[
        pd.to_numeric(details["train_no"], errors="coerce") == train
    ]

    if train_details.empty:
        print("Train details: Not found")
    else:
        print("\nTrain details:")
        print(train_details.to_string(index=False))

    if train_schedule.empty:
        print("\nSchedule: No entries found for this train")
    else:
        print("\nSchedule entries:")
        print(
            train_schedule.sort_values("station_no")
            .to_string(index=False)
        )

print("\nCheck complete.")

