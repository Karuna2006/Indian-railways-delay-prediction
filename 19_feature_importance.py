
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "results" / "cleaned_delay_data.csv"
SCHEDULE_FILE = PROJECT_DIR / "data" / "combined_schedule.csv"
RESULTS_DIR = PROJECT_DIR / "results"

print("Loading data...")

# Use a reproducible sample to avoid loading the entire dataset into memory.
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "delay"])

# Use the same chronological split as the model comparison.
train = df[df["date"] < "2025-12-01"].sample(
    n=240000, random_state=42
).copy()

schedule = pd.read_csv(SCHEDULE_FILE)
schedule["train_no"] = schedule["train_no"].astype(str).str.strip()
schedule["station_name"] = (
    schedule["station_name"].astype(str).str.strip().str.upper()
)
train["train_no"] = train["train_no"].astype(str).str.strip()
train["station_name"] = (
    train["station_name"].astype(str).str.strip().str.upper()
)

# Avoid duplicate schedule keys.
schedule = schedule.drop_duplicates(
    subset=["train_no", "station_name"], keep=False
)

schedule["arrival_time"] = schedule["arrival_time"].astype(str)
time_parts = schedule["arrival_time"].str.extract(
    r"(\d{1,2}):(\d{2})"
)
schedule["scheduled_arrival_minutes"] = (
    pd.to_numeric(time_parts[0], errors="coerce") * 60
    + pd.to_numeric(time_parts[1], errors="coerce")
)

schedule_features = schedule[
    ["train_no", "station_name", "distance_from_origin",
     "arrival_day", "scheduled_arrival_minutes"]
]

train = train.merge(
    schedule_features,
    on=["train_no", "station_name"],
    how="left"
)

train["day_of_week"] = train["date"].dt.dayofweek
train["month"] = train["date"].dt.month
train["day_of_month"] = train["date"].dt.day

features = [
    "train_no", "station_no", "day_of_week", "month",
    "day_of_month", "distance_from_origin", "arrival_day",
    "scheduled_arrival_minutes"
]

X = train[features].apply(pd.to_numeric, errors="coerce")
y = train["delay"]

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=features)

print("Training model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n--- Feature Importance ---")
print(importance.to_string(index=False))

importance.to_csv(
    RESULTS_DIR / "feature_importance.csv", index=False
)

plt.figure(figsize=(10, 6))
plt.barh(importance["feature"], importance["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title("Feature Importance for Train Delay Prediction")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=300)
plt.show()

print("\nSaved feature_importance.csv and feature_importance.png")
