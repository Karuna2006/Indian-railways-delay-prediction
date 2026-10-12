from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

DATA_FILE = RESULTS_DIR / "cleaned_delay_data.csv"
SCHEDULE_FILE = DATA_DIR / "combined_schedule.csv"

CUTOFF_DATE = pd.Timestamp("2025-12-01")
TRAIN_SAMPLE_SIZE = 240_000
TEST_SAMPLE_SIZE = 60_000

print("Loading cleaned delay data...")
df = pd.read_csv(DATA_FILE, parse_dates=["date"])

print("Total records:", f"{len(df):,}")

# Convert join keys to consistent formats.
df["train_no"] = pd.to_numeric(df["train_no"], errors="coerce")
df["station_no"] = pd.to_numeric(df["station_no"], errors="coerce")
df["station_name"] = (
    df["station_name"].astype("string").str.strip().str.upper()
)

df = df.dropna(
    subset=["date", "train_no", "station_no", "delay"]
)

# Split by date BEFORE sampling to keep the test period in the future.
earlier = df[df["date"] < CUTOFF_DATE]
future = df[df["date"] >= CUTOFF_DATE]

if earlier.empty or future.empty:
    raise ValueError("Training or test period is empty. Check the dates.")

train = earlier.sample(
    n=min(TRAIN_SAMPLE_SIZE, len(earlier)),
    random_state=42
).copy()

test = future.sample(
    n=min(TEST_SAMPLE_SIZE, len(future)),
    random_state=42
).copy()

print("Training date range:",
      train["date"].min().date(), "to", train["date"].max().date())
print("Testing date range:",
      test["date"].min().date(), "to", test["date"].max().date())
print("Training sample:", f"{len(train):,}")
print("Testing sample:", f"{len(test):,}")

# Calendar features available at prediction time.
for part in [train, test]:
    part["day_of_week"] = part["date"].dt.dayofweek
    part["month"] = part["date"].dt.month
    part["day_of_month"] = part["date"].dt.day

baseline_features = [
    "train_no",
    "station_no",
    "day_of_week",
    "month",
    "day_of_month"
]

# Read schedule features.
print("\nLoading schedule features...")
schedule = pd.read_csv(SCHEDULE_FILE)

schedule["train_no"] = pd.to_numeric(
    schedule["train_no"], errors="coerce"
)
schedule["station_name"] = (
    schedule["station_name"].astype("string")
    .str.strip().str.upper()
)

schedule["distance_from_origin"] = pd.to_numeric(
    schedule["distance_from_origin"], errors="coerce"
)
schedule["arrival_day"] = pd.to_numeric(
    schedule["arrival_day"], errors="coerce"
)

# Convert scheduled arrival time to minutes after midnight.
arrival_time = schedule["arrival_time"].astype("string").str.strip()
parsed_time = pd.to_datetime(
    arrival_time, format="%H:%M", errors="coerce"
)

# Try HH:MM:SS for any values not parsed by HH:MM.
missing_time = parsed_time.isna()
parsed_time.loc[missing_time] = pd.to_datetime(
    arrival_time.loc[missing_time],
    format="%H:%M:%S",
    errors="coerce"
)

schedule["scheduled_arrival_minutes"] = (
    parsed_time.dt.hour * 60 + parsed_time.dt.minute
)

schedule_keys = ["train_no", "station_name"]

# Keep only schedule keys that identify exactly one row.
schedule = schedule.dropna(subset=schedule_keys)
key_counts = schedule.groupby(schedule_keys).size()
unique_keys = key_counts[key_counts == 1].index

schedule_index = pd.MultiIndex.from_frame(schedule[schedule_keys])
schedule = schedule.loc[schedule_index.isin(unique_keys)].copy()

schedule_features = [
    "distance_from_origin",
    "arrival_day",
    "scheduled_arrival_minutes"
]

schedule = schedule[schedule_keys + schedule_features]

# Attach schedule information to both samples.
train = train.merge(
    schedule,
    on=schedule_keys,
    how="left",
    validate="many_to_one"
)
test = test.merge(
    schedule,
    on=schedule_keys,
    how="left",
    validate="many_to_one"
)

matched_test = train_count = 0
if len(test):
    matched_test = int(test["distance_from_origin"].notna().sum())
if len(train):
    train_count = int(train["distance_from_origin"].notna().sum())

print("\nSchedule feature coverage:")
print(f"Training records with distance: {train_count:,} / {len(train):,}")
print(f"Test records with distance: {matched_test:,} / {len(test):,}")

enhanced_features = baseline_features + schedule_features

# Train and evaluate both models on the SAME test records.
X_train_base = train[baseline_features]
X_test_base = test[baseline_features]

X_train_enhanced = train[enhanced_features]
X_test_enhanced = test[enhanced_features]

y_train = train["delay"]
y_test = test["delay"]

def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("forest", RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ))
    ])

def evaluate(name, X_train, X_test):
    print(f"\nTraining {name}...")
    model = make_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"{name} MAE:  {mae:.2f} minutes")
    print(f"{name} RMSE: {rmse:.2f} minutes")
    print(f"{name} R²:   {r2:.4f}")

    return predictions, mae, rmse, r2

base_pred, base_mae, base_rmse, base_r2 = evaluate(
    "Baseline model", X_train_base, X_test_base
)

enh_pred, enh_mae, enh_rmse, enh_r2 = evaluate(
    "Schedule-enhanced model", X_train_enhanced, X_test_enhanced
)

print("\n--- Model Comparison ---")
print(f"{'Metric':<12}{'Baseline':>15}{'Enhanced':>15}")
print(f"{'MAE':<12}{base_mae:>15.2f}{enh_mae:>15.2f}")
print(f"{'RMSE':<12}{base_rmse:>15.2f}{enh_rmse:>15.2f}")
print(f"{'R²':<12}{base_r2:>15.4f}{enh_r2:>15.4f}")

# Save predictions for later analysis and plotting.
comparison = test[
    ["date", "train_no", "station_no", "station_name", "delay"]
].copy()

comparison = comparison.rename(columns={"delay": "actual_delay"})
comparison["baseline_prediction"] = base_pred
comparison["enhanced_prediction"] = enh_pred

output_file = RESULTS_DIR / "model_comparison_predictions.csv"
comparison.to_csv(output_file, index=False)

metrics_file = RESULTS_DIR / "model_comparison_metrics.csv"
pd.DataFrame([
    {"model": "Baseline", "MAE": base_mae, "RMSE": base_rmse, "R2": base_r2},
    {"model": "Schedule-enhanced", "MAE": enh_mae, "RMSE": enh_rmse, "R2": enh_r2}
]).to_csv(metrics_file, index=False)

print("\nPredictions saved to:", output_file)
print("Metrics saved to:", metrics_file)
print("Comparison complete.")