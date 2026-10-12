
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "results" / "cleaned_delay_data.csv"

print("Loading prepared data...")

df = pd.read_csv(DATA_FILE, parse_dates=["date"])

print("Records loaded:", f"{len(df):,}")

# Use a manageable random sample for initial experimentation.
# Sampling is for computational convenience; splitting remains chronological.
if len(df) > 300_000:
    df = df.sample(n=300_000, random_state=42)

df = df.sort_values("date").reset_index(drop=True)

# Use simple features available in the dataset.
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["day_of_month"] = df["date"].dt.day

features = [
    "train_no",
    "station_no",
    "day_of_week",
    "month",
    "day_of_month"
]
target = "delay"

df = df.dropna(subset=features + [target])

# Chronological split: earlier records for training, later records for testing.
split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

print("Training records:", f"{len(train):,}")
print("Testing records:", f"{len(test):,}")
print("Training Random Forest model...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n--- Model Evaluation ---")
print(f"MAE:  {mae:.2f} minutes")
print(f"RMSE: {rmse:.2f} minutes")
print(f"R²:   {r2:.4f}")

results = test[["date", "train_no", "station_no", "delay"]].copy()
results["predicted_delay"] = predictions

output_file = PROJECT_DIR / "results" / "model_predictions.csv"
results.to_csv(output_file, index=False)

print("\nPredictions saved to:", output_file)
