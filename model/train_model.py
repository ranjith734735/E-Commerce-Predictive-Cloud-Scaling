import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# E-COMMERCE WORKLOAD FORECASTING
# RANDOM FOREST REGRESSION
# ============================================================


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATA_FILE = (
    BASE_DIR
    / "data"
    / "utilization.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "model"
    / "ecommerce_cpu_model.pkl"
)

FEATURE_FILE = (
    BASE_DIR
    / "model"
    / "feature_columns.json"
)

METRICS_FILE = (
    BASE_DIR
    / "model"
    / "model_metrics.json"
)


# ------------------------------------------------------------
# Sales type mapping
# ------------------------------------------------------------

SALES_TYPE_CODES = {

    "Normal Sales": 0,

    "Weekend Sales": 1,

    "Summer Sale": 2,

    "Mega Sale": 3,

    "Flash Sale": 4,

    "Diwali Sale": 5,

    "Black Friday": 6,

    "Christmas Sale": 7,

    "New Year Sale": 8
}


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

print()
print("=" * 70)
print("E-COMMERCE ML MODEL TRAINING")
print("=" * 70)

print(
    f"Loading dataset:\n{DATA_FILE}"
)

if not DATA_FILE.exists():

    raise FileNotFoundError(
        "utilization.csv was not found.\n"
        "Run:\n"
        "python data\\generate_data.py"
    )


df = pd.read_csv(
    DATA_FILE
)


# ------------------------------------------------------------
# Convert timestamp
# ------------------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


# ------------------------------------------------------------
# Create time features
# ------------------------------------------------------------

df["hour"] = (
    df["timestamp"].dt.hour
)

df["day_of_week"] = (
    df["timestamp"].dt.dayofweek
)

df["day_of_month"] = (
    df["timestamp"].dt.day
)

df["month"] = (
    df["timestamp"].dt.month
)

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


# ------------------------------------------------------------
# Convert sales type to numeric value
# ------------------------------------------------------------

df["sales_type_code"] = (
    df["sales_type"]
    .map(SALES_TYPE_CODES)
    .fillna(0)
)


# ------------------------------------------------------------
# Target
# ------------------------------------------------------------
#
# We want the model to predict the CPU utilization
# during the next hour.
# ------------------------------------------------------------

df["future_cpu"] = (
    df["cpu_usage"].shift(-1)
)


# ------------------------------------------------------------
# Remove final row
# ------------------------------------------------------------

df = df.dropna(
    subset=["future_cpu"]
).reset_index(
    drop=True
)


# ------------------------------------------------------------
# Features
# ------------------------------------------------------------

FEATURES = [

    "hour",

    "day_of_week",

    "day_of_month",

    "month",

    "is_weekend",

    "active_users",

    "requests_per_min",

    "orders_per_min",

    "cpu_usage",

    "memory_usage",

    "network_usage",

    "sales_type_code",

    "event_active",

    "event_announced",

    "hours_until_event",

    "event_duration_hours",

    "event_intensity"
]


TARGET = "future_cpu"


# ------------------------------------------------------------
# Validate columns
# ------------------------------------------------------------

missing_columns = [

    column

    for column in FEATURES + [TARGET]

    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing required dataset columns:\n"
        + ", ".join(missing_columns)
    )


# ------------------------------------------------------------
# Prepare X and y
# ------------------------------------------------------------

X = df[FEATURES].copy()

y = df[TARGET].copy()


# ------------------------------------------------------------
# Replace invalid values
# ------------------------------------------------------------

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


y = y.replace(
    [np.inf, -np.inf],
    np.nan
)

valid_rows = (
    y.notna()
)

X = X.loc[
    valid_rows
].reset_index(
    drop=True
)

y = y.loc[
    valid_rows
].reset_index(
    drop=True
)


# ------------------------------------------------------------
# Chronological split
# ------------------------------------------------------------
#
# We do NOT randomly shuffle time-series data.
#
# First 80% -> training
# Last 20%  -> testing
# ------------------------------------------------------------

split_index = int(
    len(X) * 0.80
)


X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]


y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


# ------------------------------------------------------------
# Print dataset information
# ------------------------------------------------------------

print()

print(
    f"Total samples    : {len(X)}"
)

print(
    f"Training samples : {len(X_train)}"
)

print(
    f"Testing samples  : {len(X_test)}"
)

print(
    f"Features         : {len(FEATURES)}"
)


# ------------------------------------------------------------
# Create Random Forest
# ------------------------------------------------------------

model = RandomForestRegressor(

    n_estimators=300,

    max_depth=18,

    min_samples_split=4,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1
)


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

print()

print(
    "Training Random Forest..."
)

model.fit(
    X_train,
    y_train
)

print(
    "Training completed."
)


# ------------------------------------------------------------
# Test prediction
# ------------------------------------------------------------

y_pred = model.predict(
    X_test
)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------

joblib.dump(
    model,
    MODEL_FILE
)


# ------------------------------------------------------------
# Save features
# ------------------------------------------------------------

with open(
    FEATURE_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        FEATURES,
        file,
        indent=4
    )


# ------------------------------------------------------------
# Save metrics
# ------------------------------------------------------------

metrics = {

    "model":
        "Random Forest Regression",

    "target":
        "Next-hour CPU utilization",

    "mae":
        round(
            float(mae),
            4
        ),

    "rmse":
        round(
            float(rmse),
            4
        ),

    "r2_score":
        round(
            float(r2),
            4
        ),

    "training_samples":
        int(len(X_train)),

    "testing_samples":
        int(len(X_test)),

    "features":
        int(len(FEATURES))
}


with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------

importance = pd.DataFrame({

    "feature":
        FEATURES,

    "importance":
        model.feature_importances_

})

importance = importance.sort_values(
    "importance",
    ascending=False
)


# ------------------------------------------------------------
# Print model performance
# ------------------------------------------------------------

print()
print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"MAE       : {mae:.2f}%"
)

print(
    f"RMSE      : {rmse:.2f}%"
)

print(
    f"R² Score  : {r2:.4f}"
)

print()
print("Top Feature Importance:")

print(
    importance
    .head(10)
    .to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Print saved files
# ------------------------------------------------------------

print()
print("=" * 70)
print("MODEL FILES SAVED")
print("=" * 70)

print(
    f"Model    : {MODEL_FILE}"
)

print(
    f"Features : {FEATURE_FILE}"
)

print(
    f"Metrics  : {METRICS_FILE}"
)

print("=" * 70)
print()