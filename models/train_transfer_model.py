import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
from datetime import datetime
import os

# Paths
DATA_PATH = "../data/"
MODEL_PATH = "../models/saved/"
os.makedirs(MODEL_PATH, exist_ok=True)

# Load datasets
players = pd.read_csv(DATA_PATH + "players.csv")
transfers = pd.read_csv(DATA_PATH + "transfers.csv")
player_valuations = pd.read_csv(DATA_PATH + "player_valuations.csv")
appearances = pd.read_csv(DATA_PATH + "appearances.csv")

# Preprocessing: ensure proper datetime formats
players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
players["contract_expiration_date"] = pd.to_datetime(players["contract_expiration_date"], errors="coerce")
transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
player_valuations["date"] = pd.to_datetime(player_valuations["date"], errors="coerce")
appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")

# Build base training dataset
cutoff_date = pd.to_datetime("2022-07-01")

# Latest valuation before cutoff
latest_vals = player_valuations[player_valuations["date"] < cutoff_date]
latest_vals = latest_vals.sort_values("date").groupby("player_id").last().reset_index()

# Merge with player info
latest_vals = latest_vals.merge(players[["player_id", "date_of_birth", "position", "foot", "height_in_cm", "contract_expiration_date"]], on="player_id", how="left")

# Calculate age and contract remaining
latest_vals["age"] = latest_vals["date_of_birth"].apply(lambda x: cutoff_date.year - x.year if pd.notnull(x) else None)
latest_vals["contract_remaining"] = (latest_vals["contract_expiration_date"] - cutoff_date).dt.days / 365

# Aggregate performance stats (last 6 months)
recent_appearances = appearances[(appearances["date"] >= cutoff_date - pd.DateOffset(months=6)) & (appearances["date"] < cutoff_date)]
perf = recent_appearances.groupby("player_id").agg({
    "goals": "sum",
    "assists": "sum",
    "minutes_played": "sum",
    "appearance_id": "count"
}).rename(columns={"appearance_id": "matches"}).reset_index()

latest_vals = latest_vals.merge(perf, on="player_id", how="left")

# Fill missing performance data with zeros
latest_vals[["goals", "assists", "minutes_played", "matches"]] = latest_vals[["goals", "assists", "minutes_played", "matches"]].fillna(0)

# Transfer history features
transfer_history = transfers[transfers["transfer_date"] < cutoff_date]
history = transfer_history.groupby("player_id").agg({
    "transfer_fee": ["count", "mean"]
})
history.columns = ["num_transfers", "avg_transfer_fee"]
history = history.reset_index()

latest_vals = latest_vals.merge(history, on="player_id", how="left")

# Fill missing transfer history with zeros
latest_vals[["num_transfers", "avg_transfer_fee"]] = latest_vals[["num_transfers", "avg_transfer_fee"]].fillna(0)

# Target: was transferred after cutoff
recent_transfers = transfers[transfers["transfer_date"] >= cutoff_date]
latest_vals["was_transferred"] = latest_vals["player_id"].isin(recent_transfers["player_id"]).astype(int)

# Drop NaNs and keep relevant features
features = [
    "age", "market_value_in_eur", "contract_remaining",
    "goals", "assists", "minutes_played", "matches",
    "num_transfers", "avg_transfer_fee", "height_in_cm",
    "position"
]
data = latest_vals.dropna(subset=["age", "market_value_in_eur", "contract_remaining", "height_in_cm", "position"] + ["was_transferred"])
X = data[features]
y = data["was_transferred"]

# One-hot encode categorical features
categorical_features = ["position"]
numerical_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

X_processed = preprocessor.fit_transform(X)

# Balance classes with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Train model
model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and preprocessor
joblib.dump(model, MODEL_PATH + "xgb_transfer_model_transfer.pkl")
joblib.dump(preprocessor, MODEL_PATH + "preprocessor_transfer.pkl")
print("Model and preprocessor saved successfully.")