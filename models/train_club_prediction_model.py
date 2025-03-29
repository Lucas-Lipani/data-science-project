import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, top_k_accuracy_score
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
clubs = pd.read_csv(DATA_PATH + "clubs.csv")

# Preprocessing: ensure datetime format
players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
players["contract_expiration_date"] = pd.to_datetime(players["contract_expiration_date"], errors="coerce")
transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
player_valuations["date"] = pd.to_datetime(player_valuations["date"], errors="coerce")
appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")

# Cutoff date to simulate prediction before transfer
cutoff_date = pd.to_datetime("2022-07-01")

# Filter transfers after cutoff for target
future_transfers = transfers[transfers["transfer_date"] >= cutoff_date].copy()
future_transfers = future_transfers.dropna(subset=["to_club_id"])

# Get latest valuation before cutoff
latest_vals = player_valuations[player_valuations["date"] < cutoff_date]
latest_vals = latest_vals.sort_values("date").groupby("player_id").last().reset_index()

# Merge with player info and transfer destination
cols_to_merge = ["player_id", "date_of_birth", "position", "sub_position", "foot", "height_in_cm", "contract_expiration_date", "country_of_citizenship", "highest_market_value_in_eur", "current_club_id"]
for col in cols_to_merge:
    if col not in players.columns:
        players[col] = np.nan

latest_vals = latest_vals.merge(players[["player_id"] + cols_to_merge[1:]], on="player_id", how="left")
print("Após merge com players:", latest_vals.shape)
latest_vals = latest_vals.merge(future_transfers[["player_id", "to_club_id", "from_club_id"]], on="player_id", how="inner")
print("Após merge com transfers:", latest_vals.shape)



# Compute age and contract remaining
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
print("Após merge com appearances:", latest_vals.shape)
latest_vals[["goals", "assists", "minutes_played", "matches"]] = latest_vals[["goals", "assists", "minutes_played", "matches"]].fillna(0)

# Additional performance metrics
latest_vals["goals_per_game"] = latest_vals["goals"] / latest_vals["matches"]
latest_vals["assists_per_game"] = latest_vals["assists"] / latest_vals["matches"]
latest_vals["minutes_per_game"] = latest_vals["minutes_played"] / latest_vals["matches"]

# Filter players with meaningful participation (at least 3 matches)
latest_vals = latest_vals[latest_vals["matches"] >= 3]
print("Após filtro de partidas >= 3:", latest_vals.shape)

# Filter top 15 most frequent target clubs
# top_clubs = latest_vals["to_club_id"].value_counts().head(15).index
# latest_vals = latest_vals[latest_vals["to_club_id"].isin(top_clubs)]
# print("Após filtro de top 15 clubes destino:", latest_vals.shape)

# Features and target
features = [
    "age", "market_value_in_eur", "contract_remaining", "highest_market_value_in_eur",
    "goals", "assists", "minutes_played", "matches",
    "goals_per_game", "assists_per_game", "minutes_per_game",
    "height_in_cm", "position", "sub_position", "foot", "country_of_citizenship"
]
target = "to_club_id"

essential_features = [
    "age", "market_value_in_eur", "contract_remaining",
    "goals", "assists", "minutes_played", "matches",
    "position", "foot"
]
print("Missing values (top 10):")
print(latest_vals[essential_features].isnull().sum().sort_values(ascending=False).head(10))
latest_vals["market_value_in_eur"] = latest_vals["market_value_in_eur"].replace({r"[^\d]": ""}, regex=True).astype(float)
latest_vals["highest_market_value_in_eur"] = latest_vals["highest_market_value_in_eur"].replace({r"[^\d]": ""}, regex=True).astype(float)
# Impute missing values for numerical and categorical features
latest_vals["contract_remaining"] = latest_vals["contract_remaining"].fillna(-1)
latest_vals["foot"] = latest_vals["foot"].fillna("Unknown")
latest_vals["sub_position"] = latest_vals["sub_position"].fillna("Unknown")
latest_vals["height_in_cm"] = latest_vals["height_in_cm"].fillna(latest_vals["height_in_cm"].median())
latest_vals["country_of_citizenship"] = latest_vals["country_of_citizenship"].fillna("Unknown")
print("Após imputação de missing values:", latest_vals.shape)
X = latest_vals[features]
y = latest_vals[target]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Preprocessing pipeline
categorical_features = ["position", "sub_position", "foot", "country_of_citizenship"]
numerical_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Only apply preprocessing if there's data
if X.shape[0] == 0:
    raise ValueError("No data available after filtering. Please check filters and input files.")

X_processed = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Hyperparameter search space
param_dist = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 500],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3, 0.5],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}

base_model = XGBClassifier(objective="multi:softprob", num_class=len(label_encoder.classes_), eval_metric='mlogloss')
random_search = RandomizedSearchCV(base_model, param_distributions=param_dist, n_iter=20, cv=3, verbose=1, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

# Evaluation
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

print(classification_report(y_test, y_pred))
top3_accuracy = top_k_accuracy_score(y_test, y_proba, k=3)
print(f"Top-3 Accuracy: {top3_accuracy:.2f}")

# Save model, preprocessor, and label encoder
joblib.dump(best_model, MODEL_PATH + "xgb_club_prediction_model.pkl")
joblib.dump(preprocessor, MODEL_PATH + "club_preprocessor.pkl")
joblib.dump(label_encoder, MODEL_PATH + "club_label_encoder.pkl")
print("Club prediction model, preprocessor, and label encoder saved successfully.")
