import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from interpret.glassbox import ExplainableBoostingClassifier

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
competitions = pd.read_csv(DATA_PATH + "competitions.csv")

# Parse dates
players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
players["contract_expiration_date"] = pd.to_datetime(players["contract_expiration_date"], errors="coerce")
transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
player_valuations["date"] = pd.to_datetime(player_valuations["date"], errors="coerce")
appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")

# Filter only successful transfers with known destination
transfers = transfers.dropna(subset=["to_club_id"])

# Define reference date before transfer
transfers["ref_date"] = transfers["transfer_date"] - pd.DateOffset(months=1)

# Prepare valuation info before transfer
latest_val = player_valuations.copy()
latest_val = latest_val.sort_values("date")
latest_val = latest_val.groupby("player_id", group_keys=False).apply(
    lambda x: x.set_index("date").resample("1ME").ffill().reset_index())

# Peak market value
peak_value = player_valuations.groupby("player_id")["market_value_in_eur"].max().reset_index().rename(columns={"market_value_in_eur": "market_value_peak"})

# Player performance stats
total_perf = appearances.groupby("player_id").agg({
    "goals": "sum",
    "assists": "sum",
    "minutes_played": "sum",
    "appearance_id": "count"
}).rename(columns={"appearance_id": "total_matches"}).reset_index()
total_perf["goals_per_game"] = total_perf["goals"] / total_perf["total_matches"]
total_perf["assists_per_game"] = total_perf["assists"] / total_perf["total_matches"]

# Merge club info
clubs = clubs.merge(competitions[["competition_id", "country_name"]], left_on="domestic_competition_id", right_on="competition_id", how="left")
clubs.rename(columns={"country_name": "club_country"}, inplace=True)
transfers = transfers.merge(clubs.rename(columns={"club_id": "from_club_id", "club_country": "from_club_country"}), on="from_club_id", how="left")

# Build dataset
samples = []
print("Building training dataset...")

for _, row in tqdm(transfers.iterrows(), total=transfers.shape[0], desc="Processing transfers"):
    pid = row["player_id"]
    ref = row["ref_date"]
    val = latest_val[(latest_val["player_id"] == pid) & (latest_val["date"] < ref)].sort_values("date").tail(6)
    if val.empty:
        continue

    avg_val = val["market_value_in_eur"].mean()
    growth = (val.iloc[-1]["market_value_in_eur"] - val.iloc[0]["market_value_in_eur"]) / val.iloc[0]["market_value_in_eur"] if val.iloc[0]["market_value_in_eur"] > 0 else 0

    p = players[players["player_id"] == pid]
    if p.empty:
        continue
    p = p.iloc[0]
    age = (ref.year - p["date_of_birth"].year) if pd.notnull(p["date_of_birth"]) else None
    contract_remaining = (p["contract_expiration_date"] - ref).days / 365 if pd.notnull(p["contract_expiration_date"]) else None

    perf = total_perf[total_perf["player_id"] == pid]
    if perf.empty:
        continue
    perf = perf.iloc[0]

    peak = peak_value[peak_value["player_id"] == pid]
    peak_val = peak["market_value_peak"].values[0] if not peak.empty else avg_val
    decline = (peak_val - val.iloc[-1]["market_value_in_eur"]) / peak_val if peak_val > 0 else 0

    samples.append({
        "player_id": pid,
        "age": age,
        "market_value_in_eur": val.iloc[-1]["market_value_in_eur"],
        "market_value_growth": growth,
        "avg_market_value_last_6m": avg_val,
        "market_value_peak": peak_val,
        "market_decline_pct": decline,
        "goals_total": perf["goals"],
        "assists_total": perf["assists"],
        "minutes_total": perf["minutes_played"],
        "matches_total": perf["total_matches"],
        "goals_per_game": perf["goals_per_game"],
        "assists_per_game": perf["assists_per_game"],
        "contract_remaining": contract_remaining,
        "height_in_cm": p["height_in_cm"],
        "position": p["position"],
        "foot": p["foot"],
        "country_of_citizenship": p["country_of_citizenship"],
        "from_club_country": row["from_club_country"],
        "to_club_id": row["to_club_id"]
    })

# DataFrame & Cleaning
data = pd.DataFrame(samples)
for col in ["age", "contract_remaining", "height_in_cm"]:
    data[col] = data[col].fillna(data[col].mean())
data = data.dropna()

# Top clubs & label
top_clubs = data["to_club_id"].value_counts().head(200).index
data["club_target"] = data["to_club_id"].apply(lambda x: x if x in top_clubs else 9999)
data["club_target"] = data["club_target"].astype(str)

# Encoding
features = [
    "age", "market_value_in_eur", "market_value_growth", "avg_market_value_last_6m",
    "market_value_peak", "market_decline_pct", "goals_total", "assists_total",
    "minutes_total", "matches_total", "goals_per_game", "assists_per_game",
    "contract_remaining", "height_in_cm", "position", "foot",
    "country_of_citizenship", "from_club_country"
]
X = data[features]
y = data["club_target"]

cat_features = ["position", "foot", "country_of_citizenship", "from_club_country"]
num_features = [col for col in features if col not in cat_features]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

print("Fitting preprocessor...")
X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, stratify=y, random_state=42)

# Train EBM
print("Training Explainable Boosting Machine...")
model = ExplainableBoostingClassifier(interactions=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
print("Evaluating model performance...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Save model & preprocessor
joblib.dump(model, MODEL_PATH + "ebm_club_model.pkl")
joblib.dump(preprocessor, MODEL_PATH + "preprocessor_club.pkl")
print("EBM club prediction model and components saved successfully.")
