import pandas as pd
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import unidecode

app = Flask(__name__)

# Load trained models
transfer_model = joblib.load("../models/saved/xgb_transfer_model.pkl")
preprocessor = joblib.load("../models/saved/preprocessor.pkl")
club_model = joblib.load("../models/saved/xgb_club_model.pkl")
scaler = joblib.load("../models/saved/xgb_scaler.pkl")
label_encoder = joblib.load("../models/saved/xgb_label_encoder.pkl")

# Load datasets
data = pd.read_csv("../data/players.csv")
data.columns = [col.strip() for col in data.columns]
data_transfers = pd.read_csv("../data/transfers.csv")
data_valuations = pd.read_csv("../data/player_valuations.csv")
data_appearances = pd.read_csv("../data/appearances.csv")

# Calculate player age
if "date_of_birth" in data.columns:
    data["date_of_birth"] = pd.to_datetime(data["date_of_birth"], errors="coerce")
    data["age"] = 2025 - data["date_of_birth"].dt.year
else:
    data["age"] = 0

numerical_features = ["age", "market_value_in_eur", "contract_remaining",
                      "goals", "assists", "minutes_played", "matches",
                      "num_transfers", "avg_transfer_fee", "height_in_cm"]

# Calculate player features consistent with training
def calculate_player_features(player):
    player_id = player["player_id"].values[0]
    history = data_transfers[data_transfers["player_id"] == player_id]
    valuation = data_valuations[data_valuations["player_id"] == player_id].sort_values("date").drop_duplicates("player_id", keep="last")
    appearance = data_appearances[data_appearances["player_id"] == player_id]

    player_features = pd.DataFrame({
        "age": [player["age"].values[0]],
        "market_value_in_eur": [valuation["market_value_in_eur"].values[0] if not valuation.empty else 0],
        "contract_remaining": [1],  # ajuste para lógica real depois
        "goals": [appearance["goals"].sum() if not appearance.empty else 0],
        "assists": [appearance["assists"].sum() if not appearance.empty else 0],
        "minutes_played": [appearance["minutes_played"].sum() if not appearance.empty else 0],
        "matches": [appearance["game_id"].nunique() if not appearance.empty else 0],
        "num_transfers": [len(history)],
        "avg_transfer_fee": [history["transfer_fee"].mean() if not history.empty else 0],
        "height_in_cm": [player["height_in_cm"].values[0]],
        "position": [player["position"].values[0]],
        "above_club_max_position_value": [0],
        "avg_player_value_diff": [0],
        "club_avg_player_value": [0],
        "club_can_afford_realistically": [0],
        "club_national_team_players": [0],
        "foreigners_percentage": [0],
        "is_affordable": [0],
        "nationality_match": [0],
        "performance_vs_value": [0],
        "player_value": [0],
        "player_value_zscore": [0]
    })

    player_features.fillna({
        "avg_transfer_fee": 0,
        "height_in_cm": data["height_in_cm"].median(),
        "market_value_in_eur": 0,
        "position": "Unknown"
    }, inplace=True)

    return player_features

@app.route("/search_player")
def search_player():
    query = request.args.get("name", "").lower()
    normalized_query = unidecode.unidecode(query)

    def match(row):
        name = unidecode.unidecode(str(row.get("name", "")).lower())
        first = unidecode.unidecode(str(row.get("first_name", "")).lower())
        last = unidecode.unidecode(str(row.get("last_name", "")).lower())
        return normalized_query in name or normalized_query in first or normalized_query in last

    matches = data[data.apply(match, axis=1)].reset_index(drop=True)

    results = []
    for idx, row in matches.iterrows():
        results.append({
            "index": idx,
            "name": row.get("name"),
            "age": int(row.get("age", 0)) if not np.isnan(row.get("age", 0)) else 0,
            "current_club": row.get("current_club_name", "Unknown")
        })

    return jsonify(results)

@app.route("/predict_transfer", methods=["POST"])
def predict_transfer():
    content = request.json
    player_index = content.get("index")

    if player_index is None or player_index >= len(data):
        return jsonify({"error": "Invalid player index."}), 400

    player = data.iloc[player_index:player_index + 1].copy()
    player_features = calculate_player_features(player)

    transfer_feature_columns = numerical_features + ["position"]
    X_transfer = preprocessor.transform(player_features[transfer_feature_columns])
    prob_transfer = transfer_model.predict_proba(X_transfer)[0][1]

    club_numerical_features = [
        "above_club_max_position_value", "avg_player_value_diff", "club_avg_player_value",
        "club_can_afford_realistically", "club_national_team_players", "age", "assists",
        "foreigners_percentage", "goals", "is_affordable", "minutes_played", "nationality_match",
        "performance_vs_value", "player_value", "player_value_zscore"
    ]

    player_scaled = scaler.transform(player_features[club_numerical_features])
    club_probs = club_model.predict_proba(player_scaled)[0]

    top_club_indices = np.argsort(club_probs)[::-1][:5]
    top_club_probs = [
        {"club": label_encoder.inverse_transform([idx])[0], "prob": float(club_probs[idx])}
        for idx in top_club_indices
    ]

    return jsonify({
        "transfer_probability": float(prob_transfer),
        "top_clubs": top_club_probs
    })

@app.route("/")
def index():
    return send_from_directory('../frontend', 'index.html')

if __name__ == "__main__":
    app.run(debug=True)
