# src/app.py

import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
from features import build_player_features, get_players_data
import unidecode
import os

# Paths
MODEL_PATH = "../models/saved/"
FRONT_PATH = "../frontend/"

app = Flask(__name__)

# Load models
model = joblib.load(MODEL_PATH + "xgb_transfer_model_transfer.pkl")
preprocessor = joblib.load(MODEL_PATH + "preprocessor_transfer.pkl")

# Load player data
players_data = get_players_data()

# Rota pra servir index.html
@app.route("/")
def index():
    return send_from_directory(FRONT_PATH, "index.html")

# Rota pra servir arquivos estáticos (css/js)
@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONT_PATH, path)

# Search player
@app.route("/search_player")
def search_player():
    query = request.args.get("name", "").lower()
    normalized_query = unidecode.unidecode(query)

    def match(row):
        name = unidecode.unidecode(str(row.get("name", "")).lower())
        first = unidecode.unidecode(str(row.get("first_name", "")).lower())
        last = unidecode.unidecode(str(row.get("last_name", "")).lower())
        return normalized_query in name or normalized_query in first or normalized_query in last

    matches = players_data[players_data.apply(match, axis=1)].reset_index(drop=True)

    results = []
    for idx, row in matches.iterrows():
        results.append({
            "index": idx,
            "player_id": row["player_id"],
            "name": row["name"],
            "age": int((datetime.today().year - row["date_of_birth"].year)) if pd.notna(row["date_of_birth"]) else None,
            "current_club": row.get("current_club_name") or row.get("from_club_name") or "Unknown"
        })

    return jsonify(results)

# Predict transfer route
@app.route("/predict_transfer", methods=["POST"])
def predict_transfer():
    content = request.json
    player_id = content.get("player_id")

    if player_id is None:
        return jsonify({"error": "Missing player_id"}), 400

    try:
        current_date = pd.to_datetime(datetime.today())
        features = build_player_features(player_id, current_date)

        df = pd.DataFrame([features])
        df_processed = preprocessor.transform(df)
        prediction_proba = model.predict_proba(df_processed)

        prob_transfer = prediction_proba[0][1] * 100
        prob_stay = prediction_proba[0][0] * 100

        return jsonify({
            "player_id": int(player_id),
            "transfer_probability": float(round(prob_transfer, 2)),
            "stay_probability": float(round(prob_stay, 2))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
