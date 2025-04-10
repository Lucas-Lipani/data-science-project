from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
from unidecode import unidecode
from difflib import get_close_matches
import os
from flask_cors import CORS  # <= importe aqui

from models.performance_model import predict_performance
from models.match_result_model import predict_match_result
from models.transfer_model import predict_transfer

app = Flask(__name__)
CORS(app, supports_credentials=True)


# Path to data files
DATA_PATH = "./data/"

# Load required datasets
try:
    appearances = pd.read_csv(DATA_PATH + "appearances.csv")
    club_games = pd.read_csv(DATA_PATH + "club_games.csv")
    clubs = pd.read_csv(DATA_PATH + "clubs.csv")
    games = pd.read_csv(DATA_PATH + "games.csv")
    players = pd.read_csv(DATA_PATH + "players.csv")
    player_valuations = pd.read_csv(DATA_PATH + "player_valuations.csv")
    transfers = pd.read_csv(DATA_PATH + "transfers.csv")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit(1)

# Player preprocessing
players = players.dropna(subset=["name", "date_of_birth", "current_club_name"])
players["name_norm"] = players["name"].apply(lambda x: unidecode(str(x)).lower())
players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
players["age"] = players["date_of_birth"].apply(
    lambda x: datetime.now().year - x.year if pd.notnull(x) else None
)
if "player_id" not in players.columns:
    players["player_id"] = players.index

# Stores the last searched players for selection by number
last_search = {}

@app.route("/")
def home():
    return "Football Prediction API Running!"

# Search for a player by name
@app.route("/search_player", methods=["GET"])
def search_player():
    query = request.args.get("q", "")
    q_norm = unidecode(query.lower())

    matched = players[players["name_norm"].str.contains(q_norm, na=False)].copy()
    matched = matched[matched["age"].notnull()]
    matched["relevance_score"] = abs(matched["age"] - 25)
    matched = matched.sort_values(by="relevance_score").head(10)

    search_results = matched.reset_index(drop=True)
    last_search["players"] = search_results

    result_lines = []
    for idx, row in search_results.iterrows():
        result_lines.append(
            f"{idx + 1}. {row['name']} - {row['current_club_name']} ({row['country_of_citizenship']}, {int(row['age'])} years)"
        )

    return jsonify({"options": result_lines})

# Select a player by the number from the previous list
@app.route("/select_player", methods=["POST"])
def select_player():
    data = request.get_json()
    index = int(data.get("option")) - 1

    if "players" not in last_search or index >= len(last_search["players"]):
        return jsonify({"error": "Invalid selection"}), 400

    selected = last_search["players"].iloc[index].to_dict()

    # Limpa os NaNs (solução segura)
    selected_clean = {k: (None if pd.isna(v) else v) for k, v in selected.items()}
    # Adiciona o logo do clube com base no club_id
    club_id = selected_clean.get("current_club_id")
    if club_id:
        selected_clean["club_logo_url"] = f"https://tmssl.akamaized.net/images/wappen/head/{int(club_id)}.png"

    last_search["selected_player"] = selected_clean
    return jsonify({"selected_player": selected_clean})


# Performance prediction (not modified)
@app.route("/predict_performance", methods=["POST"])
def performance():
    data = request.json
    player_id = data.get("player_id")

    if player_id is None:
        return jsonify({"error": "player_id is required"}), 400

    result = predict_performance(player_id, appearances)
    return jsonify(result)

# Match result prediction (not modified)
@app.route("/predict_match_result", methods=["POST"])
def match_result():
    data = request.json
    home_team = data.get("home_team")
    away_team = data.get("away_team")

    if not home_team or not away_team:
        return jsonify({"error": "home_team and away_team are required"}), 400

    result = predict_match_result(home_team, away_team, club_games, games)
    return jsonify(result)

# Transfer prediction using selected player or player_id
@app.route("/predict_transfer", methods=["POST"])
def transfer():
    data = request.get_json()
    player_id = data.get("player_id")

    # If not explicitly provided, use the selected one
    if player_id is None:
        selected = last_search.get("selected_player")
        if not selected:
            return jsonify({"error": "No player selected or provided"}), 400
        player_id = selected["player_id"]

    result = predict_transfer(player_id, player_valuations, transfers, players, clubs)

    # Enrich likely destinations with club_id for logos
    destinations = result.get("likely_destinations", {})
    enhanced_destinations = {}

    for name, score in destinations.items():
        club_name_norm = unidecode(name).lower()

        # Try direct partial match
        match = clubs[clubs["name"].apply(lambda x: club_name_norm in unidecode(str(x)).lower())]

        # If no match, try fuzzy
        if match.empty:
            possible_matches = get_close_matches(club_name_norm, clubs["name"].apply(lambda x: unidecode(str(x)).lower()), n=1, cutoff=0.7)
            if possible_matches:
                match = clubs[clubs["name"].apply(lambda x: unidecode(str(x)).lower() == possible_matches[0])]

        if not match.empty:
            club_id = int(match.iloc[0]["club_id"])
            enhanced_destinations[name] = {
                "score": score,
                "club_id": club_id
            }
        else:
            print(f"[WARN] Club not found for: {name}")
            enhanced_destinations[name] = {
                "score": score,
                "club_id": None
            }

    result["likely_destinations"] = enhanced_destinations

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
