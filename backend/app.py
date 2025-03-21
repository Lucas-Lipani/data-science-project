from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from models.performance_model import predict_performance
from models.match_result_model import predict_match_result
from models.transfer_model import predict_transfer

app = Flask(__name__)

# 📂 Caminho para os arquivos de dados
DATA_PATH = "../data/"

# 👐 Carregar os datasets necessários
try:
    appearances = pd.read_csv(DATA_PATH + "appearances.csv")
    club_games = pd.read_csv(DATA_PATH + "club_games.csv")
    clubs = pd.read_csv(DATA_PATH + "clubs.csv")
    games = pd.read_csv(DATA_PATH + "games.csv")
    players = pd.read_csv(DATA_PATH + "players.csv")
    player_valuations = pd.read_csv(DATA_PATH + "player_valuations.csv")
    transfers = pd.read_csv(DATA_PATH + "transfers.csv")
    market_trends = pd.read_csv(DATA_PATH + "market_trends.csv")
except FileNotFoundError as e:
    print(f"Erro ao carregar arquivos: {e}")
    exit(1)

@app.route("/")
def home():
    return "API de Previsão de Futebol Rodando!"

@app.route("/predict_performance", methods=["POST"])
def performance():
    data = request.json
    player_id = data.get("player_id")

    if player_id is None:
        return jsonify({"error": "player_id é obrigatório"}), 400

    result = predict_performance(player_id, appearances)
    return jsonify(result)

@app.route("/predict_match_result", methods=["POST"])
def match_result():
    data = request.json
    home_team = data.get("home_team")
    away_team = data.get("away_team")

    if not home_team or not away_team:
        return jsonify({"error": "home_team e away_team são obrigatórios"}), 400

    result = predict_match_result(home_team, away_team, club_games, games)
    return jsonify(result)

@app.route("/predict_transfer", methods=["POST"])
def transfer():
    data = request.json
    player_id = data.get("player_id")

    if player_id is None:
        return jsonify({"error": "player_id é obrigatório"}), 400

    result = predict_transfer(player_id, player_valuations, transfers, players, clubs)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
