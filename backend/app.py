from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
from unidecode import unidecode
import os

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
except FileNotFoundError as e:
    print(f"Erro ao carregar arquivos: {e}")
    exit(1)

# 🧠 Pré-processamento dos jogadores
players = players.dropna(subset=["name", "date_of_birth", "current_club_name"])
players["name_norm"] = players["name"].apply(lambda x: unidecode(str(x)).lower())
players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
players["age"] = players["date_of_birth"].apply(
    lambda x: datetime.now().year - x.year if pd.notnull(x) else None
)
if "player_id" not in players.columns:
    players["player_id"] = players.index

# 🔁 Armazena os últimos jogadores buscados para selecionar por número
last_search = {}

@app.route("/")
def home():
    return "API de Previsão de Futebol Rodando!"

# 🎯 Busca por nome de jogador
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
            f"{idx + 1}. {row['name']} - {row['current_club_name']} ({row['country_of_citizenship']}, {int(row['age'])} anos)"
        )

    return jsonify({"options": result_lines})

# ✅ Seleciona jogador pelo número da lista anterior
@app.route("/select_player", methods=["POST"])
def select_player():
    data = request.get_json()
    index = int(data.get("option")) - 1

    if "players" not in last_search or index >= len(last_search["players"]):
        return jsonify({"error": "Seleção inválida"}), 400

    selected = last_search["players"].iloc[index].to_dict()
    last_search["selected_player"] = selected  # Salva o jogador escolhido
    return jsonify({"selected_player": selected})

# ⚽ Previsão de performance (não foi alterada)
@app.route("/predict_performance", methods=["POST"])
def performance():
    data = request.json
    player_id = data.get("player_id")

    if player_id is None:
        return jsonify({"error": "player_id é obrigatório"}), 400

    result = predict_performance(player_id, appearances)
    return jsonify(result)

# 🏆 Previsão de resultado de partida (não foi alterada)
@app.route("/predict_match_result", methods=["POST"])
def match_result():
    data = request.json
    home_team = data.get("home_team")
    away_team = data.get("away_team")

    if not home_team or not away_team:
        return jsonify({"error": "home_team e away_team são obrigatórios"}), 400

    result = predict_match_result(home_team, away_team, club_games, games)
    return jsonify(result)

# 💰 Previsão de transferência usando jogador selecionado ou player_id
@app.route("/predict_transfer", methods=["POST"])
def transfer():
    data = request.get_json()
    player_id = data.get("player_id")

    # Se não foi passado explicitamente, usa o selecionado
    if player_id is None:
        selected = last_search.get("selected_player")
        if not selected:
            return jsonify({"error": "Nenhum jogador selecionado ou informado"}), 400
        player_id = selected["player_id"]

    result = predict_transfer(player_id, player_valuations, transfers, players, clubs)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
