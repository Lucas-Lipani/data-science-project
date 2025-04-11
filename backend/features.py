# src/features.py

import pandas as pd
from datetime import datetime

DATA_PATH = "../data/"

# Carrega os dados
players = pd.read_csv(DATA_PATH + "players.csv")
transfers = pd.read_csv(DATA_PATH + "transfers.csv")
player_valuations = pd.read_csv(DATA_PATH + "player_valuations.csv")
appearances = pd.read_csv(DATA_PATH + "appearances.csv")

# Corrige colunas de data
players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
players["contract_expiration_date"] = pd.to_datetime(players["contract_expiration_date"], errors="coerce")
transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
player_valuations["date"] = pd.to_datetime(player_valuations["date"], errors="coerce")
appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")


def build_player_features(player_id, current_date):
    player = players[players["player_id"] == player_id].iloc[0]

    player_val = player_valuations[(player_valuations["player_id"] == player_id) &
                                   (player_valuations["date"] < current_date)].sort_values("date").tail(1)

    if player_val.empty:
        raise ValueError("Sem valuation disponível pro jogador")

    market_value_in_eur = player_val["market_value_in_eur"].values[0]

    age = current_date.year - player["date_of_birth"].year
    contract_remaining = (player["contract_expiration_date"] - current_date).days / 365

    past_date = current_date - pd.DateOffset(months=6)
    recent_perf = appearances[(appearances["player_id"] == player_id) &
                               (appearances["date"] >= past_date) &
                               (appearances["date"] < current_date)]

    goals = recent_perf["goals"].sum()
    assists = recent_perf["assists"].sum()
    minutes_played = recent_perf["minutes_played"].sum()
    matches = recent_perf.shape[0]

    player_transfers = transfers[(transfers["player_id"] == player_id) &
                                 (transfers["transfer_date"] < current_date)]

    num_transfers = player_transfers.shape[0]
    avg_transfer_fee = player_transfers["transfer_fee"].mean() if num_transfers > 0 else 0

    features = {
        "age": age,
        "market_value_in_eur": market_value_in_eur,
        "contract_remaining": contract_remaining,
        "goals": goals,
        "assists": assists,
        "minutes_played": minutes_played,
        "matches": matches,
        "num_transfers": num_transfers,
        "avg_transfer_fee": avg_transfer_fee,
        "height_in_cm": player["height_in_cm"],
        "position": player["position"]
    }

    return features


def get_players_data():
    return players.copy()
