
import pandas as pd

DATA_PATH = "../data/"

# Carrega os dados
players = pd.read_csv(DATA_PATH + "players.csv")
transfers = pd.read_csv(DATA_PATH + "transfers.csv")
player_valuations = pd.read_csv(DATA_PATH + "player_valuations.csv")
appearances = pd.read_csv(DATA_PATH + "appearances.csv")

# Corrige colunas de data
players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
player_valuations["date"] = pd.to_datetime(player_valuations["date"], errors="coerce")
appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")

def build_club_features(player_id):
    player = players[players["player_id"] == player_id].iloc[0]

    player_val = player_valuations[player_valuations["player_id"] == player_id].sort_values("date").tail(1)

    if player_val.empty:
        raise ValueError("Sem valuation disponível pro jogador")

    market_value_in_eur = player_val["market_value_in_eur"].values[0]

    age = 2025 - player["date_of_birth"].year

    perf = appearances[appearances["player_id"] == player_id]

    goals = perf["goals"].mean() if not perf.empty else 0
    assists = perf["assists"].mean() if not perf.empty else 0
    minutes_played = perf["minutes_played"].mean() if not perf.empty else 0

    worthiness = (goals * 4 + assists * 3 + minutes_played / 90)

    valuable_player = int(market_value_in_eur > player_valuations["market_value_in_eur"].quantile(0.75))

    features = {
        "age": age,
        "position": player["position"],
        "country_of_citizenship": player["country_of_citizenship"],
        "market_value_in_eur": market_value_in_eur,
        "goals": goals,
        "assists": assists,
        "minutes_played": minutes_played,
        "worthiness": worthiness,
        "valuable_player": valuable_player
    }

    return features

def get_players_data():
    return players.copy()
