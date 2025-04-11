import numpy as np

def predict_performance(player_id, appearances_df):
    player_data = appearances_df[appearances_df["player_id"] == player_id]

    if player_data.empty:
        return {"error": "Jogador não encontrado."}

    avg_goals = player_data["goals"].mean()
    avg_assists = player_data["assists"].mean()
    avg_yellow_cards = player_data["yellow_cards"].mean()
    avg_red_cards = player_data["red_cards"].mean()

    return {
        "player_id": player_id,
        "predicted_goals": round(avg_goals, 2),
        "predicted_assists": round(avg_assists, 2),
        "predicted_yellow_cards": round(avg_yellow_cards, 2),
        "predicted_red_cards": round(avg_red_cards, 2),
    }