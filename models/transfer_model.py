import numpy as np
import pandas as pd

def predict_transfer(player_id, player_valuations_df, transfers_df, players_df):
    player_data = players_df[players_df["player_id"] == player_id]
    valuation_data = player_valuations_df[player_valuations_df["player_id"] == player_id]
    transfer_data = transfers_df[transfers_df["player_id"] == player_id]

    if player_data.empty:
        return {"error": "Jogador não encontrado."}

    last_value = valuation_data["market_value_in_eur"].max()

    # Certifique-se de que last_value é um tipo Python int e não um int64
    if not np.isnan(last_value):
        last_value = int(last_value)
    else:
        last_value = None  # Caso não tenha dados de mercado

    transfer_count = len(transfer_data)

    # Simulação baseada no histórico de transferências
    transfer_prob = min(1, (transfer_count / 5) + np.random.uniform(0, 0.3))

    # Convertendo todas as chaves e valores para tipos Python nativos
    possible_teams = {str(club): float(prob) for club, prob in transfer_data["to_club_name"].value_counts(normalize=True).items()}

    return {
        "player_id": int(player_id),  # Certifique-se de que player_id é int
        "market_value": last_value,
        "transfer_probability": round(transfer_prob * 100, 2),
        "likely_destinations": possible_teams
    }
