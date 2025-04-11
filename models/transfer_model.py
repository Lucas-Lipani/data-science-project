import pandas as pd
import numpy as np
import joblib


# Load model components once (outside the function to avoid reloading)
model_path = "models/saved/"
club_model = joblib.load(model_path + "xgb_club_prediction_model.pkl")
club_preprocessor = joblib.load(model_path + "club_preprocessor.pkl")
club_label_encoder = joblib.load(model_path + "club_label_encoder.pkl")

def predict_transfer(player_id, player_valuations_df, transfers_df, players_df, clubs_df):
    player_data = players_df[players_df["player_id"] == player_id]
    if player_data.empty:
        return {"error": "Player not found."}

    valuation_data = player_valuations_df[player_valuations_df["player_id"] == player_id]
    if valuation_data.empty:
        return {"error": "No valuation data for player."}

    # Get latest market value
    market_value = valuation_data["market_value_in_eur"].max()

    # Get player performance data (last 6 months)
    appearances = pd.read_csv("data/appearances.csv")
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    recent_cutoff = pd.to_datetime("2022-07-01")  # same cutoff as in training

    recent = appearances[(appearances["player_id"] == player_id) &
                         (appearances["date"] < recent_cutoff) &
                         (appearances["date"] >= recent_cutoff - pd.DateOffset(months=6))]

    stats = {
        "goals": recent["goals"].sum(),
        "assists": recent["assists"].sum(),
        "minutes_played": recent["minutes_played"].sum(),
        "matches": recent.shape[0]
    }

    # Calculated fields
    stats["goals_per_game"] = stats["goals"] / stats["matches"] if stats["matches"] > 0 else 0
    stats["assists_per_game"] = stats["assists"] / stats["matches"] if stats["matches"] > 0 else 0
    stats["minutes_per_game"] = stats["minutes_played"] / stats["matches"] if stats["matches"] > 0 else 0

    player_row = player_data.iloc[0]

    input_data = pd.DataFrame([{
        "age": 2022 - player_row["date_of_birth"].year,
        "market_value_in_eur": market_value,
        "contract_remaining": (
            (pd.to_datetime(player_row["contract_expiration_date"]) - recent_cutoff).days / 365
            if pd.notnull(player_row["contract_expiration_date"]) else -1
        ),
        "highest_market_value_in_eur": player_row.get("highest_market_value_in_eur", market_value),
        "goals": stats["goals"],
        "assists": stats["assists"],
        "minutes_played": stats["minutes_played"],
        "matches": stats["matches"],
        "goals_per_game": stats["goals_per_game"],
        "assists_per_game": stats["assists_per_game"],
        "minutes_per_game": stats["minutes_per_game"],
        "height_in_cm": player_row.get("height_in_cm", np.nan),
        "position": player_row["position"],
        "sub_position": player_row.get("sub_position", "Unknown"),
        "foot": player_row.get("foot", "Unknown"),
        "country_of_citizenship": player_row.get("country_of_citizenship", "Unknown")
    }])

    input_data.fillna({
        "height_in_cm": clubs_df["height_in_cm"].median() if "height_in_cm" in clubs_df else 180,
        "sub_position": "Unknown",
        "foot": "Unknown",
        "country_of_citizenship": "Unknown"
    }, inplace=True)

    # Apply preprocessing and predict
    X_processed = club_preprocessor.transform(input_data)
    y_proba = club_model.predict_proba(X_processed)[0]
    club_ids = club_label_encoder.inverse_transform(np.argsort(y_proba)[::-1][:5])
    scores = np.sort(y_proba)[::-1][:5] * 100

    likely_destinations = {
        clubs_df.loc[clubs_df["club_id"] == cid, "name"].values[0]: {
            "score": float(round(score, 2)),  # <- força tipo float nativo
            "club_id": int(cid)
        }
        for cid, score in zip(club_ids, scores)
        if cid in clubs_df["club_id"].values
    }


    return {
        "player_id": int(player_id),
        "market_value": float(market_value),
        "transfer_probability": float(round(sum(scores), 2)),
        "likely_destinations": {
            clubs_df.loc[clubs_df["club_id"] == cid, "name"].values[0]: {
                "score": float(round(score, 2)),
                "club_id": int(cid)
            }
            for cid, score in zip(club_ids, scores)
            if cid in clubs_df["club_id"].values
        }
    }
