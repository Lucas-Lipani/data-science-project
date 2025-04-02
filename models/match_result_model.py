import numpy as np


def predict_match_result(home_team, away_team, club_games_df, games_df):
    home_stats = club_games_df[club_games_df["club_id"] == home_team]
    away_stats = club_games_df[club_games_df["club_id"] == away_team]

    if home_stats.empty or away_stats.empty:
        return {"error": "Times não encontrados."}

    avg_home_goals = home_stats["own_goals"].mean()
    avg_away_goals = away_stats["own_goals"].mean()

    home_win_prob = np.random.uniform(0.3, 0.7)  # Probability Simulation
    draw_prob = 1 - home_win_prob - np.random.uniform(0.1, 0.3)
    away_win_prob = 1 - home_win_prob - draw_prob

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_score": f"{round(avg_home_goals)} - {round(avg_away_goals)}",
        "win_probability": {
            "home": round(home_win_prob * 100, 2),
            "draw": round(draw_prob * 100, 2),
            "away": round(away_win_prob * 100, 2),
        }
    }
