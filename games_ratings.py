# Recarregar os dados de eventos e jogadores
import pandas as pd

players_df = pd.read_csv("data/players.csv")
game_events_df = pd.read_csv("data/game_events.csv")

# Juntar a coluna de posição
game_events_df = game_events_df.merge(players_df[["player_id", "position"]], on="player_id", how="left")
game_events_df["position"] = game_events_df["position"].fillna("Unknown")

# Inicializar coluna de score com float
game_events_df["score"] = 0.0

# Marcar cartões
game_events_df.loc[game_events_df["type"] == "Cards", "score"] -= game_events_df["description"].str.contains("Yellow", na=False).astype(float) * 1.5
game_events_df.loc[game_events_df["type"] == "Cards", "score"] -= game_events_df["description"].str.contains("Red", na=False).astype(float) * 4

# Gols: diferentes por posição
forwards = game_events_df["position"] == "Attack"
midfielders = game_events_df["position"] == "Midfield"
defenders = game_events_df["position"] == "Defender"
goalkeepers = game_events_df["position"] == "Goalkeeper"

game_events_df.loc[forwards & (game_events_df["type"] == "Goals"), "score"] += 10
game_events_df.loc[midfielders & (game_events_df["type"] == "Goals"), "score"] += 8
game_events_df.loc[defenders & (game_events_df["type"] == "Goals"), "score"] += 6
game_events_df.loc[goalkeepers & (game_events_df["type"] == "Goals"), "score"] += 15

# Assistências
game_events_df.loc[game_events_df["type"] == "Assist", "score"] += 7

# Shootouts (pênaltis) - bônus para quem marca ou defende
game_events_df.loc[(game_events_df["type"] == "Shootout") & (forwards), "score"] += 5
game_events_df.loc[(game_events_df["type"] == "Shootout") & (goalkeepers), "score"] += 7

# Substituições no início do jogo (minuto <= 20)
game_events_df.loc[(game_events_df["type"] == "Substitutions") & (game_events_df["minute"] <= 20), "score"] -= 2

# Descrições detalhadas
game_events_df["description"] = game_events_df["description"].fillna("")

game_events_df.loc[game_events_df["description"].str.contains("Pass", na=False), "score"] += 1
game_events_df.loc[game_events_df["description"].str.contains("Cross", na=False), "score"] += 1.5
game_events_df.loc[game_events_df["description"].str.contains("Header", na=False), "score"] += 2
game_events_df.loc[game_events_df["description"].str.contains("Corner", na=False), "score"] += 1
game_events_df.loc[game_events_df["description"].str.contains("Free kick", na=False), "score"] += 1.5
game_events_df.loc[game_events_df["description"].str.contains("Fouled", na=False), "score"] += 1

# Calcular média móvel por jogador
game_events_df["date"] = pd.to_datetime(game_events_df["date"], errors="coerce")
game_events_df = game_events_df.sort_values(["player_id", "date"])
game_events_df["rolling_rating"] = game_events_df.groupby("player_id")["score"].transform(lambda x: x.rolling(10, min_periods=1).mean())

# Selecionar última nota conhecida por jogador
latest_ratings = game_events_df.dropna(subset=["rolling_rating"])[["player_id", "date", "rolling_rating"]]
latest_ratings["rolling_rating"] = latest_ratings["rolling_rating"].clip(3, 10)

# Exportar CSV
latest_ratings.to_csv("data/games_ratings.csv", index=False)

