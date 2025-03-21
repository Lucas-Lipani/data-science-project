import numpy as np
import pandas as pd

def get_market_trends(transfers_df, clubs_df):
    transfers_df["transfer_fee"] = transfers_df["transfer_fee"].fillna(0)
    transfers_df = transfers_df.merge(
        clubs_df[['club_id', 'domestic_competition_id']],
        left_on="to_club_id", right_on="club_id", how="left"
    )
    league_investments = transfers_df.groupby("domestic_competition_id")["transfer_fee"].sum().reset_index()
    all_leagues = clubs_df[['domestic_competition_id']].drop_duplicates()
    league_investments = all_leagues.merge(league_investments, on="domestic_competition_id", how="left")
    league_investments["transfer_fee"] = league_investments["transfer_fee"].fillna(0)
    max_investment = league_investments["transfer_fee"].max()
    league_investments["investment_score"] = (league_investments["transfer_fee"] / max_investment) * 10
    return league_investments.set_index("domestic_competition_id")["investment_score"].to_dict()

def get_club_spending_profile(transfers_df):
    club_stats = transfers_df.copy()
    club_stats["transfer_fee"] = club_stats["transfer_fee"].fillna(0)
    club_summary = club_stats.groupby("to_club_name")["transfer_fee"].agg(["mean", "std"]).fillna(0)
    return club_summary.to_dict(orient="index")

def get_recent_transfer_patterns(transfers_df, club_id):
    recent = transfers_df[transfers_df["from_club_id"] == club_id]
    return recent["to_club_name"].value_counts(normalize=True).to_dict()

def get_club_to_club_patterns(transfers_df, club_id):
    club_transfers = transfers_df[(transfers_df["from_club_id"] == club_id) | (transfers_df["to_club_id"] == club_id)]
    pairs = club_transfers.groupby(["from_club_name", "to_club_name"]).size().reset_index(name="count")
    pairs["probability"] = pairs["count"] / pairs["count"].sum()
    return pairs.set_index(["from_club_name", "to_club_name"])["probability"].to_dict()

def predict_transfer(player_id, player_valuations_df, transfers_df, players_df, clubs_df):
    player_data = players_df[players_df["player_id"] == player_id]
    if player_data.empty:
        return {"error": "Jogador não encontrado."}

    valuation_data = player_valuations_df[player_valuations_df["player_id"] == player_id]
    transfer_data = transfers_df[transfers_df["player_id"] == player_id]

    last_value = valuation_data["market_value_in_eur"].max()
    contract_end = player_data["contract_expiration_date"].iloc[0]
    nationality = player_data["country_of_citizenship"].iloc[0]
    age = 2025 - int(player_data["date_of_birth"].iloc[0].split("-")[0])
    last_value = int(last_value) if not np.isnan(last_value) else None
    transfer_count = len(transfer_data)
    current_club = player_data["current_club_id"].iloc[0]
    current_club_name = player_data["current_club_name"].iloc[0]
    player_league = player_data["current_club_domestic_competition_id"].iloc[0]

    # Probabilidade base
    contract_factor = 0.6 if contract_end <= "2025-06-30" else 0.2
    transfer_prob = min(1, (transfer_count / 10) + (100 - age) / 200 + contract_factor)

    # Padrões
    recent_club_patterns = get_recent_transfer_patterns(transfers_df, current_club)
    club_to_club_patterns = get_club_to_club_patterns(transfers_df, current_club)

    # Nacionalidade e continente
    same_nationality = players_df[players_df["country_of_citizenship"] == nationality]["player_id"]
    national_transfers = transfers_df[transfers_df["player_id"].isin(same_nationality)]
    national_destinations = national_transfers["to_club_name"].value_counts(normalize=True).to_dict()

    # Tendências de investimento por liga
    market_trends = get_market_trends(transfers_df, clubs_df)
    investment_factor = market_trends.get(player_league, 0.3)
    transfer_prob = min(1, transfer_prob + investment_factor)

    # Perfil de gastos dos clubes
    spending_profile = get_club_spending_profile(transfers_df)

    # Construção do score de destino
    likely_destinations = {}

    for club, prob in recent_club_patterns.items():
        likely_destinations[club] = prob * 0.5

    for (from_club, to_club), prob in club_to_club_patterns.items():
        if from_club == current_club_name:
            likely_destinations[to_club] = likely_destinations.get(to_club, 0) + prob * 0.5

    for club, prob in national_destinations.items():
        likely_destinations[club] = likely_destinations.get(club, 0) + prob * 0.3

    # Ajuste com investimento da liga e capacidade de pagamento dos clubes
    destination_to_league = clubs_df.set_index("name")["domestic_competition_id"].to_dict()
    for club in likely_destinations:
        league_id = destination_to_league.get(club)
        if league_id and league_id in market_trends:
            likely_destinations[club] *= 1 + (market_trends[league_id] / 10)

        # Ajuste pela capacidade financeira (média ± 1.5 std)
        if club in spending_profile and last_value:
            mean = spending_profile[club]["mean"]
            std = spending_profile[club]["std"]
            upper_limit = mean + 1.5 * std
            if last_value > upper_limit:
                likely_destinations[club] *= 0.2  # muito improvável de pagar
            elif last_value < mean:
                likely_destinations[club] *= 1.2  # valor acessível

    # Normalizar e retornar top 5
    total = sum(likely_destinations.values())
    if total == 0:
        return {
            "player_id": int(player_id),
            "market_value": last_value,
            "transfer_probability": round(transfer_prob * 100, 2),
            "likely_destinations": {}
        }

    normalized = {club: (score / total) * 100 for club, score in likely_destinations.items()}
    top5 = dict(sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:5])
    top5 = {club: round(score, 2) for club, score in top5.items()}

    return {
        "player_id": int(player_id),
        "market_value": last_value,
        "transfer_probability": round(transfer_prob * 100, 2),
        "likely_destinations": top5
    }
