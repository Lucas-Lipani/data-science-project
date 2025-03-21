import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load datasets
transfers = pd.read_csv("data/transfers.csv")
clubs = pd.read_csv("data/clubs.csv")

# Ensure transfer_fee is numeric and no NaNs
transfers["transfer_fee"] = transfers["transfer_fee"].fillna(0)

# Get league for both origin and destination clubs
transfers_to = transfers.merge(clubs[["club_id", "domestic_competition_id"]], left_on="to_club_id", right_on="club_id", how="left")
to_agg = transfers_to.groupby("domestic_competition_id")["transfer_fee"].sum().reset_index(name="total_bought")

transfers_from = transfers.merge(clubs[["club_id", "domestic_competition_id"]], left_on="from_club_id", right_on="club_id", how="left")
from_agg = transfers_from.groupby("domestic_competition_id")["transfer_fee"].sum().reset_index(name="total_sold")

# Merge in and out data
market_df = pd.merge(to_agg, from_agg, on="domestic_competition_id", how="outer").fillna(0)
market_df["net_spending"] = market_df["total_bought"] - market_df["total_sold"]

# Add number of clubs per league to normalize
clubs_by_league = clubs.groupby("domestic_competition_id")["club_id"].nunique().reset_index(name="n_clubs")
market_df = market_df.merge(clubs_by_league, on="domestic_competition_id", how="left")
market_df["avg_spending_per_club"] = market_df["total_bought"] / market_df["n_clubs"]

# Normalize features to scale (0 to 10)
scaler = MinMaxScaler(feature_range=(0, 10))
normalized = scaler.fit_transform(market_df[["total_bought", "total_sold", "net_spending"]])
market_df[["investment_score", "attractiveness_score", "net_score"]] = normalized

# Final composite score (weights can be adjusted)
market_df["market_trend_score"] = (
    0.5 * market_df["investment_score"] +
    0.3 * market_df["attractiveness_score"] +
    0.2 * market_df["net_score"]
)

# Clean output
market_df = market_df[[
    "domestic_competition_id", "total_bought", "total_sold", "net_spending",
    "investment_score", "attractiveness_score", "net_score", "market_trend_score"
]]

# Save
market_df.to_csv("data/market_trends.csv", index=False)

print("✅ Novo arquivo 'market_trends.csv' criado com indicadores refinados de mercado.")
