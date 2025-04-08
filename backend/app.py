# ============================================
# 📦 app.py — Club Transfer Predictor
# ============================================
import pandas as pd
import joblib
import numpy as np

# 🔹 Load models
transfer_model = joblib.load("models/xgb_transfer_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

club_model = joblib.load("models/xgb_club_model.pkl")
scaler = joblib.load("models/xgb_scaler.pkl")
label_encoder = joblib.load("models/xgb_label_encoder.pkl")

# 🔹 Load dataset with all players
data = pd.read_parquet("data/final_players_dataset.parquet")  # or CSV depending on your format

# 🔍 Step 1: Search player by name
search_term = input("🔎 Digite o nome do jogador: ").lower()
matches = data[data['name'].str.lower().str.contains(search_term)].reset_index(drop=True)

if matches.empty:
    print("Nenhum jogador encontrado.")
    exit()

print("\n🎯 Jogadores encontrados:")
for idx, row in matches.iterrows():
    print(f"{idx}. {row['name']} | Idade: {row['age']} | Clube atual: {row['from_club_name']}")

# Step 2: Select player
choice = int(input("\nDigite o número do jogador desejado: "))
player = matches.loc[choice]

# 🔄 Preprocess for transfer model
X_transfer = preprocessor.transform(pd.DataFrame([player]))
proba_transfer = transfer_model.predict_proba(X_transfer)[0][1]  # Probabilidade de sair

# 🔄 Prepare for club prediction model
features_club_model = [
    "age", "height_in_cm", "market_value_now", "market_value_mean", "market_value_growth",
    "goals", "assists", "minutes", "matches",
    "club_market_value", "club_squad_size", "club_avg_age",
    "club_foreigners_pct", "club_nat_players", "club_net_transfer_record"
]

X_club = pd.DataFrame([player])[features_club_model].copy()
X_club["club_net_transfer_record"] = X_club["club_net_transfer_record"].apply(
    lambda x: float(str(x).replace("€", "").replace("M", "000000").replace("K", "000")) if pd.notnull(x) else 0
)
X_club_scaled = scaler.transform(X_club)
probas_clubs = club_model.predict_proba(X_club_scaled)[0]

# 🎯 Top 5 clubes prováveis
top_indices = np.argsort(probas_clubs)[-5:][::-1]
top_clubs = label_encoder.inverse_transform(top_indices)
top_probs = probas_clubs[top_indices]

print(f"\n📈 Probabilidade de {player['name']} ser transferido: {proba_transfer*100:.2f}%")
print("\n🏟️ Principais clubes de destino:")
for club, prob in zip(top_clubs, top_probs):
    print(f" - {club}: {prob*100:.2f}%")
