import os
import pandas as pd
import numpy as np
import joblib  # CORRETO para carregar arquivos .pkl de ML

# Caminho base do projeto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS = os.path.join(BASE_DIR, 'saved')

# Carregar arquivos .pkl
def load_pkl(filename):
    path = os.path.join(SAVED_MODELS, filename)
    return joblib.load(path)

# Carregar modelos e scalers
transfer_model = load_pkl('xgb_transfer_model_transfer.pkl')
club_model = load_pkl('xgb_transfer_model_clubs.pkl')
scaler_transfer = load_pkl('preprocessor_transfer.pkl')
scaler_clubs = load_pkl('xgb_scaler_clubs.pkl')
encoder_clubs = load_pkl('xgb_label_encoder_clubs.pkl')

# Preparar dados do jogador
def prepare_features(player_id):
    players_df = pd.read_csv(os.path.join(BASE_DIR, '../data/players.csv'))

    player = players_df[players_df['player_id'] == player_id]

    if player.empty:
        print("Player não encontrado.")
        return None, None

    # Substituir pelas features reais do seu notebook
    features_transfer = {
        'age': player.iloc[0]['age'],
        'market_value': player.iloc[0]['market_value'],
    }

    features_transfer = pd.DataFrame([features_transfer])
    features_transfer_scaled = scaler_transfer.transform(features_transfer)
    features_clubs_scaled = scaler_clubs.transform(features_transfer)

    return features_transfer_scaled, features_clubs_scaled

# Predição
def predict(player_id):
    features_transfer, features_clubs = prepare_features(player_id)

    if features_transfer is None:
        return

    prob_transfer = transfer_model.predict_proba(features_transfer)[0][1] * 100
    club_preds = club_model.predict_proba(features_clubs)[0]
    top5_indexes = np.argsort(club_preds)[::-1][:5]
    top5_clubs = encoder_clubs.inverse_transform(top5_indexes)

    print(f'Probabilidade de Transferência: {prob_transfer:.2f}%')
    print('Clubes mais prováveis de destino:')
    for i, club in enumerate(top5_clubs, 1):
        print(f'{i}. {club}')

# Executar
if __name__ == "__main__":
    player_id = int(input("Digite o player_id: "))
    predict(player_id)
