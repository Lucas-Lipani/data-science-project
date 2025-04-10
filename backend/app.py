from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os
from utils.preprocess import prepare_club_features, prepare_player_features
from utils.predict import load_model, predict_transfer, predict_club

app = Flask(__name__)

# Define base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'saved')

# Load models and scalers
transfer_model = load_model(os.path.join(MODEL_PATH, 'rf_transfer_model.pkl'))
club_model = load_model(os.path.join(MODEL_PATH, 'club_model.pkl'))
scaler = load_model(os.path.join(MODEL_PATH, 'scaler.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_transfer', methods=['POST'])
def predict_transfer_route():
    data = request.get_json()

    player_df = pd.DataFrame([data['player_features']])
    club_df = pd.DataFrame([data['club_features']])

    player_df = prepare_player_features(player_df)
    club_df = prepare_club_features(club_df)

    transfer_prob = predict_transfer(player_df, transfer_model)
    club_prediction = predict_club(club_df, club_model, scaler)

    return jsonify({
        'transfer_probability': transfer_prob,
        'predicted_club': club_prediction[0]
    })

if __name__ == '__main__':
    app.run(debug=True)
