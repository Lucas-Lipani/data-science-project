import joblib

def load_model(path):
    return joblib.load(path)

def predict_transfer(player_features, transfer_model):
    return transfer_model.predict_proba(player_features)[0][1]

def predict_club(club_features, club_model, scaler):
    club_features = scaler.transform(club_features)
    return club_model.predict(club_features)
