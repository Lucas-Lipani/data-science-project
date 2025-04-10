import pandas as pd
import joblib

def load_expected_features(path):
    return joblib.load(path)

def prepare_club_features(club_data: pd.DataFrame) -> pd.DataFrame:
    expected_cols = load_expected_features('../../models/saved/club_features_cols.pkl')
    club_data = club_data[expected_cols]
    return club_data

def prepare_player_features(player_data: pd.DataFrame) -> pd.DataFrame:
    expected_cols = load_expected_features('../../models/saved/player_features_cols.pkl')
    player_data = player_data[expected_cols]
    return player_data
