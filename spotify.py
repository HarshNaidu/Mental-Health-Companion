import joblib
import random

kmeans = joblib.load("models/Spotify/spotify_kmeans.pkl")
scaler = joblib.load("models/Spotify/spotify_scaler.pkl")

emotion_map = {
    "Happy": 1,
    "Sad": 2,
    "Angry": 3,
    "Neutral": 0,
    "Fear": 4,
    "Surprise": 5,
    "Disgust": 6
}

songs_db = [
    "Let It Be – Beatles",
    "Fix You – Coldplay",
    "Someone Like You – Adele",
    "Happy – Pharrell Williams",
    "Believer – Imagine Dragons",
    "Blinding Lights – Weeknd",
]

def recommend_songs(emotion):
    return random.sample(songs_db, 3)
