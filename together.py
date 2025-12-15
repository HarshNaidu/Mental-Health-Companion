import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, request, jsonify
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from env
import joblib

# ---------------- CONFIG ----------------
OpenAI.api_key = "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"
spotify_kmeans = joblib.load("models/Spotify/spotify_kmeans.pkl")
spotify_scaler = joblib.load("models/Spotify/spotify_scaler.pkl")
app = Flask(__name__)

# ---------------- LOAD MODELS ----------------
emotion_model = tf.keras.models.load_model("models/mobile_net_v2_firstmodel.h5")
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Neutral", "Neutral"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Spotify
spotify_kmeans = joblib.load("models/Spotify/spotify_kmeans.pkl")
spotify_scaler = joblib.load("models/Spotify/spotify_scaler.pkl")

# Therapy
therapy_vectorizer = joblib.load("models/Therapy/therapy_vectorizer.pkl")
therapy_tfidf = joblib.load("models/Therapy/therapy_tfidf.pkl")
therapy_texts = joblib.load("models/Therapy/therapy_texts.pkl")

cap = cv2.VideoCapture(0)
current_emotion = "Neutral"
current_confidence = 0.0

# ---------------- CAMERA STREAM ----------------
def gen_frames():
    global current_emotion, current_confidence

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            preds = emotion_model.predict(face, verbose=0)[0]
            idx = np.argmax(preds)

            current_emotion = emotion_labels[idx]
            current_confidence = float(preds[idx])

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv2.putText(
                frame,
                f"{current_emotion} ({current_confidence:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 255),
                2
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

# ---------------- RECOMMENDERS ----------------
def recommend_songs(emotion):
    """
    Emotion → cluster → song mood mapping
    """
    emotion_map = {
        "Happy": 0,
        "Neutral": 1,
        "Sad": 2,
        "Angry": 3,
        "Fear": 4,
        "Disgust": 5
    }

    # Fallback safety
    if emotion not in emotion_map:
        return ["Play something calming 🌿"]

    # Fake feature vector from emotion (simple but valid)
    emotion_vector = np.zeros((1, len(emotion_map)))
    emotion_vector[0, emotion_map[emotion]] = 1

    scaled = spotify_scaler.transform(emotion_vector)
    cluster = spotify_kmeans.predict(scaled)[0]

    # Human-friendly recommendations per cluster
    cluster_songs = {
        0: ["Happy – Pharrell Williams", "Good Life – OneRepublic"],
        1: ["Let It Be – The Beatles", "Fix You – Coldplay"],
        2: ["Someone Like You – Adele", "Say Something – A Great Big World"],
        3: ["Believer – Imagine Dragons", "Stronger – Kanye West"],
        4: ["Lovely – Billie Eilish", "Breathe Me – Sia"],
        5: ["Weightless – Marconi Union", "River – Leon Bridges"]
    }

    return cluster_songs.get(cluster, ["Lo-fi Chill Beats 🎧"])

THERAPY_MAP = {
    "Happy": [
        "Practice gratitude journaling",
        "Engage in creative activities",
        "Share positive moments with others"
    ],
    "Neutral": [
        "Mindful breathing for 2 minutes",
        "Light physical activity",
        "Body scan relaxation"
    ],
    "Sad": [
        "Guided breathing exercise",
        "Journaling your thoughts",
        "Reach out to a trusted person"
    ],
    "Angry": [
        "Progressive muscle relaxation",
        "Box breathing technique",
        "Physical activity to release tension"
    ],
    "Fear": [
        "Grounding using the 5-4-3-2-1 method",
        "Slow diaphragmatic breathing",
        "Positive affirmations"
    ],
    "Disgust": [
        "Mindfulness meditation",
        "Cognitive reframing exercise",
        "Relaxation imagery"
    ]
}

import random

def recommend_therapy(emotion):
    if emotion not in THERAPY_MAP:
        emotion = "Neutral"

    return random.choice(THERAPY_MAP[emotion])

# ---------------- CHATBOT ----------------

def therapist_reply(user_text, emotion):
    prompt = f"""
You are a calm, supportive mental health assistant.

User emotion (detected): {emotion}
User message: {user_text}

Respond empathetically.
Suggest a short coping technique if appropriate.
Do NOT repeat the user's message.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a compassionate AI therapist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("together.html")

@app.route("/video")
def video():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data["message"]

    reply = therapist_reply(user_message, current_emotion)
    songs = recommend_songs(current_emotion)
    therapy = recommend_therapy(user_message)

    return jsonify({
        "reply": reply,
        "emotion": current_emotion,
        "confidence": round(current_confidence * 100, 1),
        "songs": songs,
        "therapy": therapy
    })

if __name__ == "__main__":
    app.run(debug=True)
