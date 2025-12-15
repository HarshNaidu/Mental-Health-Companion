import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load trained NLP artifacts
vectorizer = joblib.load("models/Therapy/therapy_vectorizer.pkl")
tfidf_matrix = joblib.load("models/Therapy/therapy_tfidf.pkl")
texts = joblib.load("models/Therapy/therapy_texts.pkl")

# Therapy response bank (output control)
THERAPY_RESPONSES = {
    "sad": "It sounds like you're feeling low. Try grounding yourself with slow breathing or journaling your thoughts.",
    "depressed": "You're not alone. Consider reaching out to someone you trust and focusing on small self-care steps.",
    "anxious": "Anxiety can feel overwhelming. Try the 5-4-3-2-1 grounding technique to calm your mind.",
    "stressed": "Stress builds up quietly. Taking short breaks, stretching, and deep breathing may help.",
    "angry": "Strong emotions are valid. Physical movement or writing can help release tension.",
    "lonely": "Feeling lonely is hard. Even a small social interaction today can make a difference.",
    "default": "Take a moment to breathe. Consistent self-care can improve emotional wellbeing."
}

def recommend_therapy(user_input):
    user_input = user_input.lower()

    # Step 1: Direct keyword safety check
    for key in THERAPY_RESPONSES:
        if key in user_input:
            return THERAPY_RESPONSES[key]

    # Step 2: NLP similarity (intent detection)
    user_vec = vectorizer.transform([user_input])
    sims = cosine_similarity(user_vec, tfidf_matrix)
    best_match_idx = np.argmax(sims)

    matched_text = texts[best_match_idx].lower()

    # Step 3: Map NLP intent to therapy
    for key in THERAPY_RESPONSES:
        if key in matched_text:
            return THERAPY_RESPONSES[key]

    return THERAPY_RESPONSES["default"]
