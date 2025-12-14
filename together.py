from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import time
import re
from keras.models import load_model
from openai import OpenAI

# ------------------ APP SETUP ------------------
app = Flask(__name__, static_url_path='/static')
client = OpenAI()

# ------------------ CAMERA ------------------
cam = cv2.VideoCapture(0)

# ------------------ MODELS ------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
emotion_model = load_model('mobile_net_v2_firstmodel.h5')

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Surprise", "Sad", "Neutral"]
max_emotion = "Neutral"

# ------------------ EMOTION PREDICTION ------------------
def predict_emotion(face_image):
    face_image = cv2.imdecode(np.frombuffer(face_image, np.uint8), cv2.IMREAD_COLOR)
    face_image = cv2.resize(face_image, (224, 224))
    face_image = np.expand_dims(face_image, axis=0) / 255.0

    preds = emotion_model.predict(face_image, verbose=0)
    return emotion_labels[np.argmax(preds)]

# ------------------ VIDEO STREAM ------------------
def detection():
    global max_emotion
    face_images = []
    capture_interval = 1
    last_capture = time.time()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            roi = frame[y:y+h, x:x+w]

            if time.time() - last_capture >= capture_interval:
                face_images.append(cv2.imencode('.png', roi)[1].tobytes())
                face_images = face_images[-5:]
                last_capture = time.time()

            if len(face_images) >= 3:
                counts = {e: 0 for e in emotion_labels}
                for img in face_images:
                    counts[predict_emotion(img)] += 1
                max_emotion = max(counts, key=counts.get)

            display_emotion = "Neutral" if max_emotion == "Surprise" else max_emotion
            cv2.putText(frame, display_emotion, (100, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.png', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ------------------ CHATBOT ------------------
def build_prompt(message, mood):
    mood_map = {
        "Happy": "The user is happy. Respond positively and warmly.",
        "Sad": "The user is sad. Be empathetic and comforting.",
        "Angry": "The user is angry. Stay calm and grounding.",
        "Fear": "The user is anxious. Reassure gently.",
        "Neutral": "Be friendly and supportive."
    }

    mood_context = mood_map.get(mood, "Be friendly and supportive.")

    return f"""
You are a caring AI therapist.
{mood_context}

User says: "{message}"

Reply in 60–80 words.
Ask one gentle follow-up question.
"""

def bot_answer(message, mood):
    if not mood:
        mood = "Neutral"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": build_prompt(message, mood)}],
        temperature=0.6
    )
    return response.choices[0].message.content.strip()

# ------------------ ROUTES ------------------
@app.route('/')
def home():
    return render_template('together.html')

@app.route('/video')
def video():
    return Response(detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message', '')
    reply = bot_answer(user_msg, max_emotion)
    return jsonify({'bot_message': reply})

# ------------------ MAIN ------------------
if __name__ == '__main__':
    app.run(debug=True)
