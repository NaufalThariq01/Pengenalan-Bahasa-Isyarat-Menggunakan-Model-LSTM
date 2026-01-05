import os
import cv2 # type: ignore
import numpy as np # type: ignore
import mediapipe as mp # type: ignore
import pickle
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# === KONFIGURASI PATH ===
video_path = "input.mp4"  # Ubah jika nama videomu berbeda
model_path = "model_lstm_abjad.h5"
label_encoder_path = "label_encoder_abjad.pkl"

# === INISIALISASI MEDIAPIPE ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# === LOAD MODEL DAN LABEL ENCODER ===
print("📦 Loading model dan label encoder...")
model = load_model(model_path)
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# === FUNGSI EKSTRAKSI KOORDINAT DARI VIDEO ===
def extract_xyz_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    landmarks_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            coords = []
            for lm in results.multi_hand_landmarks[0].landmark:
                coords.extend([lm.x, lm.y, lm.z])
            landmarks_sequence.append(coords)
        else:
            landmarks_sequence.append([0] * 63)  # Jika tangan tidak terdeteksi

    cap.release()

    # Hapus frame kosong jika semua nol
    filtered = [f for f in landmarks_sequence if sum(f) != 0]
    return np.array(filtered if filtered else landmarks_sequence, dtype=np.float32)

# === EKSTRAKSI DAN PREDIKSI ===
print(f"🎬 Mengekstrak koordinat dari {video_path} ...")
xyz_sequence = extract_xyz_from_video(video_path)
print(f"🧩 Total frame terdeteksi: {len(xyz_sequence)}")

# Padding
padded_sequence = pad_sequences([xyz_sequence], maxlen=100, padding="post", truncating="post", dtype="float32")

# Prediksi
print("🤖 Melakukan prediksi gesture...")
pred_probs = model.predict(padded_sequence)
pred_index = np.argmax(pred_probs[0])
pred_label = label_encoder.inverse_transform([pred_index])[0]

# Hasil
print(f"\n✅ Prediksi gesture dari video: **{pred_label}**")
print("\n📊 Probabilitas semua huruf:")
for i, prob in enumerate(pred_probs[0]):
    print(f" - {label_encoder.classes_[i]}: {prob:.4f}")
