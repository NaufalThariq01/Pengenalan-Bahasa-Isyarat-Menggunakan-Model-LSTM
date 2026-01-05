import os
import numpy as np  # type: ignore
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore

# === PATH ===
fitur_dir = r"D:/Project Data Mining/fitur_xyz"
model_save_path = "model_lstm_abjad.h5"
label_encoder_save_path = "label_encoder_abjad.pkl"

# === TARGET LABELS ===
target_labels = ["A", "D", "Q", "V", "X"]

# === LOAD DATA ===
X = []
y = []

# Telusuri setiap folder label (A, D, Q, V, X)
for label_folder in os.listdir(fitur_dir):
    folder_path = os.path.join(fitur_dir, label_folder)
    if not os.path.isdir(folder_path):
        continue  # skip kalau bukan folder

    label = label_folder.upper()
    if label not in target_labels:
        continue  # skip label di luar target

    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            file_path = os.path.join(folder_path, file)
            data = np.load(file_path)
            X.append(data)
            y.append(label)

# === VALIDASI JUMLAH SAMPLE PER LABEL ===
label_counts = Counter(y)
print("\n📊 Jumlah sample per label:")
for label in target_labels:
    print(f" - {label}: {label_counts[label]}")

# === PAD SEQUENCE ===
X_padded = pad_sequences(X, maxlen=100, dtype="float32", padding="post")

# === ENCODE LABEL ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# === SPLIT TRAIN/VALIDASI ===
try:
    X_train, X_val, y_train, y_val = train_test_split(
        X_padded, y_categorical,
        test_size=0.2,
        stratify=y_categorical,
        random_state=42
    )
except ValueError as e:
    print(f"\n⚠️ Stratified split gagal: {e}")
    print("🔁 Menggunakan split tanpa stratify...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_padded, y_categorical,
        test_size=0.2,
        random_state=42
    )

# === DEFINISI MODEL ===
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_padded.shape[1], X_padded.shape[2])),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# === LATIH MODEL ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=8
)

# === SIMPAN MODEL DAN ENCODER ===
model.save(model_save_path)
with open(label_encoder_save_path, "wb") as f:
    pickle.dump(label_encoder, f)

print("\n✅ Model dan LabelEncoder berhasil disimpan.")
