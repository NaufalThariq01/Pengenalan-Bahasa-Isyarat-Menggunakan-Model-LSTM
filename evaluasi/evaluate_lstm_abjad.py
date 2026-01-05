import os
import numpy as np # type: ignore
import pickle
from collections import Counter
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# === PATH ===
fitur_dir = r"D:/Project Data Mining/fitur_xyz"
model_path = "model_lstm_abjad.h5"
label_encoder_path = "label_encoder_abjad.pkl"

# === LOAD MODEL & LABEL ENCODER ===
model = load_model(model_path)
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# === LOAD DATA ===
X = []
y = []

# Loop ke setiap folder huruf (label)
for label_folder in os.listdir(fitur_dir):
    folder_path = os.path.join(fitur_dir, label_folder)
    if not os.path.isdir(folder_path):
        continue

    label = label_folder.upper()
    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            file_path = os.path.join(folder_path, file)
            data = np.load(file_path)
            X.append(data)
            y.append(label)

# Validasi
print(f"✅ Total file dimuat: {len(X)}")
print(f"✅ Label unik: {set(y)}")

# === PADDING ===
X_padded = pad_sequences(X, padding="post", maxlen=100, dtype="float32")

# === ENCODE LABEL ===
y_encoded = label_encoder.transform(y)

# === PREDIKSI ===
y_probs = model.predict(X_padded)
y_pred = np.argmax(y_probs, axis=1)

# === DECODE KEMBALI LABEL ===
y_true_labels = label_encoder.inverse_transform(y_encoded)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# === AKURASI PER HURUF ===
print("\n📈 Akurasi per huruf:")
correct_per_label = Counter()
total_per_label = Counter()

for true, pred in zip(y_true_labels, y_pred_labels):
    total_per_label[true] += 1
    if true == pred:
        correct_per_label[true] += 1

for label in sorted(label_encoder.classes_):
    total = total_per_label[label]
    correct = correct_per_label[label]
    acc = correct / total if total > 0 else 0
    print(f" - {label}: {acc:.2%} ({correct}/{total})")

# === CLASSIFICATION REPORT ===
print("\n📊 Classification Report:")
print(classification_report(y_true_labels, y_pred_labels))

# === CONFUSION MATRIX ===
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=label_encoder.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Per Huruf)")
plt.tight_layout()
plt.show()
