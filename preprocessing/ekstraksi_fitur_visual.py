import os
import cv2  
import numpy as np  
from tqdm import tqdm  
import mediapipe as mp  

frames_base_dir = r"D:\Project Data Mining\frames"
output_base_dir = r"D:\Project Data Mining\fitur_xyz"
os.makedirs(output_base_dir, exist_ok=True)

# === Inisialisasi MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# === Fungsi Ekstraksi Koordinat dari Frame ===
def extract_xyz_from_frames(frame_folder):
    image_files = sorted(
        [f for f in os.listdir(frame_folder) if f.lower().endswith((".jpg", ".png"))]
    )
    all_landmarks = []

    for img_name in image_files:
        img_path = os.path.join(frame_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (640, 480))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            all_landmarks.append(coords)
        else:
            all_landmarks.append([0.0] * 63)

    if len(all_landmarks) == 0:
        return np.zeros((0, 63), dtype=np.float32)

    return np.array(all_landmarks, dtype=np.float32)

# === Looping Per Kategori dan Video ===
for kategori in os.listdir(frames_base_dir):
    kategori_path = os.path.join(frames_base_dir, kategori)
    if not os.path.isdir(kategori_path):
        continue

    output_kategori_dir = os.path.join(output_base_dir, kategori)
    os.makedirs(output_kategori_dir, exist_ok=True)

    print(f"\n📂 Memproses kategori: {kategori}")
    for video_folder in tqdm(os.listdir(kategori_path)):
        video_frame_dir = os.path.join(kategori_path, video_folder)
        if not os.path.isdir(video_frame_dir):
            continue

        xyz_array = extract_xyz_from_frames(video_frame_dir)
        output_file = os.path.join(output_kategori_dir, f"{video_folder}.npy")
        np.save(output_file, xyz_array)

print("\n✅ Ekstraksi fitur X, Y, Z selesai. Semua hasil disimpan ke .npy")
