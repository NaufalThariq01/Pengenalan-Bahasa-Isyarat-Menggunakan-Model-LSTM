import os
import cv2  
import numpy as np  
from tqdm import tqdm  

# === KONFIGURASI PATH ===
base_dir = r"D:\Project Data Mining"
dataset_dir = os.path.join(base_dir, "Dataset_baru")
output_frames_dir = os.path.join(base_dir, "frames")
os.makedirs(output_frames_dir, exist_ok=True)

# === Fungsi Ekstraksi Frame berdasarkan Gerakan (Motion Detection) ===
def extract_motion_frames(video_path, output_folder, min_area=500, frame_skip=2, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    saved_frame = 0
    prev_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 🔄 Rotasi jika perlu (landscape ke portrait)
        if frame.shape[1] > frame.shape[0]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if frame_id < frame_skip:
            frame_id += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is None:
            prev_gray = gray
            frame_id += 1
            continue

        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        motion_score = cv2.countNonZero(thresh)

        if motion_score > min_area:
            resized = cv2.resize(frame, (224, 224))
            output_path = os.path.join(output_folder, f"frame_{saved_frame:04d}.jpg")
            cv2.imwrite(output_path, resized)
            saved_frame += 1

        prev_gray = gray
        frame_id += 1

        if saved_frame >= max_frames:
            break

    cap.release()
    return saved_frame

# === Looping Person dan Huruf ===
for person in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_path):
        continue

    for huruf in os.listdir(person_path):
        huruf_path = os.path.join(person_path, huruf)
        if not os.path.isdir(huruf_path):
            continue

        output_huruf_dir = os.path.join(output_frames_dir, huruf)
        os.makedirs(output_huruf_dir, exist_ok=True)

        print(f"\n👤 Memproses {person} - Huruf {huruf}")
        for video_file in tqdm(os.listdir(huruf_path)):
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                continue

            video_path = os.path.join(huruf_path, video_file)
            video_name = os.path.splitext(video_file)[0]
            output_video_dir = os.path.join(output_huruf_dir, f"{person}_{video_name}")
            os.makedirs(output_video_dir, exist_ok=True)

            saved = extract_motion_frames(video_path, output_video_dir)
            print(f"📸 {video_file} → {saved} frame disimpan")

print("\n✅ Ekstraksi semua frame selesai.")
