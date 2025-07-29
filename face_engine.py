import face_recognition
import cv2
import os
import sys
import numpy as np
import time
import json
from collections import deque
from utils import load_known_faces, get_image_paths, draw_label, save_unknown_face

MEMORY_FILE = "face_memory.json"
TEMP_FOLDER = "temp_faces"

# ==========================
#  Setup
os.makedirs(TEMP_FOLDER, exist_ok=True)

# ==========================
#  Memory Functions
# ==========================
def load_face_memory():
    """Ngarkon memorien e fytyrave nga file JSON"""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
        return [(np.array(e["encoding"]), e["name"]) for e in data]
    return []

def save_face_memory(memory):
    """Ruajtja e memorjes së fytyrave në file JSON"""
    data = [{"encoding": enc.tolist(), "name": name} for enc, name in memory]
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f)

def add_to_memory(encoding, image):
    """Shton një fytyrë të re në memorien e përkohshme"""
    memory = load_face_memory()
    new_id = f"Person_{len(memory)+1}"
    path = os.path.join(TEMP_FOLDER, new_id)
    os.makedirs(path, exist_ok=True)
    
    filename = os.path.join(path, f"{int(time.time())}.jpg")
    cv2.imwrite(filename, image)
    
    memory.append((encoding, new_id))
    save_face_memory(memory)
    print(f"[🆕] U shtua në memorie: {new_id}")

def match_with_memory(encoding, tolerance=0.45):
    """Kontrollon nëse encoding ekziston në memorie"""
    memory = load_face_memory()
    if not memory:
        return None
    encodings = [m[0] for m in memory]
    names = [m[1] for m in memory]
    
    distances = face_recognition.face_distance(encodings, encoding)
    best_index = np.argmin(distances)
    if distances[best_index] < tolerance:
        return names[best_index]
    return None

# ==========================
#  HUD & Drawing
# ==========================
fps_history = deque(maxlen=50)

def rounded_rectangle(img, top_left, bottom_right, color, radius=10, thickness=-1):
    """Vizaton një drejtkëndësh me qoshe të rrumbullakosura"""
    x1, y1 = top_left
    x2, y2 = bottom_right
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

def draw_hud(frame, total_known, total_unknown, fps, elapsed_time):
    """Vizaton panelin HUD me statistika"""
    h, w = frame.shape[:2]
    panel_w = 220
    panel_h = 150
    x1, y1 = w - panel_w - 20, 20
    x2, y2 = w - 20, 20 + panel_h

    overlay = frame.copy()
    for i in range(panel_h):
        color = (int(20 + i * 0.3), int(80 + i * 0.4), int(150 + i * 0.2))
        cv2.line(overlay, (x1, y1 + i), (x2, y1 + i), color, 1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    rounded_rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), radius=15)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.5
    cv2.putText(frame, f"🟢 Known: {total_known}", (x1 + 15, y1 + 25), font, fs, (0, 255, 0), 1)
    cv2.putText(frame, f"🔴 Unknown: {total_unknown}", (x1 + 15, y1 + 50), font, fs, (0, 100, 255), 1)
    cv2.putText(frame, f"⚡ FPS: {fps:.1f}", (x1 + 15, y1 + 75), font, fs, (255, 255, 0), 1)
    runtime_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    cv2.putText(frame, f"⏳ Uptime: {runtime_str}", (x1 + 15, y1 + 100), font, fs, (200, 200, 200), 1)

# ==========================
#  Main Function
# ==========================
def start_face_recognition(
    image_folder="known_faces",
    threshold=0.45,
    camera_index=0,
    resize_width=None,
    save_unknown=True
):
    """Funksioni kryesor i njohjes së fytyrave"""
    image_paths = get_image_paths(image_folder)
    if not image_paths:
        print(f"❌ No images in '{image_folder}'.")
        sys.exit(1)

    known_encodings, known_names = load_known_faces(image_paths)
    video = cv2.VideoCapture(camera_index)
    if not video.isOpened():
        print(f"❌ Camera index {camera_index} not available.")
        return

    fps_queue = deque(maxlen=20)
    prev_time = time.time()
    start_time = time.time()
    total_known_faces = 0
    total_unknown_faces = 0
    seen_hashes = set()

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Rikonfigurim për madhësinë
            if resize_width and frame.shape[1] > resize_width:
                scale = resize_width / frame.shape[1]
                frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, face_locations)

            for (top, right, bottom, left), encoding in zip(face_locations, encodings):
                name = "Unknown"
                score = None

                # Kontrollo njohjet e njohura
                if known_encodings:
                    distances = face_recognition.face_distance(known_encodings, encoding)
                    best_index = np.argmin(distances)
                    score = distances[best_index]
                    if score < threshold:
                        name = known_names[best_index]
                        total_known_faces += 1

                # Kontrollo memorien e përkohshme
                if name == "Unknown":
                    mem_match = match_with_memory(encoding)
                    if mem_match:
                        name = mem_match
                        total_known_faces += 1
                    else:
                        total_unknown_faces += 1
                        if save_unknown:
                            face_img = frame[top:bottom, left:right]
                            add_to_memory(encoding, face_img)
                            save_unknown_face(frame, top, right, bottom, left, seen_hashes)

                #Vizato kutinë dhe etiketën
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                draw_label(frame, name, top, left, right, score, name != "Unknown")

            # Llogaritja e FPS dhe vizatimi i HUD
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            fps_queue.append(fps)
            fps_history.append(fps)
            prev_time = curr_time
            avg_fps = np.mean(fps_queue)

            # Vizato HUD
            draw_hud(frame, total_known_faces, total_unknown_faces, avg_fps, time.time() - start_time)
            cv2.imshow(f"Camera {camera_index} - Face Recognition HUD", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n👋 Interrupted by user.")

    video.release()
    cv2.destroyAllWindows()
