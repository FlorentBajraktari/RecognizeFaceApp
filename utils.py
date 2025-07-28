# utils.py
import os
from glob import glob
import cv2
import face_recognition
from datetime import datetime

def load_known_faces(image_paths):
    encodings, names = [], []
    seen = set()
    for path in image_paths:
        if not os.path.exists(path):
            print(f"\u26a0\ufe0f File not found: {path}")
            continue
        image = face_recognition.load_image_file(path)
        face_encs = face_recognition.face_encodings(image)
        if face_encs:
            base = os.path.splitext(os.path.basename(path))[0]
            for i, enc in enumerate(face_encs):
                name = f"{base}_{i+1}" if len(face_encs) > 1 else base
                if name not in seen:
                    encodings.append(enc)
                    names.append(name)
                    seen.add(name)
        else:
            print(f"\u26a0\ufe0f No face found in: {path}")
    return encodings, names

def get_image_paths(folder, recursive=False):
    exts = ('*.jpg', '*.jpeg', '*.png')
    paths = []
    for ext in exts:
        pattern = os.path.join(folder, '**', ext) if recursive else os.path.join(folder, ext)
        paths.extend(glob(pattern, recursive=recursive))
    return paths

def draw_label(frame, name, top, left, right, score=None, is_known=True):
    label = f"{name}"
    if score is not None:
        label += f" ({score:.2f})"
    color = (0, 255, 0) if is_known else (0, 0, 255)
    bg_color = (0, 0, 0)
    (w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (left, top - 28), (left + w + 6, top - 4), bg_color, -1)
    cv2.putText(frame, label, (left + 3, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def save_unknown_face(frame, top, right, bottom, left, seen_hashes=None, output_dir="unknown_faces"):
    os.makedirs(output_dir, exist_ok=True)
    face_img = frame[top:bottom, left:right]
    face_hash = hash(face_img.tobytes())
    if seen_hashes is not None and face_hash in seen_hashes:
        return
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    cv2.imwrite(os.path.join(output_dir, filename), face_img)
    if seen_hashes is not None:
        seen_hashes.add(face_hash)
