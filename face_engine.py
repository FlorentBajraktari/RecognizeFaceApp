# face_engine.py
import face_recognition
import cv2
import os
import sys
import numpy as np
import time
from collections import deque
from datetime import datetime
from utils import load_known_faces, get_image_paths, draw_label, save_unknown_face

def start_face_recognition(
    image_folder,
    threshold=0.45,
    show_fps=True,
    show_score=False,
    camera_index=0,
    recursive=False,
    resize_width=None,
    save_unknown=False
):
    image_paths = get_image_paths(image_folder, recursive=recursive)
    if not image_paths:
        print(f"\u274c No images found in '{image_folder}'. Supported: jpg, jpeg, png.")
        sys.exit(1)

    known_encodings, known_names = load_known_faces(image_paths)
    if not known_encodings:
        print("\u274c No valid faces loaded. Exiting.")
        sys.exit(1)

    video = cv2.VideoCapture(camera_index)
    if not video.isOpened():
        print(f"\u274c Camera index {camera_index} not available. Trying index 0...")
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            print("\u274c No accessible camera found.")
            sys.exit(1)

    print("\u2705 Face Recognition Started (press 'q' or Ctrl+C to quit)")
    fps_queue = deque(maxlen=20)
    prev_time = time.time()
    seen_unknown_hashes = set()

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                print("\u274c Failed to read frame.")
                break

            if resize_width and frame.shape[1] > resize_width:
                scale = resize_width / frame.shape[1]
                frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, encodings):
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                name = "Unknown"
                score = None
                is_known = False

                if len(distances) > 0:
                    best_index = np.argmin(distances)
                    score = distances[best_index]
                    if score < threshold:
                        name = known_names[best_index]
                        is_known = True
                        print(f"\u2705 Recognized: {name} (score: {score:.2f})")
                    elif save_unknown:
                        save_unknown_face(frame, top, right, bottom, left, seen_unknown_hashes)

                color = (0, 255, 0) if is_known else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                draw_label(frame, name, top, left, right, score if show_score else None, is_known)

            if show_fps and len(fps_queue) >= 5:
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time)
                fps_queue.append(fps)
                prev_time = curr_time
                avg_fps = np.mean(fps_queue)
                cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                prev_time = time.time()

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\ud83d\udc4b Exiting.")
                break
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user.")

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python face_engine.py <image_folder>")
        sys.exit(1)
    image_folder = sys.argv[1]
    start_face_recognition(image_folder)