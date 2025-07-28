# main.py
import argparse
from face_engine import start_face_recognition

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live face recognition from webcam using known faces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--folder', type=str, default='known_faces', help="Folder with known face images.")
    parser.add_argument('--threshold', type=float, default=0.45, help="Face match threshold.")
    parser.add_argument('--no-fps', action='store_true', help="Disable FPS display.")
    parser.add_argument('--show-score', action='store_true', help="Show face match distance score.")
    parser.add_argument('--camera', type=int, default=0, help="Webcam index (default 0).")
    parser.add_argument('--recursive', action='store_true', help="Recursively search for images in subfolders.")
    parser.add_argument('--resize-width', type=int, default=None, help="Resize webcam frame to this width (for high-res cameras).")
    parser.add_argument('--save-unknown', action='store_true', help="Save unknown faces to disk.")

    args = parser.parse_args()

    start_face_recognition(
        image_folder=args.folder,
        threshold=args.threshold,
        show_fps=not args.no_fps,
        show_score=args.show_score,
        camera_index=args.camera,
        recursive=args.recursive,
        resize_width=args.resize_width,
        save_unknown=args.save_unknown
    )
