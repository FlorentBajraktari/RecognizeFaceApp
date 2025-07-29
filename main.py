# main.py
from face_engine import start_face_recognition

if __name__ == "__main__":
    print("\n📷 Face Recognition duke u nisur...")
    print("""
    ℹ️ Taste aktive:
      [q]  → Dil nga programi
    """)

    # ⚙️ Konfigurimi
    FOLDER = "known_faces"   # ku i ke fotot e njohura
    THRESHOLD = 0.45         # ndjeshmëria e njohjes
    CAMERA_INDEX = 0         # kamera kryesore (0 = default)
    RESIZE_WIDTH = 800      # ose vendos p.sh. 800 për shpejtësi
    SAVE_UNKNOWN = True      # ruaj fytyrat e panjohura

    start_face_recognition(
        image_folder=FOLDER,
        threshold=THRESHOLD,
        camera_index=CAMERA_INDEX,
        resize_width=RESIZE_WIDTH,
        save_unknown=SAVE_UNKNOWN
    )
    print("✅ Duke dalur...")