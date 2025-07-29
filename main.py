from face_engine import start_face_recognition

if __name__ == "__main__":
    print("\n📷 Face Recognition duke u nisur...")
    print("""
     Taster aktive:
      [q]  → Dil nga programi
    """)

    # Configuration
    FOLDER = "known_faces"   
    THRESHOLD = 0.45         
    CAMERA_INDEX = 0        
    RESIZE_WIDTH = 800      
    SAVE_UNKNOWN = True      

    start_face_recognition(
        image_folder=FOLDER,
        threshold=THRESHOLD,
        camera_index=CAMERA_INDEX,
        resize_width=RESIZE_WIDTH,
        save_unknown=SAVE_UNKNOWN
    )
    print("✅ Duke dalur...")