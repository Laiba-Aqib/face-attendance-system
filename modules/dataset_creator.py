# modules/dataset_creator.py
import cv2
import os
import json
import time


def create_dataset(user_id: int, user_name: str, num_samples: int = 30) -> bool:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    if face_cascade.empty():
        print("ERROR: Could not load Haar Cascade file")
        return False

    dataset_dir = os.path.join("dataset", f"user.{user_id}.{user_name}")
    os.makedirs(dataset_dir, exist_ok=True)

    # Save to names.json
    names_file = "names.json"
    if os.path.exists(names_file):
        with open(names_file, 'r') as f:
            content = f.read().strip()
        names = json.loads(content) if content else {}
    else:
        names = {}
    names[str(user_id)] = user_name
    with open(names_file, 'w') as f:
        json.dump(names, f, indent=2)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Cannot access webcam")
        return False

    sample_count = 0
    last_capture = 0
    capture_interval = 0.5

    print(f"\nRegistering: {user_name} (ID: {user_id})")
    print(f"Please look at the camera. Slowly move your head slightly.")
    print(f"Capturing {num_samples} images...")

    while sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed, skipping...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )

        display_frame = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(display_frame, f"Captured: {sample_count}/{num_samples}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Student: {user_name}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Dataset Capture - Press 'q' to cancel", display_frame)

        current_time = time.time()
        if len(faces) == 1 and current_time - last_capture >= capture_interval:
            x, y, w, h = faces[0]
            margin = int(0.1 * w)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)

            face_crop = gray_eq[y1:y2, x1:x2]
            face_resized = cv2.resize(face_crop, (200, 200),
                                      interpolation=cv2.INTER_AREA)

            sample_count += 1
            filename = f"user.{user_id}.{sample_count}.jpg"
            filepath = os.path.join(dataset_dir, filename)
            cv2.imwrite(filepath, face_resized)
            last_capture = current_time

            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 255, 0), 4)
            cv2.putText(display_frame, f"SAVED #{sample_count}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("Dataset Capture - Press 'q' to cancel", display_frame)
            cv2.waitKey(100)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return False

    print(f"\n✓ Successfully captured {num_samples} images for {user_name}")
    cap.release()
    cv2.destroyAllWindows()
    return True