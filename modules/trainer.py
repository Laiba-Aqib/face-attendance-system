# modules/trainer.py
import cv2
import numpy as np
import os
import json


def train_recognizer() -> bool:
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        print("ERROR: dataset/ folder not found.")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )

    face_images = []
    labels = []

    for folder_name in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        parts = folder_name.split('.')
        if len(parts) < 3 or parts[0] != 'user':
            continue

        try:
            user_id = int(parts[1])
            user_name = parts[2]
        except ValueError:
            continue

        print(f"Loading images for: {user_name} (ID: {user_id})")
        images_loaded = 0

        for image_file in os.listdir(folder_path):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            image_path = os.path.join(folder_path, image_file)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (200, 200):
                img = cv2.resize(img, (200, 200))

            face_images.append(img.astype(np.uint8))
            labels.append(user_id)
            images_loaded += 1

        print(f"  → Loaded {images_loaded} images")

    if len(face_images) == 0:
        print("ERROR: No training images found!")
        return False

    print(f"\nTotal training images: {len(face_images)}")
    labels_array = np.array(labels, dtype=np.int32)

    print("Training LBPH recognizer...")
    recognizer.train(face_images, labels_array)

    trainer_dir = "trainer"
    os.makedirs(trainer_dir, exist_ok=True)
    trainer_path = os.path.join(trainer_dir, "trainer.yml")
    recognizer.save(trainer_path)

    print(f"✓ Training complete! Model saved to: {trainer_path}")
    return True