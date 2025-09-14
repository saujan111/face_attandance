import face_recognition
import cv2
import os
import numpy as np

DATASET_DIR = "dataset"
ENCODINGS_PATH = "encodings.npy"
NAMES_PATH = "names.npy"

known_encodings = []
known_names = []

print("[INFO] Encoding faces...")

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")

        encodings = face_recognition.face_encodings(rgb, boxes)
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person)

print(f"[INFO] Encoded {len(known_encodings)} faces.")

# Save encodings and names
np.save(ENCODINGS_PATH, known_encodings)
np.save(NAMES_PATH, known_names)

print("[INFO] Encodings saved to", ENCODINGS_PATH, "and", NAMES_PATH)
