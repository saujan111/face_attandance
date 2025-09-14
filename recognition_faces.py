import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime

 
ENCODINGS_PATH = "encodings.npy"
NAMES_PATH = "names.npy"
ATTENDANCE_FILE = "attendance.csv"
SNAPSHOT_FOLDER = "snapshots"
 
if not os.path.exists(ENCODINGS_PATH) or not os.path.exists(NAMES_PATH):
    print("[ERROR] No encodings found. Run encode_faces.py first!")
    exit()

known_encodings = np.load(ENCODINGS_PATH, allow_pickle=True)
known_names = np.load(NAMES_PATH, allow_pickle=True)

print(f"[INFO] Loaded {len(known_encodings)} face encodings.")

 
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)
 
if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["Name", "Time", "Image"])
    df.to_csv(ATTENDANCE_FILE, index=False)

 
attendance = pd.read_csv(ATTENDANCE_FILE)

 
cap = cv2.VideoCapture(0)
print("Press 'q' to quit. Recognized people will be marked in attendance.csv")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            best_match = np.argmin(face_recognition.face_distance(known_encodings, face_encoding))
            name = known_names[best_match]

           
            if name not in attendance["Name"].values:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

               
                top, right, bottom, left = face_location
                face_img = frame[top:bottom, left:right]
                img_path = os.path.join(SNAPSHOT_FOLDER, f"{name}_{now.replace(':', '-')}.jpg")
                cv2.imwrite(img_path, face_img)
 
                new_entry = {"Name": name, "Time": now, "Image": img_path}
                attendance = pd.concat([attendance, pd.DataFrame([new_entry])], ignore_index=True)
                attendance.to_csv(ATTENDANCE_FILE, index=False)

                print(f"[INFO] Attendance marked for {name}")

       
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
