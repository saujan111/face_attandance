import cv2
import os

# Ask for person's name (used as folder name)
name = input("Enter the name (no space): ").strip()
if not name:
    print("Name required!")
    exit()

folder = os.path.join("dataset", name)
os.makedirs(folder, exist_ok=True)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

count = len(os.listdir(folder))
print("Instructions: Press 's' to save face, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.putText(frame, "Press 's' to save, 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Captured Faces", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if len(faces) > 0:
            x, y, w, h = faces[0]   # take first detected face
            face_img = frame[y:y + h, x:x + w]
            file_path = os.path.join(folder, f"{name}_{count}.jpg")
            cv2.imwrite(file_path, face_img)
            print("Saved:", file_path)
            count += 1
        else:
            print("No face detected, move closer and try again.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
