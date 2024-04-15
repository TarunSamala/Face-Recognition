import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Load images and extract names, roll numbers, and emails from filenames
image_folder = "./Student Dataset/"
known_face_encodings = []
known_faces_names = []

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):
        name, roll_number, email = filename[:-4].split("_")
        image = face_recognition.load_image_file(os.path.join(image_folder, filename))
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_faces_names.append((name, roll_number, email))

students = known_faces_names.copy()


video_capture = cv2.VideoCapture(0)

now = datetime.now()
csv_filename = "./attendance.csv"
csv_file = open(csv_filename, 'a', newline='')
csv_writer = csv.writer(csv_file)

while True:
  
    _, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        face_names.append(name)

        if name in students:
            students.remove(name)
            print(students)
            csv_writer.writerow([name[0], name[1], name[2],datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name[0], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# Close CSV file
csv_file.close()
