import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime 
import os
video_capture = cv2.VideoCapture(0)

from tqdm import tqdm 
 
known_face_encoding = []
known_faces_names = []


photos_folder = "photos"

total_images = len(os.listdir(photos_folder))
pbar = tqdm(total=total_images, desc="Loading Images and Encodings")

for filename in os.listdir(photos_folder):
    
    image = face_recognition.load_image_file(os.path.join(photos_folder, filename))

  
    encoding = face_recognition.face_encodings(image)[0]


    known_face_encoding.append(encoding)


    name_parts = filename.split("_")[0]


    known_faces_names.append(name_parts)


    pbar.update(1)

pbar.close()  

print("Images loaded and encodings generated successfully!")

# Add an empty string to the end of the names list
known_faces_names.append("loading...")
 
students = known_faces_names.copy()
 
face_locations = []
face_encodings = []
face_names = []
s=True
 
 
now = datetime.now()
current_date = now.strftime("%H:%M:%S")

filename = 'attendance.csv'
 


 
while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
 
            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (0,0,255)
                thickness              = 3
                lineType               = 2
                
                print(name)
 
                cv2.putText(frame,name+' Present', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)  
                
                with open(filename, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, current_date])
                    
                                 
                    
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
video_capture.release()
cv2.destroyAllWindows()
f.close()