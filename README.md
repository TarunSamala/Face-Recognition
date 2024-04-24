# Face Recognition Attendance System

This project is a face recognition-based attendance system developed in Python using OpenCV and face_recognition libraries. It allows for automatic marking of attendance by recognizing faces of registered students captured through a webcam.

## Features

- Automatic detection and recognition of faces using webcam
- Marking attendance for recognized students
- Saving attendance records to a CSV file
- Simple and intuitive interface

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/face-recognition-attendance-system.git
```
## Instal the Required  libraries:

```bash
pip install -r requirements.txt
```

if the installation doesnt work properly then direct to dlib-master folder and go through the README.md or ReadMeFirst.docx
## Navigate to the Script file:

```bash
cd Recognition
```

## Usage

1. Prepare the dataset:
    - Add images of students' faces to the "photos" folder. Each image file should be named in the format: Name_RollNo_Email.jpg.
2. Run the application:
```bash
python Attendance.py
```

## Configuration

- Modify the Recognition.py file to adjust parameters such as camera index, recognition threshold, etc., as needed.
- Ensure that the CSV file for storing attendance records is writable by the application.

# There's a guide Attached to this Repository 

The guid is for cmd/command prompt approach, the guide doesnt work for vscode or any other code editor approach.



