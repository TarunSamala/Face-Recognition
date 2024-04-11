import tensorflow as tf
from keras import layers, models
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def load_dataset(dataset_path, images_folder):
    dataset = pd.read_csv(dataset_path)
    images = []
    labels = []

    # Encode string labels to numerical values
    label_encoder = LabelEncoder()
    dataset['label'] = label_encoder.fit_transform(dataset['label'])

    for index, row in dataset.iterrows():
        image_path = os.path.join(images_folder, row['id'])
        label = row['label']
        
        # Read and preprocess the image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))  # Resize the image to 128x128
        image = image.astype('float32') / 255.0  # Normalize pixel values
        
        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)

input_shape = (128, 128, 3)  # Adjust input size if necessary

# Get the number of unique classes from the dataset
dataset = pd.read_csv('Dataset/Dataset.csv')
num_classes = len(dataset['label'].unique())

model = create_model(input_shape, num_classes)

# Path to the dataset folders
faces_folder = 'Dataset/Faces'
dataset_path = 'Dataset/Dataset.csv'

# Load the dataset
train_images, train_labels = load_dataset(dataset_path, faces_folder)

# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

model.save('Model/face_recognition_model')



