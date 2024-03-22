import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


# Load and preprocess data
def load_data(data_dir, labels_file):
    print("Loading data...")
    # Load labels
    labels_df = pd.read_csv(labels_file)
    filenames = labels_df['File Name']
    labels = labels_df['Category']

    print("Number of images:", len(filenames))
    print("Number of labels:", len(labels))
    print("Number of unique labels:", len(np.unique(labels)))


    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Load images and preprocess
    images = []
    for filename in filenames:
        img = load_img(os.path.join(data_dir, filename), target_size=(224, 224), color_mode='rgb')  # Resize image as per your model input size
        img_array = img_to_array(img)
        images.append(img_array)

    images = np.array(images)

    print("Data loaded.")
    return images, labels

# Define CNN model
def create_model(input_shape, num_classes):
    print("Creating model...")
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    print("Model created.")

    return model

# Train the model
def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=20):
    print("Training model...")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    
    print("Training completed.")
    return model, history

# Load data
data_dir = 'train_small'
labels_file = 'purdue-face-recognition-challenge-2024/train_small.csv'
images, labels = load_data(data_dir, labels_file)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create and compile the model
input_shape = (224, 224, 3)  # Input shape as per your resized image dimensions
num_classes = len(np.unique(labels))
print("num_classes =",num_classes )
model = create_model(input_shape, num_classes)

# Train the model
model, history = train_model(model, X_train, y_train, X_val, y_val)

# Evaluate the model on validation set
val_loss, val_acc = model.evaluate(X_val, y_val)
print("Validation Accuracy:", val_acc)

# Save the model
model.save('face_recognition_model.h5')

# Make predictions on test set and generate submission file (you need to implement this part)
# test_images = load_test_data('path/to/test')
# predictions = model.predict(test_images)
# Save predictions to submission file
