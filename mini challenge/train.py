import cv2
import os
import pickle
import numpy as np
from PIL import Image
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras_vggface.vggface import VGGFace
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping


# Set random seed for numpy
np.random.seed(42)

# Set random seed for TensorFlow
tf.random.set_seed(42)



def prep_data(data_dir, data_crop_dir):
    print("Preparing data...")

    # set dimension of images
    image_width = 224
    image_height = 224

    # for detecting faces
    facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # iterates through all the files
    for filename in os.listdir(data_dir):
        # Load the image using OpenCV
        img_path = os.path.join(data_dir, filename)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        # OpenCV loads images in BGR format by default
        imgTrain = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        if imgTrain is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Convert BGR to RGB
        imgTrain = cv2.cvtColor(imgTrain, cv2.COLOR_BGR2RGB)

        # Convert the image array to numpy array of unsigned integers
        image_array = np.array(imgTrain, "uint8")

        # get the faces detected in the image
        faces = facecascade.detectMultiScale(imgTrain, scaleFactor=1.1, minNeighbors=100)

        # if not exactly 1 face is detected, skip this photo
        if len(faces) != 1:
            print(filename,'skipped')
            # skip this photo
            continue

        # save the detected face(s) and associate them with the label
        for (x_, y_, w, h) in faces:
            # draw the face detected
            face_detect = cv2.rectangle(imgTrain, (x_, y_), (x_+ w, y_+ h), (255, 0, 255), 2)
            
            # plt.imshow(face_detect)
            # plt.show()

            # resize the detected face 
            size = (image_width, image_height)

            # detected face region
            roi = image_array[y_: y_ + h, x_: x_ + w]

            # resize the detected head to target size
            resized_image = cv2.resize(roi, size)
            image_array = np.array(resized_image, "uint8")

            # replace the image with only the face
            im = Image.fromarray(image_array)
            cv2.imwrite(os.path.join(data_crop_dir, filename), cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))


    print("Data prepared.")

def link_label_with_image(labels_file, data_crop_dir):
    print("Linking data")

    # prep for csv read and load labels
    labels_df = pd.read_csv(labels_file)
    filenames = labels_df['File Name']
    labels = labels_df['Category']

    list_images = []
    list_labels = []

    # loop through dir and get each image file's name
    for name in os.listdir(data_crop_dir):
        #load image and save in list
        img_path = os.path.join(data_crop_dir, name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        list_images.append(img)

        # find the label for the specific file name
        row = labels_df[labels_df['File Name'] == name]
        if not row.empty:
            # Extract the category from the row
            category = row['Category'].iloc[0]
            # print(f"The category of {name} is: {category}")
            list_labels.append(category)
        else:
            print(f"No category found for {name}")
    
    # for i in range(len(list_images)):
    #     print(list_images[i],list_labels[i])
    print("data linked")
    return list_images, list_labels


def train(X_train, X_val, y_train, y_val,num_classes):
    print("Training model...")

    # Load VGGFace16 model without top layers
    base_model = VGGFace(include_top=False, model='vgg16', input_shape=(224, 224, 3))

    # Add custom layers on top of the base model
    #  if training loss decrease but val loss increase = overfitting.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dense(2048, activation='relu')(x)  
    x = Dense(1024, activation='relu')(x)  
    x = Dense(512, activation='relu')(x)  
    x = Dense(256, activation='relu')(x)  
    x = Dense(128, activation='relu')(x)  
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)

    
    preds = Dense(num_classes, activation='softmax')(x) 

    # Create the final model
    model = Model(inputs=base_model.input, outputs=preds)

    # Freeze the pre-trained layers
    for layer in model.layers[:19]:
        layer.trainable = False

    # Train the custom layers
    for layer in model.layers[19:]:
        layer.trainable = True

    # Compile the model
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size = 16,verbose = 1, epochs = 100, validation_data=(X_val, y_val), callbacks=[early_stop])

# -------------------------------------creates a HDF5 file-------------------------------------
    model.save('train_crop_50.h5')
# -------------------------------------creates a HDF5 file-------------------------------------

    # Print model summary
    model.summary()
    print("Training completed.")
    return model


# -------------------------------------setup-------------------------------------
data_dir = 'train'
labels_file = 'purdue-face-recognition-challenge-2024/train.csv'
data_crop_dir = 'train_crop_50'
# -------------------------------------setup-------------------------------------

# prep_data(data_dir, labels_file, data_crop_dir)
list_images, list_labels = link_label_with_image(labels_file, data_crop_dir)

# Encode labels using np.unique with return_inverse=True
_, y_encoded = np.unique(list_labels, return_inverse=True)

# Now y_encoded contains numerical representations of the labels
# print(y_encoded)
# sorted_list = sorted(y_encoded)
# print(sorted_list)
# print("\n\nlen =",len(np.unique(y_encoded)))

# Encode labels using one-hot encoding
y_encoded_one_hot = to_categorical(y_encoded, len(np.unique(y_encoded)))

# # Open a file for writing
# with open('y_encoded_one_hot.txt', 'w') as f:
#     # Loop through each row of the array
#     for row in y_encoded_one_hot:
#         # Convert each element of the row to string and join them with a space
#         line = ' '.join(map(str, row))
#         # Write the line to the file
#         f.write(line + '\n\n\n')

# Convert input data (images) to numpy arrays
X_np = np.array(list_images)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_np, y_encoded_one_hot, test_size=0.2, random_state=42)
# for i in range(len(X_val)):
#     print(X_val[i],y_val[i])
model = train(X_train, X_val, y_train, y_val, len(np.unique(y_encoded)))
# Evaluate the model on validation set
val_loss, val_acc = model.evaluate(X_val, y_val)
print("\n\n---------------------------------------------------\nValidation Accuracy:", val_acc)