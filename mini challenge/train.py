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
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
from tensorflow.keras.layers import Flatten


# Set random seed for numpy
np.random.seed(42)

# Set random seed for TensorFlow
tf.random.set_seed(42)


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
        # load image and save in list
        img_path = os.path.join(data_crop_dir, name)
        
        # Read image using PIL to avoid color profile warnings
        img = Image.open(img_path)
        
        # Convert image to RGB explicitly to ensure consistency
        img = img.convert('RGB')
        
        # Convert PIL Image to numpy array
        img = np.array(img)
        
        # Append image to list
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



    base_model = keras.applications.ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add custom layers on top of the base model
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    # x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    preds = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=preds)
    print("Created new model.")
    model.summary()


    # Compile the model
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size = 32,verbose = 1, epochs = 100, validation_data=(X_val, y_val), callbacks=[early_stop])

# -------------------------------------creates a HDF5 file-------------------------------------
    model.save('train_RES.h5')
# -------------------------------------creates a HDF5 file-------------------------------------

    # Print model summary
    print("Training completed.")
    return model


# -------------------------------------setup-------------------------------------
labels_file = 'purdue-face-recognition-challenge-2024/train.csv'
data_crop_dir = 'train_1H'
# -------------------------------------setup-------------------------------------

list_images, list_labels = link_label_with_image(labels_file, data_crop_dir)

# Encode labels using np.unique with return_inverse=True
_, y_encoded = np.unique(list_labels, return_inverse=True)

# Now y_encoded contains numerical representations of the labels
# print(y_encoded)
# sorted_list = sorted(y_encoded)
# print(sorted_list)
print("\n\n\n\nlen =",len(np.unique(y_encoded)))

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


