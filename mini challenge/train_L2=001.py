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


np.random.seed(42)


tf.random.set_seed(42)



def prep_data(data_dir, data_crop_dir):
    print("Preparing data...")

    
    image_width = 224
    image_height = 224

    
    facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    
    for filename in os.listdir(data_dir):
        
        img_path = os.path.join(data_dir, filename)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        
        imgTrain = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        if imgTrain is None:
            print(f"Failed to load image: {img_path}")
            continue

        
        imgTrain = cv2.cvtColor(imgTrain, cv2.COLOR_BGR2RGB)

        
        image_array = np.array(imgTrain, "uint8")

        
        faces = facecascade.detectMultiScale(imgTrain, scaleFactor=1.1, minNeighbors=100)

        
        if len(faces) != 1:
            print(filename,'skipped')
            
            continue

        
        for (x_, y_, w, h) in faces:
            
            face_detect = cv2.rectangle(imgTrain, (x_, y_), (x_+ w, y_+ h), (255, 0, 255), 2)
            
            
            

            
            size = (image_width, image_height)

            
            roi = image_array[y_: y_ + h, x_: x_ + w]

            
            resized_image = cv2.resize(roi, size)
            image_array = np.array(resized_image, "uint8")

            
            im = Image.fromarray(image_array)
            cv2.imwrite(os.path.join(data_crop_dir, filename), cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))


    print("Data prepared.")

def link_label_with_image(labels_file, data_crop_dir):
    print("Linking data")

    
    labels_df = pd.read_csv(labels_file)
    filenames = labels_df['File Name']
    labels = labels_df['Category']

    list_images = []
    list_labels = []

    
    for name in os.listdir(data_crop_dir):
        
        img_path = os.path.join(data_crop_dir, name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        list_images.append(img)

        
        row = labels_df[labels_df['File Name'] == name]
        if not row.empty:
            
            category = row['Category'].iloc[0]
            
            list_labels.append(category)
        else:
            print(f"No category found for {name}")
    
    
    
    print("data linked")
    return list_images, list_labels


def train(X_train, X_val, y_train, y_val,num_classes):
    print("Training model...")

    
    base_model = VGGFace(include_top=False, model='vgg16', input_shape=(224, 224, 3))

    
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu', kernel_regularizer=l2(0.01))(x)  
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)  
    x = Dense(512, activation='relu')(x)  
    x = Dense(256, activation='relu')(x)  
    x = Dense(128, activation='relu')(x)  

    
    preds = Dense(num_classes, activation='softmax')(x) 

    
    model = Model(inputs=base_model.input, outputs=preds)

    
    for layer in model.layers[:19]:
        layer.trainable = False

    
    for layer in model.layers[19:]:
        layer.trainable = True

    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size = 16,verbose = 1, epochs = 100, validation_data=(X_val, y_val))


    model.save('normal001.h5')


    
    model.summary()
    print("Training completed.")
    return model



data_dir = 'train'
labels_file = 'purdue-face-recognition-challenge-2024/train.csv'
data_crop_dir = 'train_crop'



list_images, list_labels = link_label_with_image(labels_file, data_crop_dir)


_, y_encoded = np.unique(list_labels, return_inverse=True)








y_encoded_one_hot = to_categorical(y_encoded, len(np.unique(y_encoded)))











X_np = np.array(list_images)


X_train, X_val, y_train, y_val = train_test_split(X_np, y_encoded_one_hot, test_size=0.2, random_state=42)


model = train(X_train, X_val, y_train, y_val, len(np.unique(y_encoded)))

val_loss, val_acc = model.evaluate(X_val, y_val)
print("\n\n---------------------------------------------------\nValidation Accuracy:", val_acc)