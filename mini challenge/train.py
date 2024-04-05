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



np.random.seed(42)


tf.random.set_seed(42)


def link_label_with_image(labels_file, data_crop_dir):
    print("Linking data")

    
    labels_df = pd.read_csv(labels_file)
    filenames = labels_df['File Name']
    labels = labels_df['Category']

    list_images = []
    list_labels = []

    
    for name in os.listdir(data_crop_dir):
        
        img_path = os.path.join(data_crop_dir, name)
        
        
        img = Image.open(img_path)
        
        
        img = img.convert('RGB')
        
        
        img = np.array(img)
        
        
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



    base_model = keras.applications.ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    
    x = Dense(512, activation='relu')(x)
    preds = Dense(num_classes, activation='softmax')(x)

    
    model = Model(inputs=base_model.input, outputs=preds)
    print("Created new model.")
    model.summary()


    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size = 32,verbose = 1, epochs = 100, validation_data=(X_val, y_val), callbacks=[early_stop])


    model.save('train_RES.h5')


    
    print("Training completed.")
    return model



labels_file = 'purdue-face-recognition-challenge-2024/train.csv'
data_crop_dir = 'train_1H'


list_images, list_labels = link_label_with_image(labels_file, data_crop_dir)


_, y_encoded = np.unique(list_labels, return_inverse=True)





print("\n\n\n\nlen =",len(np.unique(y_encoded)))


y_encoded_one_hot = to_categorical(y_encoded, len(np.unique(y_encoded)))











X_np = np.array(list_images)


X_train, X_val, y_train, y_val = train_test_split(X_np, y_encoded_one_hot, test_size=0.2, random_state=42)


model = train(X_train, X_val, y_train, y_val, len(np.unique(y_encoded)))


