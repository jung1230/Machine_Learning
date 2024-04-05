import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import  preprocess_input
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from keras_vggface.vggface import VGGFace
from tensorflow.keras.models import Model


np.random.seed(42)
tf.random.set_seed(42)



def train(train_generator, val_generator, num_classes):
    print("Training model...")

    
    base_model = VGGFace(include_top=False, model='vgg16', input_shape=(224, 224, 3))

    
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)

    
    preds = Dense(num_classes, activation='softmax')(x) 

    
    model = Model(inputs=base_model.input, outputs=preds)

    
    for layer in model.layers[:19]:
        layer.trainable = False

    
    for layer in model.layers[19:]:
        layer.trainable = True

    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=0.00001,verbose = 1, restore_best_weights=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    
    custom_lr = 0.01
    optimizer = Adam(learning_rate=custom_lr)

    model.summary()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, validation_data=val_generator, batch_size = 128,verbose = 1, epochs = 1000, callbacks=[early_stop, reduce_lr])




    
    model.save('train_VGG.h5')
    print("Training completed.")
    
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')  
    return base_model

def main():
    
    labels_file = 'purdue-face-recognition-challenge-2024/train.csv'
    data_dir = 'main_data'  
    val_dir = 'validation'  

    
    labels_df = pd.read_csv(labels_file)
    num_classes = len(labels_df['Category'].unique())

    
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(224, 224),
        batch_size=128, 
        class_mode='categorical')

    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(data_dir, val_dir),
        target_size=(224, 224),
        batch_size=128, 
        class_mode='categorical')

    
    model = train(train_generator, val_generator, num_classes)


    
    


if __name__ == "__main__":
    main()
