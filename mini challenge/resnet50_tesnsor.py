import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt



np.random.seed(42)
tf.random.set_seed(42)

def train(train_generator, val_generator, num_classes):
    print("Training model...")

    
    base_model = ResNet18(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    
    x = base_model.output
    x = Flatten()(x) 
    
    
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.0625)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.0625)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.0625)(x)
    preds = Dense(num_classes, activation='softmax')(x)

    
    model = tf.keras.models.Model(inputs=base_model.input, outputs=preds)
    print("Created new model.")

    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=0.001,verbose = 1, restore_best_weights=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
    
    custom_lr = 0.01
    optimizer = Adam(learning_rate=custom_lr)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    model.fit(train_generator, validation_data=val_generator, epochs=1000,verbose = 1, callbacks=[early_stop, reduce_lr])

    
    model.save('/content/drive/MyDrive/train_RES.h5')
    model.save('/content/train_RES.h5')
    print("Training completed.")
    
    plt.figure(figsize=(10, 5))
    plt.plot(model.history.history['loss'], label='Training Loss')
    plt.plot(model.history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('/content/drive/MyDrive/loss_plot.png')  
    return model

def main():
    
    labels_file = '/content/drive/MyDrive/train.csv'
    data_dir = '/content/main_data'  
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
        batch_size=32, 
        class_mode='categorical')

    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(data_dir, val_dir),
        target_size=(224, 224),
        batch_size=32, 
        class_mode='categorical')

    
    model = train(train_generator, val_generator, num_classes)


    
    


if __name__ == "__main__":
    main()