import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers
from keras.models import Model
from keras.layers import *
from keras.regularizers import *
from keras.optimizers import RMSprop


np.random.seed(42)
tf.random.set_seed(42)



bnEps = 2e-5
bnMom = 0.9
chanDim = 1

def resnet(layer_in, n_filters,s):
    data = layer_in
    stride = (s,s)
    
    
    merge_input = Conv2D(n_filters, (1,1),strides = (s,s))(layer_in)        
    bn2 = BatchNormalization(axis = chanDim,epsilon=bnEps,momentum = bnMom)(merge_input)
    act2 = Activation('relu')(bn2)
    
    
    conv2 = Conv2D(n_filters, (3,3),strides= stride,use_bias = False, padding='same', kernel_initializer='he_normal')(act2)  
    bn3 = BatchNormalization(axis = chanDim,epsilon=bnEps,momentum = bnMom)(conv2)
    act3 = Activation('relu')(bn3)
    
    
    conv3 = Conv2D(n_filters, (1,1),use_bias = False, kernel_initializer='he_normal')(act3)  
    
    
    data = Conv2D(n_filters,(1,1),padding = 'valid',strides = (s,s))(data) 
    data = BatchNormalization()(data)  
    
    layer_out = Add()([conv3, data])
    layer_out = Activation('relu')(layer_out)
    return layer_out

def train(train_generator, val_generator, num_classes):
    print("Training model...")
    
    
    visible = Input(shape=(224,224, 3))
    layer1 = resnet(visible,64,3)
    layer2 = resnet(layer1,128,1)
    layer4 = resnet(layer2,256,1)
    layer5 = resnet(layer4,256,2)
    layer6 = resnet(layer5,512,2)
    layer7 = resnet(layer6,512,2)
    layer8 = resnet(layer7,1024,2)
    layert = Dropout(0.5)(layer8)
    layer9 = resnet(layert,2048,2)
    layert2 = Dropout(0.5)(layer9)
    layer10 = resnet(layert2,4096,2)
    x = GlobalAveragePooling2D()(layer10)
    x = Dropout(0.7)(x)
    
    den = Dense(2048,activation = 'sigmoid')(x)
    final = Dense(5, activation="softmax")(den)
    
    model = Model(inputs=visible, outputs=final)

    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.0001)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    
    custom_lr = 0.01
    optimizerrr = Adam(learning_rate=custom_lr)
    
    model.compile(optimizer=optimizerrr, loss='categorical_crossentropy', metrics=['accuracy'], batch_size=32) 
    
    
    model.fit(train_generator, validation_data=val_generator, epochs=100, callbacks=[early_stop, reduce_lr])

    
    model.save('train_RES.h5')
    print("Training completed.")
    return model

def main():
    
    labels_file = 'purdue-face-recognition-challenge-2024/train.csv'
    data_dir = 'main_data'  
    val_dir = 'validation'  

    
    labels_df = pd.read_csv(labels_file)
    num_classes = len(labels_df['Category'].unique())

    
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
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
