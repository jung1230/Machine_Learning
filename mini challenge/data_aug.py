from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2

def load_and_preprocess_image(img_path):
    
    img = cv2.imread(img_path)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    img = cv2.resize(img, (224, 224))  
    
    img = img / 255.0
    return img


data_dir = 'train_crop'


augmented_dir = 'train_crop_augmented_images'
os.makedirs(augmented_dir, exist_ok=True)


datagen = ImageDataGenerator(
    rotation_range=20,      
    width_shift_range=0.1,  
    height_shift_range=0.1, 
    shear_range=0.2,        
    zoom_range=0.2,         
    horizontal_flip=True,   
    fill_mode='constant',   
    cval=0  
)


for filename in os.listdir(data_dir):
    img_path = os.path.join(data_dir, filename)
    img = load_and_preprocess_image(img_path)  
    img = img.reshape((1,) + img.shape)        

    
    i = 0
    for batch in datagen.flow(img, batch_size=1, save_to_dir=augmented_dir, save_prefix='aug_', save_format='jpg'):
        i += 1
        if i >= 5:  
            break