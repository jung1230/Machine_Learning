from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2

def load_and_preprocess_image(img_path):
    # Load image using OpenCV
    img = cv2.imread(img_path)
    # Convert image to RGB (OpenCV loads images in BGR format by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Perform any additional preprocessing if needed
    # For example, you can resize the image to a specific size
    img = cv2.resize(img, (224, 224))  # Resize to (224, 224)
    # Normalize pixel values to range [0, 1]
    img = img / 255.0
    return img

# Directory containing your face photos
data_dir = 'train_crop'

# Create a directory to store augmented images
augmented_dir = 'train_crop_augmented_images'
os.makedirs(augmented_dir, exist_ok=True)

# Create an ImageDataGenerator instance for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,      # Random rotation in the range [-20, 20] degrees
    width_shift_range=0.1,  # Random horizontal shift by 10% of the width
    height_shift_range=0.1, # Random vertical shift by 10% of the height
    shear_range=0.2,        # Shear intensity (angle in radians)
    zoom_range=0.2,         # Random zoom in the range [0.8, 1.2]
    horizontal_flip=True,   # Randomly flip images horizontally
    fill_mode='constant',   # Fill in newly created pixels with a constant value
    cval=0  
)

# Iterate through each face photo in the directory and apply data augmentation
for filename in os.listdir(data_dir):
    img_path = os.path.join(data_dir, filename)
    img = load_and_preprocess_image(img_path)  # Load and preprocess your image (you need to define this function)
    img = img.reshape((1,) + img.shape)        # Reshape to (1, height, width, channels) for flow method

    # Generate augmented images and save them to the augmented directory
    i = 0
    for batch in datagen.flow(img, batch_size=1, save_to_dir=augmented_dir, save_prefix='aug_', save_format='jpg'):
        i += 1
        if i >= 5:  # Generate 5 augmented images for each original image
            break