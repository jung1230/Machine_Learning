import cv2
import os
import pickle
import numpy as np
from PIL import Image
import pandas as pd

import matplotlib.pyplot as plt


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
        img_path =  os.path.join(data_dir, filename)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        imgTrain = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if imgTrain is None:
            # Create a white image
            white_image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
            
            # Save the white image
            uncropped_img_path = os.path.join(data_crop_dir, filename)
            cv2.imwrite(uncropped_img_path, white_image)
            continue

        # Convert BGR to RGB
        imgTrain = cv2.cvtColor(imgTrain, cv2.COLOR_BGR2RGB)

        # Convert the image array to numpy array of unsigned integers
        image_array = np.array(imgTrain, "uint8")

# --------------------------------------- get the faces detected in the image ---------------------------------------
        faces = facecascade.detectMultiScale(imgTrain, scaleFactor=1.1, minNeighbors=25)
# --------------------------------------- get the faces detected in the image ---------------------------------------

        # if not exactly 1 face is detected
        if len(faces) != 1:
            print(filename,'not exactly 1 face is detected')
            uncropped_img_path = os.path.join(data_crop_dir, filename)
            cv2.imwrite(uncropped_img_path, cv2.cvtColor(imgTrain, cv2.COLOR_RGB2BGR))
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








# --------------------------------------- change this ---------------------------------------
data_dir = 'test'
data_crop_dir = 'test_crop_50'
# --------------------------------------- change this ---------------------------------------

prep_data(data_dir, data_crop_dir)

# data_dir = 'train'
# labels_file = 'purdue-face-recognition-challenge-2024/train.csv'
# data_crop_dir = 'train_crop'

# prep_data(data_dir, labels_file, data_crop_dir)

