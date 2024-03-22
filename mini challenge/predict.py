import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from keras_vggface.utils import preprocess_input as preprocess_input_vggface
from keras_vggface.vggface import VGGFace
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd
from tensorflow.keras.utils import to_categorical
import re


def predict_image(image_path, model,list4label,index_one_hot, output_csv):

    # Get list of filenames sorted in a natural order
    filenames = sorted(os.listdir(image_path), key=lambda x: int(re.findall(r'\d+', x)[0]))

    predictions = []

    for name in filenames:
        # Load and preprocess the image
        img = cv2.imread(os.path.join(image_path, name))
        if img is None:
            # Strip file extension from the filename
            name = os.path.splitext(name)[0]

            # Append prediction to list
            predictions.append((name, ""))
            continue


        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  # Resize image to match model input size

        # Convert image to float32
        img = img.astype('float32')

        img = preprocess_input_vggface(img)
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Perform prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)

        # Strip file extension from the filename
        name = os.path.splitext(name)[0]

        # Append prediction to list
        predictions.append((name, list4label[predicted_class]))

        # Get the confidence score (probability) of the predicted class
        confidence_score = prediction[0][predicted_class] * 100

        # # Display the image with predicted class
        # img = cv2.cvtColor(img.squeeze(), cv2.COLOR_BGR2RGB)
        # plt.imshow(img.squeeze())
        # plt.title(name + "Predicted class: " + list4label[predicted_class] + ", Confidence: {:.2f}%".format(confidence_score))
        # plt.axis('off')
        # plt.show()

        # Save predictions to CSV
        df = pd.DataFrame(predictions, columns=['Id', 'Category'])
        df.to_csv(output_csv, index=False)

def main():
# ------------------------------Load trained model------------------------------
    model = load_model('mini.h5')
# ------------------------------Load trained model------------------------------

    # Load label names
    # prep for csv read and load labels
    labels_file = 'purdue-face-recognition-challenge-2024/train_small.csv'
    labels_df = pd.read_csv(labels_file)
    filenames = labels_df['File Name']
    labels = labels_df['Category']
    list_labels = []

    data_crop_dir = 'train_small_crop'
    # loop through dir and get each image file's name
    for name in os.listdir(data_crop_dir):
        # find the label for the specific file name
        row = labels_df[labels_df['File Name'] == name]
        if not row.empty:
            # Extract the category from the row
            category = row['Category'].iloc[0]
            # print(f"The category of {name} is: {category}")
            list_labels.append(category)
        else:
            print(f"No category found for {name}")
    
    
    # Encode labels using np.unique with return_inverse=True
    list4label, index = np.unique(list_labels, return_inverse=True)
    index_one_hot = to_categorical(index, len(index))

# ------------------------------Predict on a new image------------------------------
    image_path = 'train_small'
    output_csv = 'predictions.csv'
# ------------------------------Predict on a new image------------------------------

    predict_image(image_path, model, list4label,index_one_hot, output_csv)

if __name__ == "__main__":
    main()
