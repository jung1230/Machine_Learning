import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms,models
from PIL import Image
import matplotlib.pyplot as plt
import re

def predict_celebrities(model_path, image_dir, output_csv,unique_labels):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    predictions = []
    confidence_scores = []
    image_files = os.listdir(image_dir)

    def extract_numeric_part(filename):
        return int(re.search(r'\d+', filename).group())
    image_files = sorted(os.listdir(image_dir), key=extract_numeric_part)

    for image_file in image_files:
        print(image_file,end="")
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        image = data_transform(image).unsqueeze(0)  
        with torch.no_grad():
            output = model(image)
        probabilities = torch.softmax(output, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, 0)
        predictions.append(unique_labels[predicted_class.item()])  
        confidence_scores.append(confidence.item())

        plt.imshow(image.squeeze().permute(1, 2, 0))
        plt.title(f'Predicted: {unique_labels[predicted_class.item()]}, Confidence: {confidence.item()}')
        plt.show()

    predictions_df = pd.DataFrame({'Id': range(len(image_files)), 'Category': predictions})
    predictions_df.to_csv(output_csv, index=False)

model_path = 'train_RES_main_data50.pth'  
image_dir = 'test_crop_50'  
output_csv = 'predictions_resnet18_wiithout_dense.csv'  

labels_df = pd.read_csv('purdue-face-recognition-challenge-2024/train.csv')
unique_labels = sorted(labels_df['Category'].unique())

predict_celebrities(model_path, image_dir, output_csv,unique_labels)
