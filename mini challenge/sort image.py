import os
import shutil
import pandas as pd

def sort_images(data_dir, csv_file):
    
    df = pd.read_csv(csv_file)
    
    
    train_dir = 'train1'
    val_dir = 'validation1'
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    
    
    for index, row in df.iterrows():
        image_name = row['File Name']
        category = row['Category']
        
        
        image_path = os.path.join(data_dir, image_name)
        if os.path.exists(image_path):
            
            if index % 5 == 0:  
                dest_dir = os.path.join(val_dir, category)
            else:
                dest_dir = os.path.join(train_dir, category)
                
            
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            
            
            shutil.copy(image_path, dest_dir)
            print(f"copy {image_name} to {dest_dir}")
        else:
            print(f"Image {image_name} not found!")


if __name__ == "__main__":
    data_directory = "train_crop_50"
    csv_file = "purdue-face-recognition-challenge-2024/train.csv"
    sort_images(data_directory, csv_file)
