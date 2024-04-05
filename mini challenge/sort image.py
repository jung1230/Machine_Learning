import os
import shutil
import pandas as pd

def sort_images(data_dir, csv_file):
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Create train and validation directories if they don't exist
    train_dir = 'train1'
    val_dir = 'validation1'
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    
    # Iterate through CSV rows and move images
    for index, row in df.iterrows():
        image_name = row['File Name']
        category = row['Category']
        
        # Check if the image exists
        image_path = os.path.join(data_dir, image_name)
        if os.path.exists(image_path):
            # Determine if it's a training or validation image (assuming 80% train, 20% validation)
            if index % 5 == 0:  # 1 out of 5 will go to validation
                dest_dir = os.path.join(val_dir, category)
            else:
                dest_dir = os.path.join(train_dir, category)
                
            # Create category directory if it doesn't exist
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            
            # Move image to destination directory
            shutil.copy(image_path, dest_dir)
            print(f"copy {image_name} to {dest_dir}")
        else:
            print(f"Image {image_name} not found!")

# Example usage
if __name__ == "__main__":
    data_directory = "train_crop_50"
    csv_file = "purdue-face-recognition-challenge-2024/train.csv"
    sort_images(data_directory, csv_file)
