import os
import shutil

def split_images(source_dir, dest_dir1, dest_dir2, dest_dir3):
    # Create destination directories if they don't exist
    os.makedirs(dest_dir1, exist_ok=True)
    os.makedirs(dest_dir2, exist_ok=True)
    os.makedirs(dest_dir3, exist_ok=True)

    # Get the list of image files and sort them numerically
    images = sorted([f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))],
                    key=lambda x: int(''.join(filter(str.isdigit, x))))

    # Determine how many images to move to each directory
    count = len(images) // 3

    # Copy the images to the destination directories
    for i, image in enumerate(images):
        source_path = os.path.join(source_dir, image)
        if i < count:
            dest_path = os.path.join(dest_dir1, image)
        elif  i < count*2:
            dest_path = os.path.join(dest_dir2, image)
        else:
            dest_path = os.path.join(dest_dir3, image)
        shutil.copy(source_path, dest_path)

if __name__ == "__main__":
    # Directory containing the images is assumed to be in the same directory as the script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    source_directory = os.path.join(script_directory, "train_crop_with_normal")
    destination_directory1 = os.path.join(script_directory, "train_1H")
    destination_directory2 = os.path.join(script_directory, "train_2H")
    destination_directory3 = os.path.join(script_directory, "train_3H")

    split_images(source_directory, destination_directory1, destination_directory2, destination_directory3)
    print("Images copied successfully!")
