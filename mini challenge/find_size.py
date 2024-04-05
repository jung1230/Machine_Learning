import os
import cv2

def get_image_dimensions(directory):
    min_width, min_height = float('inf'), float('inf')
    max_width, max_height = 0, 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                filepath = os.path.join(root, file)
                img = cv2.imread(filepath)
                if img is None:
                    # print(f"Unable to read image: {filepath}")
                    continue
                height, width, _ = img.shape
                min_width = min(min_width, width)
                min_height = min(min_height, height)
                max_width = max(max_width, width)
                max_height = max(max_height, height)

    return min_width, min_height, max_width, max_height

if __name__ == "__main__":
    data_dir = 'main_data/train'  # Change this to your main data directory
    min_width, min_height, max_width, max_height = get_image_dimensions(data_dir)
    print("Minimum Image Dimensions (Width x Height):", min_width, "x", min_height)
    print("Maximum Image Dimensions (Width x Height):", max_width, "x", max_height)
    data_dir = 'main_data/validation'  # Change this to your main data directory
    min_width, min_height, max_width, max_height = get_image_dimensions(data_dir)
    print("Minimum Image Dimensions (Width x Height):", min_width, "x", min_height)
    print("Maximum Image Dimensions (Width x Height):", max_width, "x", max_height)
