import os
from PIL import Image
import numpy as np
import cv2
import os

def resize_images_in_folder(input_folder, output_folder, target_size=(128, 128)):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each file in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Check if the file is an image
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            with Image.open(file_path) as img:
                # Convert to grayscale if not already
                img = img.convert('L')

                # Resize the image
                img_resized = img.resize(target_size)

                # Save the resized image to the output folder
                output_path = os.path.join(output_folder, filename)
                img_resized.save(output_path)