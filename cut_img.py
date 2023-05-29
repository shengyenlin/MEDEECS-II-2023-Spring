import sys

from PIL import Image

import os
import shutil

# Define the paths of the source directory containing the subdirectories and the destination directory
source_directory = "data"
destination_directory = "data_split"

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Iterate over the subdirectories in the source directory
for subdirectory_name in os.listdir(source_directory):
    subdirectory_path = os.path.join(source_directory, subdirectory_name)
    
    # Skip any non-directory items
    if not os.path.isdir(subdirectory_path):
        continue

    # Create the corresponding subdirectory in the destination directory
    destination_subdirectory = os.path.join(destination_directory, subdirectory_name)
    if not os.path.exists(destination_subdirectory):
        os.makedirs(destination_subdirectory)
    
    # Iterate over the files in the current subdirectory
    for file_name in os.listdir(subdirectory_path):
        file_path = os.path.join(subdirectory_path, file_name)
        
        # Skip any non-image files (you can customize the supported image extensions)
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            continue
        
        # Split the image into two sub-pictures with customized coordinates
        # Replace the 'customized_coordinates' with your actual logic for splitting the image
        
        # Example code to split the image in half vertically
        image = Image.open(file_path)
        width, height = image.size

        # Define the bounding box coordinates for the top and bottom halves
        # [xmin, ymin, xmax, ymax]
        top_box = (445, 80, 1540, 490)
        bottom_box = (445, 525, 1540, 935)

        # Extract the top and bottom halves using the defined bounding boxes
        top_half = image.crop(top_box)
        bottom_half = image.crop(bottom_box)
                
        # Define the new filenames with modified names
        top_half_filename = f"{os.path.splitext(file_name)[0]}_top.jpg"
        bottom_half_filename = f"{os.path.splitext(file_name)[0]}_down.jpg"
        
        # Move the split images to the destination subdirectory
        top_half.save(os.path.join(destination_subdirectory, top_half_filename))
        bottom_half.save(os.path.join(destination_subdirectory, bottom_half_filename))