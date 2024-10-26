# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20")

# print("Path to dataset files:", path)

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set directories
data_path = './src/mpox/mpox-skin-lesion-dataset-version-20-msld-v20/versions/4'
train_dir = os.path.join(data_path, 'Augmented Images/Augmented Images/FOLDS_AUG/fold5_AUG/Train')

# Define image size
IMG_SIZE = 150 

# List of categories (subdirectories) in the train directory
categories = [category for category in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, category))]

# Create a label dictionary to map category names to integer labels
label_dict = {category: idx for idx, category in enumerate(categories)}

# Initialize lists for images and labels
train_images = []
train_labels = []

# Load images
for category in categories:
    image_folder = os.path.join(train_dir, category)  # Path to the category folder
    for image_name in os.listdir(image_folder):
        # Skip hidden files
        if image_name.startswith('.'):
            continue
        
        image_path = os.path.join(image_folder, image_name)
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Image {image_path} could not be read.")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize the image to the target size
            train_images.append(image)
            train_labels.append(label_dict[category])  # Get the label from label_dict
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

# Convert lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

print(f"Loaded {len(train_images)} images with {len(categories)} categories.")
print("Categories:", categories)

val_dir = os.path.join(data_path, 'Original Images/Original Images/FOLDS/fold5/Valid')

# Define image size
IMG_SIZE = 150

# List of categories (subdirectories) in the train directory
categories = [category for category in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, category))]

# Create a label dictionary to map category names to integer labels
label_dict = {category: idx for idx, category in enumerate(categories)}

# Initialize lists for images and labels
val_images = []
val_labels = []

# Load images
for category in categories:
    image_folder = os.path.join(val_dir, category)  # Path to the category folder
    for image_name in os.listdir(image_folder):
        # Skip hidden files
        if image_name.startswith('.'):
            continue
        
        image_path = os.path.join(image_folder, image_name)
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Image {image_path} could not be read.")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize the image to the target size
            val_images.append(image)
            val_labels.append(label_dict[category])  # Get the label from label_dict
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

# Convert lists to numpy arrays
val_images = np.array(val_images)
val_labels = np.array(val_labels)

print(f"Loaded {len(val_images)} images with {len(categories)} categories.")
print("Categories:", categories)

test_dir = os.path.join(data_path, 'Original Images/Original Images/FOLDS/fold5/Test')

# Define image size
IMG_SIZE = 150

# List of categories (subdirectories) in the train directory
categories = [category for category in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, category))]

# Create a label dictionary to map category names to integer labels
label_dict = {category: idx for idx, category in enumerate(categories)}

# Initialize lists for images and labels
test_images = []
test_labels = []

# Load images
for category in categories:
    image_folder = os.path.join(test_dir, category)  # Path to the category folder
    for image_name in os.listdir(image_folder):
        # Skip hidden files
        if image_name.startswith('.'):
            continue
        
        image_path = os.path.join(image_folder, image_name)
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Image {image_path} could not be read.")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize the image to the target size
            test_images.append(image)
            test_labels.append(label_dict[category])  # Get the label from label_dict
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

# Convert lists to numpy arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

print(f"Loaded {len(test_images)} images with {len(categories)} categories.")
print("Categories:", categories)


# Normalize the pixel values to be between 0 and 1
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0

# Convert the labels to one-hot encoded vectors
train_labels = to_categorical(train_labels, num_classes=len(categories))
val_labels = to_categorical(val_labels, num_classes=len(categories))
test_labels = to_categorical(test_labels, num_classes=len(categories))