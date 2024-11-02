import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Declare directories path, categories and parameters
data_path = './src/mpox/resources'
categories = ['Cowpox', 'Healthy', 'HFMD', 'Measles', 'Chickenpox', 'Monkeypox']
IMG_SIZE = 64
NUM_CLASSES = len(categories)

# Load images from the resources folder
def load_resources():
    images = []
    image_names = []
    
    # Iterate through the images
    for image_name in os.listdir(data_path):
        if image_name.startswith('.'):
            continue

        if image_name.lower().endswith('.txt'):
            continue
            
        image_path = os.path.join(data_path, image_name)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Warning: Image {image_path} could not be read.")
            continue
        
        # Preprocess the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        images.append(image)
        image_names.append(image_name)
    
    # Convert images to a numpy array and normalize pixel values
    images = np.array(images).astype('float32') / 255.0
    return images, image_names

# Predict categories using the model
def predict_categories(model_path, images):
    model = load_model(model_path)
    print(f"Evaluating model: {model_path}")
    
    #Predict categories for each images
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels

# Read actual labels from the text file
def load_actual_labels(file_path):
    actual_labels = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

        # Skip the first line
        for line in lines[1:]:
            parts = line.strip().split(' - ')
            if len(parts) == 2:
                image_name, label = parts
                actual_labels[image_name.strip()] = label.strip()
    return actual_labels

# Main execution
if __name__ == "__main__":
    # Load images from the resources folder
    images, image_names = load_resources()
    print(f"Loaded {len(images)} images for evaluation.")

    # Load actual labels from text file
    answer_path = os.path.join(data_path, 'answer.txt')
    actual_labels = load_actual_labels(answer_path)
    
    final_model_path = 'my_final_model.keras'
    
    # Predict classes for the loaded images
    predicted_labels = predict_categories(final_model_path, images)

    # Count variables
    correct_count = 0
    incorrect_count = 0

    # Output the predicted classes alongside the image filenames
    for image_name, predicted_label in zip(image_names, predicted_labels):
        actual_label = actual_labels.get(image_name, "Unknown")
        predicted_label = categories[predicted_label]

        # Compare actual and predicted labels
        if actual_label == predicted_label:
            correct_count += 1
        else:
            incorrect_count += 1

        print(f"Image: {image_name}, Actual Class: {actual_label}, Predicted Class: {predicted_label}")

    # Output the counts of correct and incorrect predictions
    print(f"\nTotal Correct Predictions: {correct_count}")
    print(f"Total Incorrect Predictions: {incorrect_count}")
