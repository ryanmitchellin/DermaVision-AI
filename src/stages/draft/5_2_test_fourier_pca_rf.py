import numpy as np
import pandas as pd
import cv2
import os
import joblib
from sklearn.decomposition import PCA

# Constants
IMAGE_SIZE = 128
TEST_DIR = './src/stages/test_data'
CATEGORIES = ['macule', 'papule', 'pustule', 'scab', 'vesicles']

# Load Models
fourier_pca = joblib.load("./src/stages/draft/4_fourier_pca_model.pkl")
model = joblib.load("./src/stages/draft/5_fourier_pca_rf_stages.pkl")

def preprocess_image(image_path):
    # Load Images
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Preprocess
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image_flat = image.flatten()
    image_fourier_pca = fourier_pca.transform([image_flat])
    
    return image_fourier_pca


def predict_image(image_path):
    # Preprocess
    image_pca = preprocess_image(image_path)
    
    # Make Prediction
    prediction_result = model.predict(image_pca)[0]
    prediction_proba = model.predict_proba(image_pca)[0]
    
    predicted_class = CATEGORIES[prediction_result]
    confidence = prediction_proba[prediction_result] * 100
    
    print(f"Image: {image_path}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%\n")


def main():
    for image_name in os.listdir(TEST_DIR):
        image_path = os.path.join(TEST_DIR, image_name)
        print(f"Processing {image_path} ...")
        predict_image(image_path)


if __name__ == '__main__':
    main()
