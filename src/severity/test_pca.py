import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from sklearn.decomposition import PCA

# Constants
IMAGE_SIZE = 128
BATCH_SIZE = 16
TEST_DIR = './src/severity/test_data'

# Load model
model = load_model('PCA_mpox_severity_final_model.keras')

# Class labels
class_labels = ['macule', 'papule', 'pustule', 'scab', 'vesicles']

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.flatten()

    # Apply PCA
    pca = PCA(n_components=50)
    features_reduction = pca.fit_transform(image)
    
    return features_reduction


def predict_image(image_path):
    image = preprocess_image(image_path)
    
    if image is None:
        print(f"Error: Can't read image {image_path}")
        return
    
    try:
        # Make prediction
        predictions = model.predict(image, verbose=0)
        prediction_result = np.argmax(predictions[0])
        confidence = predictions[0][prediction_result] * 100
        
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Predicted Class: {class_labels[prediction_result]}")
        print(predictions)
        print(f"Confidence: {confidence:.2f}%\n")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")


def main():
    for image_name in os.listdir(TEST_DIR):
        image_path = os.path.join(TEST_DIR, image_name)
        print(f"Processing {image_path} ...")
        predict_image(image_path)


if __name__ == '__main__':
    main()
