import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import cv2
from sklearn.metrics import classification_report

# Directories
data_path = './src/mpox/mpox-skin-lesion-dataset-version-20-msld-v20/versions/4'
IMG_SIZE = 64
categories = ['Cowpox', 'Healthy', 'HFMD', 'Measles', 'Chickenpox', 'Monkeypox']
NUM_CLASSES = len(categories)

# Label dictionary
label_dict = {category: idx for idx, category in enumerate(categories)}

# Load test data
def load_test_data():
    test_images, test_labels = [], []
    # Interate each fold directory
    for fold in range(1, 6):
        test_dir = os.path.join(data_path, f'Original Images/Original Images/FOLDS/fold{fold}/Test')

        if not os.path.exists(test_dir):
            print(f"Warning: Directory {test_dir} does not exist.")
            continue

        for category in categories:
            category_folder = os.path.join(test_dir, category)
            if not os.path.exists(category_folder):
                print(f"Warning: Folder {category_folder} does not exist.")
                continue
            for image_name in os.listdir(category_folder):
                if image_name.startswith('.'):
                    continue
                image_path = os.path.join(category_folder, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Image {image_path} could not be read.")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                test_images.append(image)
                test_labels.append(label_dict[category])
                
    test_images = np.array(test_images).astype('float32') / 255.0  # Normalize
    test_labels = to_categorical(test_labels, num_classes=NUM_CLASSES)
    return test_images, test_labels

# Test model function
def test_model(model_path, test_images, test_labels):
    # Load the model
    model = load_model(model_path)
    print(f"Evaluating model: {model_path}")

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    return model  # Return the loaded model for further evaluation

# Evaluation function for predictions
def evaluate_predictions(model, test_images, test_labels):
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)

    # Classification Report
    report = classification_report(true_classes, predicted_classes, target_names=categories)
    print("Classification Report:\n", report)

# Load test data once for the final model
test_images, test_labels = load_test_data()
print(f"Loaded {len(test_images)} test images.")

# Path to the final model
final_model_path = 'my_final_model.keras'

# Evaluate only the final model on the test set
model = test_model(final_model_path, test_images, test_labels)
evaluate_predictions(model, test_images, test_labels)
