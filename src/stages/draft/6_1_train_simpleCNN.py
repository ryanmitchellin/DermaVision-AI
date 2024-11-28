import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold

# Constants
DATA_DIR = './src/stages/augmented_data'
CATEGORIES = ['macule', 'papule', 'pustule', 'scab', 'vesicles']
IMG_SIZE = 64
NUM_CLASSES = len(CATEGORIES)
EPOCHS = 5

# Load all images and labels from the dataset
def load_all_images():
    # Map category names to numeric labels
    label_dict = {category: i for i, category in enumerate(CATEGORIES)}

    images, labels = [], []
    for category in CATEGORIES:
        category_folder = os.path.join(DATA_DIR, category)
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

            # Preprocess the image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = image.astype('float32') / 255.0  # Normalize
            images.append(image)
            labels.append(label_dict[category])
    return np.array(images), np.array(labels)

# CNN model for image classification
def create_cnn_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Load Data
    images, labels = load_all_images()

    # Set up 5-fold cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    best_accuracy = 0

    # Perform cross-validation
    for fold_index, (train_idx, val_idx) in enumerate(kf.split(images, labels)):
        print(f"\nProcessing fold {fold_index + 1}")

        # Split data into training and validation sets for this fold
        X_train, X_val = images[train_idx], images[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
        y_val = to_categorical(y_val, num_classes=NUM_CLASSES)

        # Initialize model
        model = create_cnn_model()

        # Set up callbacks
        model_checkpoint = ModelCheckpoint(f'mpox_model_fold{fold_index + 1}.keras', save_best_only=True, monitor='val_accuracy', mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model
        history = model.fit(X_train, y_train, 
                            validation_data=(X_val, y_val),
                            epochs=EPOCHS, 
                            callbacks=[model_checkpoint, early_stopping])

        # Evaluate on the validation set
        val_loss, val_accuracy = model.evaluate(X_val, y_val)
        fold_scores.append(val_accuracy)
        print(f"Fold {fold_index + 1} - Validation Accuracy: {val_accuracy:.4f}")

        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model

    # Calculate and display cross-validation results
    print('\nCross-validation results:')
    print(f'Fold accuracies: {fold_scores}')
    print(f'Mean CV accuracy: {np.mean(fold_scores):.4f}')

    # Save the best model
    best_model.save("./src/stages/draft/6_mpox_stages_best_model.keras")
    print(f"Best model saved as: {"./src/stages/draft/6_mpox_stages_best_model.keras"}")


if __name__ == '__main__':
    main()