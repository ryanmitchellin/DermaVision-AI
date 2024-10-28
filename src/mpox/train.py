import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold

# Directories and parameters
data_path = './src/mpox/mpox-skin-lesion-dataset-version-20-msld-v20/versions/4'
fold_names = [f'fold{i}_AUG' for i in range(1, 6)]
categories = ['Cowpox', 'Healthy', 'HFMD', 'Measles', 'Chickenpox', 'Monkeypox']
IMG_SIZE = 150
NUM_CLASSES = len(categories)
EPOCHS = 2

# Label dictionary
label_dict = {category: idx for idx, category in enumerate(categories)}

# Function to load images from a specified fold
def load_images_from_fold(fold_name, split_type='Train'):
    images, labels = [], []
    category_folder_base = os.path.join(data_path, 'Augmented Images/Augmented Images/FOLDS_AUG', fold_name, split_type)

    for category in categories:
        category_folder = os.path.join(category_folder_base, category)
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
            images.append(image)
            labels.append(label_dict[category])
    return np.array(images), np.array(labels)

# Normalize images
def normalize_images(images):
    return images.astype('float32') / 255.0

# Simple CNN model
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

# K-fold Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate over each fold for cross-validation
for fold_index, fold_name in enumerate(fold_names):
    print(f"\nTraining fold {fold_index + 1} as validation set...")

    # Prepare training and validation data for the current fold
    val_images, val_labels = load_images_from_fold(fold_name, split_type='Train')
    val_images = normalize_images(val_images)  # Normalize validation images

    # Load other folds as training data
    train_images, train_labels = [], []
    for other_fold in fold_names:
        if other_fold != fold_name:
            imgs, lbls = load_images_from_fold(other_fold, split_type='Train')
            train_images.extend(imgs)
            train_labels.extend(lbls)

    # Convert lists to numpy arrays and preprocess
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    train_images = normalize_images(train_images)
    train_labels = to_categorical(train_labels, num_classes=NUM_CLASSES)
    val_labels = to_categorical(val_labels, num_classes=NUM_CLASSES)

    # Initialize and train the model
    model = create_cnn_model()

    # Callbacks for saving the best model and early stopping
    model_checkpoint = ModelCheckpoint(f'my_model_fold{fold_index + 1}.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(val_images, val_labels),
              callbacks=[model_checkpoint, early_stopping])

    # Evaluate the model on the validation set for the current fold
    val_loss, val_accuracy = model.evaluate(val_images, val_labels)
    print(f"Fold {fold_index + 1} Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Final model
model.save('my_final_model.keras')
print("Final model saved as my_final_model.keras")
