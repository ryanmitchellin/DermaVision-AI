import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model, Input

# Declare directories path, categories and parameters
data_path = './src/mpox/mpox-skin-lesion-dataset-version-20-msld-v20/versions/4'
fold_names = [f'fold{i}_AUG' for i in range(1, 6)]
categories = ['Cowpox', 'Healthy', 'HFMD', 'Measles', 'Chickenpox', 'Monkeypox']
IMG_SIZE = 64
NUM_CLASSES = len(categories)
EPOCHS = 5

# Mapping categories name into numeric labels
label_dict = {category: idx for idx, category in enumerate(categories)}

# Load images from a specified fold
def load_images_from_fold(fold_name):
    images, labels = [], []
    category_folder_base = os.path.join(data_path, 'Augmented Images/Augmented Images/FOLDS_AUG', fold_name, 'Train')

    # Iterate through each categories and load images
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

            # Preprocess image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            images.append(image)
            labels.append(label_dict[category])
    return np.array(images), np.array(labels)

# Normalize images to scale between 0 and 1
def normalize_images(images):
    return images.astype('float32') / 255.0

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

# ResNet model for image classification
# def create_resnet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
#     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
#     base_model.trainable = False

#     model = Sequential([
#         Input(shape=input_shape),
#         base_model,
#         Conv2D(32, 3, padding='same', activation='relu'),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(num_classes, activation='softmax')
#     ])

#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# 5-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate over each fold for cross-validation CNN
for fold_index, fold_name in enumerate(fold_names):
    print(f"\nTrainin CNN fold {fold_index + 1} as validation set...")

    # Load validation data from the current fold
    val_images, val_labels = load_images_from_fold(fold_name)
    val_images = normalize_images(val_images)
    val_labels = to_categorical(val_labels, num_classes=NUM_CLASSES)

    # Load other folds as training data
    train_images, train_labels = [], []
    for other_fold in fold_names:
        if other_fold != fold_name:
            imgs, lbls = load_images_from_fold(other_fold)
            train_images.extend(imgs)
            train_labels.extend(lbls)

    # Convert training data to numpy arrays and preprocess
    train_images = normalize_images(np.array(train_images))
    train_labels = to_categorical(np.array(train_labels), num_classes=NUM_CLASSES)

    # Initialize the model
    cnn_model = create_cnn_model()

    # Set callbacks for early stopping and model checkpoint
    model_checkpoint = ModelCheckpoint(f'my_cnn_model_fold{fold_index + 1}.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    cnn_model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(val_images, val_labels),
              callbacks=[model_checkpoint, early_stopping])

    # Evaluate the model on the validation set for the current fold
    val_loss, val_accuracy = cnn_model.evaluate(val_images, val_labels)
    print(f"Fold {fold_index + 1} Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Final model CNN
cnn_model.save('my_final_cnn_model.keras')
print("Final model saved as my_final_cnn_model.keras")

# Iterate over each fold for cross-validation ResNet
# for fold_index, fold_name in enumerate(fold_names):
#     print(f"\nTraining ResNet fold {fold_index + 1} as validation set...")

#     # Load validation data from the current fold
#     val_images, val_labels = load_images_from_fold(fold_name)
#     val_images = normalize_images(val_images)
#     val_labels = to_categorical(val_labels, num_classes=NUM_CLASSES)

#     # Load other folds as training data
#     train_images, train_labels = [], []
#     for other_fold in fold_names:
#         if other_fold != fold_name:
#             imgs, lbls = load_images_from_fold(other_fold)
#             train_images.extend(imgs)
#             train_labels.extend(lbls)

#     # Convert training data to numpy arrays and preprocess
#     train_images = normalize_images(np.array(train_images))
#     train_labels = to_categorical(np.array(train_labels), num_classes=NUM_CLASSES)

#     # Initialize the model
#     resnet_model = create_resnet_model()

#     # Set callbacks for early stopping and model checkpoint
#     model_checkpoint = ModelCheckpoint(f'my_resnet_model_fold{fold_index + 1}.keras', save_best_only=True, monitor='val_accuracy', mode='max')
#     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#     # Train the model
#     resnet_model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(val_images, val_labels),
#               callbacks=[model_checkpoint, early_stopping])

#     # Evaluate the model on the validation set for the current fold
#     val_loss, val_accuracy = resnet_model.evaluate(val_images, val_labels)
#     print(f"Fold {fold_index + 1} Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Final model ResNet
# resnet_model.save('my_final_resnet_model.keras')
# print("Final model saved as my_final_resnet_model.keras")
