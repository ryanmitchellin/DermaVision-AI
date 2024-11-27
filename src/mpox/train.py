import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import keras_tuner as kt

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
def create_cnn_model(hp):
    model = Sequential([
        Conv2D(hp.Int('conv_1_filters', min_value=32, max_value=128, step=32), (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(hp.Int('conv_2_filters', min_value=64, max_value=256, step=64), (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(hp.Int('dense_units', min_value=128, max_value=512, step=128), activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameter search using Random Search
tuner = kt.RandomSearch(
    create_cnn_model,
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=1,
    directory='my_dir',
    project_name='cnn_random_search'
)

# 5-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# List for storing all results
all_results = []

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

    # Perform hyperparameter search using random search
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    print(f"Starting random search on fold {fold_index + 1}...")
    tuner.search(train_images, train_labels, epochs=5, validation_data=(val_images, val_labels), callbacks=[early_stopping])

    # Retrieve the best model from the tuner
    cnn_model = tuner.get_best_models(num_models=1)[0]
    
    # Save the best model from this fold
    cnn_model.save(f'best_model_fold_{fold_index + 1}.keras')

    # Evaluate the model on the validation set for the current fold
    val_loss, val_accuracy = cnn_model.evaluate(val_images, val_labels)
    print(f"Fold {fold_index + 1} Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    # Append the results to the list
    all_results.append((fold_index + 1, val_loss, val_accuracy))

# Final model CNN with hyperparameter tuning
final_cnn_model = tuner.get_best_models(num_models=1)[0]
final_cnn_model.save('final_cnn_model.keras')
print("Final model saved as final_cnn_model.keras")

# Print out the best hyperparameters found during the tuning
best_hp = tuner.oracle.get_best_trials()[0].hyperparameters
print("\nBest Hyperparameters found:")
for hp_name, hp_value in best_hp.values.items():
    print(f"{hp_name}: {hp_value}")

