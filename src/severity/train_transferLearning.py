import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense
from sklearn.model_selection import KFold, train_test_split
import os
import cv2
import numpy as np

DATA_DIR = './src/severity/augmented_data'
CATEGORIES = ['macule', 'papule', 'pustule', 'scab', 'vesicles']
IMG_SIZE = 224
NUM_CLASSES = len(CATEGORIES)
EPOCHS = 50

# Map category names to numeric labels
label_dict = {category: idx for idx, category in enumerate(CATEGORIES)}

# Load images
def load_images():
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

            # Preprocess image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = image.astype('float32') / 255.0  # Normalize
            images.append(image)
            labels.append(label_dict[category])
    return np.array(images), np.array(labels)

def create_model():
    # Base model (Transfer Learning)
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze layers except the last 20
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Update model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.6),
        Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Callbacks
def get_callbacks(fold):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'best_mpox_classifier_fold_{fold}.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]
    return callbacks

# Main
def main():
    # Load Data
    X, y = load_images()
    y = tf.keras.utils.to_categorical(y, NUM_CLASSES)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=42)

    # Initialize Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    best_models = []

    for fold, (train_i, validation_i) in enumerate(kf.split(X_train)):
        print(f"Processing Fold {fold}")

        # Split training data
        X_train_fold, X_val = X_train[train_i], X_train[validation_i]
        y_train_fold, y_val = y_train[train_i], y_train[validation_i]

        # Create and train model
        model = create_model()
        
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            callbacks=get_callbacks(fold)
        )
        
        # Store best validation accuracy
        best_val_acc = max(history.history['val_accuracy'])
        fold_scores.append(best_val_acc)
        best_models.append(model)
        
        print(f'Fold {fold + 1} - Best Validation Accuracy: {best_val_acc:.4f}')

    # Print cross-validation results
    print('\nCross-validation results:')
    print(f'Individual fold scores: {[f"{score:.4f}" for score in fold_scores]}')
    print(f'Mean CV accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})')
    
    # Evaluate best model
    best_model_idx = np.argmax(fold_scores)
    best_model = best_models[best_model_idx]
    
    print('\nEvaluating best model on test set...')
    test_loss, test_acc = best_model.evaluate(X_test, y_test)
    print(f'Test set accuracy: {test_acc:.4f}')
    
    # Save best model
    best_model.save('best_mpox_classifier_final.keras')
    print('\nBest model saved')

if __name__ == '__main__':
    main()