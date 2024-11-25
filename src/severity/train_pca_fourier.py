import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np


# Constants
PCA_CSV_PATH = './src/severity/output_pca.csv'
CATEGORIES = ['macule', 'papule', 'pustule', 'scab', 'vesicles']
IMG_SIZE = 224
NUM_CLASSES = len(CATEGORIES)
EPOCHS = 5

# Load PCA
def load_pca():
    # Load PCA features
    pca_data = pd.read_csv(PCA_CSV_PATH)

    # Split features and labels
    X = pca_data.iloc[:, 1:].values
    y = pca_data['Severity'].values

    # Map category names to numeric labels
    label_dict = {category: i for i, category in enumerate(CATEGORIES)}
    y = np.array([label_dict[label] for label in y])

    return np.array(X), y

# CNN model for image classification
def create_cnn_model(input_shape, num_classes=NUM_CLASSES):
    model = Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Load data
    X, y = load_pca()

    # Set up 5-fold cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    best_accuracy = -1

    # Perform cross-validation
    for fold_index, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nProcessing fold {fold_index + 1}")

        # Split data into training and validation sets for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
        y_val = to_categorical(y_val, num_classes=NUM_CLASSES)

        # Initialize model
        model = create_cnn_model(input_shape=(X.shape[1],))

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
    print(f'Mean CV accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})')
    
    # Save the final model trained on all data
    best_model.save('fourierPCA_mpox_severity_final_model.keras')
    print("Final model saved as fourierPCA_mpox_severity_final_model.keras")


if __name__ == '__main__':
    main()