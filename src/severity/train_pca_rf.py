from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
import joblib

# Load the PCA-reduced data
csv_path = './src/severity/pca_output.csv'
df = pd.read_csv(csv_path)

# Separate features and labels
X = df.iloc[:, 1:].values  # All columns except the first (features)
y = df['Severity'].values  # First column (labels)

# Encode the labels to numeric values
le = LabelEncoder()
y = le.fit_transform(y)  # Convert severity levels to integers

# Set up k-fold cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5 folds
fold_scores = []  # To store validation accuracies for each fold
best_accuracy = 0.0

# Perform cross-validation
for fold_index, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nProcessing fold {fold_index + 1}")
    
    # Split data into training and validation sets for this fold
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Initialize Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model on the training set
    rf_model.fit(X_train, y_train)
    
    # Evaluate on the validation set
    y_val_pred = rf_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    fold_scores.append(val_accuracy)
    print(f"Fold {fold_index + 1} - Validation Accuracy: {val_accuracy:.4f}")
    
    # Save the best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        print(f"Saving best model for Fold {fold_index + 1} with Accuracy: {val_accuracy:.4f}")
        joblib.dump(rf_model, "./src/severity/models/pca_rf_severity.pkl")

# Summary of cross-validation
print("\nCross-Validation Results:")
print(f"Mean Validation Accuracy: {np.mean(fold_scores):.4f}")
print(f"Best Validation Accuracy: {best_accuracy:.4f}")
print(f"Best model saved as: {"./src/severity/models/pca_rf_severity.pkl"}")
