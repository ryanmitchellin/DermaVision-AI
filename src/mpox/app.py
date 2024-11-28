import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__, static_folder='../../static', template_folder='../../')

# Load models
cnn_model_path = os.path.join(os.path.dirname(__file__), '../../final_cnn_model.keras')
pca_model_path = os.path.join(os.path.dirname(__file__), '../stages/models/2_pca_model.pkl')
rf_model_path = os.path.join(os.path.dirname(__file__), '../stages/models/3_pca_rf_stages.pkl')

if os.path.exists(cnn_model_path):
    print(f"Loading CNN model from: {cnn_model_path}")
    cnn_model = load_model(cnn_model_path)
else:
    raise FileNotFoundError(f"Model file not found at path: {cnn_model_path}")

if os.path.exists(pca_model_path):
    print(f"Loading PCA model from: {pca_model_path}")
    pca = joblib.load(pca_model_path)
else:
    raise FileNotFoundError(f"PCA model file not found at path: {pca_model_path}")

if os.path.exists(rf_model_path):
    print(f"Loading Random Forest model from: {rf_model_path}")
    rf_model = joblib.load(rf_model_path)
else:
    raise FileNotFoundError(f"Random Forest model file not found at path: {rf_model_path}")

categories = ['Cowpox', 'Healthy', 'HFMD', 'Measles', 'Chickenpox', 'Monkeypox']
stages = ['Macule', 'Papule', 'Vesicle', 'Pustule', 'Crust']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Preprocess for CNN
    cnn_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cnn_image = cv2.resize(cnn_image, (64, 64))
    cnn_image = np.array(cnn_image).astype('float32') / 255.0
    cnn_image = np.expand_dims(cnn_image, axis=0)

    # CNN prediction
    cnn_prediction = cnn_model.predict(cnn_image)
    predicted_label_index = np.argmax(cnn_prediction)
    confidence = cnn_prediction[0][predicted_label_index]

    # Confidence threshold for determining if input is valid
    confidence_threshold = 0.5
    if confidence < confidence_threshold:
        return jsonify({"error": "Error: Input could not be classified. Please provide a clearer image."}), 400

    # Determine predicted label
    predicted_label = categories[predicted_label_index]

    # If the prediction is not Monkeypox, return only the prediction
    if predicted_label != "Monkeypox":
        return jsonify({"prediction": predicted_label})

    # Preprocess for PCA + RF if the case is Monkeypox
    rf_image = cv2.resize(image, (128, 128))
    rf_image = rf_image.flatten().astype('float32')
    rf_image_pca = pca.transform([rf_image])

    # Random Forest prediction for stages
    stage_prediction = rf_model.predict(rf_image_pca)[0]
    stage_proba = rf_model.predict_proba(rf_image_pca)[0]
    predicted_stage = stages[stage_prediction]
    confidence = stage_proba[stage_prediction] * 100

    # Return full details for Monkeypox cases
    return jsonify({
        "prediction": predicted_label,
        "stage": predicted_stage,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
