import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder='../../static', template_folder='../../')

model_path = os.path.join(os.path.dirname(__file__), '../../my_final_model.keras')

if os.path.exists(model_path):
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
else:
    raise FileNotFoundError(f"Model file not found at path: {model_path}")

categories = ['Cowpox', 'Healthy', 'HFMD', 'Measles', 'Chickenpox', 'Monkeypox']

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
    
    # Preprocess the image for the model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
    image = np.array(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    # Make a prediction
    prediction = model.predict(image)
    predicted_label = categories[np.argmax(prediction)]
    
    # Check if the predicted label is Monkeypox
    if predicted_label == 'Monkeypox':
        # Placeholder values for severity and explanation
        severity = "Moderate"  
        explanation = "The detected pattern and characteristics of lesions suggest a moderate case of Monkeypox."  
        return jsonify({
            "prediction": predicted_label,
            "severity": severity,
            "explanation": explanation
        })
    else:
        return jsonify({"prediction": predicted_label})

if __name__ == "__main__":
    app.run(debug=True)
