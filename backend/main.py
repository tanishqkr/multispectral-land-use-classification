import os
import io
import json
import gc
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Path to your TFLite model
MODEL_PATH = "models/single_model_quantized.tflite"

# Labels + descriptions
LABELS = [
    'AnnualCrop', 'Coastal', 'Forest', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'River', 'SeaLake', 'Urban'
]

DESCRIPTIONS = {
    'AnnualCrop': 'This image shows land used for growing crops that are harvested once a year.',
    'Coastal': 'The analysis reveals a distinct coastline with sandy beaches, shallow waters, and coastal vegetation.',
    'Forest': 'Identifies a dense forested area characterized by a healthy canopy and a river system.',
    'Highway': 'This image contains a clear representation of a highway or major road system.',
    'Industrial': 'Shows an industrial area, likely containing factories, warehouses, or warehouses.',
    'Pasture': 'This image depicts land covered with grass and low plants suitable for grazing animals.',
    'PermanentCrop': 'The image reveals land with long-term crops such as orchards or vineyards.',
    'River': 'A river is visible in this image, characterized by a winding body of water flowing through the landscape.',
    'SeaLake': 'This image contains a large body of water, indicating the presence of a sea or lake.',
    'Urban': 'The analysis identifies a dense urban landscape, with a high concentration of man-made structures.'
}

# Load TFLite model once
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        print("✅ Model loaded successfully")
        return interpreter
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

interpreter = load_model()
if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
else:
    input_details = None
    output_details = None
    input_shape = None

# Thread lock for concurrency safety
interpreter_lock = threading.Lock()

# Preprocess image
def preprocess_image(image):
    if input_shape is None:
        raise ValueError("Model not loaded.")
    img = image.resize((input_shape[1], input_shape[2]))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return jsonify({'error': 'Backend is not ready. Model failed to load.'}), 503

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        image = Image.open(io.BytesIO(file.read())).convert('RGB')

        width, height = image.size
        if width != height or width < 64:
            return jsonify({"error": "Please upload a square satellite image of at least 64x64 pixels."}), 400

        input_data = preprocess_image(image)

        # Thread-safe inference
        with interpreter_lock:
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index']).copy()

        del input_data
        gc.collect()

        prediction_index = int(np.argmax(output_data))
        predicted_class = LABELS[prediction_index]
        description = DESCRIPTIONS.get(predicted_class, "No description available.")

        # Confidence scores as dict {label: prob}
        confidences = {LABELS[i]: float(output_data[0][i]) for i in range(len(LABELS))}

        return jsonify({
            'class': predicted_class,
            'description': description,
            'confidence': confidences
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/')
def home():
    return "Backend is running!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
