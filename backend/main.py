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
    'AnnualCrop', 'Industrial', 'Pasture',
    'Residential', 'SeaLake', 'Highway', 'River'
]

DESCRIPTIONS = {
    'AnnualCrop': 'Land used for crops harvested once a year, usually fields with cultivated plants.',
    'Industrial': 'An industrial area with factories, warehouses, or other man-made infrastructure.',
    'Pasture': 'Grassland or open field suitable for grazing animals.',
    'Residential': 'Area with housing and residential structures, usually mixed with roads and small vegetation.',
    'SeaLake': 'Large body of water — either a sea or a lake, with a distinct shoreline.',
    'Highway': 'A major road system with multiple lanes and paved surfaces.',
    'River': 'A winding body of flowing water, often with riparian vegetation along the banks.'
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
