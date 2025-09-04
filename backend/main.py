import os
import io
import tensorflow as tf
import numpy as np
import json
import gc  # Import the garbage collector
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# Initialize the Flask application
app = Flask(__name__)
# Allow cross-origin requests from the frontend
CORS(app)

# The path to the TFLite model file
MODEL_PATH = "model.tflite"

# Define the labels and descriptions for the model's output
LABELS = [
    'AnnualCrop', 'Coastal', 'Forest', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'River', 'SeaLake', 'Urban'
]

DESCRIPTIONS = {
    'AnnualCrop': 'This image shows land used for growing crops that are harvested once a year. The area appears to be a field with cultivated plants, often with distinct rows or patches.',
    'Coastal': 'The analysis reveals a distinct coastline with sandy beaches, shallow waters, and coastal vegetation. The AI indicates active sedimentary processes like erosion and deposition along the shoreline.',
    'Forest': 'Identifies a dense forested area characterized by a healthy canopy and a river system, indicating rich biodiversity and a stable ecosystem.',
    'Highway': 'This image contains a clear representation of a highway or major road system, identified by its distinct paved surface and multi-lane structure.',
    'Industrial': 'Shows an industrial area, likely containing factories, warehouses, large storage tanks, or complex machinery. The terrain is dominated by man-made structures and hard surfaces.',
    'Pasture': 'This image depicts land covered with grass and other low plants suitable for grazing animals. The area is likely an open field or grassland.',
    'PermanentCrop': 'The image reveals land with crops that remain planted for many years, such as fruit orchards, vineyards, or plantations. The vegetation shows a consistent and long-term pattern.',
    'River': 'A river is visible in this image, characterized by a winding body of water flowing through the landscape. The surrounding area may show signs of erosion or riparian vegetation.',
    'SeaLake': 'This image contains a large body of water, indicating the presence of a sea or lake. It is characterized by a distinct shoreline and uniform water surface.',
    'Urban': 'The analysis identifies a dense urban landscape, with a high concentration of buildings, roads, and other man-made structures. The area is highly developed with little natural vegetation.'
}

# Load the TFLite model once when the application starts
def load_model():
    print("Loading TFLite model...")
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        print("TFLite model loaded successfully.")
        return interpreter
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the model and get input/output details
interpreter = load_model()
if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
else:
    input_details = None
    output_details = None
    input_shape = None

# Function to preprocess the image for the model
def preprocess_image(image):
    if input_shape is None:
        raise ValueError("Model not loaded, cannot preprocess image.")
    # Resize the image to the model's required input size
    img = image.resize((input_shape[1], input_shape[2]))
    # Convert to a numpy array and normalize pixel values to 0-1
    img_array = np.array(img, dtype=np.float32) / 255.0
    # Add a batch dimension to match the model's input shape
    return np.expand_dims(img_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Open the image file from the request
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Preprocess the image
        input_data = preprocess_image(image)
        
        # Set the tensor for inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference on the model
        interpreter.invoke()
        
        # Get the output tensor from the model
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # --- THE FIX ---
        # Explicitly delete the input data to prevent memory leaks
        del input_data
        
        # Get the class index with the highest probability
        prediction_index = np.argmax(output_data)
        predicted_class = LABELS[prediction_index]
        description = DESCRIPTIONS.get(predicted_class, 'No description available.')
        
        # Return the prediction results as JSON
        return jsonify({
            'class': predicted_class,
            'description': description
        })

    except Exception as e:
        # Catch any errors and return a helpful message
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

@app.route('/')
def home():
    return "Backend is running!"

if __name__ == '__main__':
    # Start the Flask app using the PORT from the environment
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
