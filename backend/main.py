from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import io
import os

# --- Configuration ---
# Set the image size to match the model's expected input shape (128x128)
IMAGE_SIZE = (128, 128)
MODEL_PATH = 'models/single_model_quantized.tflite'
CLASSES = ['AnnualCrop', 'Industrial', 'Pasture', 'Residential', 'SeaLake', 'Highway', 'River'] 

# --- Class Descriptions for Detailed Analysis ---
CLASS_DESCRIPTIONS = {
    'AnnualCrop': "The analysis of the uploaded satellite imagery reveals a distinct mosaic of agricultural fields with clear borders. The AI has detected organized crop patterns, indicating active farming and land cultivation. The texture suggests recently planted or harvested crops.",
    'Industrial': "This satellite image displays a high concentration of man-made, non-residential structures. The AI analysis identifies large buildings, factory rooftops, storage containers, and logistical infrastructure, characteristic of an industrial zone.",
    'Pasture': "The imagery shows large, open green spaces with natural ground cover. The AI analysis points to a landscape dominated by grasslands, often used for grazing or as open natural habitats, with minimal signs of tilling or dense tree cover.",
    'Residential': "This area is dominated by an organized arrangement of man-made structures with distinct geometric patterns. The AI analysis points to a suburban or urban landscape, with clear indications of roads, roofs, and a high density of buildings.",
    'SeaLake': "The analysis of the uploaded satellite imagery reveals a distinct aquatic ecosystem, characterized by a large body of water. The AI indicates a calm aquatic surface, suggesting minimal wave activity, consistent with a lake or a calm sea area.",
    'Highway': "The AI has detected long, linear features with consistent width, often intersecting with other roads. The high albedo (reflectivity) and smooth texture are characteristic of major transportation networks like highways or large roads.",
    'River': "The imagery shows a narrow, meandering body of water cutting through the landscape. The AI has identified a flowing aquatic system, characterized by its linear shape and the distinct contrast between the water and the surrounding terrain."
}

# --- Initialize Flask App and Enable CORS ---
app = Flask(__name__)
CORS(app)

# --- Load the model once to avoid reloading it on every request ---
try:
    # Use the absolute path to the TFLite model
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_PATH))
    
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("TFLite model loaded successfully.")

except Exception as e:
    print(f"Error loading TFLite model: {e}")
    interpreter = None

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # Read image from bytes
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))

            width, height = img.size
            if width != height or width < 64: 
                return jsonify({"error": "Please upload a valid satellite image. Images must be square and at least 64x64 pixels."}), 400

            img = img.resize(IMAGE_SIZE)
            
            if img.mode == 'RGBA':
                img = img.convert('RGB')
                
            img_array = np.array(img) / 255.0
            img_array = img_array.astype(np.float32)
            img_array = np.expand_dims(img_array, axis=0) 

            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            predicted_class_index = np.argmax(predictions[0])
            predicted_class = CLASSES[predicted_class_index]
            confidence = float(predictions[0][predicted_class_index] * 100)
            description = CLASS_DESCRIPTIONS.get(predicted_class, "No detailed description available.")

            return jsonify({
                "class": predicted_class.upper(),
                "confidence": f"{confidence:.2f}%",
                "description": description
            })

        except Exception as e:
            return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5050)))