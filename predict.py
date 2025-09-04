import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the pre-trained Keras model
try:
    model = keras.models.load_model('keras_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the labels
try:
    with open('labels.txt', 'r') as f:
        class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    print("Labels loaded successfully.")
except Exception as e:
    print(f"Error loading labels: {e}")
    class_names = []

# Create a dictionary for detailed descriptions for each class
class_descriptions = {
    "Annual Crop": "The analysis reveals areas of active annual crop cultivation, characterized by structured fields and distinct seasonal growth patterns. The AI has identified healthy vegetation, indicating strong agricultural productivity in the area.",
    "Forest": "The analysis indicates a dense, healthy forest with a robust tree canopy. The AI has detected a high probability of rich biodiversity and a well-established ecosystem. Minimal signs of human intervention or deforestation were observed.",
    "Herbaceous Vegetation": "The analysis identifies a region dominated by non-woody, herbaceous plants. This could include grasslands, meadows, or other low-lying plant cover. The terrain appears to be natural and undisturbed by cultivation.",
    "Highway": "The analysis has detected a clear and defined highway or major road network. These features are identified by their straight lines, consistent width, and surrounding infrastructure. They suggest a high level of human transportation activity.",
    "Industrial": "The analysis reveals an industrial area, characterized by large buildings, warehouses, paved surfaces, and machinery. These patterns suggest a zone dedicated to manufacturing, storage, or other industrial activities.",
    "Pasture": "The analysis shows a pasture or grazing land, identified by its open spaces with low-growing grass and a lack of dense forestation. This area is typically used for livestock and has a more uniform texture compared to natural grasslands.",
    "Permanent Crop": "The analysis identifies an area of permanent crop cultivation, such as orchards or vineyards. These are distinguished by their organized, long-term planting arrangements and consistent canopy, indicating sustainable farming practices.",
    "River": "The analysis has detected a clear river system, identifiable by its meandering path and the distinct difference in texture and color from the surrounding land. The AI notes the presence of water flow and the geological impact of the river on the landscape.",
    "Sea/Lake": "The analysis reveals a large body of water, classified as a sea or a lake. The AI notes its distinct color, smooth surface, and the way it interacts with the surrounding landmass, indicating a significant aquatic feature.",
    "Residential": "The analysis shows a residential area, characterized by a high density of houses and buildings, as well as a network of local roads. The AI has identified the typical suburban or urban patterns associated with human habitation."
}


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or not class_names:
        return jsonify({'error': 'Model or labels not loaded. Please check the files.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Open and process the image
            image = Image.open(file.stream).convert('RGB')
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            
            # Convert the image to a numpy array
            image_array = np.asarray(image)
            
            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            
            # Reshape into the correct format for the model
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array
            
            # Run the prediction
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            
            # Get the description from the dictionary
            description = class_descriptions.get(class_name, "No detailed description available.")

            # Return the result as a JSON response
            return jsonify({
                'class': class_name,
                'confidence': float(confidence_score),
                'description': description
            })

        except Exception as e:
            return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
