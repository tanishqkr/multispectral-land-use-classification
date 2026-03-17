import tensorflow as tf
import numpy as np

MODEL_PATH = "models/single_model_quantized.tflite"

# Load the model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get output details
output_details = interpreter.get_output_details()
output_shape = output_details[0]['shape']

print("✅ Model output shape:", output_shape)
print("👉 Number of classes:", output_shape[1])
