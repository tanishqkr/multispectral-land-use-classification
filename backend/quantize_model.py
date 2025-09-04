import tensorflow as tf
from tensorflow.python.saved_model import signature_constants

def quantize_model(model_path):
    """
    Loads a TensorFlow model, applies quantization, and saves the new, smaller model.
    
    Args:
        model_path (str): The file path to the original .h5 model.
    """
    print("Loading the original model...")
    # Load the original model
    model = tf.keras.models.load_model(model_path)
    
    print("Converting model to a TFLite flat buffer...")
    
    # Get the concrete function from the model. This is a more robust way to
    # prepare the model for conversion and avoids common graph-related errors.
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    )
    
    # Convert the Keras model to a TFLite model using the concrete function
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # Enable optimizations, including quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Perform the conversion and optimization
    quantized_tflite_model = converter.convert()
    
    # Get the original file name and create a new one for the quantized model
    original_file_name, original_extension = model_path.split('.')
    quantized_model_path = f"{original_file_name}_quantized.tflite"
    
    print(f"Saving the quantized model to: {quantized_model_path}")
    # Save the quantized model to a .tflite file
    with open(quantized_model_path, 'wb') as f:
        f.write(quantized_tflite_model)
    
    print("Model quantization complete.")

if __name__ == '__main__':
    quantize_model('models/single_model.h5')
