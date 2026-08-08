"""
model_handler.py
Handles the initialization and inference logic for the deep learning model using TFLite.
This isolates TFLite operations from the main web routing logic.
"""

import os
import numpy as np
from PIL import Image
from utils import class_names, disease_info

# Try to import tflite_runtime, fallback to full tensorflow if not available locally
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

_interpreter = None
_input_details = None
_output_details = None

def get_model():
    """
    Lazy loader for the TFLite model. Ensures the model is loaded
    only once and kept in memory for subsequent predictions.
    """
    global _interpreter, _input_details, _output_details
    if _interpreter is None:
        model_path = os.path.join(os.path.dirname(__file__), 'model.tflite')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please place 'model.tflite' in the project root.")
        print("Loading TFLite AI Model into memory (this is much faster)...")
        _interpreter = tflite.Interpreter(model_path=model_path)
        _interpreter.allocate_tensors()
        _input_details = _interpreter.get_input_details()
        _output_details = _interpreter.get_output_details()
        print("Model loaded successfully!")
    return _interpreter

def preprocess_image(image_path):
    """
    Loads and resizes an image to the dimensions required by the CNN model (128x128).
    Converts it to a numpy array batch using Pillow instead of heavy tensorflow functions.
    """
    img = Image.open(image_path).resize((128, 128))
    input_arr = np.array(img, dtype=np.float32)
    # If image has an alpha channel, convert to RGB
    if input_arr.shape[-1] == 4:
        img = img.convert('RGB')
        input_arr = np.array(img, dtype=np.float32)
    input_arr = np.expand_dims(input_arr, axis=0)  # Model expects a batch of images
    return input_arr

def predict_disease(image_path):
    """
    Takes an image file path, processes it, and returns the diagnosis using TFLite.
    Returns:
        dict: Containing 'prediction' (class name), 'cause', and 'treatment'
    """
    interpreter = get_model()
    
    # Process image
    img_array = preprocess_image(image_path)
    
    # Set the tensor
    interpreter.set_tensor(_input_details[0]['index'], img_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get predictions
    predictions = interpreter.get_tensor(_output_details[0]['index'])
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    
    # Lookup disease details
    details = disease_info.get(predicted_class, {
        'cause': 'Unknown cause.',
        'treatment': 'No specific treatment available.'
    })
    
    return {
        'prediction': predicted_class,
        'cause': details['cause'],
        'treatment': details['treatment']
    }
