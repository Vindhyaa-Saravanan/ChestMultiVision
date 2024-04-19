# Install necessary libraries
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import keras
from pathlib import Path

# Referred to tutorial by Kaushal Shah
# Accessed at https://kaushal28.github.io/Building-Multi-Output-CNN-with-Keras/
# Refer https://keras.io/api/applications/mobilenet/ for specifics of MobileNetV2.

def create_mobilenet_model():
    # Input layer
    model_input = keras.layers.Input(shape=(512, 512, 3))

    # Define MobileNetV2 base model
    base_model = keras.applications.MobileNetV2(input_shape=(512, 512, 3), include_top=False, weights=None, pooling='avg')

    # Pass preprocessed input through base model
    base_model_output = base_model(model_input)

    # Fully connected layers
    x = keras.layers.Dense(512, activation='relu')(base_model_output)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)

    # Output layers for each label
    output_layers = []
    for _ in range(0,6):
        output_layer = keras.layers.Dense(1, activation='sigmoid', name=f'output_{_}')(x)
        output_layers.append(output_layer)

    # Define model with multiple outputs
    mobilenet_model = keras.models.Model(inputs=model_input, outputs=output_layers)

    return mobilenet_model

# Load the saved TensorFlow/Keras model for 2D data
model = create_mobilenet_model()
path = Path(__file__).parent / "mobilenet_finetuned.weights.h5"
st.write(path)
model.load_weights(path, skip_mismatch=False)

# Streamlit app
st.title('Medical Image Classification App')

# Upload an image for classification
uploaded_file = st.file_uploader("Choose a medical image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the uploaded image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((512, 512))  # Resize to match the model input size

    # Use the TensorFlow/Keras model for predictions
    image_array = np.array(image)  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=-1)  # Add a channel dimension
    rgb_data = np.stack((image_array.reshape(512,512), ) * 3, axis=-1)
    rgb_data = tf.keras.applications.mobilenet_v2.preprocess_input(rgb_data)  # Preprocess the image
    prediction = model.predict(np.array([rgb_data]))
    st.write(f'TensorFlow/Keras Prediction (2D): {prediction}')