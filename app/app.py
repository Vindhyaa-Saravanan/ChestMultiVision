# Install necessary libraries
from io import BytesIO
from pathlib import Path
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import keras
import pandas as pd
import requests

logo_icon_url = 'https://raw.githubusercontent.com/Vindhyaa-Saravanan/ChestMultiVision/main/app/logo.ico?token=GHSAT0AAAAAACNZZ2C6AJ46ML6STVPGYLN4ZRKNIQQ'
response = requests.get(logo_icon_url)
#logo = Image.open(BytesIO(response.content))
path = Path(__file__).parent / "logo.ico"
logo = Image.open(path)
st.set_page_config(
    page_title="ChestMultiVision",
    page_icon=logo
)

# Refer https://keras.io/api/applications/mobilenet/ for specifics of MobileNetV2.

def create_resnet_model():
    # Input layer
    model_input = keras.layers.Input(shape=(512, 512, 3))

    # Define ResNet50V2 base model
    base_model = keras.applications.ResNet50V2(input_shape=(512, 512, 3), include_top=False, weights=None, pooling='avg')

    # Pass preprocessed input through base model
    base_model_output = base_model(model_input)

    # Fully connected layers
    x = keras.layers.Dropout(0.5)(base_model_output)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)

    # Output layers for each label
    output_layers = []
    for _ in range(0,6):
        output_layer = keras.layers.Dense(1, activation='sigmoid', name=f'output_{_}')(x)
        output_layers.append(output_layer)

    # Define model with multiple outputs
    resnet_model = keras.models.Model(inputs=model_input, outputs=output_layers)

    return resnet_model

# Load the saved TensorFlow/Keras model for 2D data
model = create_resnet_model()
path = Path(__file__).parent / "ResNet50V2_finetuned.weights.h5"
model.load_weights(path, skip_mismatch=False)

# Streamlit app

# Referred to https://discuss.streamlit.io/t/filenotfounderror-errno-2-no-such-file-or-directory-images-icon-png/36154/3
logo_picture_url = 'https://raw.githubusercontent.com/Vindhyaa-Saravanan/ChestMultiVision/main/app/logo.jpg?token=GHSAT0AAAAAACNZZ2C7P7NBQNYMKYDOOMJQZRKNMEQ'
response_picture = requests.get(logo_picture_url)
picture = Image.open(BytesIO(response_picture.content))

st.title('ChestMultiVision: Chest X-ray MultiLabel Classification App')

st.write("Product Disclaimer: ChestMultiVision is a prototype chest x ray classification app, it is NOT A MEDICAL DEVICE. Predictions made are simply to demonstrate the application and the application is not approved for medical use.")
    
# Sidebar for additional model information
with st.sidebar:
    st.image(picture, use_column_width=True)
    st.markdown("<style>h2 {font-size: 14px;}, p {font-size: 10px;}</style>", unsafe_allow_html=True)  # Reduce font size    
    st.markdown("#### About ChestMultiVision")
    st.markdown("ChestMultiVision harnesses a custom deep learning model based on the ResNet50V2 architecture. It was trained on the Chest X-ray-14 dataset. It predicts six different findings detectable on chest x-rays, that are: Atelectasis, Effusion, Infiltration, Mass, No Finding, and Nodule.")

# Upload an image for classification
uploaded_file = st.file_uploader("Upload a chest x-ray image, to receive image-level predictions for the six findings...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Preprocess the uploaded image
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        image = image.resize((512, 512))  # Resize to match the model input size

        # Use the TensorFlow/Keras model for predictions
        image_array = np.array(image)  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=-1)  # Add a channel dimension
        rgb_data = np.stack((image_array.reshape(512,512), ) * 3, axis=-1)
        rgb_data = keras.applications.resnet_v2.preprocess_input(rgb_data)  # Preprocess the image
        prediction = model.predict(np.array([rgb_data]))
        
        # Display the uploaded image and predictions side by side
        col1, col2 = st.columns(2)
        col1.image(image, caption='Uploaded Image', width=256)
        
        # Display predicted classes and probabilities
        with col2:
            st.write("## Labels and their Predicted Probabilities")
            labels = ['Atelectasis', 'Effusion', 'Infiltration', 'Mass', 'No Finding', 'Nodule']
            probabilities = [prob[0][0] for prob in prediction]

            # Print probabilities for all labels
            for label, prob in zip(labels, probabilities):
                st.write(f"{label}: {prob}")

            # Filter labels with probabilities greater than 0.5
            predicted_labels = [labels[i] for i, prob in enumerate(probabilities) if prob > 0.5]
            st.write(f"## Predicted Labels with more than 50% Probability: {str(predicted_labels)}")
    except:
        st.write("Error in processing the uploaded image. Please try again with a different image.")
        
    finally:
        st.write("Click the 'X' near the image upload point to clear this image and try another!")