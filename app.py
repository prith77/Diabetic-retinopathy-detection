import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('drdMbNetV2_40epochs_97.h5')

# Define classes
classes = ['No diabetic retinopathy', 'Mild diabetic retinopathy', 'Moderate diabetic retinopathy', 'Severe diabetic retinopathy', 'Proliferate diabetic retinopathy']

st.title('Diabetic Retinopathy Detection')

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.asarray(image) / 255.
    return np.expand_dims(image, axis=0)

# Function to make prediction
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_index = np.argmax(prediction)
    return classes[class_index], prediction[0][class_index]

# Streamlit app
st.sidebar.title('Upload Image')
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('')
    st.write('Classifying...')
    result_class, result_confidence = predict(image)
    st.write(f'Result: {result_class}')
