import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model("plant_disease_classifier.keras")  # change if your model filename is different
    return model

model = load_trained_model()

# Class labels
classes = ['bacterial_leaf_blight', 'brown spot', 'healthy']  

# Streamlit app layout

st.title("🌾 Rice Leaf Disease Detection")
st.markdown("Upload a rice leaf image and get instant disease prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose a rice leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    st.write("Processing image...")
    img = image.resize((224, 224))  # Match your model input size
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display result
    st.success(f"**Prediction:** {predicted_class} ({confidence:.2f}% confidence)")

    # Optional details
    if predicted_class == "Brown Spot":
        st.info("Brown Spot can be treated by reducing leaf wetness and using fungicides.")
    elif predicted_class == "Bacterial Leaf Blight":
        st.info("Bacterial Blight needs resistant varieties and clean water management.")
    elif predicted_class == "Healthy":
        st.info("Your leaf looks healthy! 🌱")

