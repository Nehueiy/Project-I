import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# ✅ FIRST Streamlit command
st.set_page_config(page_title="Rice Leaf Disease Detection", layout="centered")

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model("plant_disease_classifier.keras")  # change filename if needed
    return model

model = load_trained_model()

# Class labels
classes = ["Bacterial Leaf Blight", "Brown Spot", "Healthy"]

# Title and instructions
st.title("🌾 Rice Leaf Disease Detection")
st.markdown("Upload a rice leaf image and get instant disease prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose a rice leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    st.write("Processing image...")
    img = image.resize((224, 224))  # adjust to your model's input shape
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Result
    st.success(f"**Prediction:** {predicted_class} ({confidence:.2f}% confidence)")

    # Additional Info
    if predicted_class == "Brown Spot":
        st.info("Brown Spot: Control with fungicides and field sanitation.")
    elif predicted_class == "Bacterial Leaf Blight":
        st.info("Bacterial Blight: Use resistant varieties and clean irrigation.")
    elif predicted_class == "Healthy":
        st.info("Healthy Leaf! Keep monitoring regularly. 🌿")

