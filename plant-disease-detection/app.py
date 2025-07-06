from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
app = Flask(__name__)
CORS(app)

try:
    model = load_model('plant_disease_classifier.keras')  
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", e)

@app.route('/')
def home():
    return "Flask is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']
    img_path = 'temp.jpg'
    file.save(img_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_labels = ['bacterial_leaf_blight', 'Brown spot', 'healthy']  
    result = class_labels[class_index]

    os.remove(img_path)
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
