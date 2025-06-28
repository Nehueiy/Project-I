from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Load your trained model (edit the path if needed)
model = load_model("plant_disease_classifier.keras")
class_names = ['bacterial_leaf_blight', 'brown_spot', 'healthy']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = load_img(file, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction))

    return jsonify({
        'class': predicted_class,
        'confidence': f"{confidence*100:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True)
