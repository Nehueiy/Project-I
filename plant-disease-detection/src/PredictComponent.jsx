import React, { useState } from 'react';
import './PredictComponent.css';

const PredictComponent = () => {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    if (file) {
      setPreview(URL.createObjectURL(file));
      setResult('');
    }
  };

  const handlePredict = async () => {
    if (!image) {
      alert('Please select an image.');
      return;
    }

    const formData = new FormData();
    formData.append('file', image);
    setLoading(true);
    setResult('');

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setResult(data.prediction);
    } catch (error) {
      setResult('Prediction failed. Try again.');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="predict-container">
      <h1>🌿 Plant Disease Detector</h1>
      <div className="predict-card">
        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="file-input"
        />
        {preview && (
          <img src={preview} alt="Preview" className="image-preview" />
        )}

        <button onClick={handlePredict} disabled={loading}>
          {loading ? 'Predicting...' : 'Upload & Predict'}
        </button>

        {loading && <div className="loader"></div>}

        {result && (
          <div className="result-box">
            🌱 <strong>Prediction:</strong> {result}
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictComponent;
