import React from 'react';
import PredictComponent from './PredictComponent'; // 👈 important

function App() {
  return (
    <div>
      <h1>🌱 Rice Leaf Disease Detector</h1>
      <PredictComponent /> {/* 👈 this will render your JSX file */}
    </div>
  );
}

export default App;
