from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Lung Cancer Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load saved model and other assets
        model = joblib.load('hybrid_model.pkl')  # Replace with your model filename
        scaler = joblib.load('scaler.pkl')       # Replace with your scaler filename
        label_encoder = joblib.load('label_encoder.pkl')  # Replace with your label encoder filename
        
        # Get JSON data from the POST request
        data = request.get_json()

        # Validate input
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" in request data'}), 400

        features = data['features']

        # Validate feature array length
        if len(features) != 15:  # Assuming 15 features are required
            return jsonify({'error': f'Expected 15 features, but got {len(features)}'}), 400

        # Ensure all feature values are numeric
        try:
            features = [float(f) for f in features]
        except ValueError:
            return jsonify({'error': 'All features must be numeric'}), 400

        # Convert to NumPy array and reshape for model
        features = np.array(features).reshape(1, -1)

        # Scale features using the loaded scaler
        scaled_features = scaler.transform(features)

        # Make prediction using the loaded model
        prediction_numeric = model.predict(scaled_features)

        # Convert numeric prediction to label
        prediction_label = label_encoder.inverse_transform(prediction_numeric)

        # Return prediction as JSON
        return jsonify({'prediction': prediction_label[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
