"""
Flask Backend for Crop Prediction System
Provides REST API endpoint to predict crop based on soil and environmental parameters.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Global variables to store loaded model and encoder
model = None
label_encoder = None

def load_model():
    """
    Load the trained Random Forest model and label encoder.
    """
    global model, label_encoder
    
    try:
        # Path to model files (relative to backend directory)
        model_path = 'C:\cursor\model_training\crop_prediction_model.pkl'
        encoder_path = 'C:\cursor\model_training\label_encoder.pkl'
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
        
        # Load model and encoder
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        
        print("Model and label encoder loaded successfully!")
        print(f"Model classes: {label_encoder.classes_}")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def validate_input(data):
    """
    Validate input parameters.
    
    Args:
        data: Dictionary containing input parameters
        
    Returns:
        tuple: (is_valid, error_message)
    """
    required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate data types and ranges
    try:
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        
        # Basic range validation
        if N < 0 or N > 150:
            return False, "Nitrogen (N) should be between 0 and 150"
        if P < 0 or P > 150:
            return False, "Phosphorus (P) should be between 0 and 150"
        if K < 0 or K > 150:
            return False, "Potassium (K) should be between 0 and 150"
        if temperature < 0 or temperature > 50:
            return False, "Temperature should be between 0 and 50°C"
        if humidity < 0 or humidity > 100:
            return False, "Humidity should be between 0 and 100%"
        if ph < 0 or ph > 14:
            return False, "pH should be between 0 and 14"
        if rainfall < 0 or rainfall > 500:
            return False, "Rainfall should be between 0 and 500 mm"
        
        return True, None
        
    except ValueError:
        return False, "All parameters must be numeric values"

@app.route('/', methods=['GET'])
def home():
    """
    Home endpoint to check if API is running.
    """
    return jsonify({
        'message': 'Crop Prediction API is running!',
        'endpoints': {
            '/predict': 'POST - Predict crop based on soil and environmental parameters',
            '/health': 'GET - Check API health'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict crop based on input parameters.
    
    Expected JSON input:
    {
        "N": float,
        "P": float,
        "K": float,
        "temperature": float,
        "humidity": float,
        "ph": float,
        "rainfall": float
    }
    
    Returns:
        JSON response with predicted crop and confidence
    """
    try:
        # Check if model is loaded
        if model is None or label_encoder is None:
            return jsonify({
                'error': 'Model not loaded. Please ensure model files exist.'
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided. Please send JSON data with required fields.'
            }), 400
        
        # Validate input
        is_valid, error_message = validate_input(data)
        if not is_valid:
            return jsonify({
                'error': error_message
            }), 400
        
        # Extract features in the correct order
        features = np.array([[
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]])
        
        # Make prediction
        prediction_encoded = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        # Decode prediction
        predicted_crop = label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get confidence (probability) for the predicted class
        confidence = float(prediction_proba[prediction_encoded])
        
        # Get top 3 predictions with probabilities
        top_indices = np.argsort(prediction_proba)[::-1][:3]
        top_predictions = [
            {
                'crop': label_encoder.inverse_transform([idx])[0],
                'confidence': float(prediction_proba[idx])
            }
            for idx in top_indices
        ]
        
        # Prepare response
        response = {
            'success': True,
            'prediction': predicted_crop,
            'confidence': round(confidence, 4),
            'top_predictions': top_predictions,
            'input_parameters': {
                'N': float(data['N']),
                'P': float(data['P']),
                'K': float(data['K']),
                'temperature': float(data['temperature']),
                'humidity': float(data['humidity']),
                'ph': float(data['ph']),
                'rainfall': float(data['rainfall'])
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 errors.
    """
    return jsonify({
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """
    Handle 500 errors.
    """
    return jsonify({
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Load model when server starts
    print("Loading model...")
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model: {str(e)}")
        print("Please train the model first by running train_model.py")
    
    # Run Flask app
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)

