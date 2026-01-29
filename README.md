# 🌾 Crop Prediction System

A complete Machine Learning-based Crop Prediction System that recommends the most suitable crop to grow based on soil and environmental parameters using the Random Forest algorithm.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- 🤖 **Machine Learning Model**: Random Forest Classifier with hyperparameter tuning
- 🌐 **RESTful API**: Flask backend with `/predict` endpoint
- 💻 **Modern UI**: Responsive, mobile-friendly frontend with gradient design
- 📊 **Multiple Predictions**: Shows top 3 crop predictions with confidence scores
- ✅ **Input Validation**: Client-side and server-side validation
- 📱 **Mobile Responsive**: Works seamlessly on desktop, tablet, and mobile devices
- 🎨 **Beautiful Design**: Modern UI with animations and icons

## 📁 Project Structure

```
crop-prediction-system/
│
├── dataset/
│   └── crop_recommendation.csv          # Training dataset
│
├── model_training/
│   ├── train_model.py                   # ML model training script
│   ├── crop_prediction_model.pkl        # Trained model (generated)
│   └── label_encoder.pkl                # Label encoder (generated)
│
├── backend/
│   └── app.py                           # Flask API server
│
├── frontend/
│   ├── index.html                       # Main HTML file
│   ├── styles.css                       # CSS styling
│   └── script.js                        # JavaScript logic
│
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

## 🛠 Technologies Used

### Backend
- **Python 3.8+**
- **Flask**: Web framework for REST API
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **joblib**: Model serialization

### Frontend
- **HTML5**: Structure
- **CSS3**: Styling with modern gradients and animations
- **JavaScript (ES6+)**: Client-side logic
- **Font Awesome**: Icons

### Machine Learning
- **Random Forest Classifier**: Ensemble learning algorithm
- **GridSearchCV**: Hyperparameter tuning
- **Label Encoding**: Categorical encoding

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Web browser (Chrome, Firefox, Safari, Edge)

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd crop-prediction-system

# Or simply download and extract the project folder
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 📖 Usage

### Step 1: Train the Model

First, train the Random Forest model using the dataset:

```bash
cd model_training
python train_model.py
```

This will:
- Load and preprocess the dataset
- Split data into training and testing sets
- Train Random Forest with hyperparameter tuning
- Evaluate the model
- Save the trained model and label encoder

**Expected Output:**
```
Loading dataset...
Dataset loaded successfully! Shape: (2200, 8)
...
Training completed successfully!
Model accuracy: 99.XX%
Model saved at: model_training/crop_prediction_model.pkl
```

### Step 2: Start the Backend Server

Open a new terminal and start the Flask server:

```bash
cd backend
python app.py
```

The server will start on `http://localhost:5000`

**Expected Output:**
```
Loading model...
Model and label encoder loaded successfully!
Starting Flask server...
 * Running on http://0.0.0.0:5000
```

### Step 3: Open the Frontend

1. Open `frontend/index.html` in your web browser
2. Or use a local server:

```bash
# Using Python's built-in server
cd frontend
python -m http.server 8000

# Then open http://localhost:8000 in your browser
```

### Step 4: Make Predictions

1. Fill in the form with soil and environmental parameters:
   - **Nitrogen (N)**: 0-150
   - **Phosphorus (P)**: 0-150
   - **Potassium (K)**: 0-150
   - **Temperature**: 0-50°C
   - **Humidity**: 0-100%
   - **pH Value**: 0-14
   - **Rainfall**: 0-500 mm

2. Click **"Predict Crop"** button

3. View the predicted crop with confidence score and top 3 alternatives

## 📡 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### 2. Predict Crop
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.87,
  "humidity": 82.00,
  "ph": 6.50,
  "rainfall": 202.93
}
```

**Success Response (200):**
```json
{
  "success": true,
  "prediction": "rice",
  "confidence": 0.9876,
  "top_predictions": [
    {
      "crop": "rice",
      "confidence": 0.9876
    },
    {
      "crop": "maize",
      "confidence": 0.0089
    },
    {
      "crop": "chickpea",
      "confidence": 0.0035
    }
  ],
  "input_parameters": {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.87,
    "humidity": 82.00,
    "ph": 6.50,
    "rainfall": 202.93
  }
}
```

**Error Response (400/500):**
```json
{
  "error": "Error message description"
}
```

### Testing with cURL

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.87,
    "humidity": 82.00,
    "ph": 6.50,
    "rainfall": 202.93
  }'
```

## 📊 Dataset

The dataset (`dataset/crop_recommendation.csv`) contains:

- **Features**:
  - `N`: Nitrogen content (0-150)
  - `P`: Phosphorus content (0-150)
  - `K`: Potassium content (0-150)
  - `temperature`: Temperature in Celsius (0-50)
  - `humidity`: Humidity percentage (0-100)
  - `ph`: pH value (0-14)
  - `rainfall`: Rainfall in mm (0-500)

- **Target**:
  - `label`: Crop name (22 different crops)

**Crops included**: rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute

## 🧠 Model Details

### Algorithm
- **Random Forest Classifier**: Ensemble method using multiple decision trees
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Parameters Tuned**:
  - `n_estimators`: [100, 200, 300]
  - `max_depth`: [10, 20, None]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]

### Performance
- **Accuracy**: ~99%+ on test set
- **Evaluation Metrics**: Accuracy, Confusion Matrix, Classification Report

### Model Files
- `crop_prediction_model.pkl`: Trained Random Forest model
- `label_encoder.pkl`: Label encoder for crop names

## 🎨 Screenshots

### Main Interface
- Modern gradient background
- Clean form layout with icons
- Responsive design

### Prediction Result
- Large crop name display
- Confidence percentage
- Top 3 alternative predictions

## 🔧 Troubleshooting

### Issue: Model file not found
**Solution**: Make sure you've trained the model first by running `train_model.py`

### Issue: CORS errors in browser
**Solution**: The Flask app includes `flask-cors`. If issues persist, check that the backend is running on the correct port.

### Issue: Port already in use
**Solution**: Change the port in `backend/app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change port
```

### Issue: Frontend can't connect to backend
**Solution**: Update the `API_URL` in `frontend/script.js` to match your backend URL.

## 🚀 Future Enhancements

- [ ] Add more crops to the dataset
- [ ] Implement user authentication
- [ ] Add historical prediction tracking
- [ ] Create admin dashboard
- [ ] Add data visualization charts
- [ ] Implement model retraining pipeline
- [ ] Add Docker support
- [ ] Deploy to cloud platform

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Built as a complete ML project demonstrating:
- Machine Learning model training
- RESTful API development
- Modern frontend development
- Full-stack integration

## 🙏 Acknowledgments

- scikit-learn community for excellent ML tools
- Flask team for the lightweight web framework
- Font Awesome for beautiful icons

---

**Note**: This is a demonstration project. For production use, consider:
- Adding more data
- Implementing proper error handling
- Adding logging
- Setting up proper deployment infrastructure
- Adding unit tests
- Implementing CI/CD pipeline

---

Made with ❤️ using Random Forest ML Algorithm

