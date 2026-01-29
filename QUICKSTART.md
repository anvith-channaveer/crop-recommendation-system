# 🚀 Quick Start Guide

Get the Crop Prediction System up and running in 5 minutes!

## Prerequisites
- Python 3.8 or higher installed
- pip package manager

## Step-by-Step Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
cd model_training
python train_model.py
```

Wait for training to complete. You should see:
- ✅ Model accuracy (should be ~99%+)
- ✅ Model saved successfully

### 3. Start Backend Server

Open a **new terminal window** and run:

```bash
cd backend
python app.py
```

Keep this terminal open. You should see:
```
 * Running on http://0.0.0.0:5000
```

### 4. Open Frontend

**Option A: Direct File**
- Navigate to `frontend` folder
- Double-click `index.html` to open in your browser

**Option B: Local Server (Recommended)**
Open a **new terminal window** and run:

```bash
cd frontend
python -m http.server 8000
```

Then open: `http://localhost:8000` in your browser

### 5. Make a Prediction!

1. Fill in the form with sample values:
   - N: 90
   - P: 42
   - K: 43
   - Temperature: 20.87
   - Humidity: 82.00
   - pH: 6.50
   - Rainfall: 202.93

2. Click **"Predict Crop"**

3. See the result! 🎉

## Troubleshooting

**Backend not starting?**
- Make sure port 5000 is not in use
- Check that model files exist in `model_training/` folder

**Frontend can't connect?**
- Ensure backend is running on `http://localhost:5000`
- Check browser console for errors

**Model training fails?**
- Verify `dataset/crop_recommendation.csv` exists
- Check Python version (3.8+)

## Sample Test Values

Try these combinations:

**Rice:**
- N: 90, P: 42, K: 43, Temp: 20.87, Humidity: 82, pH: 6.5, Rainfall: 202.93

**Maize:**
- N: 20, P: 30, K: 20, Temp: 28.59, Humidity: 90, pH: 6.46, Rainfall: 80.55

**Banana:**
- N: 115, P: 58, K: 46, Temp: 25.60, Humidity: 71, pH: 6.31, Rainfall: 104.46

---

Happy Predicting! 🌾

