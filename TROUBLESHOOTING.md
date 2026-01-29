# Troubleshooting Guide

## Backend Connection Issues

### Issue: "Unable to connect to the server"

### Solution 1: Start the Backend Server

**Option A: Using PowerShell (Recommended)**
```powershell
cd backend
python app.py
```

**Option B: Using the provided script**
```powershell
.\start_backend.ps1
```

**Option C: Using Batch file**
Double-click `start_backend.bat` or run:
```cmd
start_backend.bat
```

You should see output like:
```
Loading model...
Model and label encoder loaded successfully!
Starting Flask server...
 * Running on http://0.0.0.0:5000
 * Debug mode: on
```

### Solution 2: Verify Backend is Running

Open a **new terminal** and test:
```powershell
# Test 1: Check if port is listening
netstat -ano | findstr :5000

# Test 2: Test health endpoint
python test_backend.py

# Test 3: Manual test
Invoke-WebRequest -Uri http://localhost:5000/health -UseBasicParsing
```

### Solution 3: Check for Port Conflicts

If port 5000 is already in use:

1. **Find what's using port 5000:**
   ```powershell
   netstat -ano | findstr :5000
   ```

2. **Kill the process** (replace PID with actual process ID):
   ```powershell
   taskkill /PID <PID> /F
   ```

3. **Or change the port** in `backend/app.py`:
   ```python
   app.run(debug=True, host='0.0.0.0', port=5001)  # Change to 5001
   ```
   
   And update `frontend/script.js`:
   ```javascript
   const API_URL = 'http://localhost:5001';  // Match the new port
   ```

### Solution 4: Check Model Files

Ensure model files exist:
```powershell
Test-Path model_training\crop_prediction_model.pkl
Test-Path model_training\label_encoder.pkl
```

If missing, train the model first:
```powershell
cd model_training
python train_model.py
```

### Solution 5: Check Firewall/Antivirus

- Windows Firewall might be blocking Python
- Antivirus might be blocking the connection
- Try temporarily disabling firewall/antivirus to test

### Solution 6: CORS Issues

If you see CORS errors in browser console:
- Make sure `flask-cors` is installed: `pip install flask-cors`
- Verify `CORS(app)` is in `backend/app.py`

## Frontend Issues

### Issue: Frontend can't connect to backend

1. **Check API URL** in `frontend/script.js`:
   ```javascript
   const API_URL = 'http://localhost:5000';  // Should match backend port
   ```

2. **Open browser console** (F12) and check for errors

3. **Verify backend is running** before opening frontend

4. **Use a local server** instead of opening HTML directly:
   ```powershell
   cd frontend
   python -m http.server 8000
   ```
   Then open: `http://localhost:8000`

## Common Errors

### Error: "Model file not found"
- Train the model: `cd model_training && python train_model.py`
- Check model files are in `model_training/` folder

### Error: "Module not found"
- Install dependencies: `pip install -r requirements.txt`

### Error: Port already in use
- Kill the process using the port (see Solution 3)
- Or change to a different port

### Error: Connection refused
- Backend is not running - start it first
- Check firewall settings

## Quick Test

Run this to test everything:
```powershell
# 1. Test backend
python test_backend.py

# 2. Test prediction (if backend is running)
python -c "import urllib.request, json; req = urllib.request.Request('http://localhost:5000/predict', json.dumps({'N':90,'P':42,'K':43,'temperature':20.87,'humidity':82,'ph':6.5,'rainfall':202.93}).encode(), {'Content-Type': 'application/json'}); print(json.loads(urllib.request.urlopen(req).read().decode()))"
```

## Still Having Issues?

1. Check all terminal windows - backend must be running
2. Check browser console (F12) for detailed error messages
3. Verify Python version: `python --version` (should be 3.8+)
4. Verify all dependencies: `pip list | findstr flask`

