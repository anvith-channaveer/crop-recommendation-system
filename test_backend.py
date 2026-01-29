"""
Simple script to test if the backend is running and accessible
"""
import urllib.request
import json

try:
    # Test health endpoint
    with urllib.request.urlopen('http://localhost:5000/health') as response:
        data = json.loads(response.read().decode())
        print("✅ Backend is running!")
        print(f"Status: {data.get('status')}")
        print(f"Model loaded: {data.get('model_loaded')}")
        
    # Test home endpoint
    with urllib.request.urlopen('http://localhost:5000/') as response:
        data = json.loads(response.read().decode())
        print("\n✅ Home endpoint working!")
        print(f"Message: {data.get('message')}")
        
except urllib.error.URLError as e:
    print(f"❌ Backend is not accessible: {e}")
    print("\nPlease make sure:")
    print("1. Backend server is running (python backend/app.py)")
    print("2. No firewall is blocking port 5000")
    print("3. Port 5000 is not used by another application")

