/**
 * Frontend JavaScript for Crop Prediction System
 * Handles form submission, API calls, and result display
 */

// API endpoint (change this if your backend runs on a different port/URL)
const API_URL = 'http://localhost:5000';

// DOM elements
const form = document.getElementById('predictionForm');
const submitBtn = document.getElementById('submitBtn');
const resultCard = document.getElementById('resultCard');
const errorCard = document.getElementById('errorCard');
const resultContent = document.getElementById('resultContent');
const errorMessage = document.getElementById('errorMessage');
const loadingSpinner = document.getElementById('loadingSpinner');

/**
 * Show loading spinner
 */
function showLoading() {
    loadingSpinner.style.display = 'block';
    resultCard.style.display = 'none';
    errorCard.style.display = 'none';
    submitBtn.disabled = true;
}

/**
 * Hide loading spinner
 */
function hideLoading() {
    loadingSpinner.style.display = 'none';
    submitBtn.disabled = false;
}

/**
 * Show error message
 */
function showError(message) {
    errorMessage.textContent = message;
    errorCard.style.display = 'block';
    resultCard.style.display = 'none';
    hideLoading();
    
    // Scroll to error card
    errorCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Show prediction result
 */
function showResult(data) {
    if (!data.success) {
        showError(data.error || 'Prediction failed');
        return;
    }

    // Hide error card
    errorCard.style.display = 'none';
    
    // Build result HTML
    let resultHTML = `
        <div class="prediction-result">
            <div class="predicted-crop">
                <i class="fas fa-seedling"></i>
                <span>${data.prediction}</span>
            </div>
            <div class="confidence">
                Confidence: <span class="confidence-value">${(data.confidence * 100).toFixed(2)}%</span>
            </div>
    `;

    // Add top predictions if available
    if (data.top_predictions && data.top_predictions.length > 0) {
        resultHTML += `
            <div class="top-predictions">
                <h3><i class="fas fa-list-ol"></i> Top Predictions</h3>
                <div class="prediction-list">
        `;
        
        data.top_predictions.forEach((pred, index) => {
            resultHTML += `
                <div class="prediction-item">
                    <div class="crop-name">${index + 1}. ${pred.crop}</div>
                    <div class="crop-confidence">${(pred.confidence * 100).toFixed(2)}% confidence</div>
                </div>
            `;
        });
        
        resultHTML += `
                </div>
            </div>
        `;
    }

    resultHTML += `</div>`;
    
    // Display result
    resultContent.innerHTML = resultHTML;
    resultCard.style.display = 'block';
    hideLoading();
    
    // Scroll to result card
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Validate form inputs
 */
function validateForm(formData) {
    const fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'];
    
    for (let field of fields) {
        const value = parseFloat(formData.get(field));
        
        if (isNaN(value)) {
            return `Please enter a valid number for ${field}`;
        }
        
        // Range validation
        switch(field) {
            case 'N':
            case 'P':
            case 'K':
                if (value < 0 || value > 150) {
                    return `${field} must be between 0 and 150`;
                }
                break;
            case 'temperature':
                if (value < 0 || value > 50) {
                    return 'Temperature must be between 0 and 50°C';
                }
                break;
            case 'humidity':
                if (value < 0 || value > 100) {
                    return 'Humidity must be between 0 and 100%';
                }
                break;
            case 'ph':
                if (value < 0 || value > 14) {
                    return 'pH must be between 0 and 14';
                }
                break;
            case 'rainfall':
                if (value < 0 || value > 500) {
                    return 'Rainfall must be between 0 and 500 mm';
                }
                break;
        }
    }
    
    return null;
}

/**
 * Handle form submission
 */
async function handleSubmit(event) {
    event.preventDefault();
    
    // Get form data
    const formData = new FormData(form);
    
    // Validate inputs
    const validationError = validateForm(formData);
    if (validationError) {
        showError(validationError);
        return;
    }
    
    // Prepare request data
    const requestData = {
        N: parseFloat(formData.get('N')),
        P: parseFloat(formData.get('P')),
        K: parseFloat(formData.get('K')),
        temperature: parseFloat(formData.get('temperature')),
        humidity: parseFloat(formData.get('humidity')),
        ph: parseFloat(formData.get('ph')),
        rainfall: parseFloat(formData.get('rainfall'))
    };
    
    // Show loading
    showLoading();
    
    try {
        // Make API request
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        // Parse response
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || `HTTP error! status: ${response.status}`);
        }
        
        // Show result
        showResult(data);
        
    } catch (error) {
        console.error('Error:', error);
        
        // Check if it's a network error
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            showError('Unable to connect to the server. Please make sure the backend is running on ' + API_URL);
        } else {
            showError(error.message || 'An error occurred while making the prediction');
        }
    }
}

/**
 * Initialize the application
 */
function init() {
    // Add form submit event listener
    form.addEventListener('submit', handleSubmit);
    
    // Add input event listeners for real-time validation
    const inputs = form.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            // Remove error styling on input
            if (this.validity.valid) {
                this.style.borderColor = '';
            }
        });
    });
    
    console.log('Crop Prediction System initialized');
    console.log('API URL:', API_URL);
}

// Initialize when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

