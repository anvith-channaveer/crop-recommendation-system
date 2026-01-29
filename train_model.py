"""
Crop Prediction Model Training Script
Uses Random Forest Classifier to predict the most suitable crop
based on soil and environmental parameters.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

def load_data(data_path):
    """
    Load the crop recommendation dataset from CSV file.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        DataFrame containing the dataset
    """
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    print(f"\nDataset Info:")
    print(df.info())
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nCrop distribution:")
    print(df['label'].value_counts())
    return df

def preprocess_data(df):
    """
    Preprocess the dataset: handle missing values, encode labels.
    
    Args:
        df: Input DataFrame
        
    Returns:
        X: Feature matrix
        y: Target vector
        label_encoder: Fitted LabelEncoder
    """
    print("\nPreprocessing data...")
    
    # Check for missing values
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Handle missing values if any (fill with median for numerical columns)
    if df.isnull().sum().any():
        print("Filling missing values...")
        df = df.fillna(df.median())
    
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y_encoded.shape}")
    print(f"Number of unique crops: {len(label_encoder.classes_)}")
    print(f"Crop classes: {label_encoder.classes_}")
    
    return X, y_encoded, label_encoder

def train_random_forest(X_train, y_train):
    """
    Train Random Forest Classifier with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Best trained model
    """
    print("\nTraining Random Forest Classifier...")
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create base Random Forest model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Perform Grid Search with cross-validation
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the trained model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        label_encoder: LabelEncoder to decode predictions
    """
    print("\nEvaluating model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_
    ))
    
    return accuracy

def save_model(model, label_encoder, model_dir='model_training'):
    """
    Save the trained model and label encoder.
    
    Args:
        model: Trained model
        label_encoder: Fitted LabelEncoder
        model_dir: Directory to save the model
    """
    print("\nSaving model...")
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, 'crop_prediction_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save label encoder
    encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    joblib.dump(label_encoder, encoder_path)
    print(f"Label encoder saved to: {encoder_path}")
    
    return model_path, encoder_path

def main():
    """
    Main function to orchestrate the training process.
    """
    # Path to dataset
    data_path = 'C:\cursor\dataset\crop_recommendation.csv'
    
    # Load data
    df = load_data(data_path)
    
    # Preprocess data
    X, y, label_encoder = preprocess_data(df)
    
    # Split data into training and testing sets (80-20 split)
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # Maintain class distribution
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    model = train_random_forest(X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test, label_encoder)
    
    # Save model
    model_path, encoder_path = save_model(model, label_encoder)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print(f"Model accuracy: {accuracy*100:.2f}%")
    print(f"Model saved at: {model_path}")
    print(f"Label encoder saved at: {encoder_path}")
    print("="*50)

if __name__ == "__main__":
    main()

