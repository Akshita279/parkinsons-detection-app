#!/usr/bin/env python3
"""
Parkinson's Disease Detection System
Complete workflow demonstration
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

class ParkinsonsDetectionSystem:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.data = None
        
    def load_data(self, filename='parkinsons.data'):
        """Load the Parkinson's dataset"""
        print("Loading dataset...")
        self.data = pd.read_csv(filename)
        print(f"Dataset loaded: {self.data.shape[0]} samples, {self.data.shape[1]} features")
        
        # Basic statistics
        print(f"Parkinson's cases: {self.data['status'].sum()}")
        print(f"Healthy cases: {len(self.data) - self.data['status'].sum()}")
        
    def explore_data(self):
        """Explore the dataset"""
        if self.data is None:
            self.load_data()
            
        print("\n=== DATA EXPLORATION ===")
        print(f"Dataset shape: {self.data.shape}")
        print(f"Missing values: {self.data.isnull().sum().sum()}")
        
        # Feature statistics
        X = self.data.drop(['name', 'status'], axis=1)
        print(f"\nFeature statistics:")
        print(X.describe())
        
    def train_model(self):
        """Train the Parkinson's detection model"""
        if self.data is None:
            self.load_data()
            
        print("\n=== MODEL TRAINING ===")
        
        # Prepare data
        X = self.data.drop(['name', 'status'], axis=1)
        y = self.data['status']
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return accuracy, feature_importance
        
    def save_model(self, filename='parkinsons_model.pkl'):
        """Save the trained model"""
        if self.model is None:
            print("No model to save. Train a model first.")
            return
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
        
    def load_model(self, filename='parkinsons_model.pkl'):
        """Load a trained model"""
        if not os.path.exists(filename):
            print(f"Model file {filename} not found. Training new model...")
            self.train_model()
            self.save_model(filename)
            return
            
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filename}")
        
    def predict(self, features):
        """Make a prediction for new data"""
        if self.model is None:
            self.load_model()
            
        # Ensure features is in the right format
        if isinstance(features, (list, np.ndarray)):
            features = pd.DataFrame([features], columns=self.feature_names)
        elif isinstance(features, pd.Series):
            features = features.to_frame().T
            
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        result = {
            'prediction': 'Parkinson\'s Disease' if prediction == 1 else 'Healthy',
            'confidence': max(probabilities),
            'probability_healthy': probabilities[0],
            'probability_parkinsons': probabilities[1]
        }
        
        return result
        
    def demo_predictions(self, n_samples=5):
        """Demonstrate predictions on sample data"""
        if self.data is None:
            self.load_data()
            
        print(f"\n=== PREDICTION DEMO ({n_samples} samples) ===")
        
        for i in range(min(n_samples, len(self.data))):
            sample = self.data.iloc[i]
            features = sample.drop(['name', 'status'])
            actual = 'Parkinson\'s Disease' if sample['status'] == 1 else 'Healthy'
            
            result = self.predict(features)
            
            print(f"\nSample {i+1} ({sample['name']}):")
            print(f"  Actual: {actual}")
            print(f"  Predicted: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Probabilities: Healthy={result['probability_healthy']:.3f}, "
                  f"Parkinson's={result['probability_parkinsons']:.3f}")
            
            # Correct prediction?
            correct = "✓" if result['prediction'] == actual else "✗"
            print(f"  Result: {correct}")

def main():
    """Main function to run the complete system"""
    print("🧠 Parkinson's Disease Detection System")
    print("=" * 50)
    
    # Initialize system
    system = ParkinsonsDetectionSystem()
    
    # Load and explore data
    system.load_data()
    system.explore_data()
    
    # Train model
    accuracy, feature_importance = system.train_model()
    
    # Save model
    system.save_model()
    
    # Demo predictions
    system.demo_predictions(10)
    
    print(f"\n🎯 System Summary:")
    print(f"   • Dataset: 195 voice recordings (147 Parkinson's, 48 Healthy)")
    print(f"   • Features: 22 voice biomarkers")
    print(f"   • Model: Random Forest Classifier")
    print(f"   • Accuracy: {accuracy:.1%}")
    print(f"   • Top feature: {feature_importance.iloc[0]['feature']}")
    print(f"\n✅ System ready for predictions!")

if __name__ == "__main__":
    main()
