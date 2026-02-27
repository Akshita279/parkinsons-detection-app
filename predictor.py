import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class ParkinsonsPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def train(self):
        """Train the model and save it"""
        # Load data
        df = pd.read_csv('parkinsons.data')
        X = df.drop(['name', 'status'], axis=1)
        y = df['status']
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        self.feature_names = X.columns.tolist()
        
        # Save model
        with open('parkinsons_model.pkl', 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        
        print("Model trained and saved successfully!")
        
    def load_model(self):
        """Load the trained model"""
        try:
            with open('parkinsons_model.pkl', 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.feature_names = data['feature_names']
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Model not found. Training new model...")
            self.train()
    
    def predict_sample(self, features):
        """Predict Parkinson's for a single sample"""
        if self.model is None:
            self.load_model()
        
        # Convert to DataFrame if it's a list/array
        if isinstance(features, (list, np.ndarray)):
            features = pd.DataFrame([features], columns=self.feature_names)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': 'Parkinson\'s Disease' if prediction == 1 else 'Healthy',
            'confidence': max(probability),
            'probabilities': {
                'Healthy': probability[0],
                'Parkinsons': probability[1]
            }
        }
    
    def predict_from_file(self, filename):
        """Predict from a CSV file"""
        df = pd.read_csv(filename)
        
        # Remove name and status columns if they exist
        feature_cols = [col for col in df.columns if col not in ['name', 'status']]
        X = df[feature_cols]
        
        results = []
        for idx, row in X.iterrows():
            result = self.predict_sample(row.values)
            result['sample_id'] = idx
            if 'name' in df.columns:
                result['name'] = df.loc[idx, 'name']
            results.append(result)
        
        return results

def demo_prediction():
    """Demo with sample data from the dataset"""
    predictor = ParkinsonsPredictor()
    
    # Load a few samples for demo
    df = pd.read_csv('parkinsons.data')
    
    print("=== Parkinson's Disease Prediction Demo ===\n")
    
    # Test with first 5 samples
    for i in range(5):
        sample = df.iloc[i]
        features = sample.drop(['name', 'status']).values
        actual = 'Parkinson\'s Disease' if sample['status'] == 1 else 'Healthy'
        
        result = predictor.predict_sample(features)
        
        print(f"Sample {i+1} ({sample['name']}):")
        print(f"  Actual: {actual}")
        print(f"  Predicted: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Probabilities: Healthy={result['probabilities']['Healthy']:.3f}, "
              f"Parkinson's={result['probabilities']['Parkinsons']:.3f}")
        print()

if __name__ == "__main__":
    demo_prediction()
