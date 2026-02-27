#!/usr/bin/env python3
"""
Test model reliability on known samples
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def test_model_reliability():
    print("🔬 Testing Model Reliability")
    print("=" * 40)
    
    # Load the best model
    model_path = Path("models/best_model_20260104_174450.pkl")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Load original dataset to test on known samples
    df = pd.read_csv('data/raw/uci_parkinsons.data')
    
    print(f"Testing on original UCI dataset...")
    print(f"Total samples: {len(df)}")
    
    # Test on known healthy samples
    healthy_samples = df[df['status'] == 0]
    parkinsons_samples = df[df['status'] == 1]
    
    print(f"Healthy samples: {len(healthy_samples)}")
    print(f"Parkinson's samples: {len(parkinsons_samples)}")
    
    # Test healthy samples
    print(f"\n🧪 Testing on known HEALTHY samples:")
    false_positives = 0
    
    for i, (idx, sample) in enumerate(healthy_samples.head(10).iterrows()):
        features = sample.drop(['name', 'status']).values
        features_scaled = scaler.transform([features])
        
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        result = 'Parkinson\'s Disease' if prediction == 1 else 'Healthy'
        confidence = max(probabilities)
        
        print(f"Sample {i+1} ({sample['name']}): {result} ({confidence:.1%})")
        
        if prediction == 1:
            false_positives += 1
    
    print(f"\n📊 False Positive Rate: {false_positives}/10 = {false_positives/10:.1%}")
    
    # Test Parkinson's samples
    print(f"\n🧪 Testing on known PARKINSON'S samples:")
    false_negatives = 0
    
    for i, (idx, sample) in enumerate(parkinsons_samples.head(10).iterrows()):
        features = sample.drop(['name', 'status']).values
        features_scaled = scaler.transform([features])
        
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        result = 'Parkinson\'s Disease' if prediction == 1 else 'Healthy'
        confidence = max(probabilities)
        
        print(f"Sample {i+1} ({sample['name']}): {result} ({confidence:.1%})")
        
        if prediction == 0:
            false_negatives += 1
    
    print(f"\n📊 False Negative Rate: {false_negatives}/10 = {false_negatives/10:.1%}")
    
    # Overall assessment
    if false_positives > 3:
        print(f"\n⚠️  HIGH FALSE POSITIVE RATE!")
        print(f"The model is over-predicting Parkinson's disease")
        print(f"Your recording result is likely a FALSE POSITIVE")
    elif false_positives == 0:
        print(f"\n🤔 Model seems accurate on dataset samples")
        print(f"Your recording might have different characteristics")
        print(f"Possible reasons:")
        print(f"• Different recording conditions")
        print(f"• Voice task differences") 
        print(f"• Audio quality/compression")
        print(f"• Individual voice variation")
    else:
        print(f"\n📊 Model shows some false positives ({false_positives}/10)")
        print(f"Your result could be a false positive")

if __name__ == "__main__":
    test_model_reliability()
