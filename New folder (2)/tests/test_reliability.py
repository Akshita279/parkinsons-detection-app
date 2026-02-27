import pandas as pd
import numpy as np
from streamlit_app import AdvancedParkinsonsDetector

def test_model_reliability():
    print("🔬 Testing Model Reliability")
    print("=" * 40)
    
    # Load the dataset to see actual distribution
    df = pd.read_csv('parkinsons.data')
    
    print(f"Dataset Info:")
    print(f"Total samples: {len(df)}")
    print(f"Parkinson's cases: {df['status'].sum()} ({df['status'].mean():.1%})")
    print(f"Healthy cases: {len(df) - df['status'].sum()} ({1-df['status'].mean():.1%})")
    
    # Test with some healthy samples from the dataset
    detector = AdvancedParkinsonsDetector()
    detector.load_and_train_advanced_model()
    
    print(f"\n🧪 Testing on Known Healthy Samples:")
    healthy_samples = df[df['status'] == 0].head(5)
    
    false_positives = 0
    for i, (idx, sample) in enumerate(healthy_samples.iterrows()):
        features = sample.drop(['name', 'status']).values
        result = detector.predict_from_features(features)
        
        if result:
            prediction = result['prediction']
            confidence = result['confidence']
            
            print(f"Sample {i+1} ({sample['name']}):")
            print(f"  Actual: Healthy")
            print(f"  Predicted: {prediction}")
            print(f"  Confidence: {confidence:.1%}")
            
            if prediction == "Parkinson's Disease":
                false_positives += 1
                print(f"  ❌ FALSE POSITIVE!")
            else:
                print(f"  ✅ Correct")
    
    print(f"\n📊 False Positive Rate on Known Healthy: {false_positives}/5 = {false_positives/5:.1%}")
    
    if false_positives > 0:
        print(f"\n⚠️  HIGH FALSE POSITIVE RATE DETECTED!")
        print(f"The model is incorrectly predicting healthy people as having Parkinson's.")
        print(f"This explains why your recording got a positive prediction.")
    
    print(f"\n🎯 Recommendations:")
    print(f"1. This is likely a FALSE POSITIVE")
    print(f"2. The model needs more training data")
    print(f"3. Voice-only diagnosis is not medically reliable")
    print(f"4. Consult a doctor if you have actual motor symptoms")

if __name__ == "__main__":
    test_model_reliability()
