#!/usr/bin/env python3
"""
Test manual feature input functionality
"""

import sys
sys.path.append('/mnt/c/Users/adity/PycharmProjects/parkison')

from src.streamlit_app import AdvancedParkinsonsDetector

def test_manual_features():
    print("🧪 Testing Manual Feature Input")
    print("=" * 40)
    
    # Initialize detector
    detector = AdvancedParkinsonsDetector()
    detector.load_and_train_advanced_model()
    
    # Test with typical healthy values
    healthy_features = [
        154.0,    # MDVP:Fo(Hz)
        197.0,    # MDVP:Fhi(Hz)
        116.0,    # MDVP:Flo(Hz)
        0.006,    # MDVP:Jitter(%)
        0.00004,  # MDVP:Jitter(Abs)
        0.003,    # MDVP:RAP
        0.003,    # MDVP:PPQ
        0.009,    # Jitter:DDP
        0.03,     # MDVP:Shimmer
        0.3,      # MDVP:Shimmer(dB)
        0.015,    # Shimmer:APQ3
        0.018,    # Shimmer:APQ5
        0.017,    # MDVP:APQ
        0.045,    # Shimmer:DDA
        0.02,     # NHR
        21.0,     # HNR
        0.5,      # RPDE
        0.7,      # DFA
        -5.0,     # spread1
        0.2,      # spread2
        2.3,      # D2
        0.2       # PPE
    ]
    
    print("Testing with typical healthy values...")
    result = detector.predict_from_features(healthy_features)
    
    if result:
        print(f"✅ Manual feature input working!")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
    else:
        print("❌ Manual feature input failed")
    
    # Test with Parkinson's-like values (higher jitter/shimmer)
    parkinsons_features = healthy_features.copy()
    parkinsons_features[3] = 0.02   # Higher jitter
    parkinsons_features[8] = 0.08   # Higher shimmer
    parkinsons_features[14] = 0.05  # Higher NHR
    parkinsons_features[15] = 15.0  # Lower HNR
    
    print("\nTesting with Parkinson's-like values...")
    result2 = detector.predict_from_features(parkinsons_features)
    
    if result2:
        print(f"✅ Second test working!")
        print(f"Prediction: {result2['prediction']}")
        print(f"Confidence: {result2['confidence']:.1%}")
    else:
        print("❌ Second test failed")

if __name__ == "__main__":
    test_manual_features()
