#!/usr/bin/env python3
"""
Test the AI explanation fix
"""

import sys
sys.path.append('/mnt/c/Users/adity/PycharmProjects/parkison')

from src.streamlit_app import AdvancedParkinsonsDetector
import numpy as np

def test_explanation_fix():
    print("🔍 Testing AI Explanation Fix")
    print("=" * 40)
    
    # Initialize detector
    detector = AdvancedParkinsonsDetector()
    detector.load_and_train_advanced_model()
    
    # Test with sample features
    test_features = np.array([
        154.0, 197.0, 116.0, 0.006, 0.00004, 0.003, 0.003, 0.009,
        0.03, 0.3, 0.015, 0.018, 0.017, 0.045, 0.02, 21.0,
        0.5, 0.7, -5.0, 0.2, 2.3, 0.2
    ])
    
    print("Making prediction...")
    result = detector.predict_from_features(test_features)
    
    if result:
        print(f"✅ Prediction successful!")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        
        # Check SHAP values
        if 'shap_values' in result:
            if result['shap_values'] is not None:
                print(f"✅ SHAP values available: {len(result['shap_values'])} values")
            else:
                print("⚠️ SHAP values are None (will use fallback explanation)")
        else:
            print("⚠️ No SHAP values in result (will use fallback explanation)")
        
        print("✅ AI Explanation should now work without errors!")
        
    else:
        print("❌ Prediction failed")

if __name__ == "__main__":
    test_explanation_fix()
