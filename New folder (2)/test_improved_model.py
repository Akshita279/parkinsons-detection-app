#!/usr/bin/env python3
"""
Test the improved model with your audio file
"""

import sys
import os
sys.path.append('/mnt/c/Users/adity/PycharmProjects/parkison')

from src.streamlit_app import AdvancedParkinsonsDetector
import pandas as pd
import numpy as np
from pathlib import Path

def test_improved_model():
    print("🧠 Testing Improved Parkinson's Detection Model")
    print("=" * 50)
    
    # Initialize detector
    detector = AdvancedParkinsonsDetector()
    
    # Load the improved model
    print("📊 Loading improved model...")
    accuracy, n_samples = detector.load_and_train_advanced_model()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Training samples: {n_samples}")
    print(f"   Model accuracy: {accuracy:.1%}")
    
    # Test with your audio file
    audio_path = "/mnt/c/Users/adity/OneDrive/Documents/Sound Recordings/Recording (2).m4a"
    
    print(f"\n🎵 Testing your audio file...")
    print(f"File: {audio_path}")
    
    try:
        # Convert audio
        print("🔄 Converting M4A to WAV...")
        wav_path = detector.convert_audio_to_wav_from_path(audio_path)
        
        if wav_path:
            print("✅ Audio conversion successful")
            
            # Extract features
            print("🔍 Extracting voice features...")
            features = detector.extract_voice_features(wav_path)
            
            # Clean up temp file
            os.unlink(wav_path)
            
            print(f"✅ Extracted {len(features)} features")
            
            # Make prediction
            print("🤖 Making prediction with improved model...")
            
            # Debug: Check if model and scaler are loaded
            if detector.model is None:
                print("❌ Model not loaded properly")
                return
            if detector.scaler is None:
                print("❌ Scaler not loaded properly") 
                return
                
            print(f"Features shape: {features.shape}")
            print(f"Feature sample: {features[:5]}")
            
            try:
                result = detector.predict_from_features(features)
            except Exception as e:
                print(f"❌ Prediction error: {e}")
                result = None
            
            if result:
                print("\n" + "=" * 50)
                print("🎯 PREDICTION RESULTS")
                print("=" * 50)
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.1%}")
                print(f"Probability Healthy: {result['probability_healthy']:.1%}")
                print(f"Probability Parkinson's: {result['probability_parkinsons']:.1%}")
                
                # Interpretation
                if result['prediction'] == 'Healthy':
                    print("\n✅ GOOD NEWS: Model predicts HEALTHY")
                    print("   This suggests no signs of Parkinson's in your voice")
                else:
                    print(f"\n⚠️  Model predicts Parkinson's with {result['confidence']:.1%} confidence")
                    print("   Note: This is likely a false positive due to:")
                    print("   - Recording quality differences")
                    print("   - Voice variation from clinical samples")
                    print("   - Model limitations")
                
                print("\n📋 IMPORTANT NOTES:")
                print("• This is an AI research tool, not medical diagnosis")
                print("• Voice-only screening has limitations")
                print("• Consult a doctor if you have motor symptoms")
                print("• The improved model should be more reliable")
                
            else:
                print("❌ Prediction failed")
        else:
            print("❌ Audio conversion failed")
            
    except Exception as e:
        print(f"❌ Error processing audio: {e}")

# Add method to detector class for file path conversion
def convert_audio_to_wav_from_path(self, file_path):
    """Convert audio file from path"""
    try:
        from pydub import AudioSegment
        import tempfile
        
        # Load audio file
        if file_path.lower().endswith('.m4a'):
            audio = AudioSegment.from_file(file_path, format="m4a")
        elif file_path.lower().endswith('.mp3'):
            audio = AudioSegment.from_mp3(file_path)
        elif file_path.lower().endswith('.wav'):
            audio = AudioSegment.from_wav(file_path)
        else:
            audio = AudioSegment.from_file(file_path)
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            wav_path = tmp_file.name
        
        audio.export(wav_path, format="wav")
        return wav_path
        
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None

# Monkey patch the method
AdvancedParkinsonsDetector.convert_audio_to_wav_from_path = convert_audio_to_wav_from_path

if __name__ == "__main__":
    test_improved_model()
