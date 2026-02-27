#!/usr/bin/env python3
"""
Test audio file upload fix
"""

import sys
sys.path.append('/mnt/c/Users/adity/PycharmProjects/parkison')

from src.streamlit_app import AdvancedParkinsonsDetector
import tempfile
import os
from pydub import AudioSegment
import numpy as np

def test_audio_upload_fix():
    print("🎵 Testing Audio Upload Fix")
    print("=" * 40)
    
    # Create a test WAV file
    print("Creating test WAV file...")
    
    # Generate a simple sine wave (simulating voice)
    sample_rate = 22050
    duration = 3  # 3 seconds
    frequency = 150  # Hz (typical voice frequency)
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.5
    
    # Convert to 16-bit integers
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create AudioSegment
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit = 2 bytes
        channels=1
    )
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        test_wav_path = tmp_file.name
    
    audio_segment.export(test_wav_path, format="wav")
    print(f"✅ Test WAV file created: {test_wav_path}")
    
    # Test the detector
    detector = AdvancedParkinsonsDetector()
    detector.load_and_train_advanced_model()
    
    # Test feature extraction directly
    print("Testing feature extraction...")
    try:
        features = detector.extract_voice_features(test_wav_path)
        print(f"✅ Feature extraction successful: {len(features)} features")
        print(f"Sample features: {features[:5]}")
        
        # Test prediction
        result = detector.predict_from_features(features)
        if result:
            print(f"✅ Prediction successful!")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.1%}")
        else:
            print("❌ Prediction failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Clean up
    try:
        os.unlink(test_wav_path)
        print("🧹 Test file cleaned up")
    except:
        pass
    
    print("\n✅ Audio upload should now work in Streamlit!")

if __name__ == "__main__":
    test_audio_upload_fix()
