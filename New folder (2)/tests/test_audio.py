import sys
sys.path.append('/mnt/c/Users/adity/PycharmProjects/parkison')

from streamlit_app_fixed import AdvancedParkinsonsDetector
import librosa
import numpy as np
from pydub import AudioSegment
import tempfile
import os

def test_audio_file():
    # Test with your audio file
    audio_path = "/mnt/c/Users/adity/OneDrive/Documents/Sound Recordings/Recording (2).m4a"
    
    print("🎵 Testing Audio File Processing")
    print("=" * 40)
    
    try:
        # Test conversion
        print("Converting M4A to WAV...")
        audio = AudioSegment.from_file(audio_path, format="m4a")
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            wav_path = tmp_file.name
        
        audio.export(wav_path, format="wav")
        print(f"✅ Conversion successful: {wav_path}")
        
        # Test feature extraction
        print("Extracting voice features...")
        detector = AdvancedParkinsonsDetector()
        detector.load_and_train_advanced_model()
        
        features = detector.extract_voice_features(wav_path)
        print(f"✅ Features extracted: {len(features)} features")
        print(f"Feature sample: {features[:5]}")
        
        # Test prediction
        print("Making prediction...")
        result = detector.predict_from_features(features)
        
        if result:
            print("✅ Prediction successful!")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Probabilities: Healthy={result['probability_healthy']:.3f}, Parkinson's={result['probability_parkinsons']:.3f}")
        else:
            print("❌ Prediction failed")
        
        # Clean up
        os.unlink(wav_path)
        print("🧹 Temporary files cleaned up")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_audio_file()
    if success:
        print("\n🎉 Audio processing test PASSED!")
        print("The web interface should now work with M4A files.")
    else:
        print("\n❌ Audio processing test FAILED!")
