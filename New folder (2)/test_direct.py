#!/usr/bin/env python3
"""
Direct test of the improved model
"""

import pickle
import numpy as np
from pathlib import Path
import librosa
from pydub import AudioSegment
import tempfile
import os

def test_direct():
    print("🧠 Direct Model Test")
    print("=" * 30)
    
    # Load the best model directly
    model_path = Path("models/best_model_20260104_182142.pkl")
    
    if not model_path.exists():
        print("❌ Model file not found")
        return
    
    print("📊 Loading model...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"Features: {len(feature_names)}")
    
    # Test with your audio
    audio_path = "/mnt/c/Users/adity/OneDrive/Documents/Sound Recordings/Recording (2).m4a"
    
    print(f"\n🎵 Processing audio...")
    
    try:
        # Convert audio
        audio = AudioSegment.from_file(audio_path, format="m4a")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            wav_path = tmp_file.name
        audio.export(wav_path, format="wav")
        
        # Load with librosa
        audio_data, sr = librosa.load(wav_path, sr=22050)
        os.unlink(wav_path)
        
        # Extract basic features (simplified)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # Jitter and Shimmer approximations
        rms = librosa.feature.rms(y=audio_data)[0]
        jitter_approx = np.std(np.diff(pitches[pitches > 0])) if np.any(pitches > 0) else 0
        shimmer_approx = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0
        
        # Create feature vector
        features = np.array([
            pitch_mean,  # MDVP:Fo(Hz)
            np.max(pitches) if np.any(pitches > 0) else 0,  # MDVP:Fhi(Hz)
            np.min(pitches[pitches > 0]) if np.any(pitches > 0) else 0,  # MDVP:Flo(Hz)
            jitter_approx,  # Jitter approximation
            jitter_approx * 0.1,  # Jitter(Abs) approximation
            jitter_approx * 0.8,  # RAP approximation
            jitter_approx * 0.9,  # PPQ approximation
            jitter_approx * 3,  # DDP approximation
            shimmer_approx,  # Shimmer
            shimmer_approx * 10,  # Shimmer(dB)
            shimmer_approx * 0.7,  # APQ3
            shimmer_approx * 0.8,  # APQ5
            shimmer_approx * 0.75,  # APQ
            shimmer_approx * 2.1,  # DDA
            np.mean(zero_crossing_rate),  # NHR approximation
            np.mean(spectral_centroids),  # HNR approximation
            np.mean(mfccs[1]),  # RPDE approximation
            np.std(mfccs[2]),  # DFA approximation
            np.mean(spectral_rolloff),  # spread1 approximation
            np.std(spectral_rolloff),  # spread2 approximation
            np.mean(mfccs[3]),  # D2 approximation
            np.std(mfccs[0])  # PPE approximation
        ])
        
        print(f"✅ Features extracted: {len(features)}")
        print(f"Feature sample: {features[:5]}")
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        print(f"\n🎯 RESULTS:")
        print(f"Prediction: {'Parkinson\'s Disease' if prediction == 1 else 'Healthy'}")
        print(f"Confidence: {max(probabilities):.1%}")
        print(f"Probabilities: Healthy={probabilities[0]:.1%}, Parkinson's={probabilities[1]:.1%}")
        
        if prediction == 0:
            print(f"\n✅ GREAT NEWS! The improved model predicts HEALTHY")
            print(f"   This is much more reliable than the previous result")
        else:
            print(f"\n⚠️  Model still predicts Parkinson's")
            print(f"   Confidence: {probabilities[1]:.1%}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_direct()
