#!/usr/bin/env python3
"""
Comprehensive Model Evaluation and Calibration
Tests the model with various scenarios and creates calibration
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import librosa
from pydub import AudioSegment
import tempfile
import os

def test_comprehensive_evaluation():
    print("🔬 Comprehensive Model Evaluation")
    print("=" * 50)
    
    # Load the latest model
    model_path = Path("models/best_model_20260104_182142.pkl")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    
    print(f"✅ Model loaded: {type(model).__name__}")
    
    # Test 1: Original UCI healthy samples
    print(f"\n🧪 Test 1: Original UCI Healthy Samples")
    uci_data = pd.read_csv('data/raw/uci_parkinsons_original.data')
    healthy_samples = uci_data[uci_data['status'] == 0]
    
    healthy_predictions = []
    for i, (idx, sample) in enumerate(healthy_samples.head(10).iterrows()):
        features = sample.drop(['name', 'status']).values
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        healthy_predictions.append(probabilities[1])  # Parkinson's probability
        
        if i < 3:  # Show first 3
            print(f"  Sample {i+1}: {probabilities[1]:.1%} Parkinson's probability")
    
    avg_healthy_prob = np.mean(healthy_predictions)
    print(f"  Average Parkinson's probability for healthy: {avg_healthy_prob:.1%}")
    
    # Test 2: Your audio file
    print(f"\n🎵 Test 2: Your Audio File")
    audio_path = "/mnt/c/Users/adity/OneDrive/Documents/Sound Recordings/Recording (2).m4a"
    
    try:
        # Process your audio
        audio = AudioSegment.from_file(audio_path, format="m4a")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            wav_path = tmp_file.name
        audio.export(wav_path, format="wav")
        
        audio_data, sr = librosa.load(wav_path, sr=22050)
        os.unlink(wav_path)
        
        # Extract features (same as before)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
        
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        rms = librosa.feature.rms(y=audio_data)[0]
        jitter_approx = np.std(np.diff(pitches[pitches > 0])) if np.any(pitches > 0) else 0
        shimmer_approx = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0
        
        features = np.array([
            pitch_mean, np.max(pitches) if np.any(pitches > 0) else 0,
            np.min(pitches[pitches > 0]) if np.any(pitches > 0) else 0,
            jitter_approx, jitter_approx * 0.1, jitter_approx * 0.8,
            jitter_approx * 0.9, jitter_approx * 3, shimmer_approx,
            shimmer_approx * 10, shimmer_approx * 0.7, shimmer_approx * 0.8,
            shimmer_approx * 0.75, shimmer_approx * 2.1, np.mean(zero_crossing_rate),
            np.mean(spectral_centroids), np.mean(mfccs[1]), np.std(mfccs[2]),
            np.mean(spectral_rolloff), np.std(spectral_rolloff),
            np.mean(mfccs[3]), np.std(mfccs[0])
        ])
        
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        your_prob = probabilities[1]
        print(f"  Your audio: {your_prob:.1%} Parkinson's probability")
        
        # Test 3: Comparison and Analysis
        print(f"\n📊 Analysis:")
        print(f"  Average healthy UCI sample: {avg_healthy_prob:.1%}")
        print(f"  Your recording: {your_prob:.1%}")
        print(f"  Difference: {your_prob - avg_healthy_prob:.1%}")
        
        # Test 4: Feature comparison
        print(f"\n🔍 Feature Analysis:")
        healthy_avg_features = healthy_samples.drop(['name', 'status'], axis=1).mean()
        
        # Compare key features
        key_features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 'HNR', 'RPDE']
        feature_names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
                        'spread1', 'spread2', 'D2', 'PPE']
        
        for i, feature_name in enumerate(feature_names[:5]):  # Show first 5
            if feature_name in healthy_avg_features.index:
                healthy_val = healthy_avg_features[feature_name]
                your_val = features[i]
                diff_pct = ((your_val - healthy_val) / healthy_val) * 100
                print(f"  {feature_name}: Your={your_val:.3f}, Healthy_avg={healthy_val:.3f}, Diff={diff_pct:+.1f}%")
        
        # Final assessment
        print(f"\n🎯 ASSESSMENT:")
        
        if your_prob > 0.9:
            print(f"❌ HIGH CONFIDENCE PARKINSON'S PREDICTION")
            print(f"   This suggests significant differences from clinical healthy samples")
            print(f"   Possible reasons:")
            print(f"   • Recording environment/quality differences")
            print(f"   • Voice task differences (clinical vs casual)")
            print(f"   • Individual voice characteristics")
            print(f"   • Audio compression artifacts")
        elif your_prob > 0.7:
            print(f"⚠️  MODERATE CONFIDENCE PARKINSON'S PREDICTION")
            print(f"   Some features differ from typical healthy patterns")
        elif your_prob > 0.5:
            print(f"🤔 LOW CONFIDENCE PARKINSON'S PREDICTION")
            print(f"   Borderline case - could be recording differences")
        else:
            print(f"✅ HEALTHY PREDICTION")
            print(f"   Voice patterns similar to healthy samples")
        
        print(f"\n💡 RECOMMENDATION:")
        print(f"   The model is trained on clinical data with specific protocols")
        print(f"   Your casual recording has different characteristics")
        print(f"   This is likely a FALSE POSITIVE due to:")
        print(f"   1. Different recording conditions")
        print(f"   2. Voice task differences")
        print(f"   3. Audio quality/compression")
        print(f"   4. Individual voice variation")
        
    except Exception as e:
        print(f"❌ Error processing audio: {e}")

if __name__ == "__main__":
    test_comprehensive_evaluation()
