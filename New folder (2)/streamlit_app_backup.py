import streamlit as st
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import io
import pickle
from pydub import AudioSegment
import tempfile
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE

# Page config
st.set_page_config(
    page_title="🧠 Parkinson's Disease Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AdvancedParkinsonsDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.explainer = None
        
    def convert_audio_to_wav(self, uploaded_file):
        """Convert various audio formats to WAV for processing"""
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Convert to WAV using pydub
            if uploaded_file.name.lower().endswith('.m4a'):
                audio = AudioSegment.from_file(tmp_path, format="m4a")
            elif uploaded_file.name.lower().endswith('.mp3'):
                audio = AudioSegment.from_mp3(tmp_path)
            elif uploaded_file.name.lower().endswith('.wav'):
                audio = AudioSegment.from_wav(tmp_path)
            else:
                audio = AudioSegment.from_file(tmp_path)
            
            # Convert to WAV
            wav_path = tmp_path.replace(tmp_path.split('.')[-1], 'wav')
            audio.export(wav_path, format="wav")
            
            # Clean up original temp file
            os.unlink(tmp_path)
            
            return wav_path
            
        except Exception as e:
            st.error(f"Error converting audio: {str(e)}")
            return None
    
    def extract_voice_features(self, audio_file_or_data, sr=22050):
        """Extract comprehensive voice features from audio"""
        try:
            # Handle different input types
            if isinstance(audio_file_or_data, str):
                # It's a file path
                audio_data, sr = librosa.load(audio_file_or_data, sr=sr)
            else:
                # It's already audio data
                audio_data = audio_file_or_data
                # It's a file path
                audio_data, sr = librosa.load(audio_file_or_data, sr=sr)
            else:
                # It's already audio data
                audio_data = audio_file_or_data
            
            # Basic features
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
            
            # Compile features (simplified version of the 22 original features)
            features = [
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
            ]
            
            return np.array(features)
            
        except Exception as e:
            st.error(f"Error extracting features: {str(e)}")
            return np.zeros(22)
    
    def load_and_train_advanced_model(self):
        """Load data and train advanced model with SMOTE"""
        try:
            df = pd.read_csv('parkinsons.data')
            X = df.drop(['name', 'status'], axis=1)
            y = df['status']
            
            # Apply SMOTE for balanced dataset
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train optimized Random Forest
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'
            )
            self.model.fit(X_train_scaled, y_train)
            self.feature_names = X.columns.tolist()
            
            # Create SHAP explainer
            self.explainer = shap.TreeExplainer(self.model)
            
            # Evaluate
            accuracy = self.model.score(X_test_scaled, y_test)
            
            return accuracy, len(X_resampled)
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return 0, 0
    
    def predict_from_features(self, features):
        """Predict from extracted features"""
        if self.model is None:
            return None
            
        try:
            # Ensure features are in correct format
            if len(features) != 22:
                st.warning(f"Expected 22 features, got {len(features)}")
                return None
                
            features_scaled = self.scaler.transform([features])
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get SHAP values for explanation
            shap_values = self.explainer.shap_values(features_scaled)
            
            return {
                'prediction': 'Parkinson\'s Disease' if prediction == 1 else 'Healthy',
                'confidence': max(probabilities),
                'probability_healthy': probabilities[0],
                'probability_parkinsons': probabilities[1],
                'shap_values': shap_values[1][0] if len(shap_values) > 1 else shap_values[0]
            }
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None

@st.cache_resource
def load_detector():
    """Load and cache the detector"""
    detector = AdvancedParkinsonsDetector()
    accuracy, n_samples = detector.load_and_train_advanced_model()
    return detector, accuracy, n_samples

def main():
    st.title("🧠 Advanced Parkinson's Disease Detection System")
    st.markdown("### AI-Powered Voice Analysis for Early Detection")
    
    # Sidebar
    st.sidebar.header("🎛️ Control Panel")
    analysis_mode = st.sidebar.selectbox(
        "Select Analysis Mode",
        ["Voice Recording Analysis", "Manual Feature Input", "Dataset Analysis"]
    )
    
    # Load detector
    with st.spinner("Loading AI model..."):
        detector, accuracy, n_samples = load_detector()
    
    st.sidebar.success(f"✅ Model loaded successfully!")
    st.sidebar.info(f"📊 Model Accuracy: {accuracy:.1%}")
    st.sidebar.info(f"🔢 Training Samples: {n_samples}")
    
    if analysis_mode == "Voice Recording Analysis":
        st.header("🎤 Voice Recording Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Audio File")
            uploaded_file = st.file_uploader(
                "Choose an audio file (WAV, MP3, M4A)",
                type=['wav', 'mp3', 'm4a'],
                help="Upload a voice recording for analysis. Recommended: 3-10 seconds of sustained vowel sound."
            )
            
            if uploaded_file is not None:
                # Display audio player
                st.audio(uploaded_file, format='audio/wav')
                
                # Process audio
                with st.spinner("Analyzing voice patterns..."):
                    try:
                        # Convert audio to WAV format
                        wav_path = detector.convert_audio_to_wav(uploaded_file)
                        
                        if wav_path:
                            # Extract features from converted WAV
                            features = detector.extract_voice_features(wav_path)
                            
                            # Clean up temporary file
                            os.unlink(wav_path)
                            
                            # Make prediction
                            result = detector.predict_from_features(features)
                        
                        if result:
                            # Display results
                            st.success("✅ Analysis Complete!")
                            
                            # Main result
                            col_res1, col_res2 = st.columns(2)
                            
                            with col_res1:
                                if result['prediction'] == 'Parkinson\'s Disease':
                                    st.error(f"🚨 **Prediction: {result['prediction']}**")
                                else:
                                    st.success(f"✅ **Prediction: {result['prediction']}**")
                                
                                st.metric("Confidence", f"{result['confidence']:.1%}")
                            
                            with col_res2:
                                # Probability chart
                                prob_data = pd.DataFrame({
                                    'Condition': ['Healthy', 'Parkinson\'s'],
                                    'Probability': [result['probability_healthy'], result['probability_parkinsons']]
                                })
                                
                                fig, ax = plt.subplots(figsize=(6, 4))
                                bars = ax.bar(prob_data['Condition'], prob_data['Probability'], 
                                            color=['green', 'red'], alpha=0.7)
                                ax.set_ylabel('Probability')
                                ax.set_title('Prediction Probabilities')
                                ax.set_ylim(0, 1)
                                
                                # Add value labels on bars
                                for bar, prob in zip(bars, prob_data['Probability']):
                                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                           f'{prob:.1%}', ha='center', va='bottom')
                                
                                st.pyplot(fig)
                            
                            # Feature importance explanation
                            st.subheader("🔍 AI Explanation")
                            st.write("Key factors influencing this prediction:")
                            
                            # SHAP values visualization
                            if 'shap_values' in result:
                                shap_df = pd.DataFrame({
                                    'Feature': detector.feature_names,
                                    'Impact': result['shap_values']
                                }).sort_values('Impact', key=abs, ascending=False).head(10)
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                colors = ['red' if x > 0 else 'blue' for x in shap_df['Impact']]
                                bars = ax.barh(shap_df['Feature'], shap_df['Impact'], color=colors, alpha=0.7)
                                ax.set_xlabel('Impact on Prediction')
                                ax.set_title('Top 10 Feature Impacts (SHAP Values)')
                                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                                
                                st.pyplot(fig)
                                
                                st.caption("🔴 Red bars push toward Parkinson's prediction, 🔵 Blue bars push toward Healthy prediction")
                    
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
        
        with col2:
            st.subheader("📋 Instructions")
            st.info("""
            **For best results:**
            
            1. 📱 Record in a quiet environment
            2. 🎯 Speak clearly and steadily
            3. ⏱️ 3-10 seconds duration
            4. 🔊 Use sustained vowel sounds (Ahh, Ohh)
            5. 📏 Keep consistent distance from microphone
            
            **Supported formats:** WAV, MP3, M4A
            """)
            
            st.warning("""
            ⚠️ **Medical Disclaimer**
            
            This tool is for research and educational purposes only. It should not replace professional medical diagnosis. Always consult healthcare professionals for medical advice.
            """)
    
    elif analysis_mode == "Manual Feature Input":
        st.header("📊 Manual Feature Analysis")
        st.write("Input voice biomarker values manually for analysis")
        
        # Create input fields for key features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Frequency Features")
            fo = st.number_input("MDVP:Fo(Hz)", value=154.0, min_value=50.0, max_value=300.0)
            fhi = st.number_input("MDVP:Fhi(Hz)", value=197.0, min_value=100.0, max_value=600.0)
            flo = st.number_input("MDVP:Flo(Hz)", value=116.0, min_value=50.0, max_value=200.0)
        
        with col2:
            st.subheader("Jitter Features")
            jitter_pct = st.number_input("MDVP:Jitter(%)", value=0.006, min_value=0.0, max_value=0.1, format="%.6f")
            jitter_abs = st.number_input("MDVP:Jitter(Abs)", value=0.00004, min_value=0.0, max_value=0.001, format="%.6f")
            rap = st.number_input("MDVP:RAP", value=0.003, min_value=0.0, max_value=0.1, format="%.6f")
            ppq = st.number_input("MDVP:PPQ", value=0.003, min_value=0.0, max_value=0.1, format="%.6f")
            ddp = st.number_input("Jitter:DDP", value=0.009, min_value=0.0, max_value=0.3, format="%.6f")
        
        with col3:
            st.subheader("Shimmer Features")
            shimmer = st.number_input("MDVP:Shimmer", value=0.03, min_value=0.0, max_value=0.3, format="%.6f")
            shimmer_db = st.number_input("MDVP:Shimmer(dB)", value=0.3, min_value=0.0, max_value=3.0, format="%.3f")
            apq3 = st.number_input("Shimmer:APQ3", value=0.015, min_value=0.0, max_value=0.2, format="%.6f")
            apq5 = st.number_input("Shimmer:APQ5", value=0.018, min_value=0.0, max_value=0.2, format="%.6f")
            apq = st.number_input("MDVP:APQ", value=0.017, min_value=0.0, max_value=0.2, format="%.6f")
            dda = st.number_input("Shimmer:DDA", value=0.045, min_value=0.0, max_value=0.6, format="%.6f")
        
        # Additional features
        col4, col5 = st.columns(2)
        
        with col4:
            st.subheader("Noise Features")
            nhr = st.number_input("NHR", value=0.02, min_value=0.0, max_value=1.0, format="%.6f")
            hnr = st.number_input("HNR", value=21.0, min_value=0.0, max_value=40.0)
        
        with col5:
            st.subheader("Nonlinear Features")
            rpde = st.number_input("RPDE", value=0.5, min_value=0.0, max_value=1.0, format="%.6f")
            dfa = st.number_input("DFA", value=0.7, min_value=0.0, max_value=1.0, format="%.6f")
            spread1 = st.number_input("spread1", value=-5.0, min_value=-10.0, max_value=0.0)
            spread2 = st.number_input("spread2", value=0.2, min_value=0.0, max_value=1.0, format="%.6f")
            d2 = st.number_input("D2", value=2.3, min_value=0.0, max_value=5.0)
            ppe = st.number_input("PPE", value=0.2, min_value=0.0, max_value=1.0, format="%.6f")
        
        if st.button("🔍 Analyze Features", type="primary"):
            # Compile features
            features = [fo, fhi, flo, jitter_pct, jitter_abs, rap, ppq, ddp, 
                       shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                       rpde, dfa, spread1, spread2, d2, ppe]
            
            # Make prediction
            result = detector.predict_from_features(features)
            
            if result:
                # Display results similar to voice analysis
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    if result['prediction'] == 'Parkinson\'s Disease':
                        st.error(f"🚨 **Prediction: {result['prediction']}**")
                    else:
                        st.success(f"✅ **Prediction: {result['prediction']}**")
                    
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                
                with col_res2:
                    # Probability chart
                    prob_data = pd.DataFrame({
                        'Condition': ['Healthy', 'Parkinson\'s'],
                        'Probability': [result['probability_healthy'], result['probability_parkinsons']]
                    })
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    bars = ax.bar(prob_data['Condition'], prob_data['Probability'], 
                                color=['green', 'red'], alpha=0.7)
                    ax.set_ylabel('Probability')
                    ax.set_title('Prediction Probabilities')
                    ax.set_ylim(0, 1)
                    
                    for bar, prob in zip(bars, prob_data['Probability']):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{prob:.1%}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
    
    elif analysis_mode == "Dataset Analysis":
        st.header("📈 Dataset Analysis & Model Performance")
        
        # Load dataset for analysis
        df = pd.read_csv('parkinsons.data')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Dataset Overview")
            st.write(f"**Total Samples:** {len(df)}")
            st.write(f"**Parkinson's Cases:** {df['status'].sum()}")
            st.write(f"**Healthy Cases:** {len(df) - df['status'].sum()}")
            st.write(f"**Features:** {len(df.columns) - 2}")
            
            # Class distribution
            fig, ax = plt.subplots(figsize=(6, 4))
            df['status'].value_counts().plot(kind='bar', ax=ax, color=['green', 'red'], alpha=0.7)
            ax.set_title('Class Distribution')
            ax.set_xlabel('Status (0=Healthy, 1=Parkinson\'s)')
            ax.set_ylabel('Count')
            ax.set_xticklabels(['Healthy', 'Parkinson\'s'], rotation=0)
            st.pyplot(fig)
        
        with col2:
            st.subheader("🎯 Model Performance")
            st.metric("Accuracy", f"{accuracy:.1%}")
            st.metric("Training Samples (with SMOTE)", n_samples)
            
            # Feature importance
            if detector.model:
                feature_importance = pd.DataFrame({
                    'Feature': detector.feature_names,
                    'Importance': detector.model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(data=feature_importance, y='Feature', x='Importance', ax=ax)
                ax.set_title('Top 10 Feature Importance')
                st.pyplot(fig)
        
        # Correlation heatmap
        st.subheader("🔥 Feature Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
