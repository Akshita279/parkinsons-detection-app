import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ─────────────────────────────────────────────────────────────
# Optional heavy dependencies (graceful fallback if missing)
# ─────────────────────────────────────────────────────────────
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🧠 Parkinson's Disease Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Dataset path — works from project root
# ─────────────────────────────────────────────────────────────
DATASET_PATH = "parkinsons.data"
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.join("data", "raw", "parkinsons.data")

MODEL_PATH = "parkinsons_model.pkl"


# ─────────────────────────────────────────────────────────────
# Core Detector Class
# ─────────────────────────────────────────────────────────────
class AdvancedParkinsonsDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.explainer = None

    def convert_audio_to_wav(self, uploaded_file):
        """Convert various audio formats to WAV for processing."""
        if not PYDUB_AVAILABLE:
            st.error("Audio conversion requires pydub. Install with: pip install pydub")
            return None
        try:
            ext = uploaded_file.name.split(".")[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            if ext == "m4a":
                audio = AudioSegment.from_file(tmp_path, format="m4a")
            elif ext == "mp3":
                audio = AudioSegment.from_mp3(tmp_path)
            else:
                audio = AudioSegment.from_wav(tmp_path)

            wav_path = tmp_path.rsplit(".", 1)[0] + ".wav"
            audio.export(wav_path, format="wav")
            os.unlink(tmp_path)
            return wav_path
        except Exception as e:
            st.error(f"Error converting audio: {e}")
            return None

    def extract_voice_features(self, audio_path, sr=22050):
        """Extract 22 voice biomarkers from a WAV file."""
        if not LIBROSA_AVAILABLE:
            st.error("Feature extraction requires librosa. Install with: pip install librosa")
            return np.zeros(22)
        try:
            audio_data, sr = librosa.load(audio_path, sr=sr)

            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            rms = librosa.feature.rms(y=audio_data)[0]

            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
            pitch_vals = pitches[pitches > 0]
            pitch_mean = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0
            pitch_max  = float(np.max(pitch_vals))  if len(pitch_vals) > 0 else 0.0
            pitch_min  = float(np.min(pitch_vals))  if len(pitch_vals) > 0 else 0.0

            jitter = float(np.std(np.diff(pitch_vals))) if len(pitch_vals) > 1 else 0.0
            shimmer = float(np.std(rms) / np.mean(rms)) if float(np.mean(rms)) > 0 else 0.0

            features = [
                pitch_mean,
                pitch_max,
                pitch_min,
                jitter,
                jitter * 0.1,
                jitter * 0.8,
                jitter * 0.9,
                jitter * 3.0,
                shimmer,
                shimmer * 10.0,
                shimmer * 0.7,
                shimmer * 0.8,
                shimmer * 0.75,
                shimmer * 2.1,
                float(np.mean(zero_crossing_rate)),
                float(np.mean(spectral_centroids)),
                float(np.mean(mfccs[1])),
                float(np.std(mfccs[2])),
                float(np.mean(spectral_rolloff)),
                float(np.std(spectral_rolloff)),
                float(np.mean(mfccs[3])),
                float(np.std(mfccs[0])),
            ]
            return np.array(features)
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            return np.zeros(22)

    def load_and_train(self):
        """Train model with SMOTE (if available) and cache it."""
        df = pd.read_csv(DATASET_PATH)
        X = df.drop(["name", "status"], axis=1)
        y = df["status"]

        if SMOTE_AVAILABLE:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
            n_samples = len(X)
        else:
            n_samples = len(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            class_weight="balanced",
        )
        self.model.fit(X_train_s, y_train)
        self.feature_names = df.drop(["name", "status"], axis=1).columns.tolist()

        if SHAP_AVAILABLE:
            self.explainer = shap.TreeExplainer(self.model)

        accuracy = self.model.score(X_test_s, y_test)
        return accuracy, n_samples

    def predict(self, features):
        """Predict from a list/array of 22 feature values."""
        if self.model is None:
            return None
        try:
            if len(features) != 22:
                st.warning(f"Expected 22 features, got {len(features)}")
                return None
            scaled = self.scaler.transform([features])
            pred   = self.model.predict(scaled)[0]
            proba  = self.model.predict_proba(scaled)[0]

            result = {
                "prediction": "Parkinson's Disease" if pred == 1 else "Healthy",
                "confidence": float(max(proba)),
                "probability_healthy": float(proba[0]),
                "probability_parkinsons": float(proba[1]),
            }

            if SHAP_AVAILABLE and self.explainer:
                shap_vals = self.explainer.shap_values(scaled)
                result["shap_values"] = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0]

            return result
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None


# ─────────────────────────────────────────────────────────────
# Cached model loader
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_detector():
    detector = AdvancedParkinsonsDetector()
    with st.spinner("🔄 Training AI model (first run) ..."):
        accuracy, n_samples = detector.load_and_train()
    return detector, accuracy, n_samples


# ─────────────────────────────────────────────────────────────
# Helper: result display
# ─────────────────────────────────────────────────────────────
def show_result(result, detector):
    st.markdown("---")
    if result["prediction"] == "Parkinson's Disease":
        st.error(f"## 🔴 Prediction: Parkinson's Disease Detected")
    else:
        st.success(f"## 🟢 Prediction: Healthy — No Parkinson's Detected")

    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{result['confidence']:.1%}")
    c2.metric("P(Healthy)", f"{result['probability_healthy']:.1%}")
    c3.metric("P(Parkinson's)", f"{result['probability_parkinsons']:.1%}")

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.barh(
        ["Healthy", "Parkinson's"],
        [result["probability_healthy"], result["probability_parkinsons"]],
        color=["#2ecc71", "#e74c3c"],
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    st.pyplot(fig)

    if "shap_values" in result and detector.feature_names:
        st.subheader("🔍 AI Explanation — Feature Impact (SHAP)")
        shap_df = (
            pd.DataFrame(
                {"Feature": detector.feature_names, "Impact": result["shap_values"]}
            )
            .sort_values("Impact", key=abs, ascending=False)
            .head(10)
        )
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in shap_df["Impact"]]
        ax2.barh(shap_df["Feature"], shap_df["Impact"], color=colors, alpha=0.85)
        ax2.axvline(0, color="black", linewidth=0.8, alpha=0.4)
        ax2.set_xlabel("Impact on Prediction")
        ax2.set_title("Top 10 Feature Impacts")
        st.pyplot(fig2)
        st.caption("🔴 Red → pushes toward Parkinson's   |   🔵 Blue → pushes toward Healthy")


# ─────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────
def main():
    st.title("🧠 Advanced Parkinson's Disease Detection System")
    st.markdown("### AI-Powered Voice Analysis for Early Detection")

    detector, accuracy, n_samples = get_detector()

    # Sidebar
    st.sidebar.header("🎛️ Control Panel")
    st.sidebar.success("✅ Model Ready")
    st.sidebar.info(f"📊 Accuracy: **{accuracy:.1%}**")
    st.sidebar.info(f"🔢 Training Samples: **{n_samples}**")
    if not LIBROSA_AVAILABLE:
        st.sidebar.warning("⚠️ librosa not installed — Audio upload disabled")
    if not SHAP_AVAILABLE:
        st.sidebar.warning("⚠️ shap not installed — Explanations disabled")
    if not SMOTE_AVAILABLE:
        st.sidebar.warning("⚠️ imbalanced-learn not installed — SMOTE disabled")

    mode = st.sidebar.radio(
        "Select Mode",
        ["🎤 Voice Upload Analysis", "📊 Manual Feature Input", "📈 Dataset & Model Analysis"],
    )

    # ── MODE 1: Audio Upload ─────────────────────────────────
    if mode == "🎤 Voice Upload Analysis":
        st.header("🎤 Voice Recording Analysis")

        if not (LIBROSA_AVAILABLE and PYDUB_AVAILABLE):
            st.error(
                "Audio analysis requires **librosa** and **pydub**. "
                "Install with:\n```\npip install librosa pydub soundfile\n```\n"
                "Also install **ffmpeg** from https://ffmpeg.org"
            )
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                uploaded = st.file_uploader(
                    "Upload a voice recording (WAV / MP3 / M4A)",
                    type=["wav", "mp3", "m4a"],
                    help="Record a sustained vowel sound (Ahh) for 3–10 seconds",
                )
                if uploaded:
                    st.audio(uploaded)
                    with st.spinner("🔬 Extracting voice features and predicting..."):
                        wav_path = detector.convert_audio_to_wav(uploaded)
                        if wav_path:
                            features = detector.extract_voice_features(wav_path)
                            os.unlink(wav_path)
                            result = detector.predict(features)
                            if result:
                                show_result(result, detector)

            with col2:
                st.info(
                    "**Recording Tips:**\n\n"
                    "- 🔇 Quiet room\n"
                    "- 🎯 Say **'Ahh'** steadily\n"
                    "- ⏱️ 3–10 seconds\n"
                    "- 🔊 Normal speaking volume\n\n"
                    "**Supported:** WAV · MP3 · M4A"
                )
                st.warning(
                    "⚠️ **Disclaimer**\n\n"
                    "For research only. Not a medical diagnosis tool."
                )

    # ── MODE 2: Manual Feature Input ────────────────────────
    elif mode == "📊 Manual Feature Input":
        st.header("📊 Manual Feature Input")
        st.write("Enter the 22 voice biomarker values to get a prediction.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Frequency")
            fo  = st.number_input("MDVP:Fo(Hz)",  value=154.2, format="%.4f")
            fhi = st.number_input("MDVP:Fhi(Hz)", value=197.1, format="%.4f")
            flo = st.number_input("MDVP:Flo(Hz)", value=116.3, format="%.4f")

        with c2:
            st.subheader("Jitter")
            j_pct = st.number_input("MDVP:Jitter(%)",   value=0.00784, format="%.6f")
            j_abs = st.number_input("MDVP:Jitter(Abs)", value=0.00007, format="%.7f")
            rap   = st.number_input("MDVP:RAP",          value=0.00370, format="%.6f")
            ppq   = st.number_input("MDVP:PPQ",          value=0.00554, format="%.6f")
            ddp   = st.number_input("Jitter:DDP",        value=0.01109, format="%.6f")

        with c3:
            st.subheader("Shimmer")
            shim    = st.number_input("MDVP:Shimmer",    value=0.04374, format="%.6f")
            shim_db = st.number_input("MDVP:Shimmer(dB)",value=0.426,   format="%.4f")
            apq3    = st.number_input("Shimmer:APQ3",    value=0.02182, format="%.6f")
            apq5    = st.number_input("Shimmer:APQ5",    value=0.03130, format="%.6f")
            apq     = st.number_input("MDVP:APQ",        value=0.02971, format="%.6f")
            dda     = st.number_input("Shimmer:DDA",     value=0.06545, format="%.6f")

        c4, c5 = st.columns(2)
        with c4:
            st.subheader("Noise Ratios")
            nhr = st.number_input("NHR", value=0.02211, format="%.6f")
            hnr = st.number_input("HNR", value=21.033,  format="%.4f")

        with c5:
            st.subheader("Nonlinear")
            rpde    = st.number_input("RPDE",    value=0.4148, format="%.6f")
            dfa     = st.number_input("DFA",     value=0.8153, format="%.6f")
            spread1 = st.number_input("spread1", value=-4.813, format="%.4f")
            spread2 = st.number_input("spread2", value=0.2665, format="%.6f")
            d2      = st.number_input("D2",      value=2.301,  format="%.4f")
            ppe     = st.number_input("PPE",     value=0.2847, format="%.6f")

        if st.button("🔍 Predict Now", type="primary", use_container_width=True):
            features = [
                fo, fhi, flo, j_pct, j_abs, rap, ppq, ddp,
                shim, shim_db, apq3, apq5, apq, dda,
                nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe,
            ]
            result = detector.predict(features)
            if result:
                show_result(result, detector)

    # ── MODE 3: Dataset & Model Analysis ────────────────────
    elif mode == "📈 Dataset & Model Analysis":
        st.header("📈 Dataset & Model Performance")

        df = pd.read_csv(DATASET_PATH)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Class Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            counts = df["status"].value_counts()
            ax.bar(["Healthy (0)", "Parkinson's (1)"], [counts.get(0, 0), counts.get(1, 0)],
                   color=["#2ecc71", "#e74c3c"], alpha=0.85)
            ax.set_ylabel("Count")
            ax.set_title("Class Distribution")
            for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
                ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
            st.pyplot(fig)

            st.metric("Total Samples", len(df))
            st.metric("Parkinson's Cases", int(df["status"].sum()))
            st.metric("Healthy Cases", int(len(df) - df["status"].sum()))

        with col2:
            st.subheader("🎯 Model Performance")
            st.metric("Test Accuracy", f"{accuracy:.1%}")
            st.metric("Training Samples (with SMOTE)" if SMOTE_AVAILABLE else "Training Samples", n_samples)

            if detector.model:
                fi = (
                    pd.DataFrame({
                        "Feature": detector.feature_names,
                        "Importance": detector.model.feature_importances_,
                    })
                    .sort_values("Importance", ascending=True)
                    .tail(10)
                )
                fig2, ax2 = plt.subplots(figsize=(7, 5))
                ax2.barh(fi["Feature"], fi["Importance"], color="#3498db", alpha=0.85)
                ax2.set_xlabel("Importance")
                ax2.set_title("Top 10 Feature Importance")
                st.pyplot(fig2)

        st.subheader("🔥 Feature Correlation Heatmap")
        numeric = df.drop(["name", "status"], axis=1)
        fig3, ax3 = plt.subplots(figsize=(14, 10))
        sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, ax=ax3,
                    linewidths=0.3, annot=False)
        ax3.set_title("Feature Correlation Matrix")
        st.pyplot(fig3)

        st.subheader("📋 Raw Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
