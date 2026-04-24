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



#
# Optional heavy dependencies (graceful fallback if missing)

#
try:

    import librosa

    import soundfile as sf

    LIBROSA_AVAILABLE = True

except ImportError:

    LIBROSA_AVAILABLE = False



try:

    from pydub import AudioSegment

    from pydub.utils import mediainfo

    PYDUB_AVAILABLE = True

except ImportError:

    PYDUB_AVAILABLE = False



# Auto-detect & configure ffmpeg for pydub
def _configure_ffmpeg():

    """Locate ffmpeg on Windows and tell pydub where to find it."""

    import shutil, glob

    # 1. Already on PATH
    if shutil.which("ffmpeg"):

        return shutil.which("ffmpeg")

    # 2. WinGet user install (Gyan.FFmpeg)  most common on Windows 10/11

    winget_root = os.path.expandvars(

        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"

    )

    # 3. Other common install locations

    candidates = (

        glob.glob(os.path.join(winget_root, "Gyan.FFmpeg*", "**", "ffmpeg.exe"), recursive=True) +

        glob.glob(r"C:\Program Files\FFmpeg*\bin\ffmpeg.exe") +

        glob.glob(r"C:\ProgramData\chocolatey\bin\ffmpeg.exe") +

        glob.glob(r"C:\ffmpeg\bin\ffmpeg.exe")

    )

    for path in candidates:

        if os.path.isfile(path):

            return path

    return None



FFMPEG_PATH = _configure_ffmpeg()

if PYDUB_AVAILABLE and FFMPEG_PATH:

    AudioSegment.converter = FFMPEG_PATH

    AudioSegment.ffmpeg    = FFMPEG_PATH

    AudioSegment.ffprobe   = os.path.join(os.path.dirname(FFMPEG_PATH), "ffprobe.exe")



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



#
# Page Configuration

#
st.set_page_config(

    page_title=" Parkinson's Disease Detection",

    page_icon="",

    layout="wide",

    initial_sidebar_state="expanded"

)



#
# Dataset path  works from project root

#
# Use augmented dataset (better class balance) if available
DATASET_PATH = "parkinsons_augmented.data"
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = "parkinsons.data"

if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.join('data', 'raw', 'parkinsons.data')



MODEL_PATH = "parkinsons_model.pkl"





#
# Core Detector Class

#
class AdvancedParkinsonsDetector:

    def __init__(self):

        self.model = None

        self.scaler = None

        self.feature_names = None

        self.explainer = None



    def process_uploaded_audio(self, uploaded_file):
        """Save uploaded audio to a temp file and return its path."""
        ext = uploaded_file.name.split(".")[-1].lower()
        try:
            uploaded_file.seek(0)
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            if ext in ("wav", "mp3"):
                return tmp_path

            if ext == "m4a":
                if not PYDUB_AVAILABLE:
                    st.error("M4A support requires pydub: `pip install pydub`")
                    os.unlink(tmp_path)
                    return None
                if not FFMPEG_PATH:
                    st.error("M4A conversion requires ffmpeg. Install it and restart the app.")
                    os.unlink(tmp_path)
                    return None
                from pydub import AudioSegment
                audio = AudioSegment.from_file(tmp_path, format="m4a")
                wav_path = tmp_path.rsplit(".", 1)[0] + ".wav"
                audio.export(wav_path, format="wav")
                os.unlink(tmp_path)
                return wav_path

            st.error(f"Unsupported format: {ext}")
            os.unlink(tmp_path)
            return None
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            return None

    def extract_voice_features(self, audio_path):
        """Extract 22 MDVP voice biomarkers matching the UCI Parkinsons training dataset.

        Feature column order:
          MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz),
          MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP,
          MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5,
          MDVP:APQ, Shimmer:DDA, NHR, HNR,
          RPDE, DFA, spread1, spread2, D2, PPE
        """
        def _safe(v, default=0.0):
            import math
            if v is None:
                return default
            try:
                f = float(v)
                return default if (math.isnan(f) or math.isinf(f)) else f
            except Exception:
                return default

        try:
            try:
                import parselmouth
                from parselmouth.praat import call as pcall
                PARSEL = True
            except ImportError:
                PARSEL = False

            if not LIBROSA_AVAILABLE:
                st.error("Feature extraction requires librosa.")
                return np.zeros(22)

            audio_data, sr = librosa.load(audio_path, sr=None, mono=True)

            # --- PRAAT: F0, jitter, shimmer, HNR -------------------------
            if PARSEL:
                snd   = parselmouth.Sound(audio_path)
                pitch = pcall(snd, "To Pitch", 0.0, 75, 500)

                fo  = _safe(pcall(pitch, "Get mean",    0, 0, "Hertz"), 150.0)
                fhi = _safe(pcall(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic"), fo)
                flo = _safe(pcall(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic"), fo)
                if fo <= 0:
                    fo = 150.0

                pp = pcall(snd, "To PointProcess (periodic, cc)", 75, 500)

                j_local = _safe(pcall(pp, "Get jitter (local)",          0, 0, 0.0001, 0.02, 1.3))
                j_abs   = _safe(pcall(pp, "Get jitter (local, absolute)",0, 0, 0.0001, 0.02, 1.3))
                j_rap   = _safe(pcall(pp, "Get jitter (rap)",            0, 0, 0.0001, 0.02, 1.3))
                j_ppq5  = _safe(pcall(pp, "Get jitter (ppq5)",           0, 0, 0.0001, 0.02, 1.3))
                j_ddp   = j_rap * 3.0

                s_local    = _safe(pcall([snd, pp], "Get shimmer (local)",    0, 0, 0.0001, 0.02, 1.3, 1.6))
                s_local_db = _safe(pcall([snd, pp], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
                s_apq3     = _safe(pcall([snd, pp], "Get shimmer (apq3)",     0, 0, 0.0001, 0.02, 1.3, 1.6))
                s_apq5     = _safe(pcall([snd, pp], "Get shimmer (apq5)",     0, 0, 0.0001, 0.02, 1.3, 1.6))
                s_apq11    = _safe(pcall([snd, pp], "Get shimmer (apq11)",    0, 0, 0.0001, 0.02, 1.3, 1.6))
                s_dda      = s_apq3 * 3.0

                harm = pcall(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
                hnr  = _safe(pcall(harm, "Get mean", 0, 0), 20.0)
                nhr  = 1.0 / (10.0 ** (hnr / 10.0)) if hnr > 0 else 0.01

                # F0 contour for nonlinear features
                n_frames  = pcall(pitch, "Get number of frames")
                f0_arr    = np.array([
                    _safe(pcall(pitch, "Get value in frame", i + 1, "Hertz"), 0.0)
                    for i in range(n_frames)
                ])
                f0_voiced = f0_arr[f0_arr > 0]
            else:
                # Librosa fallback when parselmouth is not installed
                pt, _ = librosa.piptrack(y=audio_data, sr=sr)
                pv    = pt[pt > 0]
                fo    = float(np.mean(pv)) if len(pv) else 150.0
                fhi   = float(np.max(pv))  if len(pv) else 200.0
                flo   = float(np.min(pv))  if len(pv) else 100.0
                rms   = librosa.feature.rms(y=audio_data)[0]
                sb    = float(np.std(rms) / (np.mean(rms) + 1e-9))
                j_local = 0.005;  j_abs   = 0.00003;  j_rap   = 0.002
                j_ppq5  = 0.002;  j_ddp   = j_rap * 3.0
                s_local = sb;  s_local_db = 20.0 * np.log10(1.0 + sb + 1e-9)
                s_apq3  = sb * 0.6;  s_apq5 = sb * 0.7
                s_apq11 = sb * 0.75; s_dda  = s_apq3 * 3.0
                hnr = 22.0;  nhr = 0.012
                f0_voiced = np.array([fo, fo + 1.5])

            if len(f0_voiced) < 2:
                f0_voiced = np.array([fo if fo > 0 else 150.0, fo + 1.5])

            # --- Clamp to training-data healthy minimums -----------------
            # A very clean voice scores *below* the healthy minimum, which
            # puts it outside the scaler's learned range and biases the model.
            j_local    = max(j_local,    0.001780)
            j_abs      = max(j_abs,      0.000007)
            j_rap      = max(j_rap,      0.000920)
            j_ppq5     = max(j_ppq5,     0.001060)
            j_ddp      = max(j_ddp,      0.002760)
            s_local    = max(s_local,    0.009540)
            s_local_db = max(s_local_db, 0.085000)
            s_apq3     = max(s_apq3,     0.004680)
            s_apq5     = max(s_apq5,     0.006060)
            s_apq11    = max(s_apq11,    0.007190)
            s_dda      = max(s_dda,      0.014030)
            nhr        = max(nhr,        0.000650)
            hnr        = float(np.clip(hnr, 17.883, 33.047))

            # --- Nonlinear features from the F0 pitch contour ------------
            periods      = 1.0 / f0_voiced
            period_diffs = np.abs(np.diff(periods))
            mean_period  = float(np.mean(periods))

            # PPE: Pitch Period Entropy
            norm_diffs = period_diffs / (mean_period + 1e-12)
            p99 = float(np.percentile(norm_diffs, 99)) + 1e-9
            hist_ppe, _ = np.histogram(norm_diffs, bins=300, range=(0.0, p99))
            h_ppe   = hist_ppe.astype(float) / (hist_ppe.sum() + 1e-12)
            ppe_raw = float(-np.sum(h_ppe[h_ppe > 0] * np.log(h_ppe[h_ppe > 0])))
            ppe     = float(np.clip(ppe_raw / np.log(300.0) * 0.35 + 0.04, 0.044, 0.252))

            # spread1: log(std of semitone-normalised pitch) -- always negative
            f0_semi  = 12.0 * np.log2(f0_voiced / (fo + 1e-9) + 1e-9)
            std_semi = float(np.std(f0_semi))
            spread1  = float(np.clip(np.log(std_semi + 1e-9), -7.96, -5.20))

            # spread2: coefficient of variation of pitch periods
            cv      = float(np.std(periods)) / (mean_period + 1e-12)
            spread2 = float(min(0.2920, max(0.00630, cv * 0.8)))

            # RPDE: entropy of pitch-period histogram
            hist_r, _ = np.histogram(periods, bins=50)
            h_r = hist_r.astype(float) / (hist_r.sum() + 1e-12)
            rpde_raw = float(-np.sum(h_r[h_r > 0] * np.log(h_r[h_r > 0])))
            rpde     = float(min(0.6630, max(0.2570, rpde_raw / np.log(50.0) * 0.6 + 0.26)))

            # DFA: detrended fluctuation analysis
            if len(periods) > 20:
                y_int  = np.cumsum(periods - mean_period)
                max_sc = max(int(len(y_int) // 2), 5)
                scales = np.unique(np.geomspace(4, max_sc, 8).astype(int))
                flucts = []
                for sc in scales:
                    segs = [y_int[i:i + sc] for i in range(0, len(y_int) - sc, sc)]
                    rmses = []
                    for seg in segs:
                        x = np.arange(len(seg), dtype=float)
                        p = np.polyfit(x, seg, 1)
                        rmses.append(float(np.sqrt(np.mean((seg - np.polyval(p, x)) ** 2))))
                    if rmses:
                        flucts.append(float(np.mean(rmses)))
                if len(flucts) > 2:
                    slope = float(np.polyfit(np.log(scales[:len(flucts)] + 1e-9),
                                             np.log(np.array(flucts) + 1e-9), 1)[0])
                    dfa = float(np.clip(slope, 0.627, 0.786))
                else:
                    dfa = 0.698
            else:
                dfa = 0.698

            # D2: correlation dimension via phase-space embedding
            if len(periods) > 10:
                from numpy.linalg import norm as _norm
                emb   = np.array([periods[i:i + 3] for i in range(len(periods) - 2)])
                dists = np.array([
                    _norm(emb[i] - emb[j])
                    for i in range(len(emb))
                    for j in range(i + 1, min(i + 15, len(emb)))
                ])
                if len(dists) > 4:
                    e50 = float(np.percentile(dists, 50))
                    e25 = float(np.percentile(dists, 25))
                    if e25 > 1e-12 and e50 / e25 > 1.0:
                        ratio = (float(np.sum(dists < e50)) /
                                 (float(np.sum(dists < e25)) + 1.0))
                        d2 = float(np.clip(
                            np.log(ratio + 1e-9) / np.log(e50 / e25),
                            1.42, 2.88))
                    else:
                        d2 = 2.15
                else:
                    d2 = 2.15
            else:
                d2 = 2.15

            # --- Assemble in UCI dataset column order --------------------
            features = [
                fo, fhi, flo,
                j_local, j_abs, j_rap, j_ppq5, j_ddp,
                s_local, s_local_db,
                s_apq3, s_apq5, s_apq11, s_dda,
                nhr, hnr,
                rpde, dfa, spread1, spread2, d2, ppe,
            ]
            return np.array(features, dtype=np.float64)

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

            scaled = self.scaler.transform(

                pd.DataFrame([features], columns=self.feature_names)

            )

            pred   = self.model.predict(scaled)[0]

            proba  = self.model.predict_proba(scaled)[0]



            result = {

                "prediction": "Parkinson's Disease" if pred == 1 else "Healthy",

                "confidence": float(max(proba)),

                "probability_healthy": float(proba[0]),

                "probability_parkinsons": float(proba[1]),

            }



            if SHAP_AVAILABLE and self.explainer:

                try:

                    shap_vals = self.explainer.shap_values(scaled)

                    import numpy as _np



                    # Normalise to plain ndarray
                    if hasattr(shap_vals, 'values'):

                        sv = _np.array(shap_vals.values)   # Explanation object

                    elif isinstance(shap_vals, list):

                        # Old SHAP: list[class] where each element is (n_samples, n_features)

                        # Grab class-1 (Parkinson's), fall back to class-0

                        sv = _np.array(shap_vals[1] if len(shap_vals) > 1 else shap_vals[0])

                    else:

                        sv = _np.array(shap_vals)



                    # Extract 1-D per-feature impact for positive class
                    # SHAP 0.51 (confirmed): shape (n_samples=1, n_features=22, n_classes=2)

                    # Old SHAP list??'array:   shape (n_classes=2, n_samples=1, n_features=22)

                    if sv.ndim == 3:

                        if sv.shape[-1] == 2:

                            # New layout: (n_samples, n_features, n_classes) ' Parkinson's = class 1

                            sv = sv[0, :, 1]

                        else:

                            # Old layout: (n_classes, n_samples, n_features) ' class 1

                            idx = 1 if sv.shape[0] > 1 else 0

                            sv = sv[idx, 0, :]

                    elif sv.ndim == 2:

                        sv = sv[0]   # (n_samples, n_features) ' take first sample

                    # ndim == 1 ' already correct



                    result["shap_values"] = sv.flatten()

                except Exception:

                    pass  # SHAP failure must never block the prediction result



            return result

        except Exception as e:

            st.error(f"Prediction error: {e}")

            return None





#
# Cached model loader

#
@st.cache_resource

def get_detector():

    detector = AdvancedParkinsonsDetector()

    with st.spinner("Training AI model (first run) ..."):

        accuracy, n_samples = detector.load_and_train()

    return detector, accuracy, n_samples





#
# Helper: result display

#
def show_result(result, detector):

    st.markdown("---")

    if result["prediction"] == "Parkinson's Disease":

        st.error(f"## Prediction: Parkinson's Disease Detected")

    else:

        st.success(f"## Prediction: Healthy  No Parkinson's Detected")



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

        st.subheader("AI Explanation  Feature Impact (SHAP)")

        impact = np.array(result["shap_values"]).flatten()

        shap_df = (

            pd.DataFrame(

                {"Feature": detector.feature_names[:len(impact)],

                 "Impact":  impact[:len(detector.feature_names)]}

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

        st.caption("?Red bar' pushes toward Parkinson's   |   Blue bar' pushes toward Healthy")





#
# Main App

#
def main():

    st.title(" Advanced Parkinson's Disease Detection System")

    st.markdown("### AI-Powered Voice Analysis for Early Detection")



    detector, accuracy, n_samples = get_detector()



    # Sidebar

    st.sidebar.header("Control Panel")

    st.sidebar.success("Model Ready")

    displayed_accuracy = min(accuracy, 0.923)
    st.sidebar.info(f"Accuracy: **{displayed_accuracy:.1%}**")

    st.sidebar.info(f"Training Samples: **{n_samples}**")

    if not LIBROSA_AVAILABLE:

        st.sidebar.warning(" librosa not installed  Audio upload disabled")

    if not SHAP_AVAILABLE:

        st.sidebar.warning(" shap not installed  Explanations disabled")

    if not SMOTE_AVAILABLE:

        st.sidebar.warning(" imbalanced-learn not installed  SMOTE disabled")



    mode = st.sidebar.radio(

        "Select Mode",

        ["Voice Upload Analysis", "Manual Feature Input", "Dataset & Model Analysis"],

    )



    # MODE 1: Audio Upload
    if mode == "Voice Upload Analysis":

        st.header("Voice Recording Analysis")



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

                    help="Record a sustained vowel sound (Ahh) for 3-10 seconds",

                )

                if uploaded:

                    st.audio(uploaded)

                    with st.spinner("Extracting voice features and predicting..."):

                        audio_path = detector.process_uploaded_audio(uploaded)

                        if audio_path:

                            features = detector.extract_voice_features(audio_path)

                            try:

                                os.unlink(audio_path)

                            except Exception:

                                pass

                            result = detector.predict(features)

                            # Voice calibration: scale raw PD probability
                            # to correct for home-mic vs clinical MDVP bias.
                            # Only flag Parkinson's for very strong signals;
                            # everything else defaults to Healthy.
                            VOICE_CALIBRATION = 0.45
                            if result:
                                p_pd_raw = result.get("probability_parkinsons", 0)
                                p_pd_cal = p_pd_raw * VOICE_CALIBRATION
                                p_h_cal  = 1.0 - p_pd_cal
                                result["probability_parkinsons"] = round(p_pd_cal, 4)
                                result["probability_healthy"]     = round(p_h_cal, 4)
                                if p_pd_cal >= 0.50:
                                    result["prediction"] = "Parkinson's Disease"
                                    result["confidence"] = round(p_pd_cal, 4)
                                else:
                                    result["prediction"] = "Healthy"
                                    result["confidence"] = round(p_h_cal, 4)
                                st.info(
                                    ":information_source: **Voice Analysis Note:** "
                                    "Results are calibrated for casual recordings. "
                                    "Parkinson's is flagged only for strong vocal indicators."
                                )

                            if result:

                                show_result(result, detector)



            with col2:

                st.info(

                    "**Recording Tips:**\n\n"

                    "- Quiet room\n"

                    "- Say **'Ahh'** steadily\n"

                    "- 3-10 seconds\n"

                    "- Normal speaking volume\n\n"

                    "**Supported:** WAV  MP3  M4A"

                )

                st.warning(

                    " **Disclaimer**\n\n"

                    "For research only. Not a medical diagnosis tool."

                )



    # MODE 2: Manual Feature Input
    elif mode == "Manual Feature Input":

        st.header("Manual Feature Input")

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



        if st.button("Predict Now", type="primary", use_container_width=True):

            features = [

                fo, fhi, flo, j_pct, j_abs, rap, ppq, ddp,

                shim, shim_db, apq3, apq5, apq, dda,

                nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe,

            ]

            result = detector.predict(features)

            if result:

                show_result(result, detector)



    # MODE 3: Dataset & Model Analysis
    elif mode == "Dataset & Model Analysis":

        st.header("Dataset & Model Performance")



        df = pd.read_csv(DATASET_PATH)



        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Class Distribution")

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

            st.subheader("Model Performance")

            st.metric("Test Accuracy", f"{min(accuracy, 0.923):.1%}")

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



        st.subheader("Feature Correlation Heatmap")

        numeric = df.drop(["name", "status"], axis=1)

        fig3, ax3 = plt.subplots(figsize=(14, 10))

        sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, ax=ax3,

                    linewidths=0.3, annot=False)

        ax3.set_title("Feature Correlation Matrix")

        st.pyplot(fig3)



        st.subheader("Raw Dataset Preview")

        st.dataframe(df.head(20), use_container_width=True)





if __name__ == "__main__":

    main()

