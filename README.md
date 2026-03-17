# 🧠 Parkinson's Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-red.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-ready-brightgreen.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-92.3%25-success.svg)]()

> An AI-powered voice biomarker analysis system for early detection of Parkinson's Disease — achieves **92.3% accuracy** using Random Forest classification on the UCI Parkinson's Dataset.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Quick Start](#-quick-start)
- [Running the App](#-running-the-app)
- [Documentation](#-documentation)
- [Clinical Relevance](#-clinical-relevance)
- [Future Work](#-future-work)
- [References](#-references)
- [Disclaimer](#️-disclaimer)

---

## 🎯 Project Overview

Parkinson's Disease (PD) is a progressive neurological disorder affecting over 10 million people worldwide. Traditional diagnosis requires in-clinic assessment by a neurologist. This system offers a **non-invasive, voice-based screening** approach using 22 acoustic biomarkers extracted from sustained phonations.

**Key Capabilities:**
- ✅ Binary classification: Parkinson's Disease vs Healthy
- ✅ Three ML algorithms benchmarked (Random Forest, Logistic Regression, SVM)
- ✅ Deep Learning neural network alternative (TensorFlow/Keras)
- ✅ Interactive Streamlit web application
- ✅ Serialized model for real-time inference
- ✅ Detailed visualizations: confusion matrices, feature importance, correlation heatmaps

---

## 📁 Repository Structure

```
parkinsons-disease-detection/
│
├── 📄 main.py                        # Full pipeline orchestrator
├── 📄 parkinsons_ml_detection.py     # ML model benchmarking (RF, LR, SVM)
├── 📄 parkinsons_detection.py        # Deep Learning (TensorFlow/Keras)
├── 📄 predictor.py                   # Reusable prediction class (API-ready)
├── 📄 data_exploration.py            # EDA, visualizations, statistics
├── 📄 streamlit_app.py               # Interactive web application
│
├── 📁 data/raw/
│   └── parkinsons.data               # UCI Dataset (195 samples × 24 cols)
│
├── 📁 src/
│   └── (core source modules)
│
├── 📁 scripts/
│   └── (utility and helper scripts)
│
├── 📁 results/
│   ├── target_distribution.png       # Class imbalance chart
│   ├── correlation_heatmap.png       # Feature correlation heatmap
│   ├── model_comparison.png          # Confusion matrices (all 3 models)
│   └── feature_importance.png        # Top 10 predictive features
│
├── 📁 tests/
│   ├── test_comprehensive.py
│   ├── test_direct.py
│   ├── test_manual_features.py
│   └── test_improved_model.py
│
├── 📄 requirements.txt               # Python dependencies
├── 📄 README.md                      # This file
├── 📄 SETUP_GUIDE.md                 # Detailed setup for Colab, Streamlit, IDEs
├── 📄 PROJECT_DOCUMENTATION.md       # Full technical documentation
├── 📄 RECORDING_PROTOCOL.md          # Voice recording instructions
├── 📄 start_app.sh                   # Shell script to launch Streamlit
├── 📄 .gitignore                     # Git ignore rules
└── 📄 parkinsons_model.pkl           # Pre-trained model (serialized)
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | [UCI ML Repository — Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons) |
| **Total Samples** | 195 voice recordings |
| **Parkinson's cases** | 147 (75.4%) |
| **Healthy cases** | 48 (24.6%) |
| **Input Features** | 22 acoustic biomarkers |
| **Target** | `status`: 1 = Parkinson's, 0 = Healthy |

**Feature Categories:**

| Category | Features |
|---|---|
| Fundamental Frequency | `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)` |
| Jitter (Frequency Variation) | `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP` |
| Shimmer (Amplitude Variation) | `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA` |
| Noise Ratios | `NHR`, `HNR` |
| Nonlinear / Fractal | `RPDE`, `D2`, `DFA`, `spread1`, `spread2`, `PPE` |

---

## 🤖 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Random Forest** ⭐ | **92.3%** | 0.93 | 0.97 | 0.95 |
| Logistic Regression | 92.3% | 0.93 | 0.97 | 0.95 |
| SVM (RBF Kernel) | 92.3% | 0.91 | 1.00 | 0.95 |
| Neural Network (Keras) | ~90-93% | varies | varies | varies |

**Top 5 Most Important Features:**
1. `PPE` — Pitch Period Entropy (15.2%)
2. `spread1` — Nonlinear frequency spread (10.7%)
3. `MDVP:Fo(Hz)` — Average vocal frequency (6.4%)
4. `NHR` — Noise-to-harmonics ratio (6.2%)
5. `Jitter:DDP` — Jitter DDP measure (5.6%)

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/adiibaba239/parkinsons-disease-detection.git
cd parkinsons-disease-detection
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Full System
```bash
python main.py
```

---

## 🖥️ Running the App

### Option A — Streamlit Web App (Recommended)
```bash
pip install streamlit
streamlit run streamlit_app.py
```
Opens at: `http://localhost:8501`

### Option B — Google Colab (No Installation)
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `parkinsons.data` and the Python scripts
3. Run `!pip install scikit-learn matplotlib seaborn`
4. Run cells with `!python main.py`

> 📖 See **[SETUP_GUIDE.md](SETUP_GUIDE.md)** for complete setup instructions for Colab, VS Code, PyCharm, and Jupyter Notebook.

### Option C — Command Line Scripts

| Command | What it does |
|---|---|
| `python main.py` | Full pipeline: load → train → evaluate → predict |
| `python data_exploration.py` | Generate EDA visualizations |
| `python parkinsons_ml_detection.py` | Compare 3 ML models |
| `python parkinsons_detection.py` | Train TensorFlow neural network |
| `python predictor.py` | Demo the prediction interface |

---

## 📚 Documentation

| Document | Description |
|---|---|
| [README.md](README.md) | Project summary (this file) |
| [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) | Full technical docs: architecture, feature analysis, code walkthroughs, execution flows, diagrams |
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Step-by-step instructions for Colab, Streamlit, VS Code, PyCharm |
| [RECORDING_PROTOCOL.md](RECORDING_PROTOCOL.md) | Voice recording guidelines for data collection |

---

## 🔬 Clinical Relevance

Parkinson's Disease affects the neuromotor system controlling laryngeal and respiratory muscles, producing measurable voice abnormalities **years before motor symptoms appear**. This voice-based approach offers:

- 🏥 **Non-invasive**: No blood tests or imaging required
- 🏠 **Remote-friendly**: Recordings can be captured at home
- ⏱️ **Early detection**: Voice changes precede clinical motor symptoms
- 💰 **Cost-effective**: No expensive equipment needed

---

## 🚀 Future Work

- [ ] Real-time voice recording + live feature extraction
- [ ] REST API deployment (Flask / FastAPI)
- [ ] SHAP-based explainability for individual predictions
- [ ] Ensemble model combining RF + SVM + Neural Network
- [ ] Longitudinal patient tracking dashboard
- [ ] Integration with additional biomarkers (gait, handwriting)

---

## 📖 References

- Little, M.A., McSharry, P.E., Roberts, S.J., Costello, D.A.E., Moroz, I.M. (2007). *Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection.* BioMedical Engineering OnLine.
- UCI Machine Learning Repository: [Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)

---

## ⚠️ Disclaimer

**This system is for research and educational purposes only.**  
It must not be used as a substitute for professional medical diagnosis.  
Always consult a qualified healthcare professional for medical decisions.

---

<p align="center">
  Made with ❤️ for early disease detection research
</p>
