# 🚀 Setup & Run Guide — Parkinson's Disease Detection System

> **Verified against:** [github.com/adiibaba239/parkisons_disease](https://github.com/adiibaba239/parkisons_disease)

This guide covers how to run the project on **Google Colab**, **Streamlit**, **VS Code**, **PyCharm**, and the **Command Line**.

---

## ✅ Does This Project Have Streamlit?

**YES** — The project includes a full, advanced Streamlit web application located at:
```
src/streamlit_app.py
```
The actual app file is also backed up as `streamlit_app_backup.py`.

### Streamlit App Features:
| Feature | Description |
|---|---|
| 🎤 **Voice Recording Upload** | Upload WAV / MP3 / M4A audio files for live analysis |
| 🔬 **Auto Feature Extraction** | Extracts 22 biomarkers automatically using `librosa` |
| 📊 **Manual Feature Input** | Input all 22 feature values manually via sliders |
| 📈 **Dataset Analysis** | Class distribution, correlation heatmap, model performance |
| 🧠 **SHAP Explanations** | Shows which features drove each prediction (bar chart) |
| ⚖️ **SMOTE Balancing** | Auto-balances imbalanced dataset before training |
| 🎯 **Confidence Score** | Shows probability for Healthy vs Parkinson's |

---

## 📁 Actual Repository Structure (from GitHub)

```
parkisons_disease/
│
├── 📁 data/
│   ├── raw/                    # Original UCI dataset (parkinsons.data)
│   ├── processed/              # Cleaned and combined data
│   └── external/               # Synthetic and external data
│
├── 📁 src/
│   ├── streamlit_app.py        # ← MAIN STREAMLIT WEB APP
│   ├── data/                   # Data processing modules
│   ├── features/               # Feature engineering
│   ├── models/                 # Model definitions
│   └── visualization/          # Plotting and analysis
│
├── 📁 scripts/
│   ├── collect_data.py         # Dataset collection script
│   └── train_models.py         # Model training script
│
├── 📁 models/                  # Trained model files (.pkl)
├── 📁 results/                 # Output visualizations (PNG files)
├── 📁 tests/                   # Unit tests
├── 📁 notebooks/               # Jupyter notebooks
├── 📁 docs/                    # Documentation
├── 📁 config/                  # Configuration files
├── 📁 logs/                    # Training logs
│
├── 📄 main.py                  # CLI pipeline runner
├── 📄 parkinsons_ml_detection.py
├── 📄 parkinsons_detection.py
├── 📄 predictor.py
├── 📄 streamlit_app_backup.py  # Backup of Streamlit app
├── 📄 requirements.txt
├── 📄 start_app.sh             # Shell launcher script
├── 📄 parkinsons.data          # Dataset (UCI)
├── 📄 parkinsons_model.pkl     # Pre-trained model
└── 📄 README.md
```

---

## 📦 Full Dependencies (requirements.txt)

The Streamlit app requires **extra libraries** beyond the base project:

```
# Core ML
pandas
numpy
scikit-learn
tensorflow-cpu
matplotlib
seaborn

# Streamlit App
streamlit

# Audio Processing (for voice upload feature)
librosa
soundfile
pydub

# Explainability
shap

# Data Balancing
imbalanced-learn
```

Install everything:
```bash
pip install pandas numpy scikit-learn tensorflow-cpu matplotlib seaborn \
            streamlit librosa soundfile pydub shap imbalanced-learn
```

> ⚠️ **Note for Windows**: `pydub` requires **ffmpeg** to be installed separately.
> Download ffmpeg: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
> Then add it to your system PATH.

---

## Table of Contents

1. [Run the Streamlit App (Recommended)](#1-run-the-streamlit-app-recommended)
2. [Run on Google Colab](#2-run-on-google-colab)
3. [Run on VS Code](#3-run-on-vs-code)
4. [Run on PyCharm](#4-run-on-pycharm)
5. [Run on Jupyter Notebook](#5-run-on-jupyter-notebook)
6. [Run via Command Line](#6-run-via-command-line)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Run the Streamlit App (Recommended)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/adiibaba239/parkisons_disease.git
cd parkisons_disease
```

### Step 2 — Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install All Dependencies
```bash
pip install pandas numpy scikit-learn tensorflow-cpu matplotlib seaborn \
            streamlit librosa soundfile pydub shap imbalanced-learn
```

### Step 4 — Install ffmpeg (Required for Audio Upload)

**Windows:**
1. Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract the zip
3. Add the `bin` folder to your system **PATH**:  
   `System Properties → Environment Variables → Path → Add`
4. Verify: `ffmpeg -version`

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

### Step 5 — Launch the Streamlit App

**Method A — Using the official path (as per GitHub repo):**
```bash
streamlit run src/streamlit_app.py
```

**Method B — Using the backup file (if src/ folder is missing):**
```bash
streamlit run streamlit_app_backup.py
```

**Method C — Using the start script (Linux/Mac):**
```bash
chmod +x start_app.sh
./start_app.sh
```

### Step 6 — Open in Browser
The app opens automatically at:
```
http://localhost:8501
```

### Step 7 — Using the App

**Mode 1: Voice Recording Analysis**
1. Select **"Voice Recording Analysis"** from the sidebar
2. Click **"Browse files"** to upload an audio file (WAV, MP3, or M4A)
3. The system automatically:
   - Converts your audio to WAV format using `pydub`
   - Extracts 22 voice biomarkers using `librosa`
   - Runs the prediction
4. Results show: Prediction label + Confidence % + Probability bar chart + SHAP explanation

**Mode 2: Manual Feature Input**
1. Select **"Manual Feature Input"** from the sidebar
2. Enter values for all 22 biomarker fields (grouped by category):
   - Frequency Features (Fo, Fhi, Flo)
   - Jitter Features (5 fields)
   - Shimmer Features (6 fields)
   - Noise Features (NHR, HNR)
   - Nonlinear Features (RPDE, DFA, spread1, spread2, D2, PPE)
3. Click **"🔍 Analyze Features"**
4. See prediction + SHAP explanation

**Mode 3: Dataset Analysis**
1. Select **"Dataset Analysis"** from the sidebar
2. View:
   - Class distribution bar chart
   - Model accuracy with SMOTE-balanced training
   - Top 10 feature importance chart
   - Full correlation heatmap

---

## 2. Run on Google Colab

### Step 1 — Open Google Colab
Go to: [https://colab.research.google.com](https://colab.research.google.com)  
Sign in with your Google account → Click **+ New notebook**

### Step 2 — Clone from GitHub
```python
!git clone https://github.com/adiibaba239/parkisons_disease.git
%cd parkisons_disease
```

### Step 3 — Install Dependencies
```python
!pip install pandas numpy scikit-learn matplotlib seaborn \
             librosa soundfile pydub shap imbalanced-learn
# Note: tensorflow and streamlit are pre-installed on Colab
```

### Step 4 — Install ffmpeg (for audio processing)
```python
!apt-get install -y ffmpeg
```

### Step 5 — Collect/Prepare Dataset
```python
# If parkinsons.data is not in data/raw/, upload it manually:
from google.colab import files
uploaded = files.upload()   # Upload parkinsons.data
import shutil
shutil.move('parkinsons.data', 'data/raw/parkinsons.data')
```

### Step 6 — Train the Advanced Model
```python
!python scripts/train_models.py
```

Or run the basic full pipeline:
```python
!python main.py
```

### Step 7 — Run Streamlit on Colab (via ngrok tunnel)
Streamlit needs a public URL to work on Colab:

```python
# Install dependencies
!pip install streamlit pyngrok

# Start app in the background
import subprocess
process = subprocess.Popen(['streamlit', 'run', 'src/streamlit_app.py',
                            '--server.port', '8501',
                            '--server.headless', 'true'])

# Create a public URL with ngrok
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(f"Streamlit App URL: {public_url}")
```

> 📋 Click the printed URL to open the Streamlit app in your browser.

### Step 8 — Run Data Exploration Inline
```python
!python data_exploration.py

# Show generated images directly in Colab
from IPython.display import Image, display
display(Image('results/target_distribution.png'))
display(Image('results/feature_importance.png'))
display(Image('results/model_comparison.png'))
```

### Step 9 — Run Full ML Benchmark
```python
!python parkinsons_ml_detection.py
```

### Step 10 — Download Results
```python
from google.colab import files
files.download('parkinsons_model.pkl')
files.download('results/feature_importance.png')
```

---

## 3. Run on VS Code

### Step 1 — Install VS Code
Download: [https://code.visualstudio.com](https://code.visualstudio.com)

### Step 2 — Install Python Extension
`Ctrl + Shift + X` → Search **"Python"** → Install Microsoft Python extension

### Step 3 — Clone and Open Project
```bash
git clone https://github.com/adiibaba239/parkisons_disease.git
```
Then: `File → Open Folder → Select parkisons_disease folder`

### Step 4 — Create Virtual Environment in VS Code Terminal
Press `` Ctrl + ` `` to open the terminal, then:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 5 — Install All Dependencies
```bash
pip install pandas numpy scikit-learn tensorflow-cpu matplotlib seaborn \
            streamlit librosa soundfile pydub shap imbalanced-learn
```

### Step 6 — Select the Interpreter
`Ctrl + Shift + P` → **"Python: Select Interpreter"** → Choose the `venv` interpreter

### Step 7 — Run the Data Collection Script
```bash
python scripts/collect_data.py
```

### Step 8 — Train Advanced Models
```bash
python scripts/train_models.py
```

### Step 9 — Run the Streamlit App
```bash
streamlit run src/streamlit_app.py
```

### Step 10 — Run Individual Scripts
```bash
python main.py                       # Full pipeline
python data_exploration.py           # EDA & visualizations
python parkinsons_ml_detection.py    # ML model comparison
python parkinsons_detection.py       # Neural network
python predictor.py                  # Prediction demo
```

**Recommended VS Code Extensions:**

| Extension | Purpose |
|---|---|
| Python (Microsoft) | Core Python support |
| Pylance | Autocomplete & type checking |
| Jupyter | Run `.ipynb` notebooks |
| Rainbow CSV | View `.data` / `.csv` files |
| GitLens | GitHub integration |

---

## 4. Run on PyCharm

### Step 1 — Install PyCharm (Community Edition — Free)
Download: [https://www.jetbrains.com/pycharm/download](https://www.jetbrains.com/pycharm/download)

### Step 2 — Clone from GitHub
In PyCharm: `File → New Project from Version Control`  
URL: `https://github.com/adiibaba239/parkisons_disease.git`

### Step 3 — Configure Python Interpreter
`File → Settings → Project: parkisons_disease → Python Interpreter`  
Click ⚙ → **Add Interpreter** → **Virtual Environment** → **New** → Select Python 3.x → OK

### Step 4 — Install Packages
Open **Terminal** in PyCharm (`View → Tool Windows → Terminal`):
```bash
pip install pandas numpy scikit-learn tensorflow-cpu matplotlib seaborn \
            streamlit librosa soundfile pydub shap imbalanced-learn
```

### Step 5 — Run Scripts
Right-click any Python file → **Run 'filename'**  
Or use the ▶ button in the editor toolbar.

### Step 6 — Configure Streamlit Run Configuration
`Run → Edit Configurations → + → Python`

| Field | Value |
|---|---|
| Name | `Streamlit App` |
| Module name | `streamlit` |
| Parameters | `run src/streamlit_app.py` |
| Working directory | `/path/to/parkisons_disease` |

Click **OK** → Press ▶ to launch the app.

### Step 7 — Or Run from PyCharm Terminal
```bash
streamlit run src/streamlit_app.py
```

---

## 5. Run on Jupyter Notebook

### Step 1 — Install Jupyter
```bash
pip install notebook
```

### Step 2 — Launch Jupyter
```bash
jupyter notebook
```
Opens at `http://localhost:8888`

### Step 3 — Create a New Notebook
Click **New → Python 3**

### Step 4 — Run the Project in Cells

```python
# Cell 1 — Install all libraries
!pip install pandas numpy scikit-learn matplotlib seaborn \
             librosa soundfile pydub shap imbalanced-learn streamlit
```

```python
# Cell 2 — Run data training
%run scripts/train_models.py
```

```python
# Cell 3 — Run full pipeline
%run main.py
```

```python
# Cell 4 — Direct prediction with the predictor class
from predictor import ParkinsonsPredictor
import pandas as pd

predictor = ParkinsonsPredictor()
predictor.load_model()

df = pd.read_csv('parkinsons.data')
sample = df.iloc[0].drop(['name', 'status']).values
result = predictor.predict_sample(sample)

print(f"Prediction : {result['prediction']}")
print(f"Confidence : {result['confidence']:.2%}")
print(f"P(Healthy) : {result['probabilities']['Healthy']:.2%}")
print(f"P(PD)      : {result['probabilities']['Parkinsons']:.2%}")
```

```python
# Cell 5 — Show all output images inline
from IPython.display import Image, display
display(Image('results/target_distribution.png'))
display(Image('results/feature_importance.png'))
display(Image('results/model_comparison.png'))
```

---

## 6. Run via Command Line

### Clone the Repo
```bash
git clone https://github.com/adiibaba239/parkisons_disease.git
cd parkisons_disease
```

### Windows (PowerShell)
```powershell
python -m venv venv
venv\Scripts\activate

pip install pandas numpy scikit-learn tensorflow-cpu matplotlib seaborn `
            streamlit librosa soundfile pydub shap imbalanced-learn

# Collect data (if script available)
python scripts/collect_data.py

# Train advanced models
python scripts/train_models.py

# Launch Streamlit app
streamlit run src/streamlit_app.py

# OR run individual scripts
python main.py
python data_exploration.py
python parkinsons_ml_detection.py
python parkinsons_detection.py
python predictor.py
```

### Mac / Linux (Terminal)
```bash
python3 -m venv venv
source venv/bin/activate

pip install pandas numpy scikit-learn tensorflow-cpu matplotlib seaborn \
            streamlit librosa soundfile pydub shap imbalanced-learn

# Install ffmpeg (required for audio upload in Streamlit)
sudo apt-get install ffmpeg    # Ubuntu/Debian
brew install ffmpeg            # macOS

# Train models
python3 scripts/train_models.py

# Launch Streamlit
streamlit run src/streamlit_app.py
```

---

## 7. Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'librosa'`
```bash
pip install librosa soundfile
```

### ❌ `ModuleNotFoundError: No module named 'shap'`
```bash
pip install shap
```

### ❌ `ModuleNotFoundError: No module named 'imblearn'`
```bash
pip install imbalanced-learn
```

### ❌ `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`
`pydub` requires `ffmpeg` for audio conversion. Install it:
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org), extract, add `bin/` folder to PATH
- **Mac**: `brew install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg`

### ❌ Audio file upload not working in Streamlit
Make sure `ffmpeg` is installed and on your **system PATH**.  
Test with: `ffmpeg -version` in terminal.

### ❌ `FileNotFoundError: parkinsons.data`
Make sure you are running from the **project root** folder, OR place `parkinsons.data` in the same folder as the script:
```bash
cd parkisons_disease
python main.py     # Run from root
```

### ❌ `StreamlitAPIException: port 8501 already in use`
```bash
streamlit run src/streamlit_app.py --server.port 8502
```

### ❌ Colab — `FileNotFoundError: parkinsons.data`
Upload the dataset again:
```python
from google.colab import files
uploaded = files.upload()   # Re-upload parkinsons.data
```

### ❌ `ValueError: X has N features but StandardScaler expects M features`
Delete the old model file and retrain:
```bash
del parkinsons_model.pkl     # Windows
rm parkinsons_model.pkl      # Mac/Linux
python main.py               # Retrains fresh
```

### ❌ SHAP slow to compute
SHAP `TreeExplainer` can be slow on first run. This is normal — it will complete. Do not restart the app.

---

## 🔗 Quick Reference

| Task | Command |
|---|---|
| Launch Streamlit App | `streamlit run src/streamlit_app.py` |
| Run full ML pipeline | `python main.py` |
| Train advanced models | `python scripts/train_models.py` |
| Data exploration visuals | `python data_exploration.py` |
| Compare 3 ML models | `python parkinsons_ml_detection.py` |
| Train Neural Network | `python parkinsons_detection.py` |
| Prediction demo (CLI) | `python predictor.py` |
| Clone repo | `git clone https://github.com/adiibaba239/parkisons_disease.git` |

---

> ✅ The project **does include a Streamlit app** at `src/streamlit_app.py`.  
> It supports **audio upload**, **manual feature entry**, **SHAP explanations**, and **dataset visualization**.
