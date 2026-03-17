# 🧠 Parkinson's Disease Detection System — Full Technical Documentation

> **Disclaimer**: This system is for research and educational purposes only. It should NOT be used as a substitute for professional medical diagnosis.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project File Tree Structure](#2-project-file-tree-structure)
3. [Dataset Description](#3-dataset-description)
4. [Feature Reference Table](#4-feature-reference-table)
5. [System Architecture Overview](#5-system-architecture-overview)
6. [Application Execution Flow Diagram](#6-application-execution-flow-diagram)
7. [Module Deep Dives](#7-module-deep-dives)
   - 7.1 [data_exploration.py](#71-data_explorationpy)
   - 7.2 [parkinsons_ml_detection.py](#72-parkinsons_ml_detectionpy)
   - 7.3 [parkinsons_detection.py](#73-parkinsons_detectionpy)
   - 7.4 [predictor.py](#74-predictorpy)
   - 7.5 [main.py](#75-mainpy)
8. [ML Model Comparison](#8-ml-model-comparison)
9. [Neural Network Architecture Diagram](#9-neural-network-architecture-diagram)
10. [Input / Output Reference](#10-input--output-reference)
11. [Data Preprocessing Pipeline](#11-data-preprocessing-pipeline)
12. [Prediction Flow](#12-prediction-flow)
13. [Model Persistence Flow](#13-model-persistence-flow)
14. [How to Run — Step by Step](#14-how-to-run--step-by-step)
15. [Output Files Reference](#15-output-files-reference)
16. [Clinical Background](#16-clinical-background)
17. [Future Enhancements](#17-future-enhancements)

---

## 1. Project Overview

The **Parkinson's Disease Detection System** is an AI-powered diagnostic support tool that uses voice signal analysis to classify patients as either *Parkinson's positive* or *Healthy*. It is built entirely in Python and uses classical machine learning models as well as deep learning neural networks to achieve high-accuracy predictions.

### Key Highlights

| Property | Value |
|---|---|
| **Dataset** | UCI Parkinson's Dataset |
| **Samples** | 195 voice recordings |
| **Parkinson's cases** | 147 (75.4%) |
| **Healthy cases** | 48 (24.6%) |
| **Input features** | 22 voice biomarker columns |
| **Best model** | Random Forest Classifier |
| **Best accuracy** | ~92.3% |
| **Language** | Python 3 |
| **Key Libraries** | scikit-learn, TensorFlow/Keras, Pandas, NumPy, Matplotlib, Seaborn |

---

## 2. Project File Tree Structure

```
akshita_project/
│
├── 📄 main.py                        # Primary orchestrator — runs the full pipeline
├── 📄 parkinsons_ml_detection.py     # Multi-model ML comparison engine
├── 📄 parkinsons_detection.py        # Deep Learning (TensorFlow) training module
├── 📄 predictor.py                   # Operational Prediction API class
├── 📄 data_exploration.py            # EDA — visualization and statistics
│
├── 📁 Data Files
│   └── parkinsons.data               # UCI raw CSV dataset (195 rows × 24 cols)
│
├── 📁 Saved Model Files
│   └── parkinsons_model.pkl          # Serialized Random Forest + Scaler (binary)
│
├── 📁 Generated Visualizations
│   ├── target_distribution.png       # Class imbalance bar chart
│   ├── correlation_heatmap.png       # Feature correlation heatmap
│   ├── model_comparison.png          # Confusion matrices side-by-side
│   └── feature_importance.png        # Top 10 most predictive features
│
├── 📄 requirements.txt               # Python package dependencies
├── 📄 README.md                      # Project summary
└── 📄 PROJECT_DOCUMENTATION.md       # ← This file (full documentation)
```

---

## 3. Dataset Description

The data used in this project is sourced from the **UCI Machine Learning Repository** and was originally studied by Max A. Little, Patrick E. McSharry et al. (2007).

```
Source : UCI ML Repository — Parkinson's Dataset
File   : parkinsons.data
Format : CSV (Comma-separated values)
Rows   : 195 (one voice recording per row)
Columns: 24 total (1 name + 22 features + 1 label)
```

### Data Sample (first 2 rows):

| name | MDVP:Fo(Hz) | MDVP:Fhi(Hz) | ... | status |
|---|---|---|---|---|
| phon_R01_S01_1 | 119.992 | 157.302 | ... | 1 |
| phon_R01_S07_1 | 197.076 | 206.896 | ... | 0 |

- Each **patient** may have **multiple recordings** (e.g., 6–7 recordings).
- Column `name` is a unique identifier formatted as `phon_RXXX_SXX_N`.
- Column `status` is the ground truth label: `1 = Parkinson's`, `0 = Healthy`.

---

## 4. Feature Reference Table

All 22 input features used during training are described below:

### 4A — Fundamental Frequency Features (Pitch)

| Feature | Full Name | Description |
|---|---|---|
| `MDVP:Fo(Hz)` | Average vocal frequency | Mean fundamental pitch of the voice over recording |
| `MDVP:Fhi(Hz)` | Maximum vocal frequency | Highest pitch frequency detected |
| `MDVP:Flo(Hz)` | Minimum vocal frequency | Lowest pitch frequency detected |

### 4B — Jitter Features (Frequency Variation)
> Jitter captures how much the pitch "wobbles" between cycles. Higher jitter = more neurological tremor in voice.

| Feature | Full Name | Description |
|---|---|---|
| `MDVP:Jitter(%)` | Jitter percentage | Cycle-to-cycle deviation of fundamental frequency |
| `MDVP:Jitter(Abs)` | Absolute jitter | Absolute value of cycle frequency deviation |
| `MDVP:RAP` | Relative Average Perturbation | Frequency perturbation ratio over 3 periods |
| `MDVP:PPQ` | Period Perturbation Quotient | Frequency perturbation over 5 periods |
| `Jitter:DDP` | Difference of Differences | Average absolute difference between jitter cycles |

### 4C — Shimmer Features (Amplitude Variation)
> Shimmer captures variation in loudness between cycles. Higher shimmer = unstable voice amplitude = disease signal.

| Feature | Full Name | Description |
|---|---|---|
| `MDVP:Shimmer` | Shimmer | Amplitude variation between adjacent cycles |
| `MDVP:Shimmer(dB)` | Shimmer in decibels | Log-scale version of shimmer |
| `Shimmer:APQ3` | 3-point APQ | Amplitude perturbation quotient over 3 cycles |
| `Shimmer:APQ5` | 5-point APQ | Amplitude perturbation quotient over 5 cycles |
| `MDVP:APQ` | 11-point APQ | Amplitude perturbation quotient over 11 cycles |
| `Shimmer:DDA` | DDA Shimmer | Average abs. difference between 3-cycle amplitudes |

### 4D — Noise Ratio Features

| Feature | Full Name | Description |
|---|---|---|
| `NHR` | Noise-to-Harmonics Ratio | Higher NHR = more noise, less structured voice |
| `HNR` | Harmonics-to-Noise Ratio | Lower HNR = degraded voice quality (disease indicator) |

### 4E — Nonlinear Dynamical Complexity Features
> These fractal and chaos-based features capture subtle long-range patterns in voice signal behavior.

| Feature | Full Name | Description |
|---|---|---|
| `RPDE` | Recurrence Period Density Entropy | Measures voice signal complexity and repetition |
| `D2` | Correlation Dimension | Captures dimensionality of the vocal signal attractor |
| `DFA` | Detrended Fluctuation Analysis | Measures self-affinity and turbulence in voice |
| `spread1` | Fundamental frequency variation 1 | Nonlinear measure of frequency spread |
| `spread2` | Fundamental frequency variation 2 | Secondary nonlinear spread measure |
| `PPE` | Pitch Period Entropy | Entropy-based measure of vocal irregularity (Top predictor!) |

---

## 5. System Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                              │
│              parkinsons.data (195 recordings)               │
│              22 voice biomarker features per row            │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                  PREPROCESSING LAYER                         │
│   1. Drop 'name' column (non-informative string ID)         │
│   2. Separate X (features) and y (status label)             │
│   3. Train/Test split: 80% / 20% (stratified)              │
│   4. StandardScaler: Normalize to Mean=0, Variance=1       │
└──────────────────┬───────────────────────────────────────────┘
                   │
       ┌───────────┼──────────────┐
       ▼           ▼              ▼
 Random Forest  Logistic     Neural Net
  Classifier   Regression   (TensorFlow)
    (main)       (SVM)       (keras)
       │           │              │
       └───────────┼──────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                   EVALUATION LAYER                           │
│  Accuracy Score / Classification Report / Confusion Matrix  │
│  K-Fold Cross Validation (K=5) / Feature Importance Rank   │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                SERIALIZATION LAYER                           │
│     Save model weights + scaler → parkinsons_model.pkl     │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                  INFERENCE LAYER                             │
│    Load .pkl → Apply scaler → Predict → Return result      │
│    Output: Healthy / Parkinson's + Confidence Score         │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. Application Execution Flow Diagram

```
START
  │
  ├── python main.py
  │     │
  │     ├─> load_data()
  │     │     └─> pd.read_csv('parkinsons.data')
  │     │         └─> Print shape & class counts
  │     │
  │     ├─> explore_data()
  │     │     └─> Print statistics for all features
  │     │
  │     ├─> train_model()
  │     │     ├─> Drop 'name', 'status' → X, y
  │     │     ├─> train_test_split (80/20, stratified)
  │     │     ├─> StandardScaler().fit_transform(X_train)
  │     │     ├─> RandomForestClassifier(n_estimators=100)
  │     │     ├─> model.fit(X_train_scaled, y_train)
  │     │     ├─> model.predict(X_test_scaled) → y_pred
  │     │     ├─> accuracy_score() → Print accuracy
  │     │     ├─> classification_report() → Print precision/recall
  │     │     ├─> cross_val_score() (cv=5) → Print CV means
  │     │     └─> feature_importances_ → Top 10 features
  │     │
  │     ├─> save_model()
  │     │     └─> pickle.dump({model, scaler, feature_names})
  │     │         → parkinsons_model.pkl
  │     │
  │     └─> demo_predictions(n=10)
  │           ├─> Pick 10 rows from dataset
  │           ├─> Drop label → simulate "unseen" data
  │           ├─> predict() → get result dict
  │           └─> Compare prediction vs actual → ✓ or ✗
  │
  ├── python data_exploration.py
  │     ├─> pd.read_csv()
  │     ├─> df.info() / df.shape / .isnull().sum()
  │     ├─> status.value_counts() → class distribution
  │     ├─> matplotlib bar chart → target_distribution.png
  │     └─> df.corr() → seaborn heatmap → correlation_heatmap.png
  │
  ├── python parkinsons_ml_detection.py
  │     ├─> load_and_preprocess_data()
  │     ├─> train_models()
  │     │     ├─> RandomForestClassifier()
  │     │     ├─> LogisticRegression(max_iter=1000)
  │     │     └─> SVC(probability=True)
  │     ├─> evaluate_models()
  │     │     └─> accuracy + classification_report per model
  │     ├─> best model → max(accuracy)
  │     ├─> plot_results()
  │     │     └─> 3 confusion matrices → model_comparison.png
  │     └─> feature_importances_ → feature_importance.png
  │
  ├── python parkinsons_detection.py
  │     ├─> load_and_preprocess_data()
  │     ├─> create_model(input_dim=22)
  │     │     ├─> Dense(64, relu)
  │     │     ├─> Dropout(0.3)
  │     │     ├─> Dense(32, relu)
  │     │     ├─> Dropout(0.3)
  │     │     └─> Dense(1, sigmoid)
  │     ├─> model.compile(adam, binary_crossentropy)
  │     ├─> model.fit(epochs=100, batch_size=16, val_split=0.2)
  │     ├─> model.predict() → threshold 0.5 → binary class
  │     └─> Plot accuracy + loss curves → training_history.png
  │
  └── python predictor.py
        ├─> ParkinsonsPredictor()
        ├─> load_model() → check .pkl exists
        │     ├─> YES → pickle.load()
        │     └─> NO  → train() → pickle.dump()
        └─> predict_sample([22 float values])
              ├─> Cast to DataFrame
              ├─> scaler.transform()
              ├─> model.predict() → class label
              ├─> model.predict_proba() → probabilities
              └─> Return {prediction, confidence, probabilities}

END
```

---

## 7. Module Deep Dives

### 7.1 `data_exploration.py`

**Role**: Statistical visualization and integrity check before any modeling begins.

**Input**: `parkinsons.data` (CSV file)

**Output files generated**: `target_distribution.png`, `correlation_heatmap.png`

#### Code Walkthrough:

```python
df = pd.read_csv('parkinsons.data')
```
- Loads the entire dataset into memory as a Pandas DataFrame.

```python
print(df.info())
```
- Scans the dataset for column types, non-null counts, and memory usage. Catches any missing data early.

```python
df['status'].value_counts()
```
- Groups the binary label, producing counts: `1 (Parkinson's): 147`, `0 (Healthy): 48`. Essential for checking class imbalance before training.

```python
plt.figure(figsize=(8, 6))
df['status'].value_counts().plot(kind='bar')
plt.savefig('target_distribution.png')
```
- Matplotlib renders a vertical bar chart visualizing the dataset imbalance. Saved to disk for reports.

```python
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
```
- Computes a Pearson correlation matrix. All 22 features are cross-correlated with each other and the label.  
- The heatmap color scale goes from dark blue (negative correlation -1.0) to deep red (positive correlation +1.0). Center is white (0 = no correlation).  
- This reveals that Jitter features are closely correlated with each other, as expected — they are all measuring frequency perturbation.

---

### 7.2 `parkinsons_ml_detection.py`

**Role**: Train and compare three classical ML algorithms, identify the best performing one.

**Processing Flow**:

```
Raw CSV
   │
   ├─> Drop 'name', 'status'       → X (22 features)
   ├─> Isolate 'status'            → y (0 or 1)
   │
   ├─> train_test_split(stratify=y)
   │     Train set: 156 samples
   │     Test set:  39 samples
   │
   ├─> StandardScaler
   │     fit_transform(X_train)    → X_train_scaled
   │     transform(X_test)         → X_test_scaled
   │
   ├──────────────────────────────────────────────
   │  MODEL 1: Random Forest
   │     RandomForestClassifier(n_estimators=100)
   │     Builds 100 decision trees
   │     Each tree uses random feature subsets
   │     Final class = majority vote across trees
   │
   ├──────────────────────────────────────────────
   │  MODEL 2: Logistic Regression
   │     LogisticRegression(max_iter=1000)
   │     Uses sigmoid function on linear combination:
   │     P(y=1) = 1 / (1 + e^(-wX + b))
   │     Needs high max_iter because 22 features need
   │     more gradient descent steps to converge
   │
   ├──────────────────────────────────────────────
   │  MODEL 3: Support Vector Machine
   │     SVC(probability=True)
   │     Finds a hyperplane in 22D space that maximally
   │     separates Parkinson's from Healthy classes
   │     probability=True uses Platt calibration to
   │     also output confidence percentages
   │
   └─> evaluate_models()
         │
         ├─> For each model:
         │     model.predict(X_test_scaled)
         │     accuracy_score()
         │     classification_report()
         │     confusion_matrix()
         │
         └─> Confusion Matrix Grid:
               ┌────────────────────────────┐
               │         Predicted          │
               │    Healthy │ Parkinson's   │
               ├────────────┼───────────────┤
               │    TN  │       FP          │  ← Actual Healthy
               │    FN  │       TP          │  ← Actual Parkinson's
               └────────────────────────────┘
```

**Key Metrics Explained**:

| Metric | Formula | Meaning |
|---|---|---|
| Accuracy | (TP+TN) / Total | Overall correct classifications |
| Precision | TP / (TP+FP) | Of all predicted Parkinson's, how many were right |
| Recall | TP / (TP+FN) | Of all actual Parkinson's, how many were caught |
| F1-Score | 2×(P×R)/(P+R) | Balanced score between precision and recall |

> **Clinical Note**: For medical diagnostics, **Recall matters more than Precision** — missing a true Parkinson's case (False Negative) is more dangerous than a false alarm (False Positive).

---

### 7.3 `parkinsons_detection.py`

**Role**: Deep learning approach using neural networks for classification.

**Libraries Used**: TensorFlow, Keras

#### Neural Network Architecture Breakdown:

```
INPUT LAYER
╔════════════════════════════╗
║  22 input features         ║
║  (normalized voice metrics)║
╚══════════════╤═════════════╝
               │
               ▼
HIDDEN LAYER 1 — Dense(64 neurons, ReLU)
╔════════════════════════════╗
║  64 neurons                ║
║  Activation: ReLU          ║
║  f(x) = max(0, x)          ║
║  Eliminates negative values║
╚══════════════╤═════════════╝
               │
               ▼
DROPOUT LAYER 1 — 30% Dropout
╔════════════════════════════╗
║  Randomly zeroes 30% of    ║
║  neurons each training pass║
║  Prevents overfitting      ║
╚══════════════╤═════════════╝
               │
               ▼
HIDDEN LAYER 2 — Dense(32 neurons, ReLU)
╔════════════════════════════╗
║  32 neurons                ║
║  Activation: ReLU          ║
║  Learns complex patterns   ║
║  from Layer 1 outputs      ║
╚══════════════╤═════════════╝
               │
               ▼
DROPOUT LAYER 2 — 30% Dropout
╔════════════════════════════╗
║  Second regularization pass║
╚══════════════╤═════════════╝
               │
               ▼
OUTPUT LAYER — Dense(1 neuron, Sigmoid)
╔════════════════════════════╗
║  1 output neuron           ║
║  Activation: Sigmoid       ║
║  f(x) = 1/(1+e^-x)        ║
║  Output range: 0.0 → 1.0  ║
║  < 0.5 → Healthy           ║
║  ≥ 0.5 → Parkinson's       ║
╚════════════════════════════╝
```

#### Training Configuration:

| Parameter | Value | Reason |
|---|---|---|
| Optimizer | Adam | Adaptive learning rate, efficient convergence |
| Loss Function | Binary Crossentropy | Penalizes wrong probability outputs |
| Epochs | 100 | Full training passes over dataset |
| Batch Size | 16 | Small batches generalize better on small datasets |
| Validation Split | 20% | Real-time monitoring for overfitting |

#### Training Process Step-by-step:

```
Epoch 1/100
├─> Forward Pass: Input → L1 → D1 → L2 → D2 → Output
├─> Loss = binary_crossentropy(y_true, y_pred)
├─> Backward Pass: Adjust all weights via Adam optimizer
└─> Print: loss, accuracy, val_loss, val_accuracy

Epoch 2/100 ... Epoch 100/100
(Loss decreases, accuracy increases over iterations)

After Training:
├─> model.predict(X_test) → raw probabilities [0.87, 0.12, ...]
├─> (probabilities > 0.5).astype(int) → binary labels [1, 0, ...]
└─> classification_report → accuracy, precision, recall, f1
```

---

### 7.4 `predictor.py`

**Role**: Reusable production-ready class for making predictions — designed to work like a backend API endpoint.

**Class Structure**:

```
ParkinsonsPredictor
│
├── Attributes
│     ├── self.model         = None (RandomForest instance)
│     ├── self.scaler        = None (StandardScaler instance)
│     └── self.feature_names = None (list of 22 feature names)
│
├── Methods
│     │
│     ├── train()
│     │     ├─> Read parkinsons.data
│     │     ├─> X / y split
│     │     ├─> StandardScaler → fit on X_train
│     │     ├─> RandomForestClassifier → fit
│     │     └─> pickle.dump({model, scaler, feature_names})
│     │           └─> saves parkinsons_model.pkl
│     │
│     ├── load_model()
│     │     ├─> Try: open parkinsons_model.pkl
│     │     │     └─> pickle.load() → restore model/scaler/names
│     │     └─> Except FileNotFoundError:
│     │           └─> call self.train() first
│     │
│     ├── predict_sample(features)
│     │     ├─> INPUT: list/array of 22 float values
│     │     ├─> Cast to pd.DataFrame (with column names)
│     │     ├─> scaler.transform() → standardize input
│     │     ├─> model.predict() → binary class (0 or 1)
│     │     ├─> model.predict_proba() → [P(Healthy), P(PD)]
│     │     └─> OUTPUT: {
│     │               'prediction': 'Healthy' or 'Parkinson's Disease',
│     │               'confidence': max(probabilities),
│     │               'probabilities': {'Healthy': 0.11, 'Parkinsons': 0.89}
│     │             }
│     │
│     └── predict_from_file(filename)
│           ├─> Read CSV file (without status column)
│           ├─> For each row:
│           │     └─> call predict_sample(row.values)
│           └─> Return list of result dicts
```

**Example Usage**:

```python
predictor = ParkinsonsPredictor()
predictor.load_model()

# Single patient prediction (22 float values)
features = [119.992, 157.302, 74.997, 0.00784, 0.00007,
            0.00370, 0.00554, 0.01109, 0.04374, 0.42600,
            0.02182, 0.03130, 0.02971, 0.06545, 0.02211,
            21.033, 0.414783, 0.815285, -4.813031,
            0.266482, 2.301442, 0.284654]

result = predictor.predict_sample(features)
print(result)
# Output: {
#   'prediction': "Parkinson's Disease",
#   'confidence': 0.92,
#   'probabilities': {'Healthy': 0.08, 'Parkinsons': 0.92}
# }
```

---

### 7.5 `main.py`

**Role**: Top-level orchestrator. Wraps the entire pipeline into a single cohesive class and runs a demo from start to finish.

**Class**: `ParkinsonsDetectionSystem`

**Method Execution Chain**:

```
main()
  │
  └─> ParkinsonsDetectionSystem()
        │
        ├── load_data(filename='parkinsons.data')
        │     ├─> pd.read_csv()
        │     └─> Print: shape, Parkinson's count, Healthy count
        │
        ├── explore_data()
        │     ├─> Print dataset shape
        │     ├─> Count missing values
        │     └─> Print X.describe() — stats for all features
        │
        ├── train_model()
        │     ├─> Prepare X (22 features), y (status)
        │     ├─> train_test_split(test_size=0.2, stratify=y)
        │     ├─> StandardScaler.fit_transform(X_train)
        │     ├─> RandomForestClassifier(n_estimators=100)
        │     ├─> model.fit(X_train_scaled, y_train)
        │     ├─> accuracy_score() → prints test accuracy
        │     ├─> classification_report() → per-class metrics
        │     ├─> cross_val_score(cv=5) → 5 independent scores
        │     │     └─> Prints: Mean ± Std Dev
        │     ├─> feature_importances_ → Sorted DataFrame
        │     └─> Returns: (accuracy float, feature_importance df)
        │
        ├── save_model(filename='parkinsons_model.pkl')
        │     └─> pickle.dump({model, scaler, feature_names})
        │
        ├── load_model(filename='parkinsons_model.pkl')
        │     ├─> os.path.exists() check
        │     ├─> If exists: pickle.load()
        │     └─> If not: trains + saves first
        │
        ├── predict(features)
        │     ├─> Accepts: list / np.array / pd.Series
        │     ├─> Converts to DataFrame (with correct column names)
        │     ├─> scaler.transform()
        │     ├─> model.predict()
        │     ├─> model.predict_proba()
        │     └─> Returns: {prediction, confidence, probabilities}
        │
        └── demo_predictions(n_samples=10)
              ├─> Loop over first 10 rows in dataset
              ├─> Extract features (drop name and status)
              ├─> Derive actual label
              ├─> Call self.predict(features)
              ├─> Compare result vs actual
              └─> Print: ✓ (correct) or ✗ (incorrect)
```

---

## 8. ML Model Comparison

The following table compares all three classical ML models trained in `parkinsons_ml_detection.py`:

| Model | Accuracy | Precision | Recall | F1-Score | Notes |
|---|---|---|---|---|---|
| **Random Forest** | **92.3%** | 0.93 | 0.97 | 0.95 | Best for interpretability + accuracy |
| Logistic Regression | 92.3% | 0.93 | 0.97 | 0.95 | Fast, good baseline benchmark |
| SVM (RBF kernel) | 92.3% | 0.91 | 1.00 | 0.95 | Perfect Recall — catches all PD cases |

### Top 5 Most Important Features (Random Forest):

```
Rank    Feature        Importance
──────────────────────────────────
  1.    PPE             15.2%    ████████████████████
  2.    spread1         10.7%    ██████████████
  3.    MDVP:Fo(Hz)      6.4%    █████████
  4.    NHR              6.2%    ████████
  5.    Jitter:DDP       5.6%    ████████
```

---

## 9. Neural Network Architecture Diagram

```
         22 Input Nodes  (Voice Features)
         ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○
                       │ │ │ │ ...
                       ▼ ▼ ▼ ▼
              ┌──────────────────────┐
              │  Dense Layer 1       │ ← 64 neurons, ReLU
              └──────────┬───────────┘
                         │
              ┌──────────────────────┐
              │  Dropout (30%)       │ ← Random neuron deactivation
              └──────────┬───────────┘
                         │
              ┌──────────────────────┐
              │  Dense Layer 2       │ ← 32 neurons, ReLU
              └──────────┬───────────┘
                         │
              ┌──────────────────────┐
              │  Dropout (30%)       │
              └──────────┬───────────┘
                         │
              ┌──────────────────────┐
              │  Output Layer        │ ← 1 neuron, Sigmoid
              └──────────┬───────────┘
                         │
                    ┌────┴─────┐
                    │          │
              [< 0.5]       [≥ 0.5]
                 HEALTHY   PARKINSON'S
```

---

## 10. Input / Output Reference

### Inputs (per prediction query):

```
Type: Python list / NumPy array / Pandas Series
Length: Exactly 22 float values

Order:
  [0]  MDVP:Fo(Hz)        — avg voice pitch (Hz)
  [1]  MDVP:Fhi(Hz)       — max pitch (Hz)
  [2]  MDVP:Flo(Hz)       — min pitch (Hz)
  [3]  MDVP:Jitter(%)     — jitter percentage
  [4]  MDVP:Jitter(Abs)   — absolute jitter
  [5]  MDVP:RAP           — relative average perturbation
  [6]  MDVP:PPQ           — period perturbation quotient
  [7]  Jitter:DDP         — jitter DDP
  [8]  MDVP:Shimmer       — shimmer
  [9]  MDVP:Shimmer(dB)   — shimmer (dB)
  [10] Shimmer:APQ3       — shimmer APQ3
  [11] Shimmer:APQ5       — shimmer APQ5
  [12] MDVP:APQ           — shimmer APQ11
  [13] Shimmer:DDA        — shimmer DDA
  [14] NHR                — noise-to-harmonics ratio
  [15] HNR                — harmonics-to-noise ratio
  [16] RPDE               — recurrence period density entropy
  [17] DFA                — detrended fluctuation analysis
  [18] spread1            — nonlinear frequency spread 1
  [19] spread2            — nonlinear frequency spread 2
  [20] D2                 — correlation dimension
  [21] PPE                — pitch period entropy
```

### Output (per call to `predict_sample()` or `system.predict()`):

```python
{
    'prediction'           : 'Parkinson\'s Disease'  # or 'Healthy'
    'confidence'           : 0.92                   # max probability (0.0 - 1.0)
    'probability_healthy'  : 0.08                   # P(Healthy) from Random Forest
    'probability_parkinsons': 0.92                  # P(Parkinson's) from Random Forest
}
```

---

## 11. Data Preprocessing Pipeline

```
           Raw Input Features (22 values)
                       │
          ┌────────────▼─────────────┐
          │   Step 1: Type Check     │
          │   list / array → df      │
          └────────────┬─────────────┘
                       │
          ┌────────────▼─────────────┐
          │   Step 2: Column Align   │
          │   Assign feature names   │
          │   (22 column headers)    │
          └────────────┬─────────────┘
                       │
          ┌────────────▼─────────────┐
          │   Step 3: StandardScaler │
          │   z = (x - μ) / σ       │  ← Fitted on training set
          │   Mean=0, Variance=1     │
          └────────────┬─────────────┘
                       │
          ┌────────────▼─────────────┐
          │  Step 4: Model Predict   │
          │  Scaled features pass   │
          │  through RF's 100 trees  │
          └────────────┬─────────────┘
                       │
          ┌────────────▼─────────────┐
          │  Step 5: Probability     │
          │  predict_proba() →       │
          │  [P(Healthy), P(PD)]     │
          └────────────┬─────────────┘
                       │
                   RESULT DICT
```

---

## 12. Prediction Flow

```
User provides 22 Voice Features
         │
         ▼
predictor.predict_sample(features)
         │
         ├─> load_model() [if not already loaded]
         │     └─> parkinsons_model.pkl → model + scaler
         │
         ├─> Input Validation
         │     └─> isinstance check → cast to DataFrame
         │
         ├─> scaler.transform(features)
         │     └─> Normalize to training distribution
         │
         ├─> model.predict(scaled)
         │     └─> 100 decision trees vote → majority class
         │
         ├─> model.predict_proba(scaled)
         │     └─> [0.08, 0.92] → probabilities per class
         │
         └─> Return dict
               ├─> prediction  = "Parkinson's Disease"
               ├─> confidence  = 0.92
               ├─> prob_healthy = 0.08
               └─> prob_pd     = 0.92
```

---

## 13. Model Persistence Flow

```
SAVING THE MODEL (after training)
─────────────────────────────────
 training complete
         │
         ▼
 model_data = {
     'model'        : RandomForest instance (100 trees),
     'scaler'       : StandardScaler instance,
     'feature_names': ['MDVP:Fo(Hz)', ..., 'PPE']
 }
         │
         ▼
 pickle.dump(model_data, open('parkinsons_model.pkl','wb'))
         │
         ▼
 Saved as binary .pkl file (~248 KB)


LOADING THE MODEL (for inference)
─────────────────────────────────
 predictor.load_model() called
         │
         ├─> os.path.exists('parkinsons_model.pkl')
         │         YES                     NO
         │          │                       │
         │          ▼                       ▼
         │   pickle.load(file)        self.train()
         │   restore model           (builds from CSV)
         │   restore scaler          then save_model()
         └─────────────────────────────────┘
                        │
                 Model ready to predict
```

---

## 14. How to Run — Step by Step

### Prerequisites
```bash
# Python 3.7+ required
python --version

# Install all dependencies
pip install -r requirements.txt
```

### Contents of `requirements.txt`:
```
pandas
numpy
scikit-learn
tensorflow
matplotlib
seaborn
```

---

### Run Option A — Full System Demo (`main.py`)
```bash
python main.py
```
**What happens**:
1. Loads the dataset and prints class distribution
2. Runs exploratory feature statistics
3. Trains Random Forest with 80/20 split
4. Evaluates with 5-fold cross-validation
5. Saves trained model to `.pkl`
6. Predicts on 10 random samples and prints ✓/✗

**Sample Console Output**:
```
🧠 Parkinson's Disease Detection System
==================================================
Loading dataset...
Dataset loaded: 195 samples, 24 features
Parkinson's cases: 147
Healthy cases: 48

=== MODEL TRAINING ===
Test Accuracy: 0.9231

Classification Report:
              precision    recall  f1-score
           0       0.88      0.78      0.82
           1       0.93      0.97      0.95

Cross-validation scores: [0.94 0.90 0.94 0.87 0.90]
Mean CV accuracy: 0.9103 (+/- 0.0537)

🎯 System Summary:
   • Dataset: 195 voice recordings
   • Features: 22 voice biomarkers
   • Model: Random Forest Classifier
   • Accuracy: 92.3%
   • Top feature: PPE
✅ System ready for predictions!
```

---

### Run Option B — Visualizations & EDA (`data_exploration.py`)
```bash
python data_exploration.py
```
**What happens**:
- Opens and displays class distribution chart
- Opens and displays correlation heatmap
- Saves both as `.png` in the project folder

---

### Run Option C — ML Model Comparison (`parkinsons_ml_detection.py`)
```bash
python parkinsons_ml_detection.py
```
**What happens**:
- Trains 3 classifiers (RF, LR, SVM)
- Prints accuracy & classification report for each
- Determines the best model automatically
- Generates `model_comparison.png` & `feature_importance.png`

---

### Run Option D — Neural Network Training (`parkinsons_detection.py`)
```bash
python parkinsons_detection.py
```
**What happens**:
- Compiles the Keras model & starts fitting
- Shows per-epoch logs: `loss, accuracy, val_loss, val_accuracy`
- Generates `training_history.png` after training

---

### Run Option E — Prediction Inference (`predictor.py`)
```bash
python predictor.py
```
**What happens**:
- Loads (or trains) the model from `.pkl`
- Demonstrates prediction on 5 real patient samples
- Prints actual vs predicted with confidence scores

---

## 15. Output Files Reference

| File | Generated By | Description |
|---|---|---|
| `parkinsons_model.pkl` | `main.py`, `predictor.py` | Trained Random Forest + StandardScaler + feature names (binary) |
| `target_distribution.png` | `data_exploration.py` | Bar chart of Parkinson's vs Healthy count |
| `correlation_heatmap.png` | `data_exploration.py` | Pearson correlation matrix of all 22 features |
| `model_comparison.png` | `parkinsons_ml_detection.py` | Side-by-side confusion matrices for RF, LR, SVM |
| `feature_importance.png` | `parkinsons_ml_detection.py` | Top 10 most predictive voice biomarkers (bar chart) |
| `training_history.png` | `parkinsons_detection.py` | Accuracy and loss curves per training epoch |

---

## 16. Clinical Background

### Why Voice Analysis?
Parkinson's Disease affects the neuromotor system, which controls fine muscle movements including the **larynx and vocal cords**. Voice manifestations of PD include:

```
Parkinson's Disease
        │
        ├─> Dopamine deficiency  
        │       │
        │       └─> Reduced motor control
        │               │
        │               ├─> Increased Jitter    (frequency instability)
        │               ├─> Increased Shimmer   (amplitude instability)
        │               ├─> Higher NHR          (more noise, less harmonics)
        │               ├─> Lower HNR           (degraded voice quality)
        │               └─> Higher PPE/RPDE     (irregular pitch patterns)
        │
        └─> Non-invasive voice recording can capture these early!
```

### Advantages of Voice-Based Screening:
- **Non-invasive**: No blood tests or MRI needed
- **Remote**: Recordings can be captured at home
- **Early detection**: Voice changes appear early in disease progression
- **Cost-effective**: Compared to traditional neurological assessments

---

## 17. Future Enhancements

| Enhancement | Description | Priority |
|---|---|---|
| Real-time audio capture | Record voice live via microphone and extract features in real-time | High |
| Web dashboard | Flask/FastAPI REST API with a React frontend for clinicians | High |
| Longitudinal tracking | Monitor voice changes over time per patient to track progression | Medium |
| SHAP explanations | Use SHAP values to explain each individual prediction decision | Medium |
| Ensemble model | Combine RF + SVM + Neural Net outputs for higher accuracy | Medium |
| Additional biomarkers | Integrate gait, handwriting, or eye movement data | Low |
| Multi-language analysis | Expand to non-English speech recordings | Low |

---

*Documentation generated for the Parkinson's Disease Detection project — UCI Dataset.*  
*For research and educational purposes only. Not a clinical diagnostic tool.*
