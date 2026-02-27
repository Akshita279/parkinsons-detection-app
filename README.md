# Parkinson's Disease Detection Project

This project uses machine learning to detect Parkinson's Disease from voice biomarkers, achieving **92.3% accuracy** with a Random Forest classifier.

## 🎯 Project Overview

**Objective**: Develop an AI system for early detection of Parkinson's Disease using voice signal analysis.

**Key Features**:
- ✅ Automated Parkinson's Disease detection using AI
- ✅ Enhanced early diagnosis capabilities  
- ✅ Reduces dependency on manual clinical analysis
- ✅ Practical application of ML in healthcare

## 📊 Dataset
- **Source**: UCI Machine Learning Repository - Parkinson's Dataset
- **Features**: 22 voice biomarker features
- **Samples**: 195 voice recordings (147 Parkinson's, 48 Healthy)
- **Target**: Binary classification (1=Parkinson's, 0=Healthy)

## 🚀 Quick Start

1. **Setup Environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Run Complete System**:
```bash
python main.py
```

3. **Individual Components**:
```bash
# Data exploration
python data_exploration.py

# ML model comparison
python parkinsons_ml_detection.py

# Prediction interface
python predictor.py
```

## 🧠 Voice Biomarkers Used
- **Jitter**: Fundamental frequency variations (MDVP:Jitter, RAP, PPQ, DDP)
- **Shimmer**: Amplitude variations (MDVP:Shimmer, APQ3, APQ5, DDA)
- **Noise Ratios**: NHR (Noise-to-harmonics), HNR (Harmonic-to-noise)
- **Nonlinear Features**: RPDE, DFA, PPE, D2, spread1, spread2
- **Frequency**: MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)

## 🤖 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **92.3%** | 0.93 | 0.97 | 0.95 |
| Logistic Regression | 92.3% | 0.93 | 0.97 | 0.95 |
| SVM | 92.3% | 0.91 | 1.00 | 0.95 |

**Top 5 Most Important Features**:
1. PPE (Pitch Period Entropy) - 15.2%
2. spread1 - 10.7%
3. MDVP:Fo(Hz) - 6.4%
4. NHR - 6.2%
5. Jitter:DDP - 5.6%

## 📁 Project Structure
```
parkison/
├── main.py                    # Complete system demo
├── parkinsons_ml_detection.py # Model comparison
├── predictor.py              # Prediction interface
├── data_exploration.py       # Data analysis
├── parkinsons.data          # Dataset
├── parkinsons_model.pkl     # Trained model
├── requirements.txt         # Dependencies
├── README.md               # This file
└── venv/                   # Virtual environment
```

## 🔬 Usage Examples

### Making Predictions
```python
from main import ParkinsonsDetectionSystem

# Initialize system
system = ParkinsonsDetectionSystem()

# Load pre-trained model
system.load_model()

# Make prediction (example features)
features = [119.992, 157.302, 74.997, 0.00784, ...]  # 22 features
result = system.predict(features)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Training New Model
```python
system = ParkinsonsDetectionSystem()
system.load_data()
accuracy, feature_importance = system.train_model()
system.save_model()
```

## 📈 Results & Visualizations

The system generates several visualizations:
- `target_distribution.png` - Dataset class distribution
- `correlation_heatmap.png` - Feature correlation matrix
- `model_comparison.png` - Confusion matrices for all models
- `feature_importance.png` - Top 10 most important features

## 🔬 Clinical Relevance

**Voice Biomarkers in Parkinson's**:
- Parkinson's affects vocal cord control and breathing
- Changes in voice patterns occur early in disease progression
- Non-invasive, cost-effective screening method
- Can complement traditional clinical assessments

**Key Indicators**:
- Increased jitter and shimmer (voice instability)
- Reduced harmonic-to-noise ratio
- Changes in pitch control and rhythm
- Altered nonlinear dynamics in voice signals

## 🎯 Future Enhancements

- [ ] Deep learning models (Neural Networks)
- [ ] Real-time voice recording and analysis
- [ ] Web-based interface for clinical use
- [ ] Integration with additional biomarkers
- [ ] Longitudinal tracking capabilities
- [ ] Multi-language voice analysis

## 📚 References

- Little, M.A., McSharry, P.E., Roberts, S.J., Costello, D.A.E., Moroz, I.M. (2007)
  'Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection'
- UCI Machine Learning Repository: Parkinson's Dataset

## 🏥 Disclaimer

This system is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult healthcare professionals for medical advice.
