#!/usr/bin/env python3
"""
Parkinson's Disease Detection System
=====================================
Run this file to:
  1. Train / load the ML model (backend)
  2. Automatically launch the Streamlit web app (frontend)

Usage:
    python main.py
"""

import os
import sys
import subprocess
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ─────────────────────────────────────────────────────────────
# Dataset & model paths — automatically resolved
# ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET     = os.path.join(BASE_DIR, "parkinsons.data")
if not os.path.exists(DATASET):
    DATASET = os.path.join(BASE_DIR, "data", "raw", "parkinsons.data")
MODEL_FILE  = os.path.join(BASE_DIR, "parkinsons_model.pkl")
APP_FILE    = os.path.join(BASE_DIR, "streamlit_app.py")


# ─────────────────────────────────────────────────────────────
# Backend: Train & evaluate the model
# ─────────────────────────────────────────────────────────────
class ParkinsonsDetectionSystem:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.data = None

    def load_data(self):
        print("📂 Loading dataset...")
        if not os.path.exists(DATASET):
            print(f"❌ Dataset not found at: {DATASET}")
            sys.exit(1)
        self.data = pd.read_csv(DATASET)
        total = len(self.data)
        pd_cases = int(self.data["status"].sum())
        healthy  = total - pd_cases
        print(f"   ✔ {total} samples loaded  |  PD: {pd_cases}  |  Healthy: {healthy}")

    def train_model(self):
        print("\n🤖 Training Random Forest model...")
        X = self.data.drop(["name", "status"], axis=1)
        y = self.data["status"]
        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_s, y_train)

        y_pred = self.model.predict(X_test_s)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"   ✔ Test Accuracy: {accuracy:.2%}")
        print("\n   Classification Report:")
        for line in classification_report(y_test, y_pred).split("\n"):
            print(f"   {line}")

        cv = cross_val_score(self.model, X_train_s, y_train, cv=5)
        print(f"\n   5-Fold CV: {cv.mean():.2%} ± {cv.std()*2:.2%}")

        fi = (
            pd.DataFrame({"feature": self.feature_names,
                          "importance": self.model.feature_importances_})
            .sort_values("importance", ascending=False)
        )
        print("\n   Top 5 Most Important Features:")
        for _, row in fi.head(5).iterrows():
            bar = "█" * int(row["importance"] * 200)
            print(f"   {row['feature']:25s}  {row['importance']:.3f}  {bar}")

        return accuracy

    def save_model(self):
        payload = {
            "model":         self.model,
            "scaler":        self.scaler,
            "feature_names": self.feature_names,
        }
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(payload, f)
        print(f"\n💾 Model saved → {MODEL_FILE}")

    def load_model(self):
        if not os.path.exists(MODEL_FILE):
            print("⚠️  No saved model found — training from scratch...")
            self.load_data()
            self.train_model()
            self.save_model()
            return
        with open(MODEL_FILE, "rb") as f:
            payload = pickle.load(f)
        self.model         = payload["model"]
        self.scaler        = payload["scaler"]
        self.feature_names = payload["feature_names"]
        print(f"✔ Pre-trained model loaded ← {MODEL_FILE}")

    def predict(self, features):
        if isinstance(features, (list, np.ndarray)):
            features = pd.DataFrame([features], columns=self.feature_names)
        elif isinstance(features, pd.Series):
            features = features.to_frame().T
        scaled = self.scaler.transform(features)
        pred   = self.model.predict(scaled)[0]
        proba  = self.model.predict_proba(scaled)[0]
        return {
            "prediction":             "Parkinson's Disease" if pred == 1 else "Healthy",
            "confidence":             float(max(proba)),
            "probability_healthy":    float(proba[0]),
            "probability_parkinsons": float(proba[1]),
        }

    def demo_predictions(self, n=5):
        print(f"\n🎯 Demo — predicting {n} samples:")
        correct = 0
        for i in range(min(n, len(self.data))):
            row      = self.data.iloc[i]
            feats    = row.drop(["name", "status"])
            actual   = "Parkinson's Disease" if row["status"] == 1 else "Healthy"
            result   = self.predict(feats)
            tick     = "✓" if result["prediction"] == actual else "✗"
            correct += 1 if tick == "✓" else 0
            print(
                f"   [{tick}] {row['name']:20s} | "
                f"Actual: {actual:20s} | "
                f"Predicted: {result['prediction']:20s} | "
                f"Confidence: {result['confidence']:.1%}"
            )
        print(f"\n   ✔ {correct}/{n} correct in demo")


# ─────────────────────────────────────────────────────────────
# Frontend: Launch Streamlit
# ─────────────────────────────────────────────────────────────
def launch_streamlit():
    if not os.path.exists(APP_FILE):
        print(f"❌ Streamlit app not found at: {APP_FILE}")
        print("   Make sure streamlit_app.py is in the project root.")
        return

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("⚠️  Streamlit not installed. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])

    print("\n" + "=" * 60)
    print("🌐  Launching Streamlit Web Application")
    print("=" * 60)
    print("   URL  : http://localhost:8501")
    print("   Stop : Press  Ctrl + C  in this terminal")
    print("=" * 60 + "\n")

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", APP_FILE,
         "--server.headless", "false",
         "--server.port", "8501"],
        cwd=BASE_DIR,
    )


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  🧠 Parkinson's Disease Detection System")
    print("=" * 60)

    system = ParkinsonsDetectionSystem()

    # ── Step 1: Backend — train or load the model ──────────
    if os.path.exists(MODEL_FILE):
        system.load_model()
        if system.data is None:
            system.load_data()
    else:
        system.load_data()
        system.train_model()
        system.save_model()

    # ── Step 2: Quick prediction demo ──────────────────────
    system.demo_predictions(n=5)

    print("\n✅ Backend ready.\n")

    # ── Step 3: Frontend — launch Streamlit ────────────────
    launch_streamlit()


if __name__ == "__main__":
    main()
