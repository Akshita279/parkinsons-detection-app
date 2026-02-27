#!/usr/bin/env python3
"""
Advanced Model Training Pipeline
Trains multiple models on the comprehensive dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedModelTrainer:
    def __init__(self, data_path="data/processed/large_scale_parkinsons_dataset.csv"):
        self.data_path = Path(data_path)
        self.models_dir = Path("models")
        self.results_dir = Path("results")
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = None
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("📊 Loading comprehensive dataset...")
        
        if not self.data_path.exists():
            print("❌ Large-scale dataset not found. Run collect_large_data.py first.")
            return False
        
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(self.df)} samples")
        
        # Prepare features and target
        feature_cols = [col for col in self.df.columns if col not in ['name', 'status', 'source']]
        self.X = self.df[feature_cols]
        self.y = self.df['status']
        self.feature_names = feature_cols
        
        print(f"Features: {len(self.feature_names)}")
        print(f"Class distribution: {self.y.value_counts().to_dict()}")
        
        return True
    
    def create_advanced_models(self):
        """Create a comprehensive set of models"""
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            
            'Deep Neural Network': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42,
                early_stopping=True
            ),
            
            'SVM (RBF)': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            
            'Logistic Regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        }
        
        # Create ensemble model
        ensemble = VotingClassifier(
            estimators=[
                ('rf', models['Random Forest']),
                ('gb', models['Gradient Boosting']),
                ('mlp', models['Deep Neural Network'])
            ],
            voting='soft'
        )
        
        models['Ensemble'] = ensemble
        
        return models
    
    def apply_data_balancing(self, X_train, y_train, method='smote'):
        """Apply various data balancing techniques"""
        print(f"⚖️ Applying {method} balancing...")
        
        if method == 'smote':
            balancer = SMOTE(random_state=42, k_neighbors=3)
        elif method == 'adasyn':
            balancer = ADASYN(random_state=42)
        elif method == 'smoteenn':
            balancer = SMOTEENN(random_state=42)
        elif method == 'undersample':
            balancer = RandomUnderSampler(random_state=42)
        else:
            return X_train, y_train
        
        X_balanced, y_balanced = balancer.fit_resample(X_train, y_train)
        
        print(f"Original: {len(X_train)} samples")
        print(f"Balanced: {len(X_balanced)} samples")
        print(f"New distribution: {pd.Series(y_balanced).value_counts().to_dict()}")
        
        return X_balanced, y_balanced
    
    def train_models(self, test_size=0.2, balancing_method='smote'):
        """Train all models with comprehensive evaluation"""
        print("🚀 Starting comprehensive model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        # Apply balancing
        X_train_balanced, y_train_balanced = self.apply_data_balancing(
            X_train, y_train, balancing_method
        )
        
        # Scale features
        self.scaler = RobustScaler()  # More robust to outliers
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create models
        models = self.create_advanced_models()
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train_balanced)
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train_balanced, 
                    cv=5, scoring='accuracy'
                )
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc_score': auc_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                print(f"✅ {name}: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}, CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
        
        # Find best model
        if self.results:
            best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['cv_mean'])
            self.best_model = self.results[best_model_name]['model']
            
            print(f"\n🏆 Best Model: {best_model_name}")
            print(f"   Accuracy: {self.results[best_model_name]['accuracy']:.4f}")
            print(f"   AUC Score: {self.results[best_model_name]['auc_score']:.4f}")
            print(f"   CV Score: {self.results[best_model_name]['cv_mean']:.4f}±{self.results[best_model_name]['cv_std']:.4f}")
        
        return X_test, y_test
    
    def save_models(self):
        """Save all trained models and results"""
        print("💾 Saving models and results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        for name, result in self.results.items():
            model_filename = self.models_dir / f"{name.lower().replace(' ', '_')}_{timestamp}.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump({
                    'model': result['model'],
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'metrics': {k: v for k, v in result.items() if k != 'model'}
                }, f)
        
        # Save best model separately
        if self.best_model:
            best_model_filename = self.models_dir / f"best_model_{timestamp}.pkl"
            with open(best_model_filename, 'wb') as f:
                pickle.dump({
                    'model': self.best_model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names
                }, f)
        
        # Save results summary
        results_summary = {}
        for name, result in self.results.items():
            results_summary[name] = {
                'accuracy': result['accuracy'],
                'auc_score': result['auc_score'],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std']
            }
        
        with open(self.results_dir / f"training_results_{timestamp}.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"✅ Models saved with timestamp: {timestamp}")
    
    def create_visualizations(self, X_test, y_test):
        """Create comprehensive visualizations"""
        print("📊 Creating visualizations...")
        
        # Model comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        auc_scores = [self.results[name]['auc_score'] for name in model_names]
        
        axes[0, 0].bar(model_names, accuracies, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # AUC comparison
        axes[0, 1].bar(model_names, auc_scores, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Model AUC Score Comparison')
        axes[0, 1].set_ylabel('AUC Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Confusion matrix for best model
        best_name = max(self.results.keys(), key=lambda k: self.results[k]['cv_mean'])
        cm = confusion_matrix(y_test, self.results[best_name]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_name}')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'])
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'comprehensive_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved")

def main():
    trainer = AdvancedModelTrainer()
    
    # Load data
    if not trainer.load_data():
        return
    
    # Train models
    X_test, y_test = trainer.train_models()
    
    # Save everything
    trainer.save_models()
    
    # Create visualizations
    trainer.create_visualizations(X_test, y_test)
    
    print("\n🎉 Advanced training completed!")
    print("📁 Check the 'models' and 'results' directories for outputs")

if __name__ == "__main__":
    main()
