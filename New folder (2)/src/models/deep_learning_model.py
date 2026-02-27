import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import pickle
import shap

class DeepLearningParkinsonsDetector:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.best_model = None
        self.explainer = None
        
    def create_enhanced_features(self, df):
        """Create additional engineered features"""
        # Jitter-to-Shimmer ratio
        df['jitter_shimmer_ratio'] = df['MDVP:Jitter(%)'] / (df['MDVP:Shimmer'] + 1e-8)
        
        # Frequency range
        df['freq_range'] = df['MDVP:Fhi(Hz)'] - df['MDVP:Flo(Hz)']
        
        # Noise quality index
        df['noise_quality'] = df['HNR'] / (df['NHR'] + 1e-8)
        
        # Harmonic stability
        df['harmonic_stability'] = df['HNR'] * df['MDVP:Fo(Hz)']
        
        # Voice instability index
        df['voice_instability'] = (df['MDVP:Jitter(%)'] + df['MDVP:Shimmer']) / 2
        
        # Spectral complexity
        df['spectral_complexity'] = df['spread1'] * df['spread2']
        
        return df
    
    def load_and_preprocess_data(self):
        """Load and preprocess data with feature engineering"""
        df = pd.read_csv('parkinsons.data')
        
        # Remove name column
        X = df.drop(['name', 'status'], axis=1)
        y = df['status']
        
        # Create enhanced features
        X_enhanced = pd.DataFrame(X)
        X_enhanced = self.create_enhanced_features(X_enhanced)
        
        # Apply SMOTE for balanced dataset
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_resampled, y_resampled = smote.fit_resample(X_enhanced, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.feature_names = X_enhanced.columns.tolist()
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def create_deep_neural_network(self):
        """Create deep neural network using MLPClassifier"""
        return MLPClassifier(
            hidden_layer_sizes=(128, 64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
    
    def create_optimized_random_forest(self):
        """Create optimized Random Forest"""
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    
    def create_ensemble_model(self):
        """Create ensemble of multiple models"""
        rf = self.create_optimized_random_forest()
        mlp = self.create_deep_neural_network()
        svm = SVC(probability=True, random_state=42, class_weight='balanced')
        lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('mlp', mlp),
                ('svm', svm),
                ('lr', lr)
            ],
            voting='soft'
        )
        
        return ensemble
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and compare performance"""
        print("Training models...")
        
        # Define models
        models = {
            'Deep Neural Network': self.create_deep_neural_network(),
            'Optimized Random Forest': self.create_optimized_random_forest(),
            'Ensemble Model': self.create_ensemble_model(),
            'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
            'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest Model: {best_model_name}")
        
        # Create SHAP explainer for best model
        if hasattr(self.best_model, 'predict_proba'):
            try:
                if 'Random Forest' in best_model_name:
                    self.explainer = shap.TreeExplainer(self.best_model)
                else:
                    # Use a sample for other models
                    self.explainer = shap.KernelExplainer(self.best_model.predict_proba, X_train[:100])
            except:
                print("Could not create SHAP explainer")
        
        self.models = results
        return results, y_test
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for the neural network"""
        print("Performing hyperparameter tuning...")
        
        param_grid = {
            'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01, 0.1]
        }
        
        mlp = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True)
        
        grid_search = GridSearchCV(
            mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def visualize_results(self, results, y_test):
        """Create comprehensive visualizations"""
        # Model comparison
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_means = [results[name]['cv_mean'] for name in model_names]
        cv_stds = [results[name]['cv_std'] for name in model_names]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        axes[0, 0].bar(model_names, accuracies, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Cross-validation scores
        axes[0, 1].errorbar(model_names, cv_means, yerr=cv_stds, fmt='o', capsize=5)
        axes[0, 1].set_title('Cross-Validation Scores')
        axes[0, 1].set_ylabel('CV Accuracy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Confusion matrices for top 2 models
        top_models = sorted(results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)[:2]
        
        for idx, (name, result) in enumerate(top_models):
            cm = confusion_matrix(y_test, result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, idx])
            axes[1, idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.4f}')
            axes[1, idx].set_xlabel('Predicted')
            axes[1, idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('advanced_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance for Random Forest
        if 'Optimized Random Forest' in results:
            rf_model = results['Optimized Random Forest']['model']
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=feature_importance.head(15), y='feature', x='importance')
            plt.title('Top 15 Feature Importance (Random Forest)')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('enhanced_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return feature_importance
    
    def save_model(self, filename='advanced_parkinsons_model.pkl'):
        """Save the trained model"""
        model_data = {
            'best_model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'models': {name: result['model'] for name, result in self.models.items()}
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Advanced model saved to {filename}")
    
    def predict(self, features):
        """Make prediction with the best model"""
        if self.best_model is None:
            return None
        
        # Ensure features include engineered features
        if len(features) == 22:  # Original features
            # Create a temporary dataframe to add engineered features
            temp_df = pd.DataFrame([features], columns=[
                'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
                'spread1', 'spread2', 'D2', 'PPE'
            ])
            temp_df = self.create_enhanced_features(temp_df)
            features = temp_df.iloc[0].values
        
        features_scaled = self.scaler.transform([features])
        prediction = self.best_model.predict(features_scaled)[0]
        probabilities = self.best_model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': 'Parkinson\'s Disease' if prediction == 1 else 'Healthy',
            'confidence': max(probabilities),
            'probability_healthy': probabilities[0],
            'probability_parkinsons': probabilities[1]
        }

def main():
    print("🧠 Advanced Parkinson's Disease Detection with Deep Learning")
    print("=" * 60)
    
    # Initialize detector
    detector = DeepLearningParkinsonsDetector()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = detector.load_and_preprocess_data()
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {len(detector.feature_names)}")
    
    # Train models
    results, y_test = detector.train_models(X_train, X_test, y_train, y_test)
    
    # Visualize results
    feature_importance = detector.visualize_results(results, y_test)
    
    # Hyperparameter tuning for neural network
    print("\nPerforming hyperparameter tuning...")
    tuned_mlp = detector.hyperparameter_tuning(X_train, y_train)
    
    # Evaluate tuned model
    tuned_accuracy = tuned_mlp.score(X_test, y_test)
    print(f"Tuned Neural Network Accuracy: {tuned_accuracy:.4f}")
    
    # Save model
    detector.save_model()
    
    # Demo predictions
    print("\n" + "=" * 60)
    print("🎯 Demo Predictions")
    print("=" * 60)
    
    # Load original dataset for demo
    df = pd.read_csv('parkinsons.data')
    
    for i in range(5):
        sample = df.iloc[i]
        features = sample.drop(['name', 'status']).values
        actual = 'Parkinson\'s Disease' if sample['status'] == 1 else 'Healthy'
        
        result = detector.predict(features)
        
        if result:
            print(f"\nSample {i+1} ({sample['name']}):")
            print(f"  Actual: {actual}")
            print(f"  Predicted: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            correct = "✓" if result['prediction'] == actual else "✗"
            print(f"  Result: {correct}")
    
    print(f"\n✅ Advanced system ready!")
    print(f"📊 Best Model Performance Summary:")
    best_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
    best_result = results[best_name]
    print(f"   • Model: {best_name}")
    print(f"   • Accuracy: {best_result['accuracy']:.1%}")
    print(f"   • CV Score: {best_result['cv_mean']:.1%} (±{best_result['cv_std']*2:.1%})")
    print(f"   • Enhanced Features: {len(detector.feature_names)}")

if __name__ == "__main__":
    main()
