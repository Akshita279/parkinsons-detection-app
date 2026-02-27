#!/usr/bin/env python3
"""
Large-Scale Parkinson's Dataset Collection
Downloads multiple datasets for robust training on normal recordings
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import json
from urllib.parse import urlparse
import time

class LargeScaleDataCollector:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.external_dir = self.data_dir / "external"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.external_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_uci_datasets(self):
        """Download all UCI Parkinson's datasets"""
        print("📥 Downloading UCI Parkinson's Datasets...")
        
        datasets = {
            'uci_parkinsons_original': 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data',
            'uci_parkinsons_telemonitoring': 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data'
        }
        
        downloaded = 0
        for name, url in datasets.items():
            try:
                print(f"  Downloading {name}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(self.raw_dir / f"{name}.data", "wb") as f:
                    f.write(response.content)
                
                print(f"  ✅ {name} downloaded")
                downloaded += 1
                
            except Exception as e:
                print(f"  ❌ Failed to download {name}: {e}")
        
        return downloaded
    
    def download_parkinson_speech_dataset(self):
        """Download Parkinson Speech Dataset from GitHub"""
        print("📥 Downloading Parkinson Speech Dataset...")
        
        try:
            # This is a larger dataset with more samples
            url = "https://raw.githubusercontent.com/SJTU-YONGFU-RESEARCH-GRP/Parkinson-Patient-Speech-Dataset/main/data/parkinsons_speech_data.csv"
            
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(self.raw_dir / "parkinson_speech_dataset.csv", "wb") as f:
                    f.write(response.content)
                print("  ✅ Parkinson Speech Dataset downloaded")
                return True
            else:
                print(f"  ❌ Dataset not available at URL")
                return False
                
        except Exception as e:
            print(f"  ❌ Error downloading Parkinson Speech Dataset: {e}")
            return False
    
    def create_enhanced_synthetic_data(self, n_samples=5000):
        """Create enhanced synthetic data with more variation"""
        print(f"🔬 Generating {n_samples} enhanced synthetic samples...")
        
        # Load original data to understand feature distributions
        original_data = pd.read_csv(self.raw_dir / "uci_parkinsons_original.data")
        X = original_data.drop(['name', 'status'], axis=1)
        y = original_data['status']
        
        # Separate healthy and Parkinson's samples
        healthy_samples = X[y == 0]
        parkinsons_samples = X[y == 1]
        
        synthetic_data = []
        synthetic_labels = []
        
        # Create more diverse synthetic samples
        for i in range(n_samples):
            # Randomly choose class (balanced)
            is_parkinsons = np.random.choice([0, 1])
            
            if is_parkinsons:
                # Generate Parkinson's-like sample with more variation
                base_sample = parkinsons_samples.sample(1).iloc[0]
                
                # Add different types of noise for diversity
                gaussian_noise = np.random.normal(0, 0.15, len(base_sample))
                uniform_noise = np.random.uniform(-0.1, 0.1, len(base_sample))
                
                # Combine noises
                combined_noise = 0.7 * gaussian_noise + 0.3 * uniform_noise
                
                # Apply noise with feature-specific scaling
                synthetic_sample = base_sample * (1 + combined_noise * 0.3)
                
                # Add some outliers occasionally
                if np.random.random() < 0.1:
                    outlier_indices = np.random.choice(len(base_sample), 3, replace=False)
                    synthetic_sample.iloc[outlier_indices] *= np.random.uniform(1.2, 1.8, 3)
                    
            else:
                # Generate healthy-like sample with variation
                base_sample = healthy_samples.sample(1).iloc[0]
                
                # Less noise for healthy samples
                gaussian_noise = np.random.normal(0, 0.08, len(base_sample))
                uniform_noise = np.random.uniform(-0.05, 0.05, len(base_sample))
                
                combined_noise = 0.8 * gaussian_noise + 0.2 * uniform_noise
                synthetic_sample = base_sample * (1 + combined_noise * 0.15)
            
            synthetic_data.append(synthetic_sample.values)
            synthetic_labels.append(is_parkinsons)
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(synthetic_data, columns=X.columns)
        synthetic_df['status'] = synthetic_labels
        synthetic_df['name'] = [f'synthetic_enhanced_{i:05d}' for i in range(n_samples)]
        
        # Reorder columns to match original
        synthetic_df = synthetic_df[['name'] + list(X.columns) + ['status']]
        
        # Save synthetic data
        synthetic_df.to_csv(self.external_dir / "enhanced_synthetic_parkinsons.csv", index=False)
        print(f"✅ Generated {n_samples} enhanced synthetic samples")
        
        return synthetic_df
    
    def create_telephone_quality_variations(self, base_df, n_variations=1000):
        """Create variations simulating telephone/smartphone quality"""
        print(f"📞 Creating {n_variations} telephone-quality variations...")
        
        X = base_df.drop(['name', 'status'], axis=1)
        y = base_df['status']
        
        telephone_data = []
        telephone_labels = []
        
        for i in range(n_variations):
            # Select random base sample
            idx = np.random.randint(0, len(base_df))
            base_sample = X.iloc[idx]
            label = y.iloc[idx]
            
            # Simulate telephone quality degradation
            # 1. Frequency band limiting (affects pitch features)
            pitch_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)']
            degraded_sample = base_sample.copy()
            
            for feature in pitch_features:
                if feature in degraded_sample.index:
                    # Simulate bandwidth limitation
                    degraded_sample[feature] *= np.random.uniform(0.85, 1.15)
            
            # 2. Add compression artifacts (affects jitter/shimmer)
            jitter_shimmer_features = [col for col in degraded_sample.index if 'Jitter' in col or 'Shimmer' in col]
            for feature in jitter_shimmer_features:
                # Compression increases jitter/shimmer
                degraded_sample[feature] *= np.random.uniform(1.1, 1.4)
            
            # 3. Add background noise effects
            noise_factor = np.random.uniform(0.95, 1.05)
            degraded_sample *= noise_factor
            
            # 4. Simulate microphone quality variation
            mic_quality = np.random.uniform(0.9, 1.1)
            degraded_sample *= mic_quality
            
            telephone_data.append(degraded_sample.values)
            telephone_labels.append(label)
        
        # Create DataFrame
        telephone_df = pd.DataFrame(telephone_data, columns=X.columns)
        telephone_df['status'] = telephone_labels
        telephone_df['name'] = [f'telephone_quality_{i:04d}' for i in range(n_variations)]
        
        # Reorder columns
        telephone_df = telephone_df[['name'] + list(X.columns) + ['status']]
        
        # Save telephone quality data
        telephone_df.to_csv(self.external_dir / "telephone_quality_variations.csv", index=False)
        print(f"✅ Created {n_variations} telephone-quality variations")
        
        return telephone_df
    
    def combine_all_datasets(self):
        """Combine all available datasets into one large dataset"""
        print("🔄 Combining all datasets...")
        
        combined_data = []
        
        # Load UCI original dataset
        if (self.raw_dir / "uci_parkinsons_original.data").exists():
            uci_data = pd.read_csv(self.raw_dir / "uci_parkinsons_original.data")
            uci_data['source'] = 'UCI_Original'
            combined_data.append(uci_data)
            print(f"   Added UCI Original: {len(uci_data)} samples")
        
        # Load enhanced synthetic data
        if (self.external_dir / "enhanced_synthetic_parkinsons.csv").exists():
            synthetic_data = pd.read_csv(self.external_dir / "enhanced_synthetic_parkinsons.csv")
            synthetic_data['source'] = 'Enhanced_Synthetic'
            combined_data.append(synthetic_data)
            print(f"   Added Enhanced Synthetic: {len(synthetic_data)} samples")
        
        # Load telephone quality variations
        if (self.external_dir / "telephone_quality_variations.csv").exists():
            telephone_data = pd.read_csv(self.external_dir / "telephone_quality_variations.csv")
            telephone_data['source'] = 'Telephone_Quality'
            combined_data.append(telephone_data)
            print(f"   Added Telephone Quality: {len(telephone_data)} samples")
        
        if combined_data:
            # Combine all datasets
            final_dataset = pd.concat(combined_data, ignore_index=True)
            
            # Save combined dataset
            final_dataset.to_csv(self.processed_dir / "large_scale_parkinsons_dataset.csv", index=False)
            
            print(f"✅ Large-scale dataset created: {len(final_dataset)} total samples")
            print(f"   Parkinson's: {final_dataset['status'].sum()}")
            print(f"   Healthy: {len(final_dataset) - final_dataset['status'].sum()}")
            
            # Show distribution by source
            print(f"\n📊 Data sources:")
            source_dist = final_dataset['source'].value_counts()
            for source, count in source_dist.items():
                print(f"   {source}: {count} samples")
            
            return final_dataset
        else:
            print("❌ No datasets found to combine")
            return None
    
    def download_all_large_scale(self):
        """Download and prepare large-scale dataset"""
        print("🚀 Starting large-scale data collection...")
        print("=" * 60)
        
        # Download UCI datasets
        uci_count = self.download_uci_datasets()
        
        # Try to download additional datasets
        self.download_parkinson_speech_dataset()
        
        # Generate enhanced synthetic data (5000 samples)
        self.create_enhanced_synthetic_data(n_samples=5000)
        
        # Create telephone quality variations (2000 samples)
        if (self.raw_dir / "uci_parkinsons_original.data").exists():
            base_df = pd.read_csv(self.raw_dir / "uci_parkinsons_original.data")
            self.create_telephone_quality_variations(base_df, n_variations=2000)
        
        # Combine all datasets
        combined_dataset = self.combine_all_datasets()
        
        if combined_dataset is not None:
            print(f"\n🎯 LARGE-SCALE DATASET READY!")
            print(f"=" * 60)
            print(f"Total samples: {len(combined_dataset):,}")
            print(f"Features: {len(combined_dataset.columns) - 3}")  # Excluding name, status, source
            print(f"Size increase: {len(combined_dataset)/195:.1f}x larger than original")
            
            # Class balance
            class_dist = combined_dataset['status'].value_counts()
            print(f"\nClass distribution:")
            print(f"  Healthy (0): {class_dist[0]:,} ({class_dist[0]/len(combined_dataset):.1%})")
            print(f"  Parkinson's (1): {class_dist[1]:,} ({class_dist[1]/len(combined_dataset):.1%})")
            
            print(f"\n📁 Dataset location: data/processed/large_scale_parkinsons_dataset.csv")
            print(f"🎉 Ready for training with normal recording conditions!")
        
        return combined_dataset

def main():
    collector = LargeScaleDataCollector()
    dataset = collector.download_all_large_scale()
    
    if dataset is not None:
        print(f"\n🚀 Next steps:")
        print(f"1. Run: python scripts/train_models.py")
        print(f"2. Test with your audio file")
        print(f"3. The larger dataset should handle normal recordings better!")

if __name__ == "__main__":
    main()
