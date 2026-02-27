#!/usr/bin/env python3
"""
Data Collection Script for Parkinson's Disease Detection
Downloads multiple datasets for comprehensive training
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path

class DataCollector:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.external_dir = self.data_dir / "external"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.external_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_uci_parkinsons(self):
        """Download original UCI Parkinson's dataset"""
        print("📥 Downloading UCI Parkinson's Dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
        
        try:
            response = requests.get(url)
            with open(self.raw_dir / "uci_parkinsons.data", "wb") as f:
                f.write(response.content)
            print("✅ UCI Parkinson's dataset downloaded")
            return True
        except Exception as e:
            print(f"❌ Error downloading UCI dataset: {e}")
            return False
    
    def download_parkinsons_telemonitoring(self):
        """Download Parkinson's telemonitoring dataset"""
        print("📥 Downloading Parkinson's Telemonitoring Dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"
        
        try:
            response = requests.get(url)
            with open(self.raw_dir / "parkinsons_telemonitoring.data", "wb") as f:
                f.write(response.content)
            print("✅ Telemonitoring dataset downloaded")
            return True
        except Exception as e:
            print(f"❌ Error downloading telemonitoring dataset: {e}")
            return False
    
    def create_synthetic_data(self, n_samples=1000):
        """Create synthetic data to augment the dataset"""
        print(f"🔬 Generating {n_samples} synthetic samples...")
        
        # Load original data to understand feature distributions
        original_data = pd.read_csv(self.raw_dir / "uci_parkinsons.data")
        X = original_data.drop(['name', 'status'], axis=1)
        y = original_data['status']
        
        # Separate healthy and Parkinson's samples
        healthy_samples = X[y == 0]
        parkinsons_samples = X[y == 1]
        
        synthetic_data = []
        synthetic_labels = []
        
        for i in range(n_samples):
            # Randomly choose class (balanced)
            is_parkinsons = np.random.choice([0, 1])
            
            if is_parkinsons:
                # Generate Parkinson's-like sample
                base_sample = parkinsons_samples.sample(1).iloc[0]
                # Add noise and variation
                noise = np.random.normal(0, 0.1, len(base_sample))
                synthetic_sample = base_sample * (1 + noise * 0.2)
            else:
                # Generate healthy-like sample
                base_sample = healthy_samples.sample(1).iloc[0]
                noise = np.random.normal(0, 0.05, len(base_sample))
                synthetic_sample = base_sample * (1 + noise * 0.1)
            
            synthetic_data.append(synthetic_sample.values)
            synthetic_labels.append(is_parkinsons)
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(synthetic_data, columns=X.columns)
        synthetic_df['status'] = synthetic_labels
        synthetic_df['name'] = [f'synthetic_{i:04d}' for i in range(n_samples)]
        
        # Reorder columns to match original
        synthetic_df = synthetic_df[['name'] + list(X.columns) + ['status']]
        
        # Save synthetic data
        synthetic_df.to_csv(self.external_dir / "synthetic_parkinsons.csv", index=False)
        print(f"✅ Generated {n_samples} synthetic samples")
        
        return synthetic_df
    
    def combine_datasets(self):
        """Combine all available datasets"""
        print("🔄 Combining all datasets...")
        
        combined_data = []
        
        # Load UCI dataset
        if (self.raw_dir / "uci_parkinsons.data").exists():
            uci_data = pd.read_csv(self.raw_dir / "uci_parkinsons.data")
            uci_data['source'] = 'UCI_Original'
            combined_data.append(uci_data)
            print(f"   Added UCI dataset: {len(uci_data)} samples")
        
        # Load synthetic data
        if (self.external_dir / "synthetic_parkinsons.csv").exists():
            synthetic_data = pd.read_csv(self.external_dir / "synthetic_parkinsons.csv")
            synthetic_data['source'] = 'Synthetic'
            combined_data.append(synthetic_data)
            print(f"   Added synthetic dataset: {len(synthetic_data)} samples")
        
        if combined_data:
            # Combine all datasets
            final_dataset = pd.concat(combined_data, ignore_index=True)
            
            # Save combined dataset
            final_dataset.to_csv(self.processed_dir / "combined_parkinsons_dataset.csv", index=False)
            
            print(f"✅ Combined dataset created: {len(final_dataset)} total samples")
            print(f"   Parkinson's: {final_dataset['status'].sum()}")
            print(f"   Healthy: {len(final_dataset) - final_dataset['status'].sum()}")
            
            return final_dataset
        else:
            print("❌ No datasets found to combine")
            return None
    
    def download_all(self):
        """Download and prepare all datasets"""
        print("🚀 Starting comprehensive data collection...")
        
        # Download datasets
        self.download_uci_parkinsons()
        self.download_parkinsons_telemonitoring()
        
        # Generate synthetic data
        self.create_synthetic_data(n_samples=2000)
        
        # Combine all datasets
        combined_dataset = self.combine_datasets()
        
        if combined_dataset is not None:
            print(f"\n📊 Final Dataset Statistics:")
            print(f"Total samples: {len(combined_dataset)}")
            print(f"Features: {len(combined_dataset.columns) - 3}")  # Excluding name, status, source
            print(f"Class distribution:")
            print(combined_dataset['status'].value_counts())
            print(f"Data sources:")
            print(combined_dataset['source'].value_counts())
        
        print("\n✅ Data collection completed!")
        return combined_dataset

def main():
    collector = DataCollector()
    dataset = collector.download_all()
    
    if dataset is not None:
        print(f"\n🎯 Dataset ready for training!")
        print(f"Location: data/processed/combined_parkinsons_dataset.csv")

if __name__ == "__main__":
    main()
