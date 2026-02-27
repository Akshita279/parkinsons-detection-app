import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_dataset():
    """Explore the Parkinson's dataset"""
    df = pd.read_csv('parkinsons.data')
    
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())
    
    print("\nTarget Distribution:")
    print(df['status'].value_counts())
    print(f"Parkinson's cases: {df['status'].sum()}")
    print(f"Healthy cases: {len(df) - df['status'].sum()}")
    
    # Visualize target distribution
    plt.figure(figsize=(8, 6))
    df['status'].value_counts().plot(kind='bar')
    plt.title('Distribution of Parkinson\'s vs Healthy Cases')
    plt.xlabel('Status (0=Healthy, 1=Parkinson\'s)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.savefig('target_distribution.png')
    plt.show()
    
    # Feature correlation heatmap
    plt.figure(figsize=(15, 12))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.show()

if __name__ == "__main__":
    import numpy as np
    explore_dataset()
