"""
feature_importance.py - Analyze feature importance
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance"""
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=importance_df.head(top_n),
        x='importance',
        y='feature',
        palette='viridis'
    )
    plt.title(f'Top {top_n} Features for Loan Default Prediction', fontsize=14)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()
    
    return importance_df

if __name__ == "__main__":
    model = joblib.load('model.pkl')
    
    with open('feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    importance_df = plot_feature_importance(model, feature_names)
    print("\n📊 Feature Importance:")
    print(importance_df.head(15).to_string(index=False))