"""
explore_data.py - Explore the loan dataset before loading.
"""

import pandas as pd
from pathlib import Path

# ===== Configuration =====
RAW_PATH = Path("../data/raw/lending_club_loans.csv")

def explore_data():
    """Explore the dataset."""
    
    print("=" * 70)
    print("🏦 Loan Data Exploration")
    print("=" * 70)
    
    if not RAW_PATH.exists():
        print(f"❌ File not found: {RAW_PATH}")
        print("   Please place your CSV file in data/raw/")
        return
    
    df = pd.read_csv(RAW_PATH)
    
    print(f"\n🔢 Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    print("\n📋 Columns:")
    for col in df.columns[:20]:
        print(f"   - {col}")
    if len(df.columns) > 20:
        print(f"   ... and {len(df.columns) - 20} more")
    
    print("\n📊 Data Types:")
    print(df.dtypes.value_counts().to_string())
    
    print("\n📝 First 5 rows:")
    print(df.head())
    
    print("\n🔍 Missing Values (top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    
    print("\n📊 Key Metrics:")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols[:10]:
        print(f"   {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
    
    print("\n📊 Loan Status Distribution:")
    if 'loan_status' in df.columns:
        print(df['loan_status'].value_counts().to_string())
    
    print("\n📊 Grade Distribution:")
    if 'grade' in df.columns:
        print(df['grade'].value_counts().sort_index().to_string())
    
    print("\n" + "=" * 70)
    print("✅ Exploration complete!")
    print("=" * 70)

if __name__ == "__main__":
    explore_data()