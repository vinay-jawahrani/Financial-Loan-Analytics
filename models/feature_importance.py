"""
feature_importance.py - Analyze and visualize feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.inspection import permutation_importance
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def load_model():
    """Load trained model"""
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")
    
    return joblib.load(model_path)

def get_feature_names():
    """Load feature names"""
    path = os.path.join(os.path.dirname(__file__), 'feature_names.txt')
    
    if os.path.exists(path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    else:
        return None

def plot_feature_importance(model, feature_names, top_n=15):
    """
    Plot feature importance from XGBoost model
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        top_n: Number of top features to show
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df.head(top_n))))[::-1]
    
    bars = plt.barh(
        importance_df.head(top_n)['feature'],
        importance_df.head(top_n)['importance'],
        color=colors
    )
    
    plt.xlabel('Feature Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title(f'Top {top_n} Features for Loan Default Prediction', fontsize=16)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'feature_importance.png'), dpi=150, bbox_inches='tight')
    print(f"💾 Feature importance plot saved to 'feature_importance.png'")
    
    plt.show()
    
    return importance_df

def get_permutation_importance(model, X_test, y_test, feature_names):
    """
    Calculate permutation importance
    """
    print("\n🔄 Calculating permutation importance...")
    perm_importance = permutation_importance(
        model, X_test, y_test, 
        n_repeats=5, 
        random_state=42, 
        n_jobs=-1
    )
    
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=False)
    
    return perm_df

def generate_report(importance_df, model=None, feature_names=None):
    """
    Generate a text report of feature importance
    """
    print("\n" + "="*60)
    print("📊 Feature Importance Report")
    print("="*60)
    
    print("\n📈 Top 10 Features (XGBoost):")
    print("-"*40)
    for i, row in importance_df.head(10).iterrows():
        print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    print("\n📊 Feature Categories:")
    print("-"*40)
    
    # Categorize features
    feature_categories = {
        'Loan Features': ['loan_amount', 'interest_rate', 'term', 'installment'],
        'Borrower Features': ['annual_income', 'debt_to_income', 'homeownership', 'state'],
        'Credit Features': ['delinq_2y', 'inquiries_last_12m', 'total_credit_lines', 
                           'open_credit_lines', 'total_credit_limit', 'total_credit_utilized',
                           'num_collections_last_12m', 'num_historical_failed_to_pay',
                           'months_since_90d_late', 'current_accounts_delinq',
                           'accounts_opened_24m', 'num_satisfactory_accounts'],
        'Account Features': ['num_accounts_120d_past_due', 'num_accounts_30d_past_due',
                            'num_total_cc_accounts', 'num_open_cc_accounts',
                            'num_cc_carrying_balance', 'num_mort_accounts',
                            'account_never_delinq_percent', 'tax_liens', 'public_record_bankrupt'],
        'Credit Card Features': ['num_total_cc_accounts', 'num_open_cc_accounts', 'num_cc_carrying_balance'],
        'Categorical Encoded': ['grade_encoded', 'sub_grade_encoded', 'homeownership_encoded',
                               'state_encoded', 'loan_purpose_encoded', 'application_type_encoded'],
        'Engineered Features': ['credit_utilization', 'income_to_loan', 
                               'delinquency_ratio', 'installment_to_income']
    }
    
    for category, features in feature_categories.items():
        total_importance = sum(importance_df[importance_df['feature'].isin(features)]['importance'])
        print(f"  {category}: {total_importance:.4f} ({total_importance*100:.1f}%)")

def main():
    print("🚀 Generating Feature Importance Analysis...")
    
    # Load model and feature names
    model = load_model()
    feature_names = get_feature_names()
    
    if feature_names is None:
        print("❌ Feature names not found. Run train_model.py first.")
        return
    
    # Plot feature importance
    importance_df = plot_feature_importance(model, feature_names)
    
    # Generate report
    generate_report(importance_df)
    
    print("\n✅ Feature importance analysis complete!")

if __name__ == "__main__":
    main()