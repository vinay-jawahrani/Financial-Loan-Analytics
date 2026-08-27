"""
default_prediction.py - Simplified interface for loan default prediction
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add parent directory to path if running as script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from models
from models.predict import load_model, predict_single, predict_batch

# Load model once at module import
_model, _imputer = load_model()

# Load feature names
_feature_path = os.path.join(os.path.dirname(__file__), 'feature_names.txt')
if os.path.exists(_feature_path):
    with open(_feature_path, 'r') as f:
        _feature_names = [line.strip() for line in f.readlines()]
else:
    _feature_names = None

def predict_default(loan_data):
    """
    Predict default probability for a single loan.
    
    Args:
        loan_data: Dict with loan features
    
    Returns:
        dict: {
            'default_probability': float,
            'risk_level': str,
            'recommendation': str,
            'prediction': int (0 or 1)
        }
    """
    if _model is None:
        raise Exception("Model not loaded. Run train_model.py first.")
    
    if _feature_names is None:
        raise Exception("Feature names not found. Run train_model.py first.")
    
    return predict_single(_model, _imputer, loan_data, _feature_names)

def predict_default_batch(df):
    """
    Predict default probability for multiple loans.
    
    Args:
        df: DataFrame with loan features
    
    Returns:
        DataFrame with predictions added
    """
    if _model is None:
        raise Exception("Model not loaded. Run train_model.py first.")
    
    if _feature_names is None:
        raise Exception("Feature names not found. Run train_model.py first.")
    
    return predict_batch(_model, _imputer, df, _feature_names)

def get_risk_level(proba):
    """Get risk level from probability"""
    if proba < 0.10:
        return 'Low Risk'
    elif proba < 0.25:
        return 'Medium Risk'
    elif proba < 0.50:
        return 'High Risk'
    else:
        return 'Very High Risk'

def get_recommendation(proba):
    """Get recommendation from probability"""
    if proba < 0.10:
        return "Approve loan"
    elif proba < 0.25:
        return "Approve with monitoring"
    elif proba < 0.50:
        return "Consider higher interest rate"
    else:
        return "Decline or require collateral"

if __name__ == "__main__":
    # Test prediction
    sample = {
        'loan_amount': 15000,
        'interest_rate': 12.5,
        'term': 36,
        'installment': 500,
        'grade': 'C',
        'sub_grade': 'C3',
        'annual_income': 65000,
        'debt_to_income': 25.0,
        'homeownership': 'RENT',
        'state': 'CA',
        'loan_purpose': 'debt_consolidation',
        'application_type': 'individual',
        'delinq_2y': 1,
        'inquiries_last_12m': 2,
        'total_credit_lines': 10,
        'open_credit_lines': 5,
        'total_credit_limit': 50000,
        'total_credit_utilized': 15000,
        'num_collections_last_12m': 0,
        'num_historical_failed_to_pay': 0,
        'months_since_90d_late': 12,
        'current_accounts_delinq': 0,
        'accounts_opened_24m': 3,
        'num_satisfactory_accounts': 8,
        'num_accounts_120d_past_due': 0,
        'num_accounts_30d_past_due': 1,
        'num_total_cc_accounts': 4,
        'num_open_cc_accounts': 3,
        'num_cc_carrying_balance': 2,
        'account_never_delinq_percent': 80.0,
        'tax_liens': 0,
        'public_record_bankrupt': 0
    }
    
    result = predict_default(sample)
    print("\n📊 Loan Default Prediction:")
    print(f"  Default Probability: {result['default_probability']}%")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Recommendation: {result['recommendation']}")