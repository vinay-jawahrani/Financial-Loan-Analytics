"""
predict.py - Make predictions on new loans
"""

import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

def load_model():
    """Load trained model"""
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    imputer_path = os.path.join(os.path.dirname(__file__), 'imputer.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")
    
    model = joblib.load(model_path)
    
    try:
        imputer = joblib.load(imputer_path)
    except:
        imputer = None
    
    return model, imputer

def get_risk_level(proba):
    """Map probability to risk level"""
    if proba < 0.10:
        return 'Low Risk 🟢'
    elif proba < 0.25:
        return 'Medium Risk 🟡'
    elif proba < 0.50:
        return 'High Risk 🟠'
    else:
        return 'Very High Risk 🔴'

def get_recommendation(proba):
    """Get recommendation based on risk level"""
    if proba < 0.10:
        return "✅ Approve loan"
    elif proba < 0.25:
        return "ℹ️ Approve with monitoring"
    elif proba < 0.50:
        return "⚠️ Consider higher interest rate"
    else:
        return "🚨 Decline or require collateral"

def predict_single(model, imputer, features, feature_names):
    """
    Predict default probability for a single loan
    
    Args:
        model: Trained XGBoost model
        imputer: Fitted SimpleImputer
        features: Dict with loan features
        feature_names: List of feature names from training
    
    Returns:
        dict: Prediction results
    """
    # Create DataFrame from single input
    df = pd.DataFrame([features])
    
    # Feature engineering (must match training)
    df['credit_utilization'] = df['total_credit_utilized'] / df['total_credit_limit'].replace(0, np.nan)
    df['credit_utilization'] = df['credit_utilization'].fillna(0)
    
    df['income_to_loan'] = df['annual_income'] / df['loan_amount'].replace(0, np.nan)
    df['income_to_loan'] = df['income_to_loan'].fillna(0)
    
    df['delinquency_ratio'] = df['delinq_2y'] / df['total_credit_lines'].replace(0, np.nan)
    df['delinquency_ratio'] = df['delinquency_ratio'].fillna(0)
    
    df['installment_to_income'] = df['installment'] / df['annual_income'].replace(0, np.nan)
    df['installment_to_income'] = df['installment_to_income'].fillna(0)
    
    df['months_since_90d_late'] = df['months_since_90d_late'].fillna(999)
    
    # Encode categoricals
    from sklearn.preprocessing import LabelEncoder
    categorical_cols = ['grade', 'sub_grade', 'homeownership', 'state', 'loan_purpose', 'application_type']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            df = df.drop(col, axis=1)
    
    # Ensure all features exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Impute missing values
    if imputer is not None:
        df_imputed = imputer.transform(df[feature_names])
    else:
        df_imputed = df[feature_names].fillna(0).values
    
    # Predict
    proba = model.predict_proba(df_imputed)[0, 1]
    
    return {
        'default_probability': round(proba * 100, 2),
        'risk_level': get_risk_level(proba),
        'recommendation': get_recommendation(proba),
        'prediction': 1 if proba > 0.5 else 0
    }

def predict_batch(model, imputer, df, feature_names):
    """Predict for multiple loans"""
    # Copy and engineer features
    df_pred = df.copy()
    
    # Feature engineering
    df_pred['credit_utilization'] = df_pred['total_credit_utilized'] / df_pred['total_credit_limit'].replace(0, np.nan)
    df_pred['credit_utilization'] = df_pred['credit_utilization'].fillna(0)
    
    df_pred['income_to_loan'] = df_pred['annual_income'] / df_pred['loan_amount'].replace(0, np.nan)
    df_pred['income_to_loan'] = df_pred['income_to_loan'].fillna(0)
    
    df_pred['delinquency_ratio'] = df_pred['delinq_2y'] / df_pred['total_credit_lines'].replace(0, np.nan)
    df_pred['delinquency_ratio'] = df_pred['delinquency_ratio'].fillna(0)
    
    df_pred['installment_to_income'] = df_pred['installment'] / df_pred['annual_income'].replace(0, np.nan)
    df_pred['installment_to_income'] = df_pred['installment_to_income'].fillna(0)
    
    df_pred['months_since_90d_late'] = df_pred['months_since_90d_late'].fillna(999)
    
    # Encode categoricals
    from sklearn.preprocessing import LabelEncoder
    categorical_cols = ['grade', 'sub_grade', 'homeownership', 'state', 'loan_purpose', 'application_type']
    for col in categorical_cols:
        if col in df_pred.columns:
            df_pred[col] = df_pred[col].fillna('Unknown')
            le = LabelEncoder()
            df_pred[col + '_encoded'] = le.fit_transform(df_pred[col].astype(str))
            df_pred = df_pred.drop(col, axis=1)
    
    # Ensure all features exist
    for col in feature_names:
        if col not in df_pred.columns:
            df_pred[col] = 0
    
    # Impute
    if imputer is not None:
        df_imputed = imputer.transform(df_pred[feature_names])
    else:
        df_imputed = df_pred[feature_names].fillna(0).values
    
    # Predict
    probas = model.predict_proba(df_imputed)[:, 1]
    
    df_pred['default_probability'] = (probas * 100).round(2)
    df_pred['risk_level'] = df_pred['default_probability'].apply(
        lambda x: get_risk_level(x/100)
    )
    df_pred['recommendation'] = df_pred['default_probability'].apply(
        lambda x: get_recommendation(x/100)
    )
    df_pred['prediction'] = (probas > 0.5).astype(int)
    
    return df_pred

if __name__ == "__main__":
    # Load model
    model, imputer = load_model()
    
    # Load feature names
    with open(os.path.join(os.path.dirname(__file__), 'feature_names.txt'), 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # Test prediction
    sample_loan = {
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
    
    result = predict_single(model, imputer, sample_loan, feature_names)
    print("\n📊 Prediction Result:")
    print(f"  Default Probability: {result['default_probability']}%")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Recommendation: {result['recommendation']}")