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
from sklearn.model_selection import train_test_split

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

def get_feature_names():
    """Load feature names"""
    path = os.path.join(os.path.dirname(__file__), 'feature_names.txt')
    
    if os.path.exists(path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    else:
        return None

def load_test_data():
    """Load test data from database"""
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/loan_analytics")
    engine = create_engine(db_url)
    
    query = """
    SELECT 
        CASE 
            WHEN loan_status IN ('Charged Off', 'Late (31-120 days)') THEN 1 
            ELSE 0 
        END AS is_default,
        loan_amount,
        interest_rate,
        term,
        installment,
        grade,
        sub_grade,
        annual_income,
        debt_to_income,
        homeownership,
        state,
        loan_purpose,
        application_type,
        delinq_2y,
        inquiries_last_12m,
        total_credit_lines,
        open_credit_lines,
        total_credit_limit,
        total_credit_utilized,
        num_collections_last_12m,
        num_historical_failed_to_pay,
        months_since_90d_late,
        current_accounts_delinq,
        accounts_opened_24m,
        num_satisfactory_accounts,
        num_accounts_120d_past_due,
        num_accounts_30d_past_due,
        num_total_cc_accounts,
        num_open_cc_accounts,
        num_cc_carrying_balance,
        account_never_delinq_percent,
        tax_liens,
        public_record_bankrupt
    FROM stg_loans
    WHERE loan_status IS NOT NULL
    """
    
    df = pd.read_sql(query, engine)
    return df

def engineer_features(df):
    """Create new features (must match training)"""
    df = df.replace([np.inf, -np.inf], np.nan)
    df_processed = df.copy()
    
    df_processed['credit_utilization'] = df_processed['total_credit_utilized'] / df_processed['total_credit_limit'].replace(0, np.nan)
    df_processed['credit_utilization'] = df_processed['credit_utilization'].fillna(0)
    
    df_processed['income_to_loan'] = df_processed['annual_income'] / df_processed['loan_amount'].replace(0, np.nan)
    df_processed['income_to_loan'] = df_processed['income_to_loan'].fillna(0)
    
    df_processed['delinquency_ratio'] = df_processed['delinq_2y'] / df_processed['total_credit_lines'].replace(0, np.nan)
    df_processed['delinquency_ratio'] = df_processed['delinquency_ratio'].fillna(0)
    
    df_processed['installment_to_income'] = df_processed['installment'] / df_processed['annual_income'].replace(0, np.nan)
    df_processed['installment_to_income'] = df_processed['installment_to_income'].fillna(0)
    
    df_processed['months_since_90d_late'] = df_processed['months_since_90d_late'].fillna(999)
    
    # Encode categoricals
    from sklearn.preprocessing import LabelEncoder
    categorical_cols = ['grade', 'sub_grade', 'homeownership', 'state', 'loan_purpose', 'application_type']
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna('Unknown')
            le = LabelEncoder()
            df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
            df_processed = df_processed.drop(col, axis=1)
    
    return df_processed

def plot_feature_importance(model, feature_names, top_n=15):
    """
    Plot feature importance from XGBoost model
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df.head(top_n))))[::-1]
    
    plt.barh(
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
    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'feature_importance.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Feature importance plot saved to '{save_path}'")
    
    plt.show()
    
    return importance_df

def calculate_permutation_importance():
    """
    Calculate permutation importance (requires full data)
    """
    print("\n🔄 Loading data for permutation importance...")
    df = load_test_data()
    
    # Engineer features
    df = engineer_features(df)
    
    # Separate features and target
    X = df.drop(['is_default'], axis=1)
    y = df['is_default']
    
    # Split to get test set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load model and imputer
    model, imputer = load_model()
    
    # Impute test data
    if imputer is not None:
        X_test_imputed = imputer.transform(X_test)
    else:
        X_test_imputed = X_test.fillna(0).values
    
    print("🔄 Calculating permutation importance (this may take a while)...")
    perm_importance = permutation_importance(
        model, X_test_imputed, y_test, 
        n_repeats=3, 
        random_state=42, 
        n_jobs=-1
    )
    
    perm_df = pd.DataFrame({
        'feature': X.columns,
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=False)
    
    return perm_df

def generate_report(importance_df):
    """
    Generate a text report of feature importance
    """
    print("\n" + "="*60)
    print("📊 Feature Importance Report")
    print("="*60)
    
    print("\n📈 Top 10 Features:")
    print("-"*40)
    for i, row in importance_df.head(10).iterrows():
        print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
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
        'Encoded Features': ['grade_encoded', 'sub_grade_encoded', 'homeownership_encoded',
                            'state_encoded', 'loan_purpose_encoded', 'application_type_encoded'],
        'Engineered Features': ['credit_utilization', 'income_to_loan', 
                               'delinquency_ratio', 'installment_to_income']
    }
    
    print("\n📊 Importance by Category:")
    print("-"*40)
    for category, features in feature_categories.items():
        total_importance = sum(importance_df[importance_df['feature'].isin(features)]['importance'])
        print(f"  {category}: {total_importance:.4f} ({total_importance*100:.1f}%)")

def main():
    print("🚀 Generating Feature Importance Analysis...")
    
    # Load model
    model, _ = load_model()
    feature_names = get_feature_names()
    
    if feature_names is None:
        print("❌ Feature names not found. Run train_model.py first.")
        return
    
    # Plot feature importance
    importance_df = plot_feature_importance(model, feature_names)
    
    # Generate report
    generate_report(importance_df)
    
    # Optional: calculate permutation importance
    try:
        perm_df = calculate_permutation_importance()
        print("\n📊 Permutation Importance Top 10:")
        print("-"*40)
        print(perm_df.head(10).to_string(index=False))
    except Exception as e:
        print(f"\n⚠️ Could not calculate permutation importance: {e}")
    
    print("\n✅ Feature importance analysis complete!")

if __name__ == "__main__":
    main()