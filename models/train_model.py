"""
train_model.py - Train XGBoost model to predict loan default
Handles class imbalance with SMOTE and missing values with imputation
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# ===== Database Connection =====
def get_data():
    """Load data from PostgreSQL"""
    db_url = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_rjGsBo7hdqz9@ep-royal-credit-azy6sjo9-pooler.c-3.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require")
    engine = create_engine(db_url)
    
    query = """
    SELECT 
        -- Target
        CASE 
            WHEN loan_status IN ('Charged Off', 'Late (31-120 days)') THEN 1 
            ELSE 0 
        END AS is_default,
        
        -- Loan features
        loan_amount,
        interest_rate,
        term,
        installment,
        grade,
        sub_grade,
        
        -- Borrower features
        annual_income,
        debt_to_income,
        homeownership,
        state,
        loan_purpose,
        application_type,
        
        -- Credit history
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
    print(f"📄 Loaded {len(df)} rows")
    return df

# ===== Feature Engineering =====
def engineer_features(df):
    """Create new features"""
    df = df.replace([np.inf, -np.inf], np.nan)
    df_processed = df.copy()
    
    # Credit utilization ratio
    df_processed['credit_utilization'] = df_processed['total_credit_utilized'] / df_processed['total_credit_limit'].replace(0, np.nan)
    df_processed['credit_utilization'] = df_processed['credit_utilization'].fillna(0)
    
    # Income to loan ratio
    df_processed['income_to_loan'] = df_processed['annual_income'] / df_processed['loan_amount'].replace(0, np.nan)
    df_processed['income_to_loan'] = df_processed['income_to_loan'].fillna(0)
    
    # Delinquency ratio
    df_processed['delinquency_ratio'] = df_processed['delinq_2y'] / df_processed['total_credit_lines'].replace(0, np.nan)
    df_processed['delinquency_ratio'] = df_processed['delinquency_ratio'].fillna(0)
    
    # Installment to income ratio
    df_processed['installment_to_income'] = df_processed['installment'] / df_processed['annual_income'].replace(0, np.nan)
    df_processed['installment_to_income'] = df_processed['installment_to_income'].fillna(0)
    
    # Months since delinquency (999 = never)
    df_processed['months_since_90d_late'] = df_processed['months_since_90d_late'].fillna(999)
    
    print(f"✅ Engineered {len(df_processed.columns)} features")
    return df_processed

# ===== Encode Categoricals =====
def encode_categoricals(df):
    """Encode categorical variables"""
    categorical_cols = ['grade', 'sub_grade', 'homeownership', 'state', 'loan_purpose', 'application_type']
    
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].fillna('Unknown')
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"  ✅ Encoded {col}")
    
    # Drop original categorical columns
    for col in categorical_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    return df, encoders

# ===== Train Model =====
def train_model(df):
    """Train XGBoost model with SMOTE"""
    
    print("\n" + "="*60)
    print("🏦 Training Loan Default Prediction Model")
    print("="*60)
    
    # Separate features and target
    X = df.drop(['is_default'], axis=1)
    y = df['is_default']
    
    print(f"\n📊 Target Distribution:")
    print(f"  Good Loans: {len(y[y==0]):,} ({len(y[y==0])/len(y)*100:.1f}%)")
    print(f"  Defaults: {len(y[y==1]):,} ({len(y[y==1])/len(y)*100:.1f}%)")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle missing values
    print("\n🔄 Handling missing values...")
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Convert back to DataFrame
    X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
    X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)
    print(f"  ✅ Imputed missing values with median")
    
    # Apply SMOTE
    print("\n🔄 Applying SMOTE for class imbalance...")
    smote = SMOTE(random_state=42, sampling_strategy=0.3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)
    print(f"  After SMOTE: {len(X_train_resampled):,} samples")
    
    print(f"\n📋 Data Split:")
    print(f"  Training: {len(X_train_resampled):,} samples")
    print(f"  Testing: {len(X_test):,} samples")
    
    # Calculate class weight
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    # Train XGBoost
    print("\n🤖 Training XGBoost Model...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train_resampled, y_train_resampled,
        eval_set=[(X_test_imputed, y_test)],
        verbose=False
    )
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test_imputed)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Evaluation
    print("\n📊 Model Performance:")
    print("-"*40)
    print(classification_report(y_test, y_pred, target_names=['Good', 'Default']))
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n✅ ROC-AUC Score: {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"  True Negatives: {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives: {cm[1][1]}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📈 Top 10 Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to {model_path}")
    
    # Save feature names
    with open(os.path.join(os.path.dirname(__file__), 'feature_names.txt'), 'w') as f:
        for col in X.columns:
            f.write(col + '\n')
    
    # Save imputer
    joblib.dump(imputer, os.path.join(os.path.dirname(__file__), 'imputer.pkl'))
    print(f"💾 Imputer saved to {os.path.join(os.path.dirname(__file__), 'imputer.pkl')}")
    
    return model

# ===== Main =====
def main():
    print("🚀 Starting Loan Default Prediction Pipeline...")
    
    # Load data
    df = get_data()
    
    if df.empty:
        print("❌ No data found! Please load data first.")
        return
    
    # Engineer features
    df = engineer_features(df)
    
    # Encode categoricals
    df, encoders = encode_categoricals(df)
    
    # Train model
    model = train_model(df)
    
    print("\n" + "="*60)
    print("✅ Pipeline Complete!")
    print("="*60)

if __name__ == "__main__":
    main()