"""
load_to_postgres.py - Load the full loan dataset.
"""

import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ===== FIXED: Correct path =====
RAW_PATH = Path("data/raw/loans_full_schema.csv")
DB_URL = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_rjGsBo7hdqz9@ep-royal-credit-azy6sjo9-pooler.c-3.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require")

def load_data():
    print("🚀 Loading Full Loan Dataset...")
    
    if not RAW_PATH.exists():
        print(f"❌ File not found: {RAW_PATH}")
        print(f"   Current directory: {Path.cwd()}")
        print(f"   Expected path: {RAW_PATH.absolute()}")
        return False
    
    # Read CSV
    df = pd.read_csv(RAW_PATH)
    print(f"📄 Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Replace "NA" strings with actual NaN
    df = df.replace("NA", pd.NA)
    
    # Convert numeric columns
    numeric_cols = [
        'annual_income', 'debt_to_income', 'loan_amount', 'interest_rate',
        'installment', 'balance', 'paid_total', 'paid_principal', 'paid_interest',
        'paid_late_fees', 'total_credit_limit', 'total_credit_utilized',
        'total_collection_amount_ever', 'total_debit_limit',
        'account_never_delinq_percent'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert integer columns
    int_cols = [
        'delinq_2y', 'inquiries_last_12m', 'total_credit_lines', 'open_credit_lines',
        'num_collections_last_12m', 'num_historical_failed_to_pay', 'term',
        'current_accounts_delinq', 'current_installment_accounts', 'accounts_opened_24m',
        'num_satisfactory_accounts', 'num_accounts_120d_past_due', 'num_accounts_30d_past_due',
        'num_active_debit_accounts', 'num_total_cc_accounts', 'num_open_cc_accounts',
        'num_cc_carrying_balance', 'num_mort_accounts', 'tax_liens', 'public_record_bankrupt'
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    # Connect and load
    engine = create_engine(DB_URL)
    df.to_sql('stg_loans', engine, if_exists='replace', index=False)
    print(f"✅ Loaded {len(df)} rows into stg_loans")
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT 
                COUNT(*) AS total_rows,
                COUNT(DISTINCT state) AS states,
                COUNT(DISTINCT grade) AS grades,
                AVG(loan_amount) AS avg_loan,
                AVG(interest_rate) AS avg_rate
            FROM stg_loans
        """)).fetchone()
        
        print("\n📊 Data Load Summary:")
        print(f"  Total Rows: {result[0]:,}")
        print(f"  States: {result[1]}")
        print(f"  Grades: {result[2]}")
        print(f"  Avg Loan: ${result[3]:,.2f}")
        print(f"  Avg Interest Rate: {result[4]:.2f}%")
    
    return True

if __name__ == "__main__":
    load_data()