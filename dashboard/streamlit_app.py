"""
streamlit_app.py - Financial Loan Performance Analytics Dashboard
Complete main dashboard with all tabs including ML predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import os
import joblib
from dotenv import load_dotenv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ===== Import Page Modules =====
# These will be imported after we create them in dashboard/pages/
# For now we will define the content inline or use st.import_module if needed
# But to keep it simple, we'll just render everything in the main script.

load_dotenv()

# ===== Page Config =====
st.set_page_config(
    page_title="Loan Performance Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== Custom CSS - Dark Theme =====
st.markdown("""
<style>
    .stApp {
        background-color: #0a0e17;
    }
    .css-1d391kg {
        background-color: #111927;
    }
    .css-1d391kg .st-emotion-cache-1wivap2 {
        color: #e0e7ff;
    }
    h1, h2, h3 {
        color: #e0e7ff !important;
        font-weight: 600 !important;
    }
    .metric-card {
        background: linear-gradient(145deg, #111927, #1a2332);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #2a3a5c;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #60a5fa;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        letter-spacing: 0.5px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.3rem;
    }
    .risk-low {
        color: #22c55e;
        font-weight: 600;
        padding: 4px 12px;
        background: rgba(34, 197, 94, 0.1);
        border-radius: 20px;
        border: 1px solid #22c55e;
    }
    .risk-medium {
        color: #eab308;
        font-weight: 600;
        padding: 4px 12px;
        background: rgba(234, 179, 8, 0.1);
        border-radius: 20px;
        border: 1px solid #eab308;
    }
    .risk-high {
        color: #f97316;
        font-weight: 600;
        padding: 4px 12px;
        background: rgba(249, 115, 22, 0.1);
        border-radius: 20px;
        border: 1px solid #f97316;
    }
    .risk-critical {
        color: #ef4444;
        font-weight: 600;
        padding: 4px 12px;
        background: rgba(239, 68, 68, 0.1);
        border-radius: 20px;
        border: 1px solid #ef4444;
    }
    .card {
        background: linear-gradient(145deg, #111927, #1a2332);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #2a3a5c;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    .dataframe {
        background: #111927 !important;
        color: #e0e7ff !important;
    }
    .dataframe thead tr th {
        background: #1a2332 !important;
        color: #60a5fa !important;
    }
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.8rem;
        padding: 2rem 0;
        border-top: 1px solid #1e293b;
        margin-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #111927;
        padding: 8px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a2332;
        border-radius: 6px;
        color: #94a3b8;
        padding: 8px 20px;
        border: 1px solid #2a3a5c;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3a5f;
        color: #60a5fa;
        border-color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# ===== Database Connection =====
@st.cache_resource
def get_engine():
    db_url = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_rjGsBo7hdqz9@ep-royal-credit-azy6sjo9-pooler.c-3.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require")
    return create_engine(db_url)

# ===== Load Data =====
@st.cache_data(ttl=300)
def load_data():
    engine = get_engine()
    with engine.connect() as conn:
        loans = pd.read_sql("SELECT * FROM stg_loans", conn)
        
        # dbt models in public_public schema
        try:
            performance = pd.read_sql("SELECT * FROM public_public.loan_performance", conn)
        except:
            performance = pd.DataFrame()
        try:
            risk = pd.read_sql("SELECT * FROM public_public.risk_analysis", conn)
        except:
            risk = pd.DataFrame()
        try:
            portfolio = pd.read_sql("SELECT * FROM public_public.portfolio_summary", conn)
        except:
            portfolio = pd.DataFrame()
    return loans, performance, risk, portfolio

# ===== Model Loader =====

@st.cache_resource
def load_model():
    """Load trained XGBoost model and imputer."""
    import os
    import joblib
    
    # Try multiple possible paths
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl'),
        os.path.join(os.getcwd(), 'models', 'model.pkl'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'model.pkl'),
        'models/model.pkl',
    ]
    
    model = None
    imputer = None
    model_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        st.warning("⚠️ Model not found. Please train the model first: `python models/train_model.py`")
        return None, None
    
    try:
        model = joblib.load(model_path)
        # Load imputer
        imputer_path = model_path.replace('model.pkl', 'imputer.pkl')
        if os.path.exists(imputer_path):
            imputer = joblib.load(imputer_path)
        return model, imputer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# ===== Prediction Functions =====
def get_risk_level(proba):
    if proba < 0.10:
        return 'Low Risk 🟢', 'risk-low'
    elif proba < 0.25:
        return 'Medium Risk 🟡', 'risk-medium'
    elif proba < 0.50:
        return 'High Risk 🟠', 'risk-high'
    else:
        return 'Very High Risk 🔴', 'risk-critical'

def predict_single(model, imputer, features):
    """Predict default probability for a single loan"""
    try:
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
        
        # Get feature names
        feature_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_names.txt')
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
        else:
            feature_names = [col for col in df.columns if col not in ['is_default']]
        
        # Ensure all features exist
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Impute if needed
        if imputer is not None:
            df_imputed = imputer.transform(df[feature_names])
        else:
            df_imputed = df[feature_names].fillna(0).values
        
        proba = model.predict_proba(df_imputed)[0, 1]
        return proba
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ===== Main App =====
def main():
    # Load data
    with st.spinner("Loading loan data..."):
        loans, performance, risk, portfolio = load_data()
    
    if loans.empty:
        st.error("❌ No data found! Please load data first.")
        st.stop()
    
    # Load model
    model, imputer = load_model()
    model_available = model is not None
    
    # ===== Sidebar Filters =====
    st.sidebar.title("🔍 Filters")
    grades = ['All'] + sorted(loans['grade'].dropna().unique().tolist())
    selected_grades = st.sidebar.multiselect("Loan Grade", options=grades, default=['All'])
    if 'All' in selected_grades or not selected_grades:
        selected_grades = loans['grade'].dropna().unique().tolist()
    
    states = ['All'] + sorted(loans['state'].dropna().unique().tolist())
    selected_states = st.sidebar.multiselect("State", options=states, default=['All'])
    if 'All' in selected_states or not selected_states:
        selected_states = loans['state'].dropna().unique().tolist()
    
    # Apply filters (optional)
    filtered_loans = loans[
        (loans['grade'].isin(selected_grades)) &
        (loans['state'].isin(selected_states))
    ]
    
    # ===== KPI Metrics =====
    total_loans = len(filtered_loans)
    total_funded = filtered_loans['loan_amount'].sum()
    avg_loan = filtered_loans['loan_amount'].mean()
    avg_rate = filtered_loans['interest_rate'].mean()
    default_count = filtered_loans[filtered_loans['loan_status'].isin(['Charged Off', 'Late (31-120 days)'])].shape[0]
    default_rate = (default_count / total_loans * 100) if total_loans > 0 else 0
    
    st.markdown('<h1 style="text-align: center;">🏦 Loan Performance Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #94a3b8; margin-bottom: 2rem;">Real-time insights from your loan portfolio</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Loans</div>
            <div class="metric-value">{total_loans:,}</div>
            <div class="metric-sub">Active portfolio</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Funded</div>
            <div class="metric-value">${total_funded:,.0f}</div>
            <div class="metric-sub">All loans</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Loan Amount</div>
            <div class="metric-value">${avg_loan:,.0f}</div>
            <div class="metric-sub">Per loan</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Interest Rate</div>
            <div class="metric-value">{avg_rate:.2f}%</div>
            <div class="metric-sub">Portfolio average</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        risk_color = "critical" if default_rate > 15 else "high" if default_rate > 10 else "medium" if default_rate > 5 else "low"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Default Rate</div>
            <div class="metric-value" style="color: {'#ef4444' if default_rate > 15 else '#f97316' if default_rate > 10 else '#eab308' if default_rate > 5 else '#22c55e'}">{default_rate:.1f}%</div>
            <div class="metric-sub">{default_count:,} loans</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== Tabs =====
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Risk Analysis",
        "💼 Portfolio",
        "🤖 Predictions",
        "📋 Data Explorer"
    ])
    
    # ===== TAB 1: Overview =====
    with tab1:
        st.markdown('<div class="card"><h3>📊 Loan Overview</h3>', unsafe_allow_html=True)
        
        if not filtered_loans.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Loan Status Distribution**")
                status_counts = filtered_loans['loan_status'].value_counts().reset_index()
                status_counts.columns = ['status', 'count']
                fig = px.pie(status_counts, values='count', names='status', title="", color_discrete_sequence=px.colors.qualitative.Set3, hole=0.4)
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Loans by Grade**")
                grade_counts = filtered_loans['grade'].value_counts().sort_index().reset_index()
                grade_counts.columns = ['grade', 'count']
                fig = px.bar(grade_counts, x='grade', y='count', title="", color='count', color_continuous_scale='Blues')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', height=350)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for selected filters.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== TAB 2: Risk Analysis =====
    with tab2:
        st.markdown('<div class="card"><h3>📈 Risk Analysis</h3>', unsafe_allow_html=True)
        if not risk.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Default Rate by Grade**")
                fig = px.bar(risk, x='grade', y='default_rate', title="", color='default_rate', color_continuous_scale='RdYlGn_r', text='default_rate')
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', height=350)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("**Risk Category Distribution**")
                risk_counts = risk['risk_category'].value_counts().reset_index()
                risk_counts.columns = ['category', 'count']
                fig = px.pie(risk_counts, values='count', names='category', title="",
                             color='category',
                             color_discrete_map={'Low Risk': '#22c55e', 'Medium Risk': '#eab308', 'High Risk': '#f97316', 'Very High Risk': '#ef4444'})
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', height=350)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Risk analysis data not available. Run dbt first.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== TAB 3: Portfolio =====
    with tab3:
        st.markdown('<div class="card"><h3>💼 Portfolio Performance</h3>', unsafe_allow_html=True)
        if not portfolio.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Portfolio Summary**")
                st.dataframe(
                    portfolio.style.format({
                        'total_funded': '${:,.2f}',
                        'total_received': '${:,.2f}',
                        'total_interest_received': '${:,.2f}',
                        'total_losses': '${:,.2f}',
                        'overall_default_rate': '{:.2f}%',
                        'overall_paid_rate': '{:.2f}%',
                        'portfolio_recovery_rate': '{:.2f}%',
                        'avg_interest_rate': '{:.2f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            with col2:
                st.markdown("**Performance Metrics**")
                metrics = {
                    'Metric': ['Total Funded', 'Total Received', 'Total Interest', 'Total Losses',
                               'Recovery Rate', 'Default Rate', 'Paid Rate', 'Avg Interest Rate'],
                    'Value': [
                        f"${portfolio['total_funded'].iloc[0]:,.2f}",
                        f"${portfolio['total_received'].iloc[0]:,.2f}",
                        f"${portfolio['total_interest_received'].iloc[0]:,.2f}",
                        f"${portfolio['total_losses'].iloc[0]:,.2f}",
                        f"{portfolio['portfolio_recovery_rate'].iloc[0]:.2f}%",
                        f"{portfolio['overall_default_rate'].iloc[0]:.2f}%",
                        f"{portfolio['overall_paid_rate'].iloc[0]:.2f}%",
                        f"{portfolio['avg_interest_rate'].iloc[0]:.2f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)
        else:
            st.info("Portfolio data not available. Run dbt first.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== TAB 4: Predictions =====
    with tab4:
        st.markdown('<div class="card"><h3>🤖 Loan Default Predictor</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: #94a3b8;">Enter loan details to predict default probability</p>', unsafe_allow_html=True)
        
        if not model_available:
            st.warning("⚠️ Model not found. Please train the model first: `python models/train_model.py`")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<p style="color: #e0e7ff; font-weight: 500;">Loan Details</p>', unsafe_allow_html=True)
                loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=100000, value=15000, step=1000)
                interest_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.5, step=0.5)
                term = st.selectbox("Term (months)", [36, 60], index=0)
                installment = st.number_input("Monthly Installment ($)", min_value=50, max_value=5000, value=500, step=50)
                grade = st.selectbox("Grade", ['A','B','C','D','E','F','G'], index=2)
                sub_grade = st.selectbox("Sub-Grade", ['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5',
                                                       'C1','C2','C3','C4','C5','D1','D2','D3','D4','D5',
                                                       'E1','E2','E3','E4','E5','F1','F2','F3','F4','F5',
                                                       'G1','G2','G3','G4','G5'], index=10)
                loan_purpose = st.selectbox("Loan Purpose",
                                            ['debt_consolidation','credit_card','home_improvement','major_purchase',
                                             'small_business','medical','car','moving','vacation','wedding',
                                             'house','educational','renewable_energy'], index=0)
                application_type = st.selectbox("Application Type", ['individual', 'joint'], index=0)
            
            with col2:
                st.markdown('<p style="color: #e0e7ff; font-weight: 500;">Borrower Details</p>', unsafe_allow_html=True)
                annual_income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=65000, step=5000)
                debt_to_income = st.slider("Debt-to-Income Ratio (%)", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
                homeownership = st.selectbox("Homeownership", ['RENT', 'OWN', 'MORTGAGE'], index=0)
                state = st.selectbox("State", ['CA','TX','FL','NY','IL','PA','OH','GA','NC','MI','NJ','VA','WA','AZ','MA'], index=0)
                delinq_2y = st.number_input("Delinquencies in last 2 years", min_value=0, max_value=10, value=1, step=1)
                inquiries_last_12m = st.number_input("Credit inquiries in last 12 months", min_value=0, max_value=20, value=2, step=1)
                total_credit_lines = st.number_input("Total Credit Lines", min_value=1, max_value=50, value=10, step=1)
                open_credit_lines = st.number_input("Open Credit Lines", min_value=0, max_value=30, value=5, step=1)
                total_credit_limit = st.number_input("Total Credit Limit ($)", min_value=1000, max_value=200000, value=50000, step=5000)
                total_credit_utilized = st.number_input("Total Credit Utilized ($)", min_value=0, max_value=200000, value=15000, step=1000)
                num_collections_last_12m = st.number_input("Collections in last 12 months", min_value=0, max_value=10, value=0, step=1)
                months_since_90d_late = st.number_input("Months since 90-day late", min_value=0, max_value=999, value=999, step=1)
                accounts_opened_24m = st.number_input("Accounts opened in 24 months", min_value=0, max_value=20, value=3, step=1)
                num_accounts_30d_past_due = st.number_input("Accounts 30+ days past due", min_value=0, max_value=10, value=0, step=1)
                num_cc_carrying_balance = st.number_input("Credit cards carrying balance", min_value=0, max_value=10, value=2, step=1)
                tax_liens = st.number_input("Tax Liens", min_value=0, max_value=5, value=0, step=1)
            
            if st.button("🔮 Predict Default Risk", type="primary", use_container_width=True):
                # Prepare features
                features = {
                    'loan_amount': loan_amount,
                    'interest_rate': interest_rate,
                    'term': term,
                    'installment': installment,
                    'grade': grade,
                    'sub_grade': sub_grade,
                    'annual_income': annual_income,
                    'debt_to_income': debt_to_income,
                    'homeownership': homeownership,
                    'state': state,
                    'loan_purpose': loan_purpose,
                    'application_type': application_type,
                    'delinq_2y': delinq_2y,
                    'inquiries_last_12m': inquiries_last_12m,
                    'total_credit_lines': total_credit_lines,
                    'open_credit_lines': open_credit_lines,
                    'total_credit_limit': total_credit_limit,
                    'total_credit_utilized': total_credit_utilized,
                    'num_collections_last_12m': num_collections_last_12m,
                    'num_historical_failed_to_pay': 0,
                    'months_since_90d_late': months_since_90d_late,
                    'current_accounts_delinq': 0,
                    'accounts_opened_24m': accounts_opened_24m,
                    'num_satisfactory_accounts': 0,
                    'num_accounts_120d_past_due': 0,
                    'num_accounts_30d_past_due': num_accounts_30d_past_due,
                    'num_total_cc_accounts': 0,
                    'num_open_cc_accounts': 0,
                    'num_cc_carrying_balance': num_cc_carrying_balance,
                    'account_never_delinq_percent': 100.0,
                    'tax_liens': tax_liens,
                    'public_record_bankrupt': 0
                }
                try:
                    proba = predict_single(model, imputer, features)
                    if proba is not None:
                        risk_level, risk_class = get_risk_level(proba)
                        st.markdown("---")
                        col1, col2, col3 = st.columns([1,2,1])
                        with col2:
                            st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
                            st.markdown(f"""
                            <div class="metric-card" style="border-color: {'#22c55e' if proba < 0.10 else '#eab308' if proba < 0.25 else '#f97316' if proba < 0.50 else '#ef4444'};">
                                <div class="metric-label">Default Probability</div>
                                <div class="metric-value" style="color: {'#22c55e' if proba < 0.10 else '#eab308' if proba < 0.25 else '#f97316' if proba < 0.50 else '#ef4444'}">{proba * 100:.1f}%</div>
                                <div class="metric-sub"><span class="{risk_class}">{risk_level}</span></div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        if proba < 0.10:
                            st.success("✅ Low risk - Loan is very likely to be repaid")
                        elif proba < 0.25:
                            st.info("ℹ️ Medium risk - Monitor regularly")
                        elif proba < 0.50:
                            st.warning("⚠️ High risk - Consider higher interest rate or additional verification")
                        else:
                            st.error("🚨 Very High risk - Consider declining or requiring collateral")
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== TAB 5: Data Explorer =====
    with tab5:
        st.markdown('<div class="card"><h3>📋 Data Explorer</h3>', unsafe_allow_html=True)
        columns = st.multiselect(
            "Select columns to display",
            options=filtered_loans.columns.tolist(),
            default=['loan_amount', 'interest_rate', 'grade', 'loan_status', 'state', 'loan_purpose']
        )
        if columns:
            st.dataframe(filtered_loans[columns], use_container_width=True, height=400)
        st.markdown(f'<p style="color: #64748b;">Showing {len(filtered_loans):,} records</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== Footer =====
    st.markdown(f"""
    <div class="footer">
        <p>🏦 Financial Loan Performance Analytics • Built with Streamlit • {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()