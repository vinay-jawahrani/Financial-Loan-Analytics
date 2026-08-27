"""
streamlit_app.py - Financial Loan Performance Analytics Dashboard
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
import os
import joblib
from dotenv import load_dotenv
from datetime import datetime

# Import page modules
from pages import overview, risk_analysis, portfolio

load_dotenv()

# ===== Page Config =====
st.set_page_config(
    page_title="Loan Performance Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== Custom CSS =====
st.markdown("""
<style>
    .stApp { background-color: #0a0e17; }
    .css-1d391kg { background-color: #111927; }
    h1, h2, h3 { color: #e0e7ff !important; }
    .metric-card {
        background: linear-gradient(145deg, #111927, #1a2332);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #2a3a5c;
        text-align: center;
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
    }
    .card {
        background: linear-gradient(145deg, #111927, #1a2332);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #2a3a5c;
        margin-bottom: 1rem;
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
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.8rem;
        padding: 2rem 0;
        border-top: 1px solid #1e293b;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ===== Database Connection =====
@st.cache_resource
def get_engine():
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/loan_analytics")
    return create_engine(db_url)

# ===== Load Data =====
@st.cache_data(ttl=300)
def load_data():
    engine = get_engine()
    with engine.connect() as conn:
        loans = pd.read_sql("SELECT * FROM stg_loans", conn)
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

# ===== Load Data =====
with st.spinner("Loading data..."):
    loans, performance, risk, portfolio = load_data()

if loans.empty:
    st.error("❌ No data found! Please load data first.")
    st.stop()

# ===== KPI Metrics =====
total_loans = len(loans)
total_funded = loans['loan_amount'].sum()
avg_loan = loans['loan_amount'].mean()
avg_rate = loans['interest_rate'].mean()
default_count = loans[loans['loan_status'].isin(['Charged Off', 'Late (31-120 days)'])].shape[0]
default_rate = (default_count / total_loans * 100) if total_loans > 0 else 0

st.markdown('<h1 style="text-align: center;">🏦 Loan Performance Analytics</h1>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Loans</div>
        <div class="metric-value">{total_loans:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Funded</div>
        <div class="metric-value">${total_funded:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Loan Amount</div>
        <div class="metric-value">${avg_loan:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Interest Rate</div>
        <div class="metric-value">{avg_rate:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    color = '#ef4444' if default_rate > 15 else '#f97316' if default_rate > 10 else '#eab308' if default_rate > 5 else '#22c55e'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Default Rate</div>
        <div class="metric-value" style="color: {color}">{default_rate:.1f}%</div>
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

with tab1:
    overview.render()

with tab2:
    risk_analysis.render()

with tab3:
    portfolio.render()

with tab4:
    st.info("🤖 Prediction tab - Coming soon!")

with tab5:
    st.markdown('<div class="card"><h3>📋 Data Explorer</h3>', unsafe_allow_html=True)
    columns = st.multiselect(
        "Select columns to display",
        options=loans.columns.tolist(),
        default=['loan_amount', 'interest_rate', 'grade', 'loan_status', 'state', 'loan_purpose']
    )
    if columns:
        st.dataframe(loans[columns], use_container_width=True, height=400)
    st.markdown(f'<p style="color: #64748b;">Showing {len(loans):,} records</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ===== Footer =====
st.markdown(f"""
<div class="footer">
    <p>🏦 Financial Loan Performance Analytics • Built with Streamlit • {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>
""", unsafe_allow_html=True)