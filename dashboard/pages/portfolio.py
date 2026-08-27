"""
portfolio.py - Portfolio Performance Dashboard Tab
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    db_url = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_rjGsBo7hdqz9@ep-royal-credit-azy6sjo9-pooler.c-3.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require")
    return create_engine(db_url)

@st.cache_data(ttl=300)
def load_portfolio_data():
    engine = get_engine()
    with engine.connect() as conn:
        loans = pd.read_sql("SELECT * FROM stg_loans", conn)
        try:
            portfolio = pd.read_sql("SELECT * FROM public_public.portfolio_summary", conn)
        except:
            portfolio = pd.DataFrame()
    return loans, portfolio

def render():
    st.markdown('<div class="card"><h3>💼 Portfolio Performance</h3>', unsafe_allow_html=True)
    
    loans, portfolio = load_portfolio_data()
    
    if loans.empty:
        st.warning("No loan data available. Please load data first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Portfolio Summary")
        if not portfolio.empty:
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
        else:
            st.info("Portfolio data not available")
    
    with col2:
        st.markdown("#### Performance Metrics")
        if not portfolio.empty:
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
            metrics_df = pd.DataFrame(metrics)
            st.dataframe(
                metrics_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Performance metrics not available")
    
    st.markdown("#### Recovery Rate by Grade")
    try:
        recovery_by_grade = loans.groupby('grade').apply(
            lambda x: (x['paid_total'].sum() / x['loan_amount'].sum() * 100) if x['loan_amount'].sum() > 0 else 0
        ).reset_index()
        recovery_by_grade.columns = ['grade', 'recovery_rate']
        recovery_by_grade = recovery_by_grade.sort_values('grade')
        
        fig = px.bar(
            recovery_by_grade,
            x='grade',
            y='recovery_rate',
            title="",
            color='recovery_rate',
            color_continuous_scale='Greens'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            height=350,
            xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b')
        )
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Recovery rate data not available")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    render()