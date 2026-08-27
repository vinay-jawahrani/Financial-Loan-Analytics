"""
risk_analysis.py - Risk Analysis Dashboard Tab
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
def load_risk_data():
    engine = get_engine()
    with engine.connect() as conn:
        loans = pd.read_sql("SELECT * FROM stg_loans", conn)
        try:
            risk = pd.read_sql("SELECT * FROM public_public.risk_analysis", conn)
        except:
            risk = pd.DataFrame()
    return loans, risk

def render():
    st.markdown('<div class="card"><h3>📈 Risk Analysis</h3>', unsafe_allow_html=True)
    
    loans, risk = load_risk_data()
    
    if loans.empty:
        st.warning("No loan data available. Please load data first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Default Rate by Grade")
        if not risk.empty:
            fig = px.bar(
                risk,
                x='grade',
                y='default_rate',
                title="",
                color='default_rate',
                color_continuous_scale='RdYlGn_r',
                text='default_rate'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8',
                height=400,
                xaxis=dict(gridcolor='#1e293b'),
                yaxis=dict(gridcolor='#1e293b')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Risk analysis data not available")
    
    with col2:
        st.markdown("#### Risk Category Distribution")
        if not risk.empty:
            risk_counts = risk['risk_category'].value_counts().reset_index()
            risk_counts.columns = ['category', 'count']
            fig = px.pie(
                risk_counts,
                values='count',
                names='category',
                title="",
                color='category',
                color_discrete_map={
                    'Low Risk': '#22c55e',
                    'Medium Risk': '#eab308',
                    'High Risk': '#f97316',
                    'Very High Risk': '#ef4444'
                }
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8',
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Risk category data not available")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### Average DTI by Grade")
        if not risk.empty:
            fig = px.bar(
                risk,
                x='grade',
                y='avg_dti',
                title="",
                color='avg_dti',
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8',
                height=400,
                xaxis=dict(gridcolor='#1e293b'),
                yaxis=dict(gridcolor='#1e293b')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("DTI data not available")
    
    with col4:
        st.markdown("#### Default Rate by Purpose")
        purpose_risk = loans.groupby('loan_purpose').apply(
            lambda x: (x['loan_status'].isin(['Charged Off', 'Late (31-120 days)']).sum() / len(x) * 100)
        ).reset_index()
        purpose_risk.columns = ['purpose', 'default_rate']
        purpose_risk = purpose_risk.sort_values('default_rate', ascending=False).head(10)
        
        fig = px.bar(
            purpose_risk,
            x='default_rate',
            y='purpose',
            title="",
            orientation='h',
            color='default_rate',
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            height=400,
            xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    render()