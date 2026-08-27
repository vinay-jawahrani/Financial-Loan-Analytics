"""
overview.py - Overview Dashboard Tab
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/loan_analytics")
    return create_engine(db_url)

@st.cache_data(ttl=300)
def load_overview_data():
    engine = get_engine()
    with engine.connect() as conn:
        loans = pd.read_sql("SELECT * FROM stg_loans", conn)
        try:
            performance = pd.read_sql("SELECT * FROM public_public.loan_performance", conn)
        except:
            performance = pd.DataFrame()
    return loans, performance

def render():
    st.markdown('<div class="card"><h3>📊 Loan Overview</h3>', unsafe_allow_html=True)
    
    loans, performance = load_overview_data()
    
    if loans.empty:
        st.warning("No loan data available. Please load data first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Loan Status Distribution")
        status_counts = loans['loan_status'].value_counts().reset_index()
        status_counts.columns = ['status', 'count']
        fig = px.pie(
            status_counts,
            values='count',
            names='status',
            title="",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Loans by Grade")
        grade_counts = loans['grade'].value_counts().sort_index().reset_index()
        grade_counts.columns = ['grade', 'count']
        fig = px.bar(
            grade_counts,
            x='grade',
            y='count',
            title="",
            color='count',
            color_continuous_scale='Blues'
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
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### Top Loan Purposes")
        purpose_counts = loans['loan_purpose'].value_counts().head(10).reset_index()
        purpose_counts.columns = ['purpose', 'count']
        fig = px.bar(
            purpose_counts,
            x='count',
            y='purpose',
            title="",
            orientation='h',
            color='count',
            color_continuous_scale='Viridis'
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
    
    with col4:
        st.markdown("#### Homeownership Distribution")
        home_counts = loans['homeownership'].value_counts().reset_index()
        home_counts.columns = ['homeownership', 'count']
        fig = px.pie(
            home_counts,
            values='count',
            names='homeownership',
            title="",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    render()