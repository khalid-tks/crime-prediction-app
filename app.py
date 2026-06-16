import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# --- 1. Page Configuration (Polished Look) ---
st.set_page_config(page_title="Crime Analytics AI", layout="wide", initial_sidebar_state="collapsed")
st.title("🛡️ Bangladesh Crime Analytics & AI Prediction Portal")
st.markdown("A Data-Driven Decision Support System for Law Enforcement")

# --- 2. Data Loading & Cleaning ---
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('main_crime_data.csv')
    df['Year'] = df['Year'].astype(str).str.split('.').str[0].str.strip()
    df['Month'] = df['Month'].astype(str).str.strip()
    df['Date'] = pd.to_datetime(df['Year'] + ' ' + df['Month'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Specific Crime Categories
    crime_cols = ['Dacoity', 'Robbery', 'Murder', 'Woman & Child Repression', 'Kidnapping', 
                  'Burglary', 'Theft', 'Arms Act', 'Explosive', 'Narcotics', 'Smuggling']
    
    for col in crime_cols + ['Total Cases']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        
    return df, crime_cols

df, crime_cols = load_and_process_data()

# --- 3. UI Layout (Using Tabs for Cleanliness) ---
tab1, tab2 = st.tabs(["📊 1. Hotspots & Concerns (Overview)", "🤖 2. Dynamic AI Prediction (Forecast)"])

# ==========================================
# TAB 1: HOTSPOTS & CONCERNS
# ==========================================
with tab1:
    st.subheader("🚨 Current Crime Landscape & Vulnerability Analysis")
    
    # Calculating Top Concerns and Hotspots
    total_crimes_by_type = df[crime_cols].sum().sort_values(ascending=False)
    top_crime_concern = total_crimes_by_type.index[0]
    
    hotspots = df.groupby('Unit Name')['Total Cases'].sum().sort_values(ascending=False)
    top_hotspot = hotspots.index[0]
    
    # Clean KPI Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric(label="⚠️ Highest Concern (Crime Type)", value=top_crime_concern)
    col2.metric(label="📍 Most Vulnerable Hotspot", value=top_hotspot)
    col3.metric(label="📉 Total Cases Recorded", value=f"{df['Total Cases'].sum():,}")
    
    st.divider()
    
    # Clean Visuals using Streamlit's native beautiful charts
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**🔥 Top 5 Crime Hotspots (Where to deploy forces?)**")
        st.bar_chart(hotspots.head(5), color="#ff4b4b")
    
    with col_b:
        st.markdown(f"**📈 Crimes of Highest Concern (What to focus on?)**")
        st.bar_chart(total_crimes_by_type.head(5), color="#1f77b4")

# ==========================================
# TAB 2: DYNAMIC AI PREDICTION
# ==========================================
with tab2:
    st.subheader("🔮 Dynamic AI Forecasting Studio")
    st.markdown("Select a specific crime category and forecast range to generate live predictions.")
    
    # User Inputs (Clean side-by-side layout)
    col_sel, col_slide = st.columns(2)
    with col_sel:
        target_crime = st.selectbox("📌 Select Crime Type to Predict:", ['Total Cases'] + crime_cols)
    with col_slide:
        forecast_period = st.slider("📅 Select Forecast Range (Months):", min_value=3, max_value=36, value=12, step=1)
        
    # Prepare Data for selected crime
    monthly_data = df.groupby('Date')[target_crime].sum().reset_index()
    ml_data = monthly_data.rename(columns={'Date': 'ds', target_crime: 'y'})
    
    # AI Model Training
    m = Prophet(interval_width=0.95, yearly_seasonality=True)
    m.fit(ml_data)
    future = m.make_future_dataframe(periods=forecast_period, freq='MS')
    forecast = m.predict(future)
    
    # Clean, Minimalist Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ml_data['ds'], ml_data['y'], label='Historical Data', color='#1f77b4', linewidth=2)
    ax.plot(forecast['ds'].tail(forecast_period), forecast['yhat'].tail(forecast_period), label='AI Prediction', color='#ff4b4b', linestyle='--', linewidth=2)
    ax.fill_between(forecast['ds'].tail(forecast_period), forecast['yhat_lower'].tail(forecast_period), forecast['yhat_upper'].tail(forecast_period), color='#ff4b4b', alpha=0.15)
    
    # Removing clunky borders for a polished look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(f"Number of Cases")
    ax.legend()
    st.pyplot(fig)
    
    # Actionable Insight Text
    next_month_val = int(forecast['yhat'].iloc[-forecast_period])
    st.info(f"💡 **AI Insight:** Based on historical patterns, the model predicts approximately **{next_month_val:,}** cases of **{target_crime}** in the upcoming month. Special administrative focus is recommended.")
