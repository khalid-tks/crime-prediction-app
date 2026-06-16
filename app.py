import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ১. ওয়েবসাইটের হেডার এবং কনফিগারেশন
st.set_page_config(page_title="Crime Analytics Pipeline", layout="wide")
st.title("🚨 Bangladesh Crime Data & AI Prediction Portal")
st.markdown("An End-to-End Machine Learning Pipeline: From Raw Data to Future Forecasts.")

# ২. মেইন ডেটা লোড করা
# ২. মেইন ডেটা লোড করা
# ২. মেইন ডেটা লোড করা
# ২. মেইন ডেটা লোড করা
@st.cache_data
def load_and_process_data():
    # মূল ফাইলটি রিড করা
    df = pd.read_csv('main_crime_data.csv')
    
    # Year কলামের .0 সমস্যা দূর করা (যেমন 2020.0 থেকে শুধু 2020 বানানো)
    df['Year'] = df['Year'].astype(str).str.split('.').str[0].str.strip()
    
    # Month কলামের অদৃশ্য স্পেস ক্লিন করা
    df['Month'] = df['Month'].astype(str).str.strip()
    
    # ডেট তৈরি করা
    df['Date'] = pd.to_datetime(df['Year'] + ' ' + df['Month'], errors='coerce')
    
    # যেসব রো-তে আসল ডেট নেই, সেগুলো স্কিপ করা
    df = df.dropna(subset=['Date'])
    
    # Total Cases কলামটি নাম্বার (Numeric) হিসেবে নিশ্চিত করা
    df['Total Cases'] = pd.to_numeric(df['Total Cases'].astype(str).str.replace(',', ''), errors='coerce')
    df = df.dropna(subset=['Total Cases'])
    
    # সব ডিপার্টমেন্টের ডেটা মাস অনুযায়ী যোগ করা
    monthly_data = df.groupby('Date')['Total Cases'].sum().reset_index()
    return df, monthly_data

# ডেটা ফাংশন কল করা
raw_df, monthly_df = load_and_process_data()

# ৩. সেকশন ১: হিস্টোরিকাল ডেটা (Historical Overview)
st.divider()
st.header("📊 1. Historical Data Overview (2020 - 2025)")

col1, col2 = st.columns(2)
with col1:
    st.write("Preview of Raw National Dataset:")
    st.dataframe(raw_df.head(10)) # মূল ডেটার প্রথম ১০ লাইন দেখাবে
with col2:
    st.write("Aggregated Monthly Trend:")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(monthly_df['Date'], monthly_df['Total Cases'], marker='o', color='blue')
    ax1.set_title("Total Crimes Per Month")
    st.pyplot(fig1)

# ৪. সেকশন ২: মেশিন লার্নিং প্রেডিকশন (Advanced AI Forecast)
st.divider()
st.header("🤖 2. Advanced AI Prediction Analysis")
st.markdown("Interactive forecasting powered by Meta's Prophet algorithm. Adjust the slider to forecast further into the future.")

# ডায়নামিক স্লাইডার (কত মাসের প্রেডিকশন চান)
forecast_period = st.slider("Select Forecast Period (Months):", min_value=3, max_value=36, value=12, step=1)

# Prophet এর জন্য ডেটা রেডি করা
ml_data = monthly_df.rename(columns={'Date': 'ds', 'Total Cases': 'y'})

# মডেল ট্রেইন ও প্রেডিক্ট করা
m = Prophet(interval_width=0.95, yearly_seasonality=True)
m.fit(ml_data)
future = m.make_future_dataframe(periods=forecast_period, freq='MS')
forecast = m.predict(future)

# --- নতুন: KPI মেট্রিক্স (Smart Insights) ---
st.subheader("💡 Key Future Insights")
col1, col2, col3 = st.columns(3)

# আগামী মাসের প্রেডিকশন বের করা
next_month_pred = int(forecast['yhat'].iloc[-forecast_period])
# ফোরকাস্টের মধ্যে সর্বোচ্চ সংখ্যা
max_pred = int(forecast['yhat'].tail(forecast_period).max())

col1.metric(label="Next Month's Expected Cases", value=f"{next_month_pred:,}")
col2.metric(label="Peak Expected Cases (Within Forecast)", value=f"{max_pred:,}")
col3.metric(label="Model Confidence Interval", value="95%")

# --- প্রেডিকশন গ্রাফ আঁকা ---
st.subheader("📈 Interactive Future Trend")
fig2, ax2 = plt.subplots(figsize=(12, 5))
# হিস্টোরিকাল লাইন
ax2.plot(ml_data['ds'], ml_data['y'], label='Historical Data', color='blue', marker='o')
# প্রেডিকশন লাইন
ax2.plot(forecast['ds'].tail(forecast_period), forecast['yhat'].tail(forecast_period), label='AI Forecast', color='red', linestyle='--', marker='o')
ax2.fill_between(forecast['ds'].tail(forecast_period), forecast['yhat_lower'].tail(forecast_period), forecast['yhat_upper'].tail(forecast_period), color='red', alpha=0.2, label='Confidence Interval')

ax2.set_xlabel("Year & Month")
ax2.set_ylabel("Total Crime Cases")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig2)

# --- নতুন: AI Components (Trend & Seasonality) ---
st.subheader("⚙️ Model Decomposition (Why does the AI predict this?)")
st.write("This section breaks down the AI's prediction into overall growth trends and yearly seasonal patterns.")
with st.expander("Click to view AI Trend & Seasonality Analysis"):
    fig3 = m.plot_components(forecast)
    st.pyplot(fig3)

# --- নতুন: ডেটা ডাউনলোড অপশন ---
st.subheader("📋 Forecasted Dataset Export")
with st.expander("View & Download Future Data"):
    forecast_export = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period)
    forecast_export = forecast_export.rename(columns={'ds': 'Date', 'yhat': 'Predicted Cases', 'yhat_lower': 'Minimum Range', 'yhat_upper': 'Maximum Range'})
    
    # ডেট ফরম্যাট সুন্দর করা
    forecast_export['Date'] = forecast_export['Date'].dt.strftime('%B %Y')
    # দশমিক সংখ্যা রাউন্ড (Round) করা
    forecast_export.iloc[:, 1:] = forecast_export.iloc[:, 1:].round(0).astype(int)
    
    st.dataframe(forecast_export, use_container_width=True)
    
    csv = forecast_export.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Predicted Data as CSV", data=csv, file_name='AI_Crime_Forecast.csv', mime='text/csv')

st.success("✅ Defense-Ready AI Pipeline Executed Successfully!")
