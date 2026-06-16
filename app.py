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

# ৪. সেকশন ২: মেশিন লার্নিং প্রেডিকশন (Live AI Forecast)
st.divider()
st.header("🤖 2. Live Machine Learning Forecast (Next 12 Months)")
st.write("Our integrated Meta Prophet model is analyzing the historical patterns to predict future trends...")

# Prophet এর জন্য ডেটা রেডি করা
ml_data = monthly_df.rename(columns={'Date': 'ds', 'Total Cases': 'y'})

# মডেল ট্রেইন ও প্রেডিক্ট করা
m = Prophet(interval_width=0.95, yearly_seasonality=True)
m.fit(ml_data)
future = m.make_future_dataframe(periods=12, freq='MS')
forecast = m.predict(future)

# প্রেডিকশন গ্রাফ আঁকা
fig2, ax2 = plt.subplots(figsize=(12, 5))
# হিস্টোরিকাল লাইন
ax2.plot(ml_data['ds'], ml_data['y'], label='Historical Data', color='blue', marker='o')
# প্রেডিকশন লাইন
ax2.plot(forecast['ds'].tail(12), forecast['yhat'].tail(12), label='AI Forecast', color='red', linestyle='--', marker='o')
ax2.fill_between(forecast['ds'].tail(12), forecast['yhat_lower'].tail(12), forecast['yhat_upper'].tail(12), color='red', alpha=0.2, label='Confidence Interval')

ax2.set_title("Historical vs Predicted Crime Trends")
ax2.set_xlabel("Year")
ax2.set_ylabel("Total Cases")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

st.success("✅ End-to-End Pipeline Executed Successfully!")
