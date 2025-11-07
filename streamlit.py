# ==========================================
# 📊 Streamlit App: Stock Price Movement Predictor (Final & Fixed)
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Stock Movement Predictor", layout="centered")

# -----------------------------------
# Sidebar Inputs
# -----------------------------------
st.sidebar.header("📈 Stock Configuration")
ticker = st.sidebar.text_input("Enter NSE Stock Symbol", "RELIANCE.NS")
period = st.sidebar.selectbox("Data Period", ["6mo", "1y", "2y"], index=0)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk"], index=0)
st.sidebar.markdown("---")

# -----------------------------------
# Fetch Data
# -----------------------------------
st.title("💹 Stock Price Movement Prediction using ML")

st.write(f"Fetching {ticker} data for last {period} ...")
data = yf.download(ticker, period=period, interval=interval)

if data.empty:
    st.error("Could not download data. Please check ticker symbol.")
    st.stop()

data.reset_index(inplace=True)

# 🩹 Handle MultiIndex or prefixed columns safely
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

data.columns = [str(c).split("_")[0].title().strip() for c in data.columns]

# 🧩 Normalize expected column names
rename_map = {
    "Adjclose": "Adj_Close",
    "Adj Close": "Adj_Close",
    "Closeprice": "Close",
}
data.rename(columns=rename_map, inplace=True)

# ✅ Check if Close column exists
close_col = None
for col in data.columns:
    if "Close" in col:
        close_col = col
        break

if not close_col:
    st.error("❌ Could not find 'Close' column in the data. Check the ticker.")
    st.stop()

# -----------------------------------
# Feature Engineering
# -----------------------------------
data["Rsi"] = ta.momentum.RSIIndicator(data[close_col], 14).rsi()
data["Sma20"] = data[close_col].rolling(20).mean()
data["Sma50"] = data[close_col].rolling(50).mean()

macd = ta.trend.MACD(data[close_col])
data["Macd"] = macd.macd()
data["Macd_Signal"] = macd.macd_signal()

data["Return"] = data[close_col].pct_change()
data["Lag1"] = data["Return"].shift(1)
data.dropna(inplace=True)

# Target variable
data["Target"] = (data[close_col].shift(-1) > data[close_col]).astype(int)

# -----------------------------------
# Prepare features
# -----------------------------------
features = ["Rsi", "Sma20", "Sma50", "Macd", "Macd_Signal", "Lag1"]
X = data[features]
y = data["Target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------
# Train Models (fast)
# -----------------------------------
logreg = LogisticRegression(max_iter=500)
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

split = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

logreg.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# -----------------------------------
# Predict next day
# -----------------------------------
last_row = X_scaled[-1].reshape(1, -1)
pred_logreg = logreg.predict(last_row)[0]
pred_xgb = xgb.predict(last_row)[0]
proba_xgb = xgb.predict_proba(last_row)[0][1]

# -----------------------------------
# Display Results
# -----------------------------------
st.subheader(f"📅 Latest Data till: {data['Date'].iloc[-1].date()}")
st.line_chart(data.set_index("Date")[close_col], height=300)

col1, col2 = st.columns(2)
with col1:
    st.metric("Logistic Regression Prediction",
              "📈 UP" if pred_logreg == 1 else "📉 DOWN")

with col2:
    st.metric("XGBoost Prediction",
              "📈 UP" if pred_xgb == 1 else "📉 DOWN",
              f"Confidence: {proba_xgb*100:.1f}%")

# -----------------------------------
# Show Data
# -----------------------------------
st.markdown("---")
st.write("### Recent Data Snapshot")
st.dataframe(data.tail(10))

# -----------------------------------
# About
# -----------------------------------
st.markdown("""
---
👨‍💻 **Developed for GoalFi – IIT Kanpur ML Intern Assignment**  
Uses: Logistic Regression + XGBoost | Technical Indicators (RSI, SMA, MACD)
""")
