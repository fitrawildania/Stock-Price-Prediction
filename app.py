import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import load_model
import pickle

# --- Page config full width ---
st.set_page_config(page_title="📊 Stock Monitoring Dashboard", layout="wide")

# --- Load Model dan Scaler ---
model = load_model("lstm_model.keras")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --- Title ---
st.title("📊 Stock Price Monitoring Dashboard")
st.markdown("Real-time stock prediction with **LSTM Model**")

# --- Input Section ---
col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Enter stock code:", "BBCA.JK")
with col2:
    period = st.selectbox("Choose data period:", ["1y", "2y", "5y", "10y", "max"], index=2)
with col3:
    future_days = st.slider("Prediction days:", 7, 60, 30)

if ticker:
    data = yf.download(ticker, period=period)
    data = data.reset_index()

    if data.empty:
        st.error("❌ Data not found. Try another stock code.")
    else:
        st.success(f"{ticker} data successfully retrieved!")

        # --- Data Processing ---
        data = data.rename(columns={
            "Close": "close",
            "High": "high",
            "Low": "low",
            "Open": "open",
            "Volume": "volume"
        })
        features = ["close", "high", "low", "open", "volume"]
        dataset = data[features].values

        scaled_data = scaler.transform(dataset)
        time_steps = 60
        last_sequence = scaled_data[-time_steps:]
        current_sequence = last_sequence.copy()
        future_predictions = []

        for _ in range(future_days):
            pred_scaled = model.predict(
                current_sequence.reshape(1, time_steps, len(features)), verbose=0
            )
            pred_scaled_value = pred_scaled[0][0]
            future_predictions.append(pred_scaled_value)

            new_row = np.zeros((1, len(features)))
            new_row[0, 0] = pred_scaled_value
            new_row[0, 1:] = current_sequence[-1, 1:]
            current_sequence = np.vstack((current_sequence[1:], new_row))

        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions_rescaled = scaler.inverse_transform(
            np.concatenate(
                (future_predictions, np.zeros((future_days, len(features) - 1))),
                axis=1
            )
        )[:, 0]

        future_dates = pd.date_range(start=data["Date"].iloc[-1] + pd.Timedelta(days=1),
                                     periods=future_days)
        pred_df = pd.DataFrame({"date": future_dates, "predicted_close": future_predictions_rescaled})

        # --- Dashboard Layout ---
        latest_close = float(data["close"].iloc[-1])
        max_future = float(pred_df["predicted_close"].max())
        min_future = float(pred_df["predicted_close"].min())
        avg_future = float(pred_df["predicted_close"].mean())

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Latest Close", f"Rp {latest_close:,.0f}")
        kpi2.metric("Highest Prediction", f"Rp {max_future:,.0f}")
        kpi3.metric("Lowest Prediction", f"Rp {min_future:,.0f}")
        kpi4.metric("Average Prediction", f"Rp {avg_future:,.0f}")

        # --- Chart full width ---
        st.subheader("📈 Price Chart")
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(data["Date"], data["close"], label="Real Data", linewidth=2)
        ax.plot(pred_df["date"], pred_df["predicted_close"], label="Prediction", linestyle="-", linewidth=2)
        ax.set_title(f"Stock Predictions {ticker}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (IDR)")
        ax.legend()
        st.pyplot(fig)

        # --- Tables side by side ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Historical Data")
            st.dataframe(data.tail(30), height=250)

        with col2:
            st.subheader("🔮 Prediction Table")
            st.dataframe(pred_df, height=250)

        # --- Download button ---
        st.download_button("💾 Download Prediction Results",
                           pred_df.to_csv(index=False).encode("utf-8"),
                           "Stock_Prediction.csv",
                           "text/csv")
