# 📈 Stock Price Prediction with LSTM

This project is a **web-based dashboard** built using **Streamlit**, designed to predict Indonesian stock prices using a trained **LSTM (Long Short-Term Memory) model**.

The application fetches stock data directly from **Yahoo Finance (yfinance)**, processes it with a pre-trained model, and visualizes both **historical data** and **future price predictions** in an interactive dashboard.

---

## 🚀 Features

* 📊 **Historical Data Viewer** – Display the latest stock data from Yahoo Finance.
* 🔮 **Stock Price Prediction** – Predict future stock closing prices for up to 60 days.
* 📉 **Interactive Chart** – Compare real vs predicted stock prices in a single chart.
* 🖥 **Monitoring Dashboard** – KPI metrics, charts, and tables in one view.
* 💾 **Download Option** – Export prediction results as CSV.

---

## 🛠 Tech Stack

* **Python 3.10+**
* **Streamlit** – UI framework
* **TensorFlow / Keras** – Deep learning model (LSTM)
* **scikit-learn** – Data preprocessing & scaling
* **yfinance** – Stock market data API
* **matplotlib** – Data visualization

---

## 📦 Installation & Run

1. Clone this repository:

   ```bash
   git clone https://github.com/fitrawildania/stock-price-prediction.git
   cd stock-price-prediction
   ```
2. Create virtual environment & install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:

   ```bash
   streamlit run app.py
   ```
4. Open in browser: `http://localhost:8501`

---

## 📂 Project Structure

```
├── app.py                # Main Streamlit application
├── lstm_model.keras      # Trained LSTM model
├── scaler.pkl            # Scaler used for preprocessing
├── requirements.txt      # Python dependencies
└── README.md             # Project description
```
---

## 🌍 Deployment

This app deployed on **Streamlit Community Cloud** 

---

## ✨ Author

Developed by Fitra Wildania 💻
