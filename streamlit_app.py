# streamlit_app.py

import streamlit as st
import yfinance as yf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests
from textblob import TextBlob
from datetime import datetime, timedelta

# Set your News API key
NEWS_API_KEY = "d4257bcf169c4556978178d542003740"

# Function to fetch historical commodity data
def fetch_data(ticker, period='2y'):
    data = yf.download(ticker, period=period)
    return data[['Close']].dropna()

# Compute sentiment score from news
def compute_sentiment(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&pageSize=10&sortBy=publishedAt"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    sentiments = []
    for article in articles:
        content = f"{article.get('title', '')} {article.get('description', '')}"
        if content:
            polarity = TextBlob(content).sentiment.polarity
            sentiments.append(polarity)
    return np.mean(sentiments) if sentiments else 0.0

# Prepare data
def prepare_lstm_data(series, window_size=60):
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series)
    X, y = [], []
    for i in range(window_size, len(scaled_series)):
        X.append(scaled_series[i-window_size:i, 0])
        y.append(scaled_series[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Build and train LSTM model
def train_lstm_model(X, y):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model

# Predict future prices
def predict_future(model, recent_data, steps, scaler):
    predictions = []
    input_seq = recent_data[-60:].reshape(1, 60, 1)
    for _ in range(steps):
        pred = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Streamlit UI
def main():
    st.title("📈 Commodity Price Forecasting with LSTM")
    st.markdown("Predict future commodity prices using LSTM models and news sentiment analysis.")

    commodity = st.selectbox("Select Commodity", ['Crude Oil', 'Gold', 'Natural Gas', 'Brent Oil'])
    days_ahead = st.slider("Days Ahead", 1, 30, 5)

    if st.button("Predict"):
        with st.spinner("Fetching data and predicting..."):
            ticker_map = {
                'Crude Oil': 'CL=F',
                'Gold': 'GC=F',
                'Natural Gas': 'NG=F',
                'Brent Oil': 'BZ=F'
            }
            ticker = ticker_map.get(commodity, 'CL=F')
            data = fetch_data(ticker)
            sentiment = compute_sentiment(commodity)
            X, y, scaler = prepare_lstm_data(data.values)
            model = train_lstm_model(X, y)
            predictions = predict_future(model, X[-1], days_ahead, scaler)

            dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days_ahead)
            forecast_df = pd.DataFrame({'Date': dates, 'Predicted Price': predictions})

            # Output results
            st.subheader("📊 Forecast Results")
            st.write(f"**Sentiment Score:** {sentiment:.2f}")
            st.dataframe(forecast_df)

            # Plotting
            st.subheader("📉 Price Forecast Chart")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data.index[-60:], data['Close'].values[-60:], label='Historical Prices')
            ax.plot(forecast_df['Date'], forecast_df['Predicted Price'], label='Predicted Prices')
            ax.set_title(f'{commodity} Price Forecast')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
