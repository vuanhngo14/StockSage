import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Package for additional features
from talib import abstract as ta
from talib import RSI
from talib import MACD

# Define the ticker symbol for the company (e.g., Apple Inc. with the symbol AAPL)
ticker_symbol = "AAPL"
start_date = "2010-01-01"
end_date = "2023-02-06"

# Download historical stock data
data = yf.download(ticker_symbol, start=start_date, end=end_date)
df = pd.DataFrame(data)

# Load the trained LSTM model
model = load_model('model.h5')

# Feature engineering
df['RSI'] = RSI(df['Close'], timeperiod=14)
macd, signal, hist = MACD(df['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['MACD'] = macd
df['Signal'] = signal
df['MACD_Hist'] = hist
close_prices = df["Adj Close"]
high_prices = df["High"]
low_prices = df["Low"]
true_range = pd.Series(
    [max(hi - lo, abs(hi - close_prev), abs(lo - close_prev))
     for hi, lo, close_prev in zip(high_prices, low_prices, close_prices.shift(1))]
)
window = 14
atr = true_range.rolling(window=window).mean()
atr_df = pd.DataFrame({'ATR': atr.values}, index=df.index)
df = pd.merge(df, atr_df, left_index=True, right_index=True)
new_order = ["Open", "High", "Low", "Volume", "RSI", 'MACD', 'Signal', 'MACD_Hist', "ATR", "Adj Close"]
df = df[new_order]
df.dropna(inplace=True)

# Selecting features
features = ["Open", "High", "Low", "Volume", "RSI", 'MACD', 'Signal', 'MACD_Hist', "ATR"]
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df[features])
feature_transform = pd.DataFrame(data=feature_transform, columns=features, index=df.index)
feature_transform.head()

# Prepare data for prediction
last_data_point = feature_transform.iloc[-1].values.reshape(1, 1, len(features))

# Make prediction
predicted_price = model.predict(last_data_point)

print("Predicted Price on 2024-02-09:", predicted_price[0][0])
