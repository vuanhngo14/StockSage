from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import TimeSeriesSplit
from keras.models import load_model


# Package for additional features
from talib import abstract as ta
from talib import RSI
from talib import MACD

app = Flask(__name__)

# Load the pre-trained LSTM model
model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker_symbol = request.form['ticker_symbol']
    end_date = request.form['end_date']

    # Fetch historical data
    start_date = "2010-01-01"
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    df = pd.DataFrame(data)

    # Feature engineering and preprocessing
    # Relative Strength Index (RSI): price movement over a given period.
    df['RSI'] = RSI(df['Close'], timeperiod=14)

    # Moving Average Convergence Divergence (MACD): Identifies trend strength and potential turning points.
    macd, signal, hist = MACD(df['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['Signal'] = signal
    df['MACD_Hist'] = hist

    # Average True Range (ATR): Measures market volatility.
    # Higher ATR values indicate higher market volatility, meaning prices are fluctuating more significantly.
    # Lower ATR values indicate lower volatility, suggesting a calmer market with smaller price swings.
    # You can use ATR to set stop-loss orders, manage risk, and identify potential trading opportunities based on volatility changes.

    close_prices = df["Adj Close"]
    high_prices = df["High"]
    low_prices = df["Low"]

    true_range = pd.Series(
        [max(hi - lo, abs(hi - close_prev), abs(lo - close_prev))
        for hi, lo, close_prev in zip(high_prices, low_prices, close_prices.shift(1))]
    )

    # Common window size, which can balance.
    window = 14
    atr = true_range.rolling(window=window).mean()

    atr_df = pd.DataFrame({'ATR': atr.values}, index=df.index)

    # Merge the original DataFrame with the new ATR DataFrame
    df = pd.merge(df, atr_df, left_index=True, right_index=True)

    # Re-order the data frame
    new_order = ["Open", "High", "Low", "Volume", "RSI", 'MACD', 'Signal', 'MACD_Hist', "ATR", "Adj Close"]
    df = df[new_order]

    # Drop null values
    df.dropna(inplace=True)

    # Set Target Variable
    output_var = pd.DataFrame(df["Adj Close"])

    # Selecting the Features
    features = ["Open", "High", "Low", "Volume", "RSI", 'MACD', 'Signal', 'MACD_Hist', "ATR"]

    # Scaling
    scaler = MinMaxScaler()
    feature_transform = scaler.fit_transform(df[features])
    feature_transform = pd.DataFrame(data=feature_transform, columns=features, index=df.index)

    # Selecting relevant features and scaling
    features = ["Open", "High", "Low", "Volume", "RSI", 'MACD', 'Signal', 'MACD_Hist', "ATR"]
    scaler = MinMaxScaler()
    feature_transform = scaler.fit_transform(df[features])

    # Reshape data for LSTM input
    X = np.array(feature_transform)
    X = X.reshape(X.shape[0], 1, X.shape[1])

    # Make predictions
    predictions = model.predict(X)

    # Extract the last predicted price
    predicted_price = predictions[-1][0]

    return render_template('index.html', prediction=f'Predicted price for {ticker_symbol} on {end_date}: {predicted_price:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
