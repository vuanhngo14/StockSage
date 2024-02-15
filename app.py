from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import TimeSeriesSplit
from keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import plotly.graph_objs as go
from plotly.subplots import make_subplots


# Package for additional features
from talib import abstract as ta
from talib import RSI
from talib import MACD
import finnhub # Extract economical news 
import json

def evaluate_model(X, y, model):
    tscv = TimeSeriesSplit(n_splits=10)
    mse_scores = []
    mae_scores = []
    r2_scores = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        mae_scores.append(mae)
        r2_scores.append(r2)

    return np.mean(mse_scores), np.mean(mae_scores), np.mean(r2_scores)

def news_sentiment(date, company_code):

    api_key = "cn0ah7pr01qkcvkfucv0cn0ah7pr01qkcvkfucvg"; 

    finnhub_client = finnhub.Client(api_key=api_key)

    # Get all the news of that day for the company
    data = finnhub_client.company_news(company_code, _from =date, to=date)

    return data


app = Flask(__name__)

# Load the pre-trained LSTM model
model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Get the input values
    ticker_symbol = request.form['ticker_symbol']
    end_date = request.form['end_date']

    start_date = "2010-01-01"
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    df = pd.DataFrame(data)

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

    # Common window size, which can balance.
    window = 14
    atr = true_range.rolling(window=window).mean()
    atr_df = pd.DataFrame({'ATR': atr.values}, index=df.index)

    # Merge the original DataFrame with the new ATR DataFrame
    df = pd.merge(df, atr_df, left_index=True, right_index=True)

    # Re-order the data frame
    new_order = ["Open", "High", "Low", "Volume", "RSI", 'MACD', 'Signal', 'MACD_Hist', "ATR", "Adj Close"]
    df = df[new_order]
    df.dropna(inplace=True)
    output_var = pd.DataFrame(df["Adj Close"])
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

    # ================================================================================================= # 
    # ###################################### DISPLAYING RESULT # ###################################### #
    # ================================================================================================= #

    # Create an interactive plot with Plotly
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Past Stock Prices'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[end_date], y=predictions[-1], mode='markers', marker=dict(color='red', size=8), name='Predicted Price'), row=1, col=1)
    fig.update_layout(title='Past Stock Prices and Predicted Price', xaxis_title='Date', yaxis_title='Stock Price', showlegend=True)

    # Save the plot as an HTML file
    plot_path = 'static/prediction_plot.html'  # Save the plot in the static directory
    fig.write_html(plot_path)

    # Extract the last predicted price
    predicted_price = predictions[-1][0]

    # Display the prediction and accuracy
    mse_avg, mae_avg, r2_avg = evaluate_model(X, df['Adj Close'].values, model)
    accuracy = r2_avg * 100

    # Retrive the model metadata 
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    model_version = metadata['version']
    model_date_modified = metadata['date_modified']



    # ================================================================================================= # 
    # ####################################### RENDER TO HTML # ######################################## #
    # ================================================================================================= #

    # Render to the template 
    return render_template('index.html',
                           prediction=f'Predicted price for {ticker_symbol} on {end_date}: {predicted_price:.2f}',
                           accuracy=f'Accuracy: {accuracy:.2f}',
                           plot_path = plot_path,
                           # headlines = headlines,
                            model_version = model_version,
                            model_date_modified = model_date_modified
                           )

if __name__ == '__main__':
    app.run(debug=True)


