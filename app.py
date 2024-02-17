from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

    start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=2)).strftime('%Y-%m-%d')


    # Get all the news of that day for the company
    data = finnhub_client.company_news(company_code, _from = start_date, to=date)
    return data

# Function to prepare data for prediction
def prepare_data(ticker_symbol, end_date, prediction_days=70):
    start_date = end_date - dt.timedelta(days=prediction_days)
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    
    # Define and fit scaler on training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[['Open', 'High', 'Low', 'Volume', 'Close']].values)
    
    # Scale the data
    scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Volume', 'Close']].values)
    
    x_predict = scaled_data[-prediction_days:, :]
    x_predict = np.reshape(x_predict, (1, x_predict.shape[0], x_predict.shape[1]))
    return x_predict, scaler

app = Flask(__name__)

# Load the pre-trained LSTM model
model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        ticker_symbol = request.form['ticker_symbol']
        end_date = datetime.strptime(request.form['end_date'], '%Y-%m-%d')
        
        # Prepare data
        x_predict, scaler = prepare_data(ticker_symbol, end_date)
        
        # Make prediction
        prediction = model.predict(x_predict)
        
        # Inverse transform the prediction to get the actual value
        prediction = scaler.inverse_transform(np.array([[0, 0, 0, 0, prediction[0, 0]]]))
        predicted_price = prediction[0, -1]
        
        return render_template('index.html', prediction=predicted_price)



    # ================================================================================================= # 
    # ####################################### DISPLAYING NEWS # ####################################### #
    # ================================================================================================= #

    news_data = news_sentiment(end_date, ticker_symbol)
    print("ELEMENTS IN THE NEWS DATA: ", len(news_data))

    news_info = []
    for news_item in news_data:

        timestamp = news_item['datetime']
        news_date = datetime.utcfromtimestamp(timestamp)

        news_info.append({
            'datetime': news_date.strftime('%Y-%m-%d %H:%M:%S'),  
            'headline': news_item['headline'],
            'image': news_item['image'],
            'url': news_item['url']
        })

    # ================================================================================================= # 
    # ####################################### RENDER TO HTML # ######################################## #
    # ================================================================================================= #

    # Render to the template 
    return render_template('index.html',
                           prediction=f'Predicted price for {ticker_symbol} on {end_date}: {predicted_price:.2f}',
                           accuracy=f'Accuracy: {accuracy:.2f}',
                           # plot_path = plot_path,
                           model_version = model_version,
                           model_date_modified = model_date_modified,
                           news_info = news_info
                           )

if __name__ == '__main__':
    app.run(debug=True)


