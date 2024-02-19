from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

from sklearn.metrics import mean_absolute_error, mean_squared_error
import json


app = Flask(__name__)

# Load the trained LSTM model
model = load_model('final_model.h5')

# Load the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(yf.download('AAPL', start='2010-01-01', end='2022-01-01')[['Open', 'High', 'Low', 'Volume', 'Close']].values)

with open('model_metadata.json', 'r') as file:
    metadata = json.load(file)

# Route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and display prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    ticker_symbol = request.form['ticker_symbol']
    end_date = request.form['end_date']

    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    end_date_predict = end_date - timedelta(days=1)

    # Get historical data
    data = yf.download(ticker_symbol, start='2010-01-01', end=end_date_predict)
    scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Volume', 'Close']].values)
    
    # Prepare input for prediction
    prediction_days = 70
    x_input = scaled_data[-prediction_days:].reshape(1, prediction_days, 5)
    
    # Make prediction
    prediction = model.predict(x_input)
    
    # Inverse transform the prediction
    predicted_price = scaler.inverse_transform(np.concatenate((scaled_data[-1,:-1], prediction), axis=None).reshape(1, 5))[-1,-1]
    formatted_predicted_price = "${:,.2f}".format(predicted_price)

    # ================================================ #
    # Evaluate the model
    # ================================================ # 

    # Display the model version from json file 
    # Evaluate the model
    actual_price = data.iloc[-1]['Close']
    mae = mean_absolute_error([actual_price], [predicted_price])
    mse = mean_squared_error([actual_price], [predicted_price])
    rmse = np.sqrt(mse)

    # ================================================ #
    # Add the plot 
    # ================================================ #

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Past Stock Prices'))
    fig.add_trace(go.Scatter(x=[end_date], y=[predicted_price], mode='markers', marker=dict(color='red', size=8), name='Predicted Price'))
    fig.update_layout(title='Past Stock Prices and Predicted Price', xaxis_title='Date', yaxis_title='Stock Price', showlegend=True)

    # Save the plot as an HTML file
    plot_path = 'static/prediction_plot.html'  # Save the plot in the static directory
    fig.write_html(plot_path)

    return render_template('index.html', 
                           ticker_symbol=ticker_symbol, 
                           end_date=end_date, 
                           predicted_price=formatted_predicted_price,
                           plot_path = plot_path,
                           mae = mae,
                           mse = mse,
                           rmse = rmse,
                           meta_data = metadata)

if __name__ == '__main__':
    app.run(debug=True)
