from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

from sklearn.metrics import mean_absolute_error, mean_squared_error


app = Flask(__name__)



start_date = datetime(2010, 1, 1)
end_date = datetime(2022, 1, 1)

def general_scaler():
    # AAPL, GGLE and NVDA
    df1 = yf.download('AAPL', start=start_date, end=end_date)
    df2 = yf.download('NFLX', start=start_date, end=end_date)
    df3 = yf.download('GOOGL', start=start_date, end=end_date)
    df4 = yf.download('META', start=start_date, end=end_date)

    # Combine 3 df into final_df (append them below each other)
    final_df = pd.concat([df1, df2, df3, df4], axis=0)

    scaler = MinMaxScaler(feature_range=(0, 1))
    general_data = scaler.fit_transform(final_df[['Open', 'High', 'Low', 'Volume', 'Close']].values)

    return scaler 

def loadModel():
    # Load the trained LSTM model
    model1 = load_model('model1.h5')
    model2 = load_model('model2.h5')
    model3 = load_model('model3.h5')

    return model1, model2, model3

scaler = general_scaler()

def make_prediction(scaled_data, model):

    prediction_days = 70
    x_input = scaled_data[-prediction_days:].reshape(1, prediction_days, 5)

    prediction = model.predict(x_input)

    return prediction


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

    model1, model2, model3 = loadModel()

    # Get historical data 
    data = yf.download(ticker_symbol, start=start_date, end=end_date_predict)

    # Prepare input for prediction
    scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Volume', 'Close']].values)
    
    prediction1 = make_prediction(scaled_data, model1)
    prediction2 = make_prediction(scaled_data, model2)
    prediction3 = make_prediction(scaled_data, model3)  

    # Inverse transform the prediction
    predicted_price_1 = scaler.inverse_transform(np.concatenate((scaled_data[-1,:-1], prediction1), axis=None).reshape(1, 5))[-1,-1]
    predicted_price_2 = scaler.inverse_transform(np.concatenate((scaled_data[-1,:-1], prediction2), axis=None).reshape(1, 5))[-1,-1]
    predicted_price_3 = scaler.inverse_transform(np.concatenate((scaled_data[-1,:-1], prediction3), axis=None).reshape(1, 5))[-1,-1]
    predicted_price = (predicted_price_1 + predicted_price_2 + predicted_price_3) / 3

    formatted_predicted_price = "${:,.2f}".format(predicted_price)

    # ================================================ #
    # Evaluate the model
    # ================================================ #

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
                           rmse = rmse)

if __name__ == '__main__':
    app.run(debug=True)
