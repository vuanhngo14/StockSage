from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import plotly.graph_objs as go
import plotly.io as pio


app = Flask(__name__)

# Load the trained LSTM model
model = load_model('final_model.h5')

# Load the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(yf.download('AAPL', start='2010-01-01', end='2022-01-01')[['Open', 'High', 'Low', 'Volume', 'Close']].values)

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
    
    # Get historical data
    data = yf.download(ticker_symbol, start='2010-01-01', end='2022-01-01')
    scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Volume', 'Close']].values)
    
    # Prepare input for prediction
    prediction_days = 70
    x_input = scaled_data[-prediction_days:].reshape(1, prediction_days, 5)
    
    # Make prediction
    prediction = model.predict(x_input)
    
    # Inverse transform the prediction
    predicted_price = scaler.inverse_transform(np.concatenate((scaled_data[-1,:-1], prediction), axis=None).reshape(1, 5))[-1,-1]
    predicted_price = "${:,.2f}".format(predicted_price)

    # ================================================ #
    # Add the plot 
    # ================================================ #

    # Get historical data
    data = yf.download('AAPL', start='2010-01-01', end='2022-01-01')
    scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Volume', 'Close']].values)

    # Get dates for historical prices
    dates = data.index.strftime('%Y-%m-%d')

    # Prepare input for prediction
    prediction_days = 70
    x_input = scaled_data[-prediction_days:].reshape(1, prediction_days, 5)

    # Make prediction
    prediction = model.predict(x_input)

    # Inverse transform the prediction
    predicted_price = scaler.inverse_transform(np.concatenate((scaled_data[-1,:-1], prediction), axis=None).reshape(1, 5))[-1,-1]

    # Create trace for past prices
    trace_past_prices = go.Scatter(
        x=dates[-prediction_days:], y=data['Close'].values[-prediction_days:],  mode='lines',name='Past Prices')

    # Create trace for predicted price
    trace_predicted_price = go.Scatter(x=[dates[-1]],  y=[predicted_price],mode='markers',name='Predicted Price',marker=dict(color='red',size=10))

    # Combine traces
    data = [trace_past_prices, trace_predicted_price]

    # Configure layout
    layout = go.Layout(
        title='Historical Prices and Predicted Price',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price'),
        showlegend=True
    )

    # Create the figure
    fig = go.Figure(data=data, layout=layout)

    # Save the figure to an HTML file
    pio.write_html(fig, file='graph.html', auto_open=True)

    


    return render_template('index.html', 
                           ticker_symbol=ticker_symbol, 
                           end_date=end_date, 
                           predicted_price=predicted_price,
                           graph_json=graph_json)

if __name__ == '__main__':
    app.run(debug=True)
