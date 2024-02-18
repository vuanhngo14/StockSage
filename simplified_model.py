import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from scikeras.wrappers import KerasRegressor
from datetime import datetime
import json

def create_lstm_model(units=50, optimizer='adam', batch_size=32, epochs=25):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer with 1 unit for predicting 'Close'
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Load Data
ticker_symbol = 'AAPL'
start_date = dt.datetime(2010, 1, 1)
end_date = dt.datetime(2022, 1, 1)
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Prepare Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Volume', 'Close']].values)

prediction_days = 70

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, :])  # Using all four features
    y_train.append(scaled_data[x, 4])  # 'Close' is the fourth column (index 3)

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

# Wrap the Keras model so it can be used by scikit-learn GridSearchCV
model = KerasRegressor(model=create_lstm_model, epochs=10, batch_size=32, verbose=1)  # Set verbose to 1 for more output

# Define the hyperparameter grid
param_grid = {
    'batch_size': [32],  
    'epochs': [25],  
    'optimizer': ['adam'], 
}

# Create Time Series Split for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Create Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=tscv, verbose=1)


# Fit the grid search to the data
print("Starting grid search...")
grid_result = grid.fit(x_train, y_train)
print("Grid search completed!")

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Retrieve best hyperparameters
best_batch_size = grid_result.best_params_['batch_size']
best_optimizer = grid_result.best_params_['optimizer']
best_epochs = grid_result.best_params_['epochs']

# Build the final model with the best hyperparameters
print("Building final model with best hyperparameters...")
final_model = create_lstm_model(epochs=best_epochs, optimizer=best_optimizer, batch_size=best_batch_size)
final_model.fit(x_train, y_train)
print("Final model training completed!")

# Save the model with metadate to a JSON file

model_version = "v2.0"
model_date_modified = datetime.today().strftime('%Y-%m-%d')

# Save metadata
metadata = {
    "version": model_version,
    "date_modified": model_date_modified
}

# Save metadata to a JSON file
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f)

final_model.save('final_model.h5')