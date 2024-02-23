# Import libraries

# Baiscs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

# Stock market 
import yfinance as yf

# ML 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from scikeras.wrappers import KerasRegressor

# Extras
from datetime import datetime
import json

# ================================================ #
#                    Main Mode l                   #
# ================================================ #

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

def perform_grid_search_and_fit(x_train, y_train, param_grid, n_splits=5):
    # Wrap the Keras model so it can be used by scikit-learn GridSearchCV
    model = KerasRegressor(model=create_lstm_model, epochs=10, batch_size=32, verbose=1)  # Set verbose to 1 for more output

    # Create Time Series Split for cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)

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
    
    # Fit the final model
    final_model.fit(x_train, y_train)
    print("Final model training completed!")

    return final_model


# ================================================ #
#          Fit a general min-max scaler            #
# ================================================ #

start_date = dt.datetime(2010, 1, 1)
end_date = dt.datetime(2022, 1, 1)

# AAPL, GGLE and NVDA
df1 = yf.download('AAPL', start=start_date, end=end_date)
df2 = yf.download('NFLX', start=start_date, end=end_date)
df3 = yf.download('NVDA', start=start_date, end=end_date)
df4 = yf.download('META', start=start_date, end=end_date)

# Combine 3 df into final_df (append them below each other)
final_df = pd.concat([df1, df2, df3, df4], axis=0)

scaler = MinMaxScaler(feature_range=(0, 1))
general_data = scaler.fit_transform(final_df[['Open', 'High', 'Low', 'Volume', 'Close']].values)

# ================================================ #
#                     Explain                      #
# ================================================ #

# Since stock code varies between different companies, for the best generalization, 
# we will use ensemble methods for for multiple companies. 


# # ================================================ #
# #                     Model 1                      #
# # ================================================ #

# # Load Data 1
# ticker_symbol = 'AAPL'
# start_date = dt.datetime(2010, 1, 1)
# end_date = dt.datetime(2022, 1, 1)
# data = yf.download(ticker_symbol, start=start_date, end=end_date)

# # Prepare Data
# scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Volume', 'Close']].values)

# prediction_days = 70

# x_train = []
# y_train = []

# for x in range(prediction_days, len(scaled_data)):
#     x_train.append(scaled_data[x - prediction_days:x, :])  # Using all four features
#     y_train.append(scaled_data[x, 4])  # 'Close' is the fourth column (index 3)

# x_train, y_train = np.array(x_train), np.array(y_train)
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

# param_grid = {
#     'batch_size': [32],  
#     'epochs': [25],  
#     'optimizer': ['adam'], 
# }

# final_model_1 = perform_grid_search_and_fit(x_train, y_train, param_grid)

# final_model_1.save("model1.h5")

# ================================================ #
#                     Model 2                      #
# ================================================ #

# Prepare the data for the second stock code 

# Load Data
ticker_symbol = 'NVDA'
start_date = dt.datetime(2010, 1, 1)
end_date = dt.datetime(2022, 1, 1)
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Prepare Data
scaled_data_2 = scaler.transform(data[['Open', 'High', 'Low', 'Volume', 'Close']].values)

x_train_2 = []
y_train_2 = []

prediction_days = 70

for x in range(prediction_days, len(scaled_data_2)):
    x_train_2.append(scaled_data_2[x - prediction_days:x, :])  # Using all four features
    y_train_2.append(scaled_data_2[x, 4])  # 'Close' is the fourth column (index 3)

x_train_2, y_train_2 = np.array(x_train_2), np.array(y_train_2)
x_train_2 = np.reshape(x_train_2, (x_train_2.shape[0], x_train_2.shape[1], x_train_2.shape[2]))

x_train = x_train_2
y_train = y_train_2


param_grid = {
    'batch_size': [32],  
    'epochs': [25],  
    'optimizer': ['adam'], 
}

final_model_2 = perform_grid_search_and_fit(x_train_2, y_train_2, param_grid)

final_model_2.save("model2.h5")

# ================================================ #
#                     Model 3                      #
# ================================================ #

# Third stock code

# Load Data
ticker_symbol = 'META'
start_date = dt.datetime(2010, 1, 1)
end_date = dt.datetime(2022, 1, 1)
data3 = yf.download(ticker_symbol, start=start_date, end=end_date)

# Prepare Data
scaled_data_3 = scaler.fit_transform(data3[['Open', 'High', 'Low', 'Volume', 'Close']].values)

x_train_3 = []
y_train_3 = []



prediction_days = 70

for x in range(prediction_days, len(scaled_data_3)):
    x_train_3.append(scaled_data_3[x - prediction_days:x, :])  # Using all four features
    y_train_3.append(scaled_data_3[x, 4])  # 'Close' is the fourth column (index 3)

x_train_3, y_train_3 = np.array(x_train_3), np.array(y_train_3)
x_train_3 = np.reshape(x_train_3, (x_train_3.shape[0], x_train_3.shape[1], x_train_3.shape[2]))

x_train = x_train_3
y_train = y_train_3

param_grid = {
    'batch_size': [32],  
    'epochs': [25],  
    'optimizer': ['adam'], 
}

final_model_3 = perform_grid_search_and_fit(x_train_3, y_train_3, param_grid)

final_model_3.save("model3.h5")

