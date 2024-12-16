import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import TimeseriesGenerator
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# Load dataset
filename = "TSLA.csv"
df = pd.read_csv(filename)

# Data preprocessing
df.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
close_data = df['Close'].values.reshape((-1, 1))

# Split data into training and testing sets
split_percent = 0.80
split = int(split_percent * len(close_data))

close_train, close_test = close_data[:split], close_data[split:]
date_train, date_test = df['Date'][:split], df['Date'][split:]

# Generate time series data
look_back = 31
train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

# Build the LSTM model
model = Sequential([
    LSTM(10, activation='relu', input_shape=(look_back, 1)),
    Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_generator, epochs=150, verbose=1)

# Make predictions
prediction = model.predict(test_generator)
close_train = close_train.flatten()
close_test = close_test.flatten()
prediction = prediction.flatten()

# Plot training vs prediction
trace1 = go.Scatter(x=date_train, y=close_train, mode='lines', name='Data')
trace2 = go.Scatter(x=date_test, y=prediction, mode='lines', name='Prediction')
trace3 = go.Scatter(x=date_test, y=close_test, mode='lines', name='Ground Truth')

layout = go.Layout(title="Tesla Stock Prediction", xaxis={'title': "Date"}, yaxis={'title': "Close"})
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
fig.show()

# Forecast next 30 days
def predict_future(num_prediction, model, data, look_back):
    prediction_list = data[-look_back:]
    for _ in range(num_prediction):
        x = prediction_list[-look_back:].reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    return prediction_list[look_back-1:]

def generate_future_dates(start_date, num_days):
    return pd.date_range(start=start_date, periods=num_days).tolist()

num_prediction = 30
forecast = predict_future(num_prediction, model, close_data, look_back)
forecast_dates = generate_future_dates(date_test.iloc[-1], num_prediction)

# Plot forecast
trace4 = go.Scatter(x=forecast_dates, y=forecast, mode='lines', name='Forecast')
fig.add_trace(trace4)
fig.show()
