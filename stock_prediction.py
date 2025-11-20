import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import math

# Load and preprocess data
filename = "TSLA.csv"
df = pd.read_csv(filename)

# Ensure dates are datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# Drop unused columns
df.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)

# Scale the 'Close' prices
scaler = MinMaxScaler()
close_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Split data
split_percent = 0.80
split = int(split_percent * len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = df['Date'][:split]
date_test = df['Date'][split:]

# Sequence length
look_back = 31

# Create TensorFlow time series datasets
train_dataset = tf.keras.utils.timeseries_dataset_from_array(
    data=close_train,
    targets=close_train[look_back:],
    sequence_length=look_back,
    batch_size=20,
)

test_dataset = tf.keras.utils.timeseries_dataset_from_array(
    data=close_test,
    targets=close_test[look_back:],
    sequence_length=look_back,
    batch_size=1,
)

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(10, activation='relu', input_shape=(look_back, 1)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping to avoid overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Train model
model.fit(train_dataset, epochs=150, verbose=1, callbacks=[early_stop])

# Predict
predictions = model.predict(test_dataset, verbose=0)
predictions = scaler.inverse_transform(predictions).flatten()

# Inverse transform train/test for plotting
close_train_inv = scaler.inverse_transform(close_train).flatten()
close_test_inv = scaler.inverse_transform(close_test).flatten()

# Plot results
trace1 = go.Scatter(x=date_train, y=close_train_inv, mode='lines', name='Training Data')
trace2 = go.Scatter(x=date_test[look_back:], y=predictions, mode='lines', name='Predictions')
trace3 = go.Scatter(x=date_test, y=close_test_inv, mode='lines', name='Ground Truth')

layout = go.Layout(title="Tesla Stock Price Prediction",
                   xaxis={'title': "Date"},
                   yaxis={'title': "Close Price"})

fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
fig.show()

# Evaluate RMSE
rmse = math.sqrt(mean_squared_error(close_test_inv[look_back:], predictions))
print(f"Test RMSE: {rmse:.2f}")

# Forecast future values
def forecast_future(num_prediction, model, close_data, look_back):
    input_sequence = close_data[-look_back:]
    predictions = []

    for _ in range(num_prediction):
        input_sequence = input_sequence.reshape((1, look_back, 1))
        next_pred = model.predict(input_sequence, verbose=0)[0][0]
        predictions.append(next_pred)
        input_sequence = np.append(input_sequence[0][1:], next_pred).reshape((look_back, 1))

    return predictions

# Forecast next 30 days
num_prediction = 30
future_predictions = forecast_future(num_prediction, model, close_data, look_back)
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

future_dates = pd.date_range(start=date_test.iloc[-1], periods=num_prediction + 1).tolist()[1:]

# Add forecast to plot
trace4 = go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Forecast')
fig.add_trace(trace4)
fig.show()
