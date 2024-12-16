import tensorflow as tf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# Load and preprocess data
filename = "TSLA.csv"
df = pd.read_csv(filename)
print(df.info())
print(df.describe())

# Drop unnecessary columns
df.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)

# Data preparation
close_data = df['Close'].values.reshape((-1, 1))
split_percent = 0.80
split = int(split_percent * len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = df['Date'][:split]
date_test = df['Date'][split:]

look_back = 31  # Number of timesteps to look back

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

# Train the model
num_epochs = 150
model.fit(train_dataset, epochs=num_epochs, verbose=1)

# Make predictions
predictions = model.predict(test_dataset)

# Flatten the arrays for visualization
close_train = close_train.flatten()
close_test = close_test.flatten()
predictions = predictions.flatten()

# Plot the results
trace1 = go.Scatter(x=date_train, y=close_train, mode='lines', name='Training Data')
trace2 = go.Scatter(x=date_test, y=predictions, mode='lines', name='Predictions')
trace3 = go.Scatter(x=date_test, y=close_test, mode='lines', name='Ground Truth')

layout = go.Layout(title="Tesla Stock Price Prediction",
                   xaxis={'title': "Date"},
                   yaxis={'title': "Close Price"})

fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
fig.show()

# Forecast future values
def forecast_future(num_prediction, model, close_data, look_back):
    input_sequence = close_data[-look_back:]
    predictions = []

    for _ in range(num_prediction):
        input_sequence = input_sequence.reshape((1, look_back, 1))
        next_prediction = model.predict(input_sequence)[0][0]
        predictions.append(next_prediction)
        input_sequence = np.append(input_sequence[0][1:], next_prediction).reshape((look_back, 1))

    return predictions

# Predict the next 30 days
num_prediction = 30
future_predictions = forecast_future(num_prediction, model, close_data, look_back)
future_dates = pd.date_range(start=date_test.iloc[-1], periods=num_prediction + 1).tolist()

# Add forecast to the plot
trace4 = go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Forecast')
fig.add_trace(trace4)
fig.show()
