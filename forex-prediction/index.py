import service
import normalizators
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import asyncio


tf.config.run_functions_eagerly(True)
window_size = 50


def setup_model():
    # Load historical chart data
    data = service.get_forex_data()
    x_train, y_train, x_test, y_test = normalizators.normalize_forex_data(data)

    print(len(x_train), len(y_train), len(x_test), len(y_test))
    # set the window size
    # print(normalized_data)
    #

    # define model architecture
    model = Sequential()
    model.add(LSTM(128, input_shape=(window_size, 4), return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(1, activation='relu'))

    model.summary()

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['binary_accuracy'], run_eagerly=True)

    # train the model
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    return model


setup_model()

# # Define the model architecture
# model = tf.keras.Sequential([
#     tf.keras.layers.LSTM(64, input_shape=(None, 1)),
#     tf.keras.layers.Dense(1)
# ])

# # Compile the model
# model.compile(loss='mean_squared_error', optimizer='adam')

# # Train the model
# model.fit(train_data, epochs=10)

# # Evaluate the model
# mse = model.evaluate(test_data)
# print(f"Mean Squared Error: {mse}")

# # Use the model to make predictions
# predictions = model.predict(new_data)
