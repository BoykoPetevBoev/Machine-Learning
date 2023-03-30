import service
import normalizators
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import asyncio

tf.config.run_functions_eagerly(True)
window_size = 50

def setup_model():
    train_data = service.get_forex_train_data()
    x_train, y_train = normalizators.normalize_forex_data(train_data)

    test_data = service.get_forex_test_data()
    x_test, y_test = normalizators.normalize_forex_data(test_data)
    
    predict_data = service.get_forex_predict_data()
    x_predict, y_predict = normalizators.normalize_forex_data(predict_data)
    
    

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=32, input_shape=(50,1), activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
    
    model.compile(
        optimizer=Adam(0.001), 
        loss='mean_squared_error'
    )

    model.fit(
        x_train, 
        y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.1,
        shuffle=False
    )
    
    result = model.evaluate(
        x_test, 
        y_test, 
        batch_size=16, 
        verbose=2
    )
    print('Result:', result)
    
    predictions = model.predict(x_predict)
    
    print(predictions)
    
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
