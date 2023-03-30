import numpy as np

def normalize_forex_data(data):
    PERIODS = 50
    # Define the data
    time = np.array(data[0])
    open_prices = np.array(data[1])
    close_prices = np.array(data[2])
    high_prices = np.array(data[3])
    low_prices = np.array(data[4])

    # Normalize the data    
    normalized_time = (time - np.min(time)) / (np.max(time) - np.min(time))
    normalized_open = (open_prices - np.min(open_prices)) / (np.max(open_prices) - np.min(open_prices))
    normalized_close = (close_prices - np.min(close_prices)) / (np.max(close_prices) - np.min(close_prices))
    normalized_high = (high_prices - np.min(high_prices)) / (np.max(high_prices) - np.min(high_prices))
    normalized_low = (low_prices - np.min(low_prices)) / (np.max(low_prices) - np.min(low_prices))
    
    # Combine the 4 arrays into a single NumPy array
    # normalized_data = np.column_stack((normalized_open, normalized_close, normalized_high, normalized_low))

    x_data = []
    y_data = []

    for i in range(len(normalized_close)-PERIODS):
        x_data.append(normalized_close[i:i+PERIODS])
        y_data.append(normalized_close[i+PERIODS])

    # Split the data into training and testing sets 
    # train_size = int(len(normalized_close) * 0.8)
    # x_train, y_train = x[:train_size], y[:train_size]
    # x_test, y_test = x[train_size:], y[train_size:]
     
     # create X and y data from normalized_data
    # x, y = [], []
    # window_size = 50
    # for i in range(window_size, len(normalized_data)):
    #     x.append(normalized_data[i])
    #     y.append(normalized_time[i])
    # x, y = np.array(x), np.array(y)
    
    # # split the data into training and testing sets
    # train_size = int(len(x) * 0.8)
    # x_train, y_train = x[:train_size], y[:train_size]
    # X_test, y_test = x[train_size:], y[train_size:]
    
    return [
        np.array(x_data),
        np.array(y_data),
    ]