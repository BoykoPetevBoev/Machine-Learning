import numpy as np

def normalize_forex_data(data):
    # Define the data
    open_prices = np.array(data[0])
    close_prices = np.array(data[1])
    high_prices = np.array(data[2])
    low_prices = np.array(data[3])

    # Normalize the data
    normalized_open = (open_prices - np.min(open_prices)) / (np.max(open_prices) - np.min(open_prices))
    normalized_close = (close_prices - np.min(close_prices)) / (np.max(close_prices) - np.min(close_prices))
    normalized_high = (high_prices - np.min(high_prices)) / (np.max(high_prices) - np.min(high_prices))
    normalized_low = (low_prices - np.min(low_prices)) / (np.max(low_prices) - np.min(low_prices))

    # Combine the 4 arrays into a single NumPy array
    normalized_data = np.column_stack((normalized_open, normalized_close, normalized_high, normalized_low))

     # create X and y data from normalized_data
    x, y = [], []
    window_size = 50
    for i in range(window_size, len(normalized_data)):
        x.append(normalized_data[i-window_size:i, :])
        y.append(1 if normalized_close[i] > normalized_close[i-1] else 0)
    x, y = np.array(x), np.array(y)
    
    # split the data into training and testing sets
    train_size = int(len(x) * 0.8)
    x_train, y_train = x[:train_size], y[:train_size]
    X_test, y_test = x[train_size:], y[train_size:]
    
    return [
        x_train,
        y_train,
        X_test,
        y_test
    ]