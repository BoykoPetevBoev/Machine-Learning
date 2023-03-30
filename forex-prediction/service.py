import MetaTrader5 as mt5
import pandas as pd
import numpy as np

def get_forex_data (
    start_time,
    end_time
):
    
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()

    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_D1
    start_time = pd.Timestamp("2020-02-01")
    end_time = pd.Timestamp("2021-02-01")

    data =  mt5.copy_rates_range(symbol, timeframe, start_time, end_time)

    mt5.shutdown()
    return [
        data["time"],
        data["open"],
        data["close"],
        data["high"],
        data["low"]
    ]

def get_forex_train_data ():
    start_time = pd.Timestamp("2020-01-01")
    end_time = pd.Timestamp("2021-01-01")
    return  get_forex_data(
        start_time=start_time,
        end_time=end_time
    )

def get_forex_test_data ():
    start_time = pd.Timestamp("2021-01-01")
    end_time = pd.Timestamp("2022-01-01")
    return  get_forex_data(
        start_time=start_time,
        end_time=end_time
    )

def get_forex_predict_data ():
    start_time = pd.Timestamp("2022-01-01")
    end_time = pd.Timestamp("2022-03-01")
    return  get_forex_data(
        start_time=start_time,
        end_time=end_time
    )
