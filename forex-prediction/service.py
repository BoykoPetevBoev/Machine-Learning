import MetaTrader5 as mt5
import pandas as pd
import numpy as np

def get_forex_data ():
    
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()

    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_D1
    start_time = pd.Timestamp("2020-02-01")
    end_time = pd.Timestamp("2022-02-01")

    data =  mt5.copy_rates_range(symbol, timeframe, start_time, end_time)

    mt5.shutdown()
    return [
        data["open"],
        data["close"],
        data["high"],
        data["low"]
    ]
