import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import json
import os

# Load credentials from the config file
def load_credentials():
    with open('config.json', 'r') as file:
        return json.load(file)

# Initialize MetaTrader 5 and login with credentials
def initialize_mt5():
    credentials = load_credentials()
    login = credentials["login"]
    password = credentials["password"]
    server = credentials["server"]

    # Initialize MT5 platform
    if not mt5.initialize():
        print("MT5 Initialization failed")
        return False

    # Log in to the trade account
    authorized = mt5.login(login, password, server)
    if not authorized:
        print(f"Login failed, error: {mt5.last_error()}")
        return False

    print("MT5 initialized and login successful")
    return True

# Download historical data
def download_historical_data(symbol, timeframe, start_date, end_date):
    try:
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None:
            print(f"Failed to download historical data for {symbol}, error: {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    except Exception as e:
        print(f"Error while downloading historical data: {e}")
        return None

# Download real-time data
def download_real_time_data(symbol, timeframe):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)  # Fetch the latest 1 tick
        if rates is None:
            print(f"Failed to download real-time data for {symbol}, error: {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    except Exception as e:
        print(f"Error while downloading real-time data: {e}")
        return None

# Example usage
if initialize_mt5():
    # Historical data
    symbol = "USTEC"
    timeframe = mt5.TIMEFRAME_M5
    start_date = datetime(2024, 10, 1)
    end_date = datetime.now()

    historical_data = download_historical_data(symbol, timeframe, start_date, end_date)
    if historical_data is not None:
        historical_data.to_csv(f"{symbol}_historical.csv", index=False)
        print(f"Historical data saved to {symbol}_historical.csv")

    # Real-time data collection
    live_data = download_real_time_data(symbol, timeframe)
    if live_data is not None:
        print(live_data)

# Shutdown MetaTrader 5 connection
mt5.shutdown()
