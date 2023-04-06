import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import talib
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()

alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")


# Fetch historical data for a stock
def fetch_data(symbol, start, end):
    try:
        ts = TimeSeries(key=alpha_vantage_api_key, output_format='pandas')
        df, _ = ts.get_daily_adjusted(symbol, outputsize='full')

        # Convert the date strings to datetime objects, sort the index, and filter the date range
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.loc[start:end]

        if df.empty:
            print(f"No data found for {symbol} in the specified date range.")
            return None
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None


# Preprocess the data and create features
def preprocess_data(df):
    # Create a new DataFrame to avoid SettingWithCopyWarning
    df_processed = df.copy()
    
    # Calculate moving averages
    df_processed['SMA_10'] = talib.SMA(df_processed['4. close'], timeperiod=10)
    df_processed['SMA_20'] = talib.SMA(df_processed['4. close'], timeperiod=20)

    # Calculate Bollinger Bands
    df_processed['upper_BB'], df_processed['middle_BB'], df_processed['lower_BB'] = talib.BBANDS(df_processed['4. close'], timeperiod=20)

    # Calculate RSI
    df_processed['RSI'] = talib.RSI(df_processed['4. close'], timeperiod=14)

    # Calculate MACD
    df_processed['MACD'], df_processed['MACD_signal'], _ = talib.MACD(df_processed['4. close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # Drop missing values
    df_processed = df_processed.dropna()

    # Define the target variable (e.g., next day's close)
    df_processed['target'] = df_processed['4. close'].shift(-1)

    # Drop the last row, which will have a missing target value
    df_processed = df_processed[:-1]

    return df_processed


# Train and evaluate a machine learning model
def train_model(df):
    features = df.columns.drop(['target']).tolist()
    target = 'target'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Plot the stock data with the predicted values
    plt.figure(figsize=(14, 8))
    plt.plot(df.index[-len(y_test):], y_test, label='Actual Prices', color='blue')
    plt.plot(df.index[-len(y_test):], y_pred, label='Predicted Prices', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    # plt.show()

    return model, features



# Fetch historical data for multiple stocks
def fetch_data_multiple(symbols, start, end):
    data = {}
    for symbol in symbols:
        df = fetch_data(symbol, start, end)
        if df is not None:
            data[symbol] = df
    return data

# Train and evaluate a machine learning model for multiple stocks
def train_models(data):
    models = {}
    feature_names = {}
    for symbol, df in data.items():
        print(f'Training model for {symbol}')
        processed_data = preprocess_data(df)
        model, features = train_model(processed_data)
        models[symbol] = model
        feature_names[symbol] = features
    return models, feature_names
