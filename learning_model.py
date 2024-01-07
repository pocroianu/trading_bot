import matplotlib.pyplot as plt
import talib
import yfinance as yf
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

load_dotenv()


# Fetch historical data for a stock
def fetch_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end)

        if df.empty:
            print(f"No data found for {symbol} in the specified date range.")
            return None
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None


# Fetch historical data for multiple stocks
def fetch_data_multiple(symbols, start, end):
    data = {}
    for symbol in symbols:
        df = fetch_data(symbol, start, end)
        if df is not None:
            data[symbol] = df
    return data


# Preprocess the data and create features
def preprocess_data(df):
    # Create a new DataFrame to avoid SettingWithCopyWarning
    df_processed = df.copy()

    # Calculate moving averages
    df_processed["SMA_10"] = talib.SMA(df_processed["Close"], timeperiod=10)
    df_processed["SMA_20"] = talib.SMA(df_processed["Close"], timeperiod=20)

    # Calculate Bollinger Bands
    (
        df_processed["upper_BB"],
        df_processed["middle_BB"],
        df_processed["lower_BB"],
    ) = talib.BBANDS(df_processed["Close"], timeperiod=20)

    # Calculate RSI
    df_processed["RSI"] = talib.RSI(df_processed["Close"], timeperiod=14)

    # Calculate MACD
    df_processed["MACD"], df_processed["MACD_signal"], _ = talib.MACD(
        df_processed["Close"], fastperiod=12, slowperiod=26, signalperiod=9
    )

    df_processed["EMA_10"] = talib.EMA(df_processed["Close"], timeperiod=10)
    df_processed["EMA_30"] = talib.EMA(df_processed["Close"], timeperiod=30)
    df_processed["Stoch"], df_processed["Stoch_signal"] = talib.STOCH(df_processed["High"], df_processed["Low"],
                                                                      df_processed["Close"])
    df_processed["OBV"] = talib.OBV(df_processed["Close"], df_processed["Volume"])
    df_processed["ATR"] = talib.ATR(df_processed["High"], df_processed["Low"], df_processed["Close"])
    df_processed["ROC"] = talib.ROC(df_processed["Close"])
    df_processed["MACD"], df_processed["MACD_signal"], df_processed["MACD_hist"] = talib.MACD(df_processed["Close"],
                                                                                              fastperiod=12,
                                                                                              slowperiod=26,
                                                                                              signalperiod=9)

    # Drop missing values
    df_processed = df_processed.dropna()

    # Define the target variable (e.g., next day's close)
    df_processed["target"] = df_processed["Close"].shift(-1)

    # Drop the last row, which will have a missing target value
    df_processed = df_processed[:-1]

    return df_processed


# Train and evaluate a machine learning model
def train_model(df):
    features = df.columns.drop(["target"]).tolist()
    target = "target"

    x = df[features]
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(random_state=42)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 4, 5],
        "subsample": [0.8, 0.9, 1],
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_

    y_prediction = best_model.predict(x_test)
    mse = mean_squared_error(y_test, y_prediction)
    print(f"Mean Squared Error: {mse}")

    # Plot the stock data with the predicted values
    plt.figure(figsize=(14, 8))
    plt.plot(df.index[-len(y_test):], y_test, label="Actual Prices", color="blue")
    plt.plot(df.index[-len(y_test):], y_prediction, label="Predicted Prices", color="red")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Stock Price Prediction")
    plt.legend()
    # plt.show()

    return best_model, features


# Train and evaluate a machine learning model for multiple stocks
def train_models(data):
    models = {}
    feature_names = {}
    for symbol, df in data.items():
        print(f"Training model for {symbol}")
        processed_data = preprocess_data(df)
        model, features = train_model(processed_data)
        models[symbol] = model
        feature_names[symbol] = features
    return models, feature_names
