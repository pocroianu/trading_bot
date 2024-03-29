from datetime import datetime
import alpaca_trade_api as trade_api

from learning_model import preprocess_data, train_models, fetch_data_multiple, save_model

from dotenv import load_dotenv
import os

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL: str = "https://paper-api.alpaca.markets"  # Use paper trading for testing

QUANTITY = 2

# Connect to the API
api = trade_api.REST(key_id=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET, base_url=BASE_URL, api_version="v2")


# Placeholder
def should_buy(symbol):
    # Implement your buy logic here
    return True


# Placeholder
def should_sell(symbol):
    # Implement your sell logic here
    return False


def generate_signals(models, feature_names, data):
    symbols_signals = {}
    for symbol, model in models.items():
        # Get the latest available data for the stock
        df = data[symbol]

        # Preprocess the data and remove the target column
        x = preprocess_data(df).drop(columns="target")

        # Set the feature names for X to match the feature names used during model training
        x.columns = feature_names[symbol]

        # Make a prediction for the next day
        next_day_prediction = model.predict(x.iloc[-1].values.reshape(1, -1))[0]

        # Implement a trading strategy based on the prediction
        current_price = df.iloc[-1]["Close"]  # Update this line
        signal = "buy" if next_day_prediction > current_price else "sell"
        symbols_signals[symbol] = signal

    return symbols_signals


def execute_trades(signals):
    for symbol, signal in signals.items():
        # Define the trade parameters
        trade_type = "buy" if signal == "buy" else "sell"
        qty = QUANTITY  # Number of shares to trade

        # Check if the stock is already in the portfolio
        positions = api.list_positions()
        position = [p for p in positions if p.symbol == symbol]

        if trade_type == "buy" and not position:
            # Submit a buy order
            print(f"Submitting a buy order for {symbol}, quantity: {qty}, time_in_force: gtc")
            api.submit_order(
                symbol=symbol, qty=qty, side="buy", type="market", time_in_force="gtc"
            )
        elif trade_type == "sell" and position:
            # Get the actual quantity of shares owned
            owned_qty = int(position[0].qty)

            # Sell the available shares, even if it's less than the requested quantity
            sell_qty = min(owned_qty, qty)

            # Submit a sell order
            print(f"Submitting a sell order for {symbol}, quantity: {sell_qty}, time_in_force: gtc")

            try:
                api.submit_order(
                    symbol=symbol, qty=sell_qty, side="sell", type="market", time_in_force="gtc"
                )
            except trade_api.rest.APIError as e:
                print(f"trade_api.rest.APIError {e}")
        else:
            print(f"No trading done for {symbol}, position was: {position}")
