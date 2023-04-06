import alpaca_trade_api as tradeapi

from learning_model import fetch_data_multiple, preprocess_data, train_models

from dotenv import load_dotenv
import os

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = 'https://paper-api.alpaca.markets'  # Use paper trading for testing

# Connect to the API
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, BASE_URL, api_version='v2')

# Define your trading strategy
def should_buy(symbol):
    # Implement your buy logic here
    return True

def should_sell(symbol):
    # Implement your sell logic here
    return False

# Execute trades based on your strategy
def execute_trades():
    # Get a list of your positions
    positions = api.list_positions()

    # Check if we should sell any positions
    for position in positions:
        if should_sell(position.symbol):
            api.submit_order(
                symbol=position.symbol,
                qty=position.qty,
                side='sell',
                type='market',
                time_in_force='gtc'
            )

    # Check if we should buy any new stocks
    symbols = ['AAPL', 'GOOG', 'MSFT']  # Replace with your desired symbols
    for symbol in symbols:
        if should_buy(symbol):
            api.submit_order(
                symbol=symbol,
                qty=1,  # Replace with your desired quantity
                side='buy',
                type='market',
                time_in_force='gtc'
            )


def generate_signals(models, feature_names, data):
    signals = {}
    for symbol, model in models.items():
        # Get the latest available data for the stock
        df = data[symbol]

        # Preprocess the data and remove the target column
        X = preprocess_data(df).drop(columns='target')
        
        # Set the feature names for X to match the feature names used during model training
        X.columns = feature_names[symbol]

        # Make a prediction for the next day
        next_day_prediction = model.predict(X.iloc[-1].values.reshape(1, -1))[0]

        # Implement a trading strategy based on the prediction
        current_price = df.iloc[-1]['4. close']
        signal = 'buy' if next_day_prediction > current_price else 'sell'
        signals[symbol] = signal

    return signals



def execute_trades(signals):
    for symbol, signal in signals.items():
        # Define the trade parameters
        trade_type = 'buy' if signal == 'buy' else 'sell'
        qty = 4  # Number of shares to trade

        # Check if the stock is already in the portfolio
        positions = api.list_positions()
        position = [p for p in positions if p.symbol == symbol]

        if trade_type == 'buy' and not position:
            # Submit a buy order
            print(f"Submitting a buy order for {symbol}")
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
        elif trade_type == 'sell' and position:
            # Submit a sell order
            print(f"Submitting a sell order for {symbol}")
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='gtc'
            )

# Schedule the bot to run once per day
if __name__ == '__main__':
    # Define the list of stock symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']

    # Fetch historical data for multiple stocks
    start = '2018-01-01'
    end = '2022-01-01'
    data = fetch_data_multiple(symbols, start, end)

    # Train models for multiple stocks
    models, feature_names = train_models(data)

    # Generate buy/sell signals
    signals = generate_signals(models, feature_names, data)

    # Execute the trades
    execute_trades(signals)
