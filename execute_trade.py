iimport random
import time
import ccxt
import logging
import talib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create an instance of the ndax.io exchange
exchange = ccxt.ndax({
    'apiKey': '',
    'secret': '',
    # 'password': 'YOUR_API_PASSWORD',
    # Additional exchange-specific options if needed
})

def calculate_sma(symbol, timeframe, period):
    # Get the OHLCV data for the specified symbol and timeframe
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=period)

    # Extract the closing prices from the OHLCV data
    close_prices = [candle[4] for candle in ohlcv]

    # Calculate the simple moving average (SMA) for the closing prices
    sma = sum(close_prices) / len(close_prices)

    return sma

def calculate_rsi(symbol, timeframe, period):
    # Get the OHLCV data for the specified symbol and timeframe
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=period)

    # Extract the closing prices from the OHLCV data
    close_prices = np.array([candle[4] for candle in ohlcv])

    # Calculate the RSI using the closing prices
    rsi = talib.RSI(close_prices, timeperiod=period)

    return rsi[-1]  # Get the latest RSI value

def execute_trade():
    # Get the list of available trading pairs
    markets = exchange.load_markets()
    symbols = list(markets.keys())

    # Create an empty dataframe to store market data
    market_data = pd.DataFrame()

    # Prepare data for machine learning
    features = []
    labels = []

    for symbol in symbols:
        try:
            # Get the latest ticker data
            ticker = exchange.fetch_ticker(symbol)
            logger.info(f"Ticker data for {symbol}: {ticker}")

            # Append ticker data to the market data dataframe
            market_data = market_data.append(pd.Series(ticker), ignore_index=True)

            current_price = ticker['close']
            buy_price = current_price

            # Calculate sell prices
            sell_price_1 = buy_price * 1.01  # 1% higher than buy price
            sell_price_2 = buy_price * 1.011  # 1.1% higher than buy price

            # Calculate stop loss and take profit prices
            stop_loss_price = buy_price * 0.9991  # 0.09% lower than buy price
            take_profit_price = buy_price * 1.013  # 1.3% higher than buy price

            # Check candlestick market data
            candle_data = exchange.fetch_ohlcv(symbol, '1m', limit=2)  # Fetch last 2 candles
            current_candle = candle_data[-1]
            previous_candle = candle_data[-2]

            # Check buy condition based on candlestick market data
            if current_candle[1] > previous_candle[4]:  # If current candle opens higher than the previous candle's close
                # Generate random amounts for buy and sell
                buy_amount = int(random.uniform(1, 100000))  # Random amount between 1 and 100,000 as an integer

                # Place a market buy order with the random buy amount
                buy_order = exchange.create_market_buy_order(symbol, buy_amount)
                logger.info(f"Buy order placed for {symbol} at market price: {buy_order}")

                # Generate random amounts for sell
                sell_amount_1 = random.uniform(0.00001, buy_amount + 1)
                sell_amount_2 = random.uniform(0.00001, buy_amount + 1)

                # Place sell orders at specified prices with random sell amounts
                sell_order_1 = exchange.create_limit_sell_order(symbol, sell_amount_1, sell_price_1)
                logger.info(f"Sell order placed for {symbol} at price: {sell_price_1}")
                sell_order_2 = exchange.create_limit_sell_order(symbol, sell_amount_2, sell_price_2)
                logger.info(f"Sell order placed for {symbol} at price: {sell_price_2}")

                # Prepare data for machine learning
                features.append([buy_amount, sell_amount_1, sell_amount_2])
                labels.append(1)  # 1 indicates a successful trade

                # Set stop loss and take profit prices based on real-time market data
                if symbol in market_data.columns:
                    symbol_data = market_data.loc[:, symbol]
                    symbol_high = symbol_data['high']
                    symbol_low = symbol_data['low']

                    stop_loss_price = symbol_low * 0.9991  # Set stop loss 0.09% below the symbol's low price
                    take_profit_price = symbol_high * 1.013  # Set take profit 1.3% above the symbol's high price

                exchange.create_order(
                    symbol,
                    'stop',
                    'sell',
                    sell_amount_1,
                    stopPrice=stop_loss_price,
                    price=buy_price,
                    params={'stopLoss': True}
                )
                logger.info(f"Stop loss order placed for {symbol} at price: {stop_loss_price}")
                exchange.create_order(
                    symbol,
                    'limit',
                    'sell',
                    sell_amount_2,
                    price=take_profit_price,
                    params={'takeProfit': True}
                )
                logger.info(f"Take profit order placed for {symbol} at price: {take_profit_price}")

            else:
                # Prepare data for machine learning
                features.append([0, 0, 0])
                labels.append(0)  # 0 indicates an unsuccessful trade

        except ccxt.InsufficientFunds as e:
            logger.info(f"Insufficient funds for {symbol}. Skipping to the next trading pair.")

        except ccxt.BaseError as e:
            if "symbol" in str(e).lower():
                logger.info(f"Invalid symbol ({symbol}). Skipping to the next trading pair.")
            else:
                logger.info(f"An error occurred for {symbol}. Skipping to the next trading pair.")

        except Exception as e:
            logger.info(f"An error occurred: {str(e)}")

        time.sleep(1)

    # Train a random forest classifier
    clf = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    clf.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy: {accuracy}")

execute_trade()
