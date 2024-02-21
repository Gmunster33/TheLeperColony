import yfinance as yf
import backtrader as bt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import ta


extended_tickers = [
    "INTC",  # Intel Corporation
    "QCOM",  # Qualcomm Incorporated
    "AVGO",  # Broadcom Inc.
    "NVDA",  # NVIDIA Corporation
    "AMD",   # Advanced Micro Devices, Inc.
    "MU",    # Micron Technology, Inc.
    "000660.KS", # SK Hynix Inc.
    "TXN",   # Texas Instruments Incorporated
    "AMAT",  # Applied Materials, Inc.
    "ASML",  # ASML Holding N.V.
    "LRCX",  # Lam Research Corporation
    "SONY",  # Sony Group Corporation
    "2317.TW", # Foxconn (Hon Hai Precision Industry Co., Ltd.)
    "GLW",   # Corning Incorporated
    "TSM",   # Taiwan Semiconductor Manufacturing Company (TSMC)
    "005930.KS", # Samsung Electronics Co., Ltd.
    "WDC",   # Western Digital Corporation
    "STX",   # Seagate Technology Holdings plc
    "KEYS",  # Keysight Technologies
    "KLAC",  # KLA Corporation
    "GOOG",  # Alphabet Inc. (Google)
    "MSFT",  # Microsoft Corporation
    "META",  # Meta Platforms, Inc. (formerly Facebook)
    "1810.HK", # Xiaomi Corporation
    "2357.TW", # ASUS (ASUSTeK Computer Inc.)
    "2353.TW", # Acer Inc.
    "DELL",  # Dell Technologies Inc.
    "HPQ",   # HP Inc.
    "0992.HK", # Lenovo Group Limited
    "ADBE",  # Adobe Inc.
    "CRM",   # Salesforce
    "ORCL",  # Oracle Corporation
    "IBM",   # International Business Machines Corporation
    "CSCO",  # Cisco Systems, Inc.
    "SAP",   # SAP SE
    "INTU",  # Intuit Inc.
    "VMW",   # VMware, Inc.
    "SQ",    # Block, Inc. (formerly Square, Inc.)
    "SHOP",  # Shopify Inc.
    "TWTR",  # Twitter, Inc. (Note: As of my last update in April 2023, Twitter was taken private by Elon Musk, so this might not be current.)
    "SNAP",  # Snap Inc.
    "TSLA",  # Tesla, Inc. (significant in tech through its advancements in electric vehicles and energy storage solutions)
    "PYPL",  # PayPal Holdings, Inc.
    "ADSK",  # Autodesk, Inc.
    "ANSS",  # ANSYS, Inc.
    "CTSH",  # Cognizant Technology Solutions Corporation
    "INFY",  # Infosys Limited
    "TSM",   # Repeated for emphasis, Taiwan Semiconductor Manufacturing Company
    "ERIC",  # Telefonaktiebolaget LM Ericsson (publ)
    "NOK",   # Nokia Corporation
    "V",     # Visa Inc. (significant in tech through digital payment technologies)
    "MA",    # Mastercard Incorporated (similarly significant as Visa)
    "AAPL",  # Apple Inc. (for completeness, adding to this extended list)
    "AMZN",  # Amazon.com, Inc.
    "ZM",    # Zoom Video Communications, Inc.
    "UBER",  # Uber Technologies, Inc.
    "LYFT"   # Lyft, Inc.
]


def download_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

def add_technical_indicators(df):
    # Bollinger Bands
    indicator_bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()

    # MACD
    indicator_macd = ta.trend.MACD(close=df["Close"])
    df['macd'] = indicator_macd.macd()

    # RSI
    df['rsi'] = ta.momentum.rsi(close=df["Close"], window=14)

    return df

def preprocess_data(df):
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Drop rows with NaN values that might be created by moving averages and indicators
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

def prepare_features(df):
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', '50_MA', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'macd', 'rsi']
    X = df[feature_columns][:-1].values  # Use the updated feature set
    y = df['Close'].shift(-1).dropna().values
    return X, y

def split_data(X, y, test_size=250):
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train, C, gamma):
    model = SVR(kernel='rbf', C=C, gamma=gamma)
    model.fit(X_train, y_train)
    return model

class SVRStrategy(bt.Strategy):
    params = (
        ('model', None),
        ('scaler', None),
        ('dataframe', None),
        ('threshold', 0.005),
    )

    def __init__(self):
        # Assuming the 'model', 'scaler', and 'dataframe' parameters are passed correctly and exist
        df = self.params.dataframe
        self.data_predicted = self.params.model.predict(self.params.scaler.transform(df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', '50_MA', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'macd', 'rsi']].values[-250:]))
        self.iterations = 0

    def next(self):
        predicted_price = self.data_predicted[self.iterations]
        self.iterations += 1
        if predicted_price > self.data.close[0] * (1 + self.params.threshold):
            self.buy()
        elif predicted_price < self.data.close[0] * (1 - self.params.threshold):
            self.sell()

def plot_predictions(df, model, scaler):
    dates = df.index[-250:].tolist()
    prices = df['Close'].values[-250:]
    X = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', '50_MA', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'macd', 'rsi']].values[-250:]
    predicted_prices = model.predict(scaler.transform(X))

    plt.figure(figsize=(15, 5))
    plt.plot(dates, prices, label='Actual Prices')
    plt.plot(dates, predicted_prices, label='Predicted Prices', alpha=0.7)
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def backtest_strategy(df, strategy_class, strategy_params, perform_plot=False):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class, **strategy_params)
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.broker.set_cash(10000)
    cerebro.broker.setcommission(commission=0)
    starting_portfolio_value = cerebro.broker.getvalue()
    cerebro.run()
    ending_portfolio_value = cerebro.broker.getvalue()
    if perform_plot:
        cerebro.plot()
    print(f'Starting Portfolio Value: {starting_portfolio_value:.2f}')
    print(f'Ending Portfolio Value: {ending_portfolio_value:.2f}')
    
    # Return the profit, which is the difference between ending and starting portfolio values
    return ending_portfolio_value - starting_portfolio_value


def grid_search_C_gamma_min_error(X_train_scaled, y_train, X_test_scaled, y_test, scaler, parameter_grid):
    best_error = np.inf
    best_params = {'C': None, 'gamma': None}
    
    for C in parameter_grid['C']:
        for gamma in parameter_grid['gamma']:
            # Train the model with current C and gamma values
            model = train_model(X_train_scaled, y_train, C, gamma)
            
            # Make predictions on the test set
            predictions = model.predict(X_test_scaled)
            
            # Calculate mean squared error (MSE)
            error = mean_squared_error(y_test, predictions)
            
            # Update best parameters if current model has lower error
            if error < best_error:
                best_error = error
                best_params['C'] = C
                best_params['gamma'] = gamma

    return best_params, best_error


def calculate_directional_accuracy(y_test, predictions):
    """
    Calculates the directional accuracy of predictions against actual next day prices.
    
    y_test: Actual next day prices.
    predictions: Predicted prices, aligned with y_test indices.
    """
    # Ensure predictions are aligned with y_test for direct comparison
    actual_direction = np.sign(y_test[1:] - y_test[:-1])
    predicted_direction = np.sign(predictions[1:] - y_test[:-1])
    
    accuracy = np.mean(actual_direction == predicted_direction)
    return accuracy

# Use params: C=5994.8425031894085, gamma=0.001
# Parameters specified
C_param = 5994.8425031894085
gamma_param = 0.001

for ticker in extended_tickers:
    # Download stock data
    df = download_stock_data(ticker, '2012-01-01', '2023-01-01')
    
    # Preprocess data and add technical indicators
    df = preprocess_data(df)
    
    # Prepare features and labels
    X, y = prepare_features(df)
    
    # Split data into training and test sets (adjust according to your needs)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = train_model(X_train_scaled, y_train, C=C_param, gamma=gamma_param)
    
    # Make predictions
    predictions = model.predict(X_test_scaled)
    
    # Calculate directional accuracy
    directional_accuracy = calculate_directional_accuracy(y_test, predictions)
    print(f'{ticker} - Directional Accuracy: {directional_accuracy * 100:.2f}%')