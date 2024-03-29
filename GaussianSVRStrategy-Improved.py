import yfinance as yf
import backtrader as bt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import ta


tickers = [
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
    "0992.HK" # Lenovo Group Limited
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


# Profit oriented grid search..... NOT ACCURACY!!!
def grid_search_C_gamma(df, X_train_scaled, y_train, scaler, parameter_grid):
    best_profit = -float('inf')
    best_params = {'C': None, 'gamma': None}
    for C in parameter_grid['C']:
        for gamma in parameter_grid['gamma']:
            # Train the model with current C and gamma values
            model = train_model(X_train_scaled, y_train, C, gamma)
            
            # Prepare strategy parameters with the current model
            strategy_params = {
                'model': model,
                'scaler': scaler,
                'dataframe': df
            }
            
            # Backtest the strategy with the current model
            profit = backtest_strategy(df[-250:], SVRStrategy, strategy_params, perform_plot=False)
            
            # Update best parameters if current model is better
            if profit > best_profit:
                best_profit = profit
                best_params['C'] = C
                best_params['gamma'] = gamma

    return best_params, best_profit

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

def grid_search_for_directional_accuracy(X_train_scaled, y_train, X_test_scaled, y_test, scaler, parameter_grid):
    best_directional_accuracy = 0
    best_params = {'C': None, 'gamma': None}
    
    for C in parameter_grid['C']:
        for gamma in parameter_grid['gamma']:
            # Train the model
            model = train_model(X_train_scaled, y_train, C, gamma)
            
            # Make predictions on the test set
            predictions = model.predict(X_test_scaled)
            
            # Calculate directional accuracy
            directional_accuracy = calculate_directional_accuracy(y_test, predictions)
            
            # Update best parameters if current model has higher directional accuracy
            if directional_accuracy > best_directional_accuracy:
                best_directional_accuracy = directional_accuracy
                best_params['C'] = C
                best_params['gamma'] = gamma

    return best_params, best_directional_accuracy


# Example usage:
# 1. Download stock data
df = download_stock_data('AAPL', '2019-01-01', '2023-01-01')

# 2. Preprocess data
df = preprocess_data(df)

# 3. Prepare features
X, y = prepare_features(df)

# 4. Split data
X_train, X_test, y_train, y_test = split_data(X, y)

# 5. Scale features
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# Now that all prerequisites are defined, you can define your parameter grid and perform the grid search.
# 6. Define your parameter grid for C and gamma
# Generating logarithmic range for C values around 1000 (10^3)
C_values = np.logspace(2, 4, 10)  # 10 values between 10^2 and 10^4

# Generating linear range for gamma values around 0.01
gamma_values = np.linspace(0.001, 0.1, 10)  # 10 values between 0.001 and 0.1

parameter_grid = {
    'C': C_values,
    'gamma': gamma_values
}

# 7. Perform grid search to find the best C and gamma values
best_params_mse, best_error_mse = grid_search_C_gamma_min_error(X_train_scaled, y_train, X_test_scaled, y_test, scaler, parameter_grid)
print(f'Best Parameters for MSE: C={best_params_mse["C"]}, gamma={best_params_mse["gamma"]}')
print(f'Best Error (MSE): {best_error_mse}')

# Retrain the model with the best parameters for MSE
model = train_model(X_train_scaled, y_train, best_params_mse['C'], best_params_mse['gamma'])

# Use the plot_predictions function to visualize the results
plot_predictions(df, model, scaler)

# Make predictions on the test set
predictions = model.predict(X_test_scaled)

# Calculate directional accuracy
directional_accuracy = calculate_directional_accuracy(y_test, predictions)

print(f'Directional Accuracy of MSE-based model: {directional_accuracy * 100:.2f}%')


# Example of using the new grid search function
parameter_grid = {
    'C': np.logspace(2, 4, 10),
    'gamma': np.linspace(0.001, 0.1, 10)
}
best_params_direction, best_directional_accuracy = grid_search_for_directional_accuracy(X_train_scaled, y_train, X_test_scaled, y_test, scaler, parameter_grid)

print(f'Best Parameters: C={best_params_direction["C"]}, gamma={best_params_direction["gamma"]}')
print(f'Best Directional Accuracy: {best_directional_accuracy * 100:.2f}%')

# Retrain the model with the best parameters for directional accuracy
model = train_model(X_train_scaled, y_train, best_params_direction['C'], best_params_direction['gamma'])

# Use the plot_predictions function to visualize the results
plot_predictions(df, model, scaler)
