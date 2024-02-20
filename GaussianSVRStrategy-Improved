import yfinance as yf
import backtrader as bt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def download_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

def preprocess_data(df):
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

def prepare_features(df):
    X = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', '50_MA']][:-1].values
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
        self.data_predicted = self.params.model.predict(self.params.scaler.transform(df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', '50_MA']].values[-250:]))
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
    X = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', '50_MA']].values[-250:]
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
            
            # Calculate mean absolute error (MAE)
            error = mean_absolute_error(y_test, predictions)
            
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

# Example usage:
# 1. Download stock data
df = download_stock_data('MSFT', '2020-01-01', '2023-01-01')

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
best_params, best_error = grid_search_C_gamma_min_error(X_train_scaled, y_train, X_test_scaled, y_test, scaler, parameter_grid)
print(f'Best Parameters: C={best_params["C"]}, gamma={best_params["gamma"]}')
print(f'Best Error (MAE): {best_error}')

# Retrain the model with the best parameters
model = train_model(X_train_scaled, y_train, best_params['C'], best_params['gamma'])

# Use the plot_predictions function to visualize the results
plot_predictions(df, model, scaler)

# Assuming best_params are found and the model is retrained
model = train_model(X_train_scaled, y_train, best_params['C'], best_params['gamma'])

# Make predictions on the test set
predictions = model.predict(X_test_scaled)

# Calculate directional accuracy
directional_accuracy = calculate_directional_accuracy(y_test, predictions)

print(f'Directional Accuracy: {directional_accuracy * 100:.2f}%')