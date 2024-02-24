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
 
# TYLER IS GAAYYYYYYY

extended_tickers = [
    "INTC",  # Intel Corporation
    "QCOM",  # Qualcomm Incorporated
    "AVGO",  # Broadcom Inc.
    "NVDA",  # NVIDIA Corporation
    "AMD",   # Advanced Micro Devices, Inc.
    "MU",    # Micron Technology, Inc.
    # "000660.KS", # SK Hynix Inc.
    "TXN",   # Texas Instruments Incorporated
    "AMAT",  # Applied Materials, Inc.
    "ASML",  # ASML Holding N.V.
    "LRCX",  # Lam Research Corporation
    "SONY",  # Sony Group Corporation
    # "2317.TW", # Foxconn (Hon Hai Precision Industry Co., Ltd.)
    "GLW",   # Corning Incorporated
    "TSM",   # Taiwan Semiconductor Manufacturing Company (TSMC)
    # "005930.KS", # Samsung Electronics Co., Ltd.
    "WDC",   # Western Digital Corporation
    "STX",   # Seagate Technology Holdings plc
    "KEYS",  # Keysight Technologies
    "KLAC",  # KLA Corporation
    "GOOG",  # Alphabet Inc. (Google)
    "MSFT",  # Microsoft Corporation
    "META",  # Meta Platforms, Inc. (formerly Facebook)
    # "1810.HK", # Xiaomi Corporation
    # "2357.TW", # ASUS (ASUSTeK Computer Inc.)
    # "2353.TW", # Acer Inc.
    "DELL",  # Dell Technologies Inc.
    "HPQ",   # HP Inc.
    # "0992.HK", # Lenovo Group Limited
    "ADBE",  # Adobe Inc.
    "CRM",   # Salesforce
    "ORCL",  # Oracle Corporation
    "IBM",   # International Business Machines Corporation
    "CSCO",  # Cisco Systems, Inc.
    "SAP",   # SAP SE
    "INTU",  # Intuit Inc.
    # "VMW",   # VMware, Inc.
    "SQ",    # Block, Inc. (formerly Square, Inc.)
    "SHOP",  # Shopify Inc.
    # "TWTR",  # Twitter, Inc. (Note: As of my last update in April 2023, Twitter was taken private by Elon Musk, so this might not be current.)
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
    "AMZN",  # Amazon.com, Inc.
    "ZM",    # Zoom Video Communications, Inc.
    "UBER",  # Uber Technologies, Inc.
    "LYFT"   # Lyft, Inc.
]
test_ticker = ['AAPL']


def download_and_preprocess_data(tickers, start_date, end_date):
    all_data = []
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date)
        print(f'Downloaded {ticker} data from {start_date} to {end_date}')
        df = preprocess_data(df)
        df['Ticker'] = ticker  # Add ticker identifier
        all_data.append(df)
    combined_df = pd.concat(all_data)
    return combined_df

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

    # Absolute Price Oscillator (APO)
    df['apo'] = ta.trend.APO(df["Close"], fast=12, slow=26, matype=0)  # SMA version
    df['apo_ema'] = ta.trend.APO(df["Close"], fast=12, slow=26, matype=1)  # EMA version

    # Commodity Channel Index (CCI)
    df['cci'] = ta.trend.CCI(df["High"], df["Low"], df["Close"], window=20)

    # Chaikin A/D Line
    df['ad'] = ta.volume.ChaikinMoneyFlowIndicator(df["High"], df["Low"], df["Close"], df["Volume"], window=20).chaikin_money_flow()

    # Average Directional Index (ADX)
    df['adx'] = ta.trend.ADX(df["High"], df["Low"], df["Close"], window=14)

    # Stochastic Oscillator (STOCH)
    stoch_indicator = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_window=3)
    df['stoch'] = stoch_indicator.stoch()

    # On Balance Volume (OBV)
    df['obv'] = ta.volume.on_balance_volume(df["Close"], df["Volume"])

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

def prepare_features_sequences(df, sequence_length=5):
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', '50_MA', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'macd', 'rsi']
    sequences = []
    labels = []
    
    # Group by ticker to ensure sequences are ticker-specific
    grouped = df.groupby('Ticker')
    print(f"Total tickers found: {len(grouped)}")  # Debugging print

    for ticker, group in grouped:
        data = group[feature_columns].values
        print(f"Processing ticker: {ticker} with data length: {len(data)}")  # Debugging print

        for i in range(len(data) - sequence_length):
            sequence = data[i:i+sequence_length]
            label = group['Close'].iloc[i+sequence_length]
            sequences.append(sequence)
            labels.append(label)

            if i == 0:
                print(f"First sequence: {sequence}")
                print(f"First label: {label}")
                print(f"!!!!group['Close'][i:i+sequence_length+1]: {group['Close'][i:i+sequence_length+1]}")


        print(f"Generated {len(sequences)} sequences and {len(labels)} labels for ticker: {ticker}")  # Debugging print
    
    print("Completed processing all tickers.")
    print(f"Total sequences[0].shape: {sequences[0].shape}")
    print(f"Total labels[0]: {labels[0]}")
    return np.array(sequences), np.array(labels)

def draw_random_samples(X, y, num_samples):
    indices = np.random.choice(np.arange(len(X)), size=num_samples, replace=False)
    X_sampled = X[indices]
    y_sampled = y[indices]
    return X_sampled, y_sampled

# Adjust the split_data function to split before drawing random samples, ensuring the test set simulates future unseen data
def split_data(X, y, train_ratio=0.8, sequence_length=5, test_size=250):
    """
    Adjusted to ensure the test set is the most recent data, simulating future unseen data.
    The function now first separates a portion of data for testing, then draws random samples from the remaining data for training.
    """
    # Calculate the index to start the test set
    test_start_index = len(X) - test_size
    
    # Separate the test set
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    # Use the remaining data for training
    X_train = X[:test_start_index]
    y_train = y[:test_start_index]
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train, C, gamma):
    # No need to reshape X_train here as it should already be 2D after scaling
    # Just directly use X_train for fitting the model
    model = SVR(kernel='rbf', C=C, gamma=gamma)
    model.fit(X_train, y_train)  # Use the already 2D scaled X_train
    return model

# Adjust the plot_predictions function if necessary to accommodate the testing data
def plot_predictions(actual_prices, predicted_prices, dates):
    plt.figure(figsize=(15, 5))
    plt.plot(dates, actual_prices, label='Actual Prices')
    plt.plot(dates, predicted_prices, label='Predicted Prices', alpha=0.7)
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

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

# Test usage:

# Parameters specified
C_param = 59948.425031894085
gamma_param = 0.001
num_samples = 10000  # Number of random samples for training

# Download and preprocess data from all tickers
combined_df = download_and_preprocess_data(extended_tickers, '2012-01-01', '2023-01-01')

test_df = download_and_preprocess_data(test_ticker, '2012-01-01', '2023-12-31')

# Prepare sequenced features and labels
X, y = prepare_features_sequences(combined_df, sequence_length=5)

X1, y1 = prepare_features_sequences(test_df, sequence_length=5)

# Use the adjusted split_data function before drawing random samples
X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8, sequence_length=5, test_size=500)

# Now, apply draw_random_samples only to the training data to ensure diversity
X_train_sampled, y_train_sampled = draw_random_samples(X_train, y_train, num_samples)

print(f'X_train_sampled: {X_train_sampled}')
print(f'X_train_sampled.shape: {X_train_sampled.shape}')

# Ensure the data passed to the SVR model is appropriately reshaped
nsamples, nx, ny = X_train_sampled.shape
X_train_reshaped = X_train_sampled.reshape((nsamples, nx*ny))
X_test_reshaped = X_test.reshape((X_test.shape[0], nx*ny))

X1_reshaped = X1.reshape((X1.shape[0], nx*ny))

print(f'X_train_reshaped: {X_train_reshaped}')
print(f'X_train_reshaped.shape: {X_train_reshaped.shape}')
print(f'X_test_reshaped: {X_test_reshaped}')
print(f'X_test_reshaped.shape: {X_test_reshaped.shape}')
# Scale features after reshaping
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)

X1_scaled = scaler.transform(X1_reshaped)

print(f'X_train_scaled: {X_train_scaled}')
print(f'X_train_scaled.shape: {X_train_scaled.shape}')
print(f'X_test_scaled: {X_test_scaled}')
print(f'X_test_scaled.shape: {X_test_scaled.shape}')

# Train the SVR model with reshaped and scaled training data
model = train_model(X_train_scaled, y_train_sampled, C=C_param, gamma=gamma_param)

# Make predictions with the reshaped and scaled test data
predictions = model.predict(X_test_scaled)

predictions1 = model.predict(X1_scaled)

# Evaluate the model
directional_accuracy = calculate_directional_accuracy(y_test, predictions)
mse_error = mean_squared_error(y_test, predictions)
mae_error = mean_absolute_error(y_test, predictions)
print(f'Directional Accuracy: {directional_accuracy * 100:.2f}% - MSE: {mse_error:.2f} - MAE: {mae_error:.2f}')

directional_accuracy1 = calculate_directional_accuracy(y1, predictions1)
mse_error1 = mean_squared_error(y1, predictions1)
mae_error1 = mean_absolute_error(y1, predictions1)
print(f'Directional Accuracy: {directional_accuracy1 * 100:.2f}% - MSE: {mse_error1:.2f} - MAE: {mae_error1:.2f}')

# Optional: Plot predictions
plot_predictions(y_test, predictions, combined_df.index[-len(y_test):])

plot_predictions(y1, predictions1, test_df.index[-len(y1):])
