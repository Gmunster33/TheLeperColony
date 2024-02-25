import yfinance as yf
import backtrader as bt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import ta
 
# TYLER IS seggsy and smort

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

    # Manually calculate the APO based on EMAs
    fast_ema = ta.trend.ema_indicator(close=df["Close"], window=12)
    slow_ema = ta.trend.ema_indicator(close=df["Close"], window=26)
    df['apo'] = fast_ema - slow_ema

    # Commodity Channel Index (CCI)
    df['cci'] = ta.trend.cci(df["High"], df["Low"], df["Close"], window=20)

    # Chaikin A/D Line - Use ChaikinMoneyFlow with window=1 for similar effect
    df['ad'] = ta.volume.ChaikinMoneyFlowIndicator(df["High"], df["Low"], df["Close"], df["Volume"], window=1).chaikin_money_flow()

    # Average Directional Index (ADX)
    df['adx'] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)

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
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
        '50_MA', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'macd', 'rsi', 
        'apo', 'cci', 'ad', 'adx', 'stoch', 'obv'  # Include new indicators here
    ]
    X = df[feature_columns].values  # Extract feature values
    y = df['Close'].shift(-1).dropna().values  # Target values (next day's close)
    return X, y

def prepare_features_sequences(df, sequence_length=5):
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
        '50_MA', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'macd', 'rsi', 
        'apo', 'cci', 'ad', 'adx', 'stoch', 'obv'  # Include new indicators here
    ]    
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
            label = group['Close'].iloc[i+sequence_length] # list of closing prices for the day after the last day in the sequence
            sequences.append(sequence)
            labels.append(label)

        print(f"Generated {len(sequences)} sequences and {len(labels)} labels after processing ticker: {ticker}")  # Debugging print
    
    print("Completed processing all tickers.")
    print(f"Total sequences: {len(sequences)}")
    print(f"Total labels: {len(labels)}")
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

# Define a custom scorer function for directional accuracy
def directional_accuracy_scorer(y_true, y_pred):
    actual_direction = np.sign(y_true[1:] - y_true[:-1])
    predicted_direction = np.sign(y_pred[1:] - y_true[:-1])
    accuracy = np.mean(actual_direction == predicted_direction)
    return accuracy

# Convert the custom scorer into a scorer function that can be used in BayesSearchCV
directional_accuracy = make_scorer(directional_accuracy_scorer, greater_is_better=True)

def perform_bayesian_optimization(X_train, y_train):
    # Define the search space for C and gamma
    search_spaces = {'C': Real(1e-6, 1e+6, prior='log-uniform'),
                     'gamma': Real(1e-6, 1e+1, prior='log-uniform')}
    
    # Initialize the model
    svr = SVR(kernel='rbf')

    # Setup the repeated K-Fold cross-validator
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

    # Initialize the Bayesian Search
    bayes_search = BayesSearchCV(estimator=svr,
                                 search_spaces=search_spaces,
                                 scoring=directional_accuracy,
                                 cv=cv,
                                 n_iter=32,
                                 n_jobs=-1,
                                 return_train_score=True,
                                 random_state=0,
                                 verbose=3  # Verbose mode
                                 )

    # Perform the search
    bayes_search.fit(X_train, y_train)

    # Print the best score found
    print(f"Best Directional Accuracy: {bayes_search.best_score_* 100:.2f}%")

    # Return the best parameters
    return bayes_search.best_params_

# Test usage:
# Parameters specified by Bayesian optimization round 2 with 1000 samples
C_param = 1000000.0
gamma_param = 2.9788804908841073e-05

# Parameters specified by Bayesian optimization round 1 with 500 samples
# C_param = 840.5406521370916
# gamma_param = 2.747707867669414e-05
######################
# Parameters specified by grid search
# C_param = 59948.425031894085
# gamma_param = 0.001
num_samples = 500  # Number of random samples for training

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


# Ensure the data passed to the SVR model is appropriately reshaped
nsamples, nx, ny = X_train_sampled.shape
X_train_reshaped = X_train_sampled.reshape((nsamples, nx*ny))
X_test_reshaped = X_test.reshape((X_test.shape[0], nx*ny))

X1_reshaped = X1.reshape((X1.shape[0], nx*ny))

# Scale features after reshaping
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)

X1_scaled = scaler.transform(X1_reshaped)

# Assuming X_train_scaled and y_train_sampled are available from the previous steps
best_params = perform_bayesian_optimization(X_train_scaled, y_train_sampled)
print("Best Parameters:", best_params)

C_param = best_params['C']
gamma_param = best_params['gamma']

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
