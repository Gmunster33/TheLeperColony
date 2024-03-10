import yfinance as yf
import backtrader as bt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
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
    "INSG",  # Inseego Corp
    "APPS",  # Digital Turbine, Inc.
    "PLUG",  # Plug Power Inc
    "EXTR",  # Extreme Networks, Inc.
    "CRNC",  # Cerence Inc
    "RMBS",  # Rambus Inc.
    # "SWIR",  # Sierra Wireless, Inc. delisted
    "ATEN",  # A10 Networks, Inc.
    # "NPTN",  # NeoPhotonics Corporation delisted
    "LSCC",  # Lattice Semiconductor Corporation
    "AMBA",  # Ambarella, Inc.
    "ITRI",  # Itron, Inc.
    "FORM",  # FormFactor, Inc.
    "COHU",  # Cohu, Inc.
    "SYNA",  # Synaptics Incorporated
    "LASR",  # nLIGHT, Inc.
    "LITE",  # Lumentum Holdings Inc.
    # "IIVI",  # II-VI Incorporated delisted
    # "QUOT",  # Quotient Technology Inc. delisted
    "SLAB",  # Silicon Laboratories Inc.
    "RPD",   # Rapid7, Inc.
    # "MIME",  # Mimecast Limited delisted
    "BOX",   # Box, Inc.
    "EVBG",  # Everbridge, Inc.
    "PD",    # PagerDuty, Inc.
    "SMAR",  # Smartsheet Inc.
    "TWLO",  # Twilio Inc.
    "NTNX",  # Nutanix, Inc.
    "PSTG",  # Pure Storage, Inc. 
    # "PFPT",  # Proofpoint, Inc. delisted
    "ESTC",  # Elastic N.V.
    "NET",   # Cloudflare, Inc.
    "CRWD",  # CrowdStrike Holdings, Inc.
    "OKTA",  # Okta, Inc.
    "ZS",    # Zscaler, Inc.
    # "SUMO",  # Sumo Logic, Inc. delisted
    # "SAIL",  # SailPoint Technologies Holdings, Inc.
    "VRNS",  # Varonis Systems, Inc.
    "TENB",  # Tenable Holdings, Inc.
    "CASA",  # Casa Systems, Inc.
    "CALX",  # Calix, Inc.
    "COMM",  # CommScope Holding Company, Inc.
    "CIEN",  # Ciena Corporation
    "ADTN",  # ADTRAN, Inc.
    "AKAM",  # Akamai Technologies, Inc.
    "ALLT",  # Allot Communications Ltd.
    "ANET",  # Arista Networks, Inc.
    "FFIV",  # F5 Networks, Inc.
    "JNPR",  # Juniper Networks, Inc.
    "NTGR",  # Netgear, Inc.
    "NTCT",  # NETSCOUT Systems, Inc.
    "RDWR",  # Radware Ltd.
    # "RVBD",  # Riverbed Technology, Inc. delisted
    "SWI",   # SolarWinds Corporation
    # "SONS",  # Sonus Networks, Inc. delisted
    "SPT",   # Spirent Communications plc
    "VIAV",  # Viavi Solutions Inc.
    # "WGRD",  # WatchGuard Technologies, Inc.
    # "ZYXEL", # Zyxel Communications Corp.
    "FTNT",  # Fortinet, Inc.
    "PANW",  # Palo Alto Networks, Inc.
    "CHKP",  # Check Point Software Technologies Ltd.
    "CYBR",  # CyberArk Software Ltd.
    # "FEYE",  # FireEye, Inc.
    "CUDA",  # Barracuda Networks, Inc.
    "SOPH",  # Sophos Group plc
    # "KASP",  # Kaspersky Lab delisted
    # "TMIC",  # Trend Micro Incorporated
    # "MCFE",  # McAfee Corp.
    # "CBLK",  # Carbon Black, Inc.
    "CYL",   # Cylance Inc.
    # "S",     # SentinelOne, Inc.
    # "BITD",  # Bitdefender
    # "ESET",  # ESET, spol. s r.o.
    # "FSC1V", # F-Secure Corporation
    # "MBAM",  # Malwarebytes Inc.
    # "PANDA", # Panda Security, S.L.
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
    """
    Adjusts technical indicators to be more illuminating for day trading,
    focusing on whether the price of the stocks will rise or fall.
    """
    # Bollinger Bands: Remain unchanged as they're useful for spotting volatility and price extremes.
    indicator_bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()

    # MACD: Shortened the signal line for faster response. This can help identify trend changes more quickly.
    indicator_macd = ta.trend.MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = indicator_macd.macd_diff()  # Using MACD diff to get the difference between MACD and signal lines

    # RSI: Window remains as it balances sensitivity with avoiding too much noise.
    df['rsi'] = ta.momentum.rsi(close=df["Close"], window=14)

    # APO: No change needed, as it's already suited for identifying short-term price momentum.
    fast_ema = ta.trend.ema_indicator(close=df["Close"], window=12)
    slow_ema = ta.trend.ema_indicator(close=df["Close"], window=26)
    df['apo'] = fast_ema - slow_ema

    # CCI: Reduced window to make it more responsive to daily price movements.
    df['cci'] = ta.trend.cci(df["High"], df["Low"], df["Close"], window=14)

    # Chaikin A/D Line: Useful for day trading as it combines price and volume to show buying/selling pressure.
    df['ad'] = ta.volume.ChaikinMoneyFlowIndicator(df["High"], df["Low"], df["Close"], df["Volume"], window=3).chaikin_money_flow()

    # ADX: Lowered the window to make the indicator more responsive to short-term trends.
    df['adx'] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=10)

    # Stochastic Oscillator: No change as it's already suitable for spotting overbought/oversold conditions in a short term.
    df['stoch'] = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_window=3).stoch()

    # OBV: Remains unchanged, as volume trends can be a good indicator of price movements.
    df['obv'] = ta.volume.on_balance_volume(df["Close"], df["Volume"])

    # Correct initialization of AroonIndicator
    aroon_indicator = ta.trend.AroonIndicator(high=df["High"], low=df["Low"], window=25)
    df['aroon_up'] = aroon_indicator.aroon_up()
    df['aroon_down'] = aroon_indicator.aroon_down()
    df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    return df

def preprocess_data(df):
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Drop rows with NaN values that might be created by moving averages and indicators
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

def prepare_features_sequences(df, sequence_length=5):
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
        'bb_bbm', 'bb_bbh', 'bb_bbl', 'macd', 'rsi', 
        'apo', 'cci', 'ad', 'adx', 'stoch', 'obv', 'aroon_up',
        'aroon_down', 'vwap' # Include new indicators here
    ]    
    sequences = []
    labels = []
    tickers = []  # To keep track of the ticker for each sequence
    
    grouped = df.groupby('Ticker')

    for ticker, group in grouped:
        data = group[feature_columns].values

        for i in range(len(data) - sequence_length):
            sequence = data[i:i+sequence_length]
            label = group['Close'].iloc[i+sequence_length]
            sequences.append(sequence)
            labels.append(label)
            tickers.append(ticker)  # Add the ticker for the current sequence
    
    return np.array(sequences), np.array(labels), np.array(tickers)


def draw_random_samples(X, y, num_samples, tickers):
    indices = np.random.choice(np.arange(len(X)), size=num_samples, replace=False)
    X_sampled = X[indices]
    y_sampled = y[indices]
    tickers_sampled = tickers[indices]
    return X_sampled, y_sampled, tickers_sampled

# Splits BEFORE drawing random samples, ensuring the test set simulates future unseen data
def split_data(sequences, labels, tickers, train_ratio=0.8, test_size=250):
    total_size = len(sequences)
    test_start_index = total_size - test_size
    
    X_train = sequences[:test_start_index]
    y_train = labels[:test_start_index]
    tickers_train = tickers[:test_start_index]
    
    unique_tickers = np.unique(tickers)

    X_test_arrays = []
    y_test_arrays = []
    tickers_test_arrays = []

    #append the last test_size sequences, labels, and tickers to the test set of each
    i = 0
    for ticker in unique_tickers:
        X_test_arrays.append([])  # Initialize a list for each ticker
        y_test_arrays.append([])
        tickers_test_arrays.append([])
        ticker_indices = np.where(tickers == ticker)[0] # Get indices of the current ticker
        ticker_test_indices = ticker_indices[-test_size:] # Get the last test_size indices for the current ticker
        X_test_arrays[i] = sequences[ticker_test_indices]
        y_test_arrays[i] = labels[ticker_test_indices]
        tickers_test_arrays[i] = (tickers[ticker_test_indices])
        i+=1
    
    return X_train, X_test_arrays, y_train, y_test_arrays, tickers_train, tickers_test_arrays

def train_model(X_train, y_train, C, gamma):
    # No need to reshape X_train here as it should already be 2D after scaling
    # Just directly use X_train for fitting the model
    model = SVR(kernel='rbf', C=C, gamma=gamma)
    model.fit(X_train, y_train)  # Use the already 2D scaled X_train
    return model

# Adjust the plot_predictions function if necessary to accommodate the testing data
def plot_predictions(actual_prices, predicted_prices, dates, ticker=''):
    plt.figure(figsize=(15, 5))
    plt.plot(dates, actual_prices, label='Actual Prices')
    plt.plot(dates, predicted_prices, label='Predicted Prices', alpha=0.7)
    plt.title(f'Stock Price Prediction for {ticker}')
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
    search_spaces = {'C': Real(1e+4, 3e+6, prior='log-uniform'),
                     'gamma': Real(1e-7, 1e-4, prior='log-uniform')}
    
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

# Test usage:#############################################################################
# Parameters specified by Bayesian optimization round 2 with 1000 samples
C_param = 200000.0
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
combined_df = download_and_preprocess_data(extended_tickers, '2015-01-01', '2023-01-01')

test_df = download_and_preprocess_data(test_ticker, '2008-01-01', '2018-12-31')

# Prepare sequenced features and labels
X, y, tickers = prepare_features_sequences(combined_df, sequence_length=5)
X1, y1, tickers1 = prepare_features_sequences(test_df, sequence_length=5)

# Use the adjusted split_data function before drawing random samples
X_train, X_test_arrays, y_train, y_test_arrays, tickers_train, tickers_test_arrays = split_data(X, y, tickers, train_ratio=0.8, test_size=500)

# Now, apply draw_random_samples only to the training data to ensure diversity
X_train_sampled, y_train_sampled, tickers_sampled = draw_random_samples(X_train, y_train, num_samples, tickers_train)


# Ensure the data passed to the SVR model is appropriately reshaped
nsamples, nx, ny = X_train_sampled.shape
X_train_reshaped = X_train_sampled.reshape((nsamples, nx*ny))
# X_test_reshaped = X_test.reshape((X_test.shape[0], nx*ny))

X1_reshaped = X1.reshape((X1.shape[0], nx*ny))

# Scale features after reshaping
standardScaler = StandardScaler()
minMaxScaler = MinMaxScaler()
robustScaler = RobustScaler()

X_train_scaled_standard = standardScaler.fit_transform(X_train_reshaped)
X_train_scaled_minmax = minMaxScaler.fit_transform(X_train_reshaped)
X_train_scaled_robust = robustScaler.fit_transform(X_train_reshaped)

# print the dimensions of X_train_sampled
print(f'X_train_sampled dimensions are: {X_train_sampled.shape}')

# print dimensions of X_train_scaled
print(f'X_train_scaled_standard dimensions are: {X_train_scaled_standard.shape}')

# Output X_train_scaled to a csv file
# np.savetxt('Bigger-X_train_scaled.csv', X_train_scaled, delimiter=',')

# X_test_scaled = scaler.transform(X_test_reshaped)

X1_scaled = standardScaler.transform(X1_reshaped)

# Assuming X_train_scaled_standard and y_train_sampled are available from the previous steps
print("Performing Bayesian Optimization for Standard Scaler")
best_params_standard = perform_bayesian_optimization(X_train_scaled_standard, y_train_sampled)
print("Best Parameters:", best_params_standard)
C_param_standard = best_params_standard['C']
gamma_param_standard = best_params_standard['gamma']


# Assuming X_train_scaled_minmax and y_train_sampled are available from the previous steps
print("Performing Bayesian Optimization for MinMax Scaler")
best_params_minmax = perform_bayesian_optimization(X_train_scaled_minmax, y_train_sampled)
print("Best Parameters:", best_params_minmax)
C_param_minmax = best_params_minmax['C']
gamma_param_minmax = best_params_minmax['gamma']


# Assuming X_train_scaled_robust and y_train_sampled are available from the previous steps
print("Performing Bayesian Optimization for Robust Scaler")
best_params_robust = perform_bayesian_optimization(X_train_scaled_robust, y_train_sampled)
print("Best Parameters:", best_params_robust)
C_param_robust = best_params_robust['C']
gamma_param_robust = best_params_robust['gamma']


# Train the SVR models with reshaped and scaled training data
standardModel = train_model(X_train_scaled_standard, y_train_sampled, C=C_param_standard, gamma=gamma_param_standard)
minMaxModel = train_model(X_train_scaled_minmax, y_train_sampled, C=C_param_minmax, gamma=gamma_param_minmax)
robustModel = train_model(X_train_scaled_robust, y_train_sampled, C=C_param_robust, gamma=gamma_param_robust)

# Iterate through each ticker's data in the test sets
for i, X_test in enumerate(X_test_arrays):
    # Reshape each ticker's test data and scale it using the same scaler
    X_test_reshaped = X_test.reshape((X_test.shape[0], nx*ny))
    X_test_scaled_standard = standardScaler.transform(X_test_reshaped)  # Assuming scaler is already fitted to the training data
    X_test_scaled_minmax = minMaxScaler.transform(X_test_reshaped)
    X_test_scaled_robust = robustScaler.transform(X_test_reshaped)
    
    # Make predictions for the current ticker's data
    predictions_standard = standardModel.predict(X_test_scaled_standard)
    predictions_minmax = minMaxModel.predict(X_test_scaled_minmax)
    predictions_robust = robustModel.predict(X_test_scaled_robust)
    
    # Get the corresponding y_test data for the current ticker
    current_y_test = y_test_arrays[i]
    
    # Calculate evaluation metrics for the current ticker
    mse_error_standard = mean_squared_error(current_y_test, predictions_standard)
    mae_error_standard = mean_absolute_error(current_y_test, predictions_standard)
    directional_accuracy_standard = calculate_directional_accuracy(current_y_test, predictions_standard)

    mse_error_minmax = mean_squared_error(current_y_test, predictions_minmax)
    mae_error_minmax = mean_absolute_error(current_y_test, predictions_minmax)
    directional_accuracy_minmax = calculate_directional_accuracy(current_y_test, predictions_minmax)

    mse_error_robust = mean_squared_error(current_y_test, predictions_robust)
    mae_error_robust = mean_absolute_error(current_y_test, predictions_robust)
    directional_accuracy_robust = calculate_directional_accuracy(current_y_test, predictions_robust)

    # Print evaluation metrics
    print(f"{tickers_test_arrays[i][0]} - MSE: {mse_error_standard:.2f}, MAE: {mae_error_standard:.2f}, DA: {directional_accuracy_standard}% -- Standard Scaler")  
    print(f"{tickers_test_arrays[i][0]} - MSE: {mse_error_minmax:.2f}, MAE: {mae_error_minmax:.2f}, DA: {directional_accuracy_minmax}% -- MinMax Scaler")
    print(f"{tickers_test_arrays[i][0]} - MSE: {mse_error_robust:.2f}, MAE: {mae_error_robust:.2f}, DA: {directional_accuracy_robust}% -- Robust Scaler")

    # Plot predictions for '2023-01-01' minus the test size to '2023-01-01'
    # plot_predictions(current_y_test, predictions, combined_df.index[-len(current_y_test):], ticker=tickers_test_arrays[i][0])
