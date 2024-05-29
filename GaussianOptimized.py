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
    "ADTN",  # ADTRAN, Inc. Why does testing start with this stock as opposed to the first stock in the list? 
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
    print(f"tickers length is: {len(tickers)}")
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date)
        print(f'Downloaded {ticker} data from {start_date} to {end_date}')
        df = preprocess_data(df)
        df['Ticker'] = ticker  # Add ticker identifier
        all_data.append(df)
    print(f"all_data length is: {len(all_data)}")
    combined_df = pd.concat(all_data)

    print(f"combined_df.shape is: {combined_df.shape}")

    # The number of tickers in combined_df should be equal to the number of tickers
    print(f"Number of tickers in combined_df: {combined_df['Ticker'].nunique()}")

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
    """
    Prepares sequences of features for time series prediction, including the delta in closing price.
    
    :param df: DataFrame containing stock data with technical indicators.
    :param sequence_length: Length of the feature sequences.
    :return: Sequences of features, labels (delta in closing price), and corresponding tickers.
    """
    # Calculate the delta in closing price from one day to the next
    df['Close_delta'] = df.groupby('Ticker')['Close'].diff().fillna(0)

    # Extend feature_columns to include 'Close_delta'
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
        'bb_bbm', 'bb_bbh', 'bb_bbl', 'macd', 'rsi', 
        'apo', 'cci', 'ad', 'adx', 'stoch', 'obv', 'aroon_up',
        'aroon_down', 'vwap', 'Close_delta'  # Include 'Close_delta'
    ]    
    sequences = []
    returns = []
    tickers = []  # To keep track of the ticker for each sequence
    
    grouped = df.groupby('Ticker')

    for ticker, group in grouped:
        data = group[feature_columns].values

        for i in range(len(data) - sequence_length):
            sequence = data[i:i+sequence_length]
            # Calculate the label as the delta in closing price from the last day in the sequence to the next day
            priceDelta = group['Close'].iloc[i+sequence_length] - group['Close'].iloc[i+sequence_length-1]
            sequences.append(sequence)
            returns.append(priceDelta)
            tickers.append(ticker)  # Add the ticker for the current sequence
    
    return np.array(sequences), np.array(returns), np.array(tickers)

# This refactored function will now correctly prepare feature sequences including the day-to-day price change as a feature,
# and the labels will represent the change in closing price from the last day in the input sequence to the next day.
# This modification requires the initial data preprocessing to be done as before, ensuring all necessary columns are present.
# The rest of the code that handles training and evaluating the model can remain largely the same,
# with the understanding that the target variable has now changed to predicting price changes rather than absolute prices.


def draw_random_samples(X, y, num_samples, tickers):
    indices = np.random.choice(np.arange(len(X)), size=num_samples, replace=False)
    X_sampled = X[indices]
    y_sampled = y[indices]
    tickers_sampled = tickers[indices]
    return X_sampled, y_sampled, tickers_sampled

# Splits BEFORE drawing random samples, ensuring the test set simulates future unseen data
def split_data(sequences, returns, tickers, test_size=250):
    # Get unique tickers to split data by ticker    
    unique_tickers = np.unique(tickers)

    X_train = []
    y_train = []
    tickers_train = []
    X_test_arrays = []
    y_test_arrays = []
    tickers_test_arrays = []

    #append the last test_size sequences, returns, and tickers to the test set of each, while appending the rest to the training set of each
    i = 0
    for ticker in unique_tickers:
        X_test_arrays.append([])  # Initialize a list for each ticker
        y_test_arrays.append([])
        tickers_test_arrays.append([])
        X_train.append([])
        y_train.append([])
        tickers_train.append([])

        ticker_indices = np.where(tickers == ticker)[0] # Get indices of the current ticker 
        ticker_test_indices = ticker_indices[-test_size:] # Get the last test_size indices for the current ticker
        ticker_train_indices = ticker_indices[:-test_size] # Get the training indices for the current ticker
        X_test_arrays[i] = sequences[ticker_test_indices]
        y_test_arrays[i] = returns[ticker_test_indices]
        tickers_test_arrays[i] = (tickers[ticker_test_indices])
        X_train[i] = sequences[ticker_train_indices]
        y_train[i] = returns[ticker_train_indices]
        tickers_train[i] = (tickers[ticker_train_indices])
        i+=1

    print(f"X_test_arrays length: {len(X_test_arrays)}, y_test_arrays length: {len(y_test_arrays)}, tickers_test_arrays length: {len(tickers_test_arrays)}")
    print(f"FINAL X_train length: {len(X_train)}, y_train length: {len(y_train)}, tickers_train shape: {len(tickers_train)}")
    # Convert training lists to arrays as inputs to the draw_random_samples function
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    tickers_train = np.concatenate(tickers_train)
    print(f"FINAL (Correct??) X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, tickers_train shape: {tickers_train.shape}")
    
    return X_train, X_test_arrays, y_train, y_test_arrays, tickers_train, tickers_test_arrays

def train_model(X_train, y_train, C, gamma):
    # No need to reshape X_train here as it should already be 2D after scaling
    # Just directly use X_train for fitting the model
    model = SVR(kernel='rbf', C=C, gamma=gamma)
    model.fit(X_train, y_train)  # Use the already 2D scaled X_train
    return model

# Plot the actual and predicted prices for a given ticker along the testing date range
def plot_predictions(actual_prices, predicted_prices, dates, ticker=''):
    plt.figure(figsize=(15, 5))
    plt.plot(dates, actual_prices, label='Actual Daily Returns')
    plt.plot(dates, predicted_prices, label='Predicted Daily Returns', alpha=0.7)
    plt.title(f'Daily Stock Return Prediction for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the actual and predicted prices for a given ticker along the testing date range
# only in this function, the plot will show each return as a vector representing the direction and magnitude of each day's actual and predicted return
def plot_prediction_scatter(actual_prices, predicted_prices, dates, ticker=''):
    plt.figure(figsize=(15, 5))
    plt.scatter(dates, actual_prices, label='Actual Daily Returns', color='blue')
    plt.scatter(dates, predicted_prices, label='Predicted Daily Returns', color='red')
    plt.title(f'Daily Stock Return Prediction for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
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
    
    y_test: actual next day price changes
    predictions: predicted price changes
    """
    # Ensure predictions are aligned with y_test for direct comparison
    actual_direction = np.sign(y_test[1:] - y_test[:-1]) # (tomorrow's returns - today's returns) = direction of actual change in returns
    predicted_direction = np.sign(predictions[1:] - y_test[:-1]) # (predicted_tomorrow's returns - today's returns) = direction of predicted change in returns

    # actual_direction = np.sign(y_test[:])
    # predicted_direction = np.sign(predictions[:])
    
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

def scale_test_sequences(X_train, X_test_sequences, ticker):
    """
    Scales test sequences sequentially.
    
    X_train: Training data, used to initialize the scaler.
    X_test_sequences: Test sequences, each sequence is scaled based on past information.
    """

    scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))

    X_test_scaled_sequences = []
    for sequence in X_test_sequences:
        # Flatten the sequence to 2D if it's 3D: (sequence_length, num_features)
        sequence_reshaped = sequence.reshape(-1, sequence.shape[-1])
        
        # Scale the sequence
        sequence_scaled = scaler.transform(sequence_reshaped)
        
        # Update scaler with the original sequence data (simulates receiving this sequence in real-time)
        scaler.partial_fit(sequence_reshaped)

        # Append the scaled sequence (optionally reshape back to original shape)
        X_test_scaled_sequences.append(sequence_scaled.reshape(sequence.shape))
        
    return np.array(X_test_scaled_sequences)


# Test usage:#############################################################################

optimize = False # Set to True to perform Bayesian optimization, False to use pre-optimized parameters
num_samples = 1000  # Number of random samples for training

# Download and preprocess data from all tickers
combined_df = download_and_preprocess_data(extended_tickers, '2015-01-01', '2023-01-01')

# Prepare sequenced features and labels
X, y, tickers = prepare_features_sequences(combined_df, sequence_length=5)

# Use the adjusted split_data function before drawing random samples
X_train, X_test_arrays, y_train, y_test_arrays, tickers_train, tickers_test_arrays = split_data(X, y, tickers, test_size=500) # TODO X_test_arrays currently does not have the last 500 days of data from ALL stocks; rather, just ADTN and on in the list above...

# Now, apply draw_random_samples only to the training data to ensure diversity
X_train_sampled, y_train_sampled, tickers_sampled = draw_random_samples(X_train, y_train, num_samples, tickers_train)

# Ensure the data passed to the SVR model is appropriately reshaped
nsamples, nx, ny = X_train_sampled.shape
X_train_reshaped = X_train_sampled.reshape((nsamples, nx*ny))

# print the dimensions of X_train_sampled
print(f'X_train_sampled dimensions are: {X_train_sampled.shape}')

# Scale features after reshaping
# Standard Scaler SVR Model #############################################################################
best_params_standard = {'C' : 468222.5644361616, 'gamma' : 1.2981542464559332e-06}
standardScaler = StandardScaler()
X_train_scaled_standard = standardScaler.fit_transform(X_train_reshaped)
if optimize: # Perform Bayesian optimization and overwrite the best parameters for each scaler hard-coded above
    print("Performing Bayesian Optimization for Standard Scaler")
    best_params_standard = perform_bayesian_optimization(X_train_scaled_standard, y_train_sampled)
    print("Best Standard Parameters:", best_params_standard)
C_param_standard = best_params_standard['C']
gamma_param_standard = best_params_standard['gamma']
print("Using Standard Parameters:", best_params_standard)
standardModel = train_model(X_train_scaled_standard, y_train_sampled, C=C_param_standard, gamma=gamma_param_standard)


# MinMax Scaler SVR Model #############################################################################
best_params_minmax = {'C' : 292636.19156268163, 'gamma' : 5.371764360760396e-05}
minMaxScaler = MinMaxScaler()
X_train_scaled_minmax = minMaxScaler.fit_transform(X_train_reshaped)
if optimize: # Perform Bayesian optimization and overwrite the best parameters for each scaler hard-coded above
    print("Performing Bayesian Optimization for MinMax Scaler")
    best_params_minmax = perform_bayesian_optimization(X_train_scaled_minmax, y_train_sampled)
    print("Best MinMax Parameters:", best_params_minmax)
C_param_minmax = best_params_minmax['C']
gamma_param_minmax = best_params_minmax['gamma']
print("Using MinMax Parameters:", best_params_minmax)
minMaxModel = train_model(X_train_scaled_minmax, y_train_sampled, C=C_param_minmax, gamma=gamma_param_minmax)


# Robust Scaler SVR Model #############################################################################
best_params_robust = {'C' : 26070.948197887286, 'gamma' : 2.6637677914313985e-06}
robustScaler = RobustScaler()
X_train_scaled_robust = robustScaler.fit_transform(X_train_reshaped)
if optimize: # Perform Bayesian optimization and overwrite the best parameters for each scaler hard-coded above
    print("Performing Bayesian Optimization for Robust Scaler")
    best_params_robust = perform_bayesian_optimization(X_train_scaled_robust, y_train_sampled)
    print("Best Robust Parameters:", best_params_robust)
C_param_robust = best_params_robust['C']
gamma_param_robust = best_params_robust['gamma']
print("Using Robust Parameters:", best_params_robust)
robustModel = train_model(X_train_scaled_robust, y_train_sampled, C=C_param_robust, gamma=gamma_param_robust) 

  

# Iterate through each ticker's data in the test sets
for i, X_test in enumerate(X_test_arrays):
    # Reshape each ticker's test data and scale it using the same scaler
    X_test_reshaped = X_test.reshape((X_test.shape[0], nx*ny))
    X_test_scaled_standard = scale_test_sequences(X_train_reshaped, X_test_reshaped, tickers_test_arrays[i][0])
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
    print(f"{tickers_test_arrays[i][0]} - MSE: {mse_error_standard:.2f}, MAE: {mae_error_standard:.2f}, DA: {directional_accuracy_standard*100}% -- Standard Scaler")  
    print(f"{tickers_test_arrays[i][0]} - MSE: {mse_error_minmax:.2f}, MAE: {mae_error_minmax:.2f}, DA: {directional_accuracy_minmax*100}% -- MinMax Scaler")
    print(f"{tickers_test_arrays[i][0]} - MSE: {mse_error_robust:.2f}, MAE: {mae_error_robust:.2f}, DA: {directional_accuracy_robust*100}% -- Robust Scaler")

    # Plot predictions for the current ticker
    # Inputs to the plot_predictions function are: (actual_prices, predicted_prices, dates, ticker='')
    # The dates can be obtained from the combined_df dataframe
    # The actual_prices can be obtained from the current_y_test variable
    # The predicted_prices are the predictions from the model
    # The ticker is the current ticker symbol
    dates = combined_df.loc[combined_df['Ticker'] == tickers_test_arrays[i][0]].index[-500:] # Get the last 500 dates for the current ticker
    plot_predictions(current_y_test, predictions_standard, dates, ticker=tickers_test_arrays[i][0])

    # Plot prediction vectors for the current ticker
    plot_prediction_scatter(current_y_test, predictions_standard, dates, ticker=tickers_test_arrays[i][0])
    # Explaining how we get the dates: combined_df is the dataframe with all the data, and we're using the index of the current ticker's test data by using the .loc method with the ticker symbol (tickers_test_arrays[i][0]) to get the corresponding dates. We're starting from the 5th index to match the sequence length used for training and predictions.