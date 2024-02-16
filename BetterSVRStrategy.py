import yfinance as yf
import backtrader as bt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. Fetch the stock data
data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')

# 2. Compute the MACD and RSI
class IndicatorsStrategy(bt.Strategy):
    params = (
        ("short_period", 12),
        ("long_period", 26),
        ("signal_period", 9),
        ("printlog", False),
    )
    
    def __init__(self):
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.short_period,
            period_me2=self.params.long_period,
            period_signal=self.params.signal_period,
        )
        self.rsi = bt.indicators.RSI(self.data.close)
        self.macd_diff_values = []
        self.rsi_values = []

    def next(self):
        self.macd_diff_values.append(self.macd.macd[0] - self.macd.signal[0])
        self.rsi_values.append(self.rsi[0])

cerebro = bt.Cerebro()
feed = bt.feeds.PandasData(dataname=data)
cerebro.adddata(feed)
cerebro.addstrategy(IndicatorsStrategy)
results = cerebro.run()
data['macd_diff'] = [np.nan] * (len(data) - len(results[0].macd_diff_values)) + results[0].macd_diff_values
data['RSI'] = [np.nan] * (len(data) - len(results[0].rsi_values)) + results[0].rsi_values

# 3. Prepare the data

# Drop rows with NaN values
data = data.dropna()

# Prepare the dataset using MACD difference and RSI
n = 5
X = [data[['macd_diff', 'RSI']].values[i-n:i].reshape(-1).tolist() for i in range(n, len(data))]
y = data['Close'].values[n:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train the SVM model with 'rbf' kernel
model = SVR(kernel='rbf', C=1e3, gamma=0.1)
model.fit(X_train, y_train)

# 5. Predict the stock price
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 6. Add SVM Predictions to the dataframe
data['SVM Predictions'] = np.concatenate((np.array([np.nan]*n), y_train, y_pred))

# Use this strategy to plot the predictions
class SVM_Predictions(bt.Indicator):
    lines = ('predictions',)

class SVMPredictStrategy(bt.Strategy):

    def __init__(self):
        # Define SVM predictions line to be plotted
        self.svm_predictions = SVM_Predictions(self.data)

        # Fetch the SVM predictions
        self.predicted_data = data['SVM Predictions'].values.tolist()

    def next(self):
        self.svm_predictions.lines.predictions[0] = self.predicted_data[len(self.data) - 1]

cerebro = bt.Cerebro()
feed = bt.feeds.PandasData(dataname=data)
cerebro.adddata(feed)
cerebro.addstrategy(SVMPredictStrategy)
cerebro.run()
cerebro.plot(style='candle', title='SVM Predictions vs Actual Prices')