import yfinance as yf
import backtrader as bt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd  # Added to use pd.to_datetime


# Fetch historical data
ticker = 'MSFT'
df = yf.download(ticker, start='2020-01-01', end='2023-01-01')

# Compute a 50-day moving average as a basic technical indicator
df['50_MA'] = df['Close'].rolling(window=50).mean()

# Drop the rows with NaN values (due to the rolling mean)
df = df.dropna()

X = df[['Open','High','Low','Close','Adj Close','Volume','50_MA']][:-1].values  # drop the last row
y = df['Close'].shift(-1).dropna().values

X_train = X[:-250]  # Just an example split, adjust as needed
y_train = y[:-250]

X_test = X[-250:]
y_test = y[-250:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = SVR(kernel='rbf', C=278.2559402207126, gamma=0.05600000000000001)
model.fit(X_train, y_train)

df.index = pd.to_datetime(df.index)

data = bt.feeds.PandasData(dataname=df[-250:])

class SVRStrategy(bt.Strategy):
    def __init__(self):
        # self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=15)
        self.data_predicted = model.predict(scaler.transform(df[['Open','High','Low','Close','Adj Close','Volume','50_MA']].values[-250:]))
        self.threshold = 0  # 0.5% threshold for trades. Adjust as needed.
        self.iterations = 0
        print("Init called")


    def next(self):
        current_date = self.datas[0].datetime.datetime(0)
        # Find the index of the current date in the dates list
        current_index = self.iterations
        predicted_price = self.data_predicted[current_index]
        self.iterations += 1

        # If predicted price is significantly higher, buy. If significantly lower, sell.
        if predicted_price > self.data.close[0] * (1 + self.threshold):
            self.buy()
            print("Buy")
        elif predicted_price < self.data.close[0] * (1 - self.threshold):
            self.sell()
            print("Sell")

# Create a cerebro entity and add strategy
cerebro = bt.Cerebro()
cerebro.addstrategy(SVRStrategy)

# Set our desired cash start
cerebro.broker.set_cash(10000.0)

# Add the data feed
cerebro.adddata(data)

# Set the commission
cerebro.broker.setcommission(commission=0)
# Print out the starting conditions
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Run the backtest
cerebro.run()

# Print out the final result
print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Use Cerebro's plotting capabilities
cerebro.plot()

# Finally, visualize the result
# Extract the data
dates = df.index[-249:].tolist() 
prices = df['Close'].values[-249:]
predicted_prices = model.predict(scaler.transform(df[['Open','High','Low','Close','Adj Close','Volume','50_MA']].values[-250:-1]))

# Plot using matplotlib
plt.figure(figsize=(15, 5))
plt.plot(dates, prices, label='True Prices')
plt.plot(dates, predicted_prices, label='Predicted Prices', alpha=0.7)
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


# Extract the data for plotting
dates_train = df.index[:-249].tolist()
prices_train = df['Close'].values[:-249]
ma_train = df['50_MA'].values[:-249]

dates_test = df.index[-249:].tolist()
prices_test = df['Close'].values[-249:]
predicted_prices = model.predict(scaler.transform(df[['Open','High','Low','Close','Adj Close','Volume','50_MA']].values[-250:-1]))

# Plot
plt.figure(figsize=(15, 7))

# Plotting the training data
plt.plot(dates_train, prices_train, label='Train Prices', color='blue')
plt.plot(dates_train, ma_train, label='Train 50-day MA', color='blue', alpha=0.5)

# Plotting the test data
plt.plot(dates_test, prices_test, label='True Test Prices', color='green')
plt.plot(dates_test, predicted_prices, label='Predicted Prices', color='red', alpha=0.7)

plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

differences = prices[-249:] - predicted_prices[-1:]
print(differences)