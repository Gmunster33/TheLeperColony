import backtrader as bt

class MovingAverageCrossoverStrategy(bt.Strategy):
    params = (
        ("short_period", 50),
        ("long_period", 200)
    )

    def __init__(self):
        self.short_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.short_period
        )
        self.long_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.long_period
        )
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)

    def next(self):
        if self.crossover > 0:  # If short MA crosses above long MA
            self.buy()
        elif self.crossover < 0:  # If short MA crosses below long MA
            self.sell()


if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MovingAverageCrossoverStrategy)

    # Load data from CSV
    data = bt.feeds.YahooFinanceCSVData(dataname="AAPL.csv")
    cerebro.adddata(data)

    # Set starting cash
    cerebro.broker.set_cash(10000)

    # Set the commission - 0.1% ... divide by 100 to remove the percentage
    cerebro.broker.setcommission(commission=0.001)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Plot the result
    cerebro.plot()