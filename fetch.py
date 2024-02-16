import yfinance as yf

data = yf.download("AAPL", start="2010-01-01", end="2023-01-01")
data.to_csv("AAPL.csv")