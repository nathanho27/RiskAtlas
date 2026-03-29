"""
Risk Atlas - Stock Load

Downloads historical OHLCV price data for selected equities and
a market benchmark using the yfinance API.

Output:
data/raw/market_prices.csv
"""

import os
import yfinance as yf
import pandas as pd


TICKERS = [
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN",
    "V", "MA", "JPM", "XOM", "SPY"
]

START_DATE = "2015-01-01"
OUTPUT_PATH = "data/raw/market_prices.csv"


def download_stock_data():

    data = yf.download(
        TICKERS,
        start=START_DATE,
        auto_adjust=True,
        progress=False
    )

    data = data.stack(level=0).rename_axis(["Date", "Ticker"]).reset_index()

    return data


def save_data(df):

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)


def main():

    print("Downloading stock data...")

    df = download_stock_data()

    print(f"Dataset shape: {df.shape}")

    print("Saving dataset...")

    save_data(df)

    print("Stock data saved successfully.")


if __name__ == "__main__":
    main()