# RiskAtlas
Market risk modeling project that analyzes historical equity data to estimate forward-looking risk signals such as drawdowns and volatility regimes. Uses engineered financial features and machine learning models to generate interpretable risk scores through a reproducible Python workflow and interactive Streamlit dashboards.

## Data Ingestion (Temporary)

Right now the project includes a simple script (`stock_load.py`) that pulls historical stock data using the `yfinance` API.

This is just a temporary script to get some initial market data into the project while the rest of the pipeline is being built.

The script downloads daily price data for a small set of equities and saves the dataset to:

data/raw/market_prices.csv

This dataset will later be used for feature engineering and machine learning experiments.