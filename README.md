# RiskAtlas
Market risk modeling project that analyzes historical equity data to estimate forward-looking risk signals such as drawdowns and volatility regimes. Uses engineered financial features and machine learning models to generate interpretable risk scores through a reproducible Python workflow and interactive Streamlit dashboards.

## Data Ingestion (Temporary)

Right now the project includes a simple script (`stock_load.py`) that pulls historical stock data using the `yfinance` API.

This is just a temporary script to get some initial market data into the project while the rest of the pipeline is being built.

The script downloads daily price data for a small set of equities and saves the dataset to:

data/raw/market_prices.csv

This dataset will later be used for feature engineering and machine learning experiments.

## Feature Engineering (In Progress)

The next stage of the project focuses on generating financial features from historical price data.

These features will be used as inputs for machine learning models and are designed to capture key signals such as momentum, volatility, and trend behavior.

The initial feature set includes:

• Rolling returns over multiple time horizons  
• Volatility measures based on rolling standard deviation  
• Moving averages and trend indicators  
• Relative positioning of price compared to moving averages  

The feature engineering pipeline is currently being developed and will output a structured dataset for modeling.

## AI Layer (Planned)

The project will incorporate an AI layer to enhance model interpretability and provide natural language insights.

Planned capabilities include:

• Generating explanations for predicted risk signals based on model outputs and feature importance  
• Summarizing overall market risk conditions and trends  
• Translating quantitative signals into human-readable insights  

This layer will complement the machine learning models by making predictions more interpretable and actionable.
