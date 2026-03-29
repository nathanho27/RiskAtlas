# Risk Atlas

Risk Atlas is an end-to-end financial analytics system designed to model market risk using historical S&P 500 data.

The project builds a full data pipeline that transforms raw price data into interpretable risk signals, including volatility regimes, drawdowns, and market-wide trend indicators.

---

## Pipeline Architecture

The system follows a layered data architecture:

Python → PostgreSQL → SQL transformations → analytics outputs

### Data Layers

- **Raw Layer**
  - `raw_market_prices`
  - Historical price data for S&P 500 equities

- **Staging Layer**
  - `stg_market_prices`
  - Cleaned and standardized price data

- **Feature Layer**
  - `price_features`
  - Daily returns
  - Rolling volatility (20, 30, 60 day)
  - Moving averages (20, 50, 200 day)
  - Price relative to trend

- **Analytics Layer**
  - `risk_signals`
    - Drawdowns from peak
    - Volatility regimes
    - Trend regimes
    - Risk classification (risk_on, risk_off, neutral)

  - `market_metrics`
    - Average market return
    - Cross-sectional volatility
    - Market breadth (% above moving averages)

- **Mart Layer**
  - `market_summary`
  - Final classification of market regime (high, moderate, low risk)
  - Directional trend (bullish, bearish, neutral)

---

## Key Features

- Rolling volatility and return-based feature engineering
- Drawdown-based risk measurement
- Market breadth analysis using moving averages
- Regime classification combining volatility and trend signals
- Fully SQL-driven analytics pipeline on top of Python ingestion

---

## Tech Stack

- Python (pandas, yfinance)
- SQL (PostgreSQL)
- Data Engineering Design (layered architecture)

---

## Project Structure

- `src/data/` – ingestion pipeline  
- `sql/schema/` – table definitions  
- `sql/staging/` – cleaned data layer  
- `sql/features/` – feature engineering  
- `sql/analytics/` – risk signals and market metrics  
- `sql/marts/` – final outputs  

---

## Future Improvements

- Machine learning models for predictive risk scoring  
- Backtesting strategies using risk signals  
- Streamlit dashboard for interactive exploration  
- AI-generated explanations for market conditions  

---

## Outcome

Built a scalable financial analytics system capable of transforming raw market data into interpretable risk signals at both the individual asset and market level.
