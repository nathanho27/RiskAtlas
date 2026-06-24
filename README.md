# Risk Atlas

## Market Risk Modeling System for Equity Markets

Market risk modeling system focused on identifying stocks with elevated downside risk using historical S&P 500 data. The project combines Python, PostgreSQL, SQL-based feature engineering, and machine learning to transform raw market data into forward-looking risk predictions.

---

## Status

**In Progress**

The core data pipeline, analytics layer, and machine learning models have been implemented.

Historical market data is ingested through Python, transformed through a layered PostgreSQL analytics architecture, and used to train machine learning models that estimate downside risk probabilities.

Current outputs include engineered financial features, downside-risk labels, model-ready datasets, and prediction tables containing stock-level risk scores.

Planned next steps include building an interactive application layer, automating pipeline execution, cloud deployment, and AI-generated explanations for model outputs.

---

## Overview

Risk Atlas is a financial analytics and machine learning project focused on modeling downside risk in equity markets.

Rather than analyzing historical performance alone, the project approaches market risk as a forward-looking prediction problem. Historical stock price data is transformed into volatility, trend, and momentum-based features which are then used to train machine learning models capable of estimating the probability of future downside moves.

The system is designed as an end-to-end workflow integrating data engineering, SQL analytics, and predictive modeling.

---

## Analytical Objectives

- Identify stocks with elevated downside risk over short-term horizons
- Model relationships between volatility, trend, and future drawdowns
- Generate interpretable risk signals using market data
- Build an end-to-end financial analytics workflow integrating Python, PostgreSQL, SQL, and machine learning
- Develop a foundation for automated market risk monitoring

---

## Data Sources

- Historical stock price data from Yahoo Finance
- S&P 500 constituent data from publicly available sources
- Derived datasets generated through SQL transformations and feature engineering

---

## Methodology

- Pull historical market data using Python.
- Store raw market data in PostgreSQL.
- Clean and standardize datasets using SQL transformations.
- Compute derived metrics such as daily returns, rolling volatility, and moving averages using SQL window functions.
- Generate forward-looking downside-risk labels based on future returns.
- Train machine learning models to estimate downside-risk probabilities.
- Store model predictions in PostgreSQL for downstream applications.

---

## Pipeline Architecture

The system follows a layered data architecture:

### Raw Layer
- `raw_market_prices`
- Historical daily stock price data

### Staging Layer
- `stg_market_prices`
- Cleaned and standardized market data

### Feature Layer
- `price_features`
- Daily returns
- Rolling volatility (20, 30, 60 day)
- Moving averages (20, 50, 200 day)
- Price relative to trend

### Label Layer
- `labels`
- Forward returns
- Binary downside-risk events

### Modeling Layer
- `model_dataset`
- Combined feature and label dataset used for training

### Prediction Layer
- `predictions`
- Risk probabilities
- Binary risk classifications
- Model-generated outputs

---

## Machine Learning

The project currently includes two classification models:

### Logistic Regression
- Baseline model used for downside-risk prediction

### Random Forest
- Nonlinear model used to capture interactions between volatility, trend, and momentum features

Key modeling considerations include:

- Time-based train/test splits
- Class imbalance handling through class weighting
- Evaluation focused on identifying rare downside-risk events

The final model output is a probability representing the likelihood of a significant downside move over a future 10-day horizon.

---

## Tools & Technologies

- Python
- PostgreSQL
- SQL
- pandas
- scikit-learn
- yfinance
- Git & GitHub

---

## Project Structure

```text
RiskAtlas/
├── src/
│   ├── data/
│   │   └── stock_load.py
│   │
│   ├── models/
│   │   ├── model_training.py
│   │   └── model_training_rf.py
│   │
│   ├── app/
│   │   └── app.py (planned)
│   │
│   └── ai/
│       └── ai_explanations.py (planned)
│
├── sql/
│   ├── staging/
│   ├── features/
│   ├── analytics/
│   └── marts/
│
└── README.md
```

---

## Future Improvements

### Application Layer
- Streamlit dashboard for risk monitoring
- Interactive stock lookup
- Historical risk visualization

### Automation
- Scheduled data ingestion
- Automated model retraining
- Daily risk prediction updates

### Cloud Deployment
- Hosted PostgreSQL database
- Cloud-based application deployment
- End-to-end automated pipeline execution

### AI Layer
- AI-generated explanations for risk predictions
- Natural language summaries of market conditions
- Model interpretation and reasoning support

---

## Outcome

Built an end-to-end market risk modeling system that transforms raw market data into predictive risk signals using SQL-based feature engineering and machine learning. The project serves as a foundation for a production-style financial analytics platform capable of generating forward-looking risk assessments for individual stocks and broader market conditions.