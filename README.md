# RiskAtlas

## AI-Powered Stock Risk Intelligence Platform

RiskAtlas is an end-to-end market risk intelligence system designed to identify stocks exhibiting elevated downside-risk conditions.

The project combines data engineering, SQL analytics, machine learning, explainability, and AI-powered interpretation into a single production-style workflow.

Rather than simply describing what has already happened in the market, RiskAtlas attempts to answer a more useful question:

> Which stocks currently show the highest modeled downside risk, how unusual is that risk, and what signals are driving it?

The system transforms raw market data into stock-level risk scores, risk rankings, and eventually AI-generated explanations that help users understand why a stock is being flagged.

---

## Table of Contents

- [Current Status](#current-status)
- [Latest Results](#latest-results)
- [Why I Built This](#why-i-built-this)
- [Business Problem](#business-problem)
- [System Architecture](#system-architecture)
- [How RiskAtlas Works](#how-riskatlas-works)
- [Streamlit Application](#streamlit-application)
- [AI Explanation Layer](#ai-explanation-layer)
- [RiskAtlas V3](#riskatlas-v3)
- [Future Modeling](#future-modeling)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How To Run](#how-to-run)
- [Future Development](#future-development)
- [Notes](#notes)

---

## Current Status

### Completed

- S&P 500 universe ingestion
- Historical market data ingestion
- PostgreSQL database architecture
- SQL cleaning and transformation layers
- Financial feature engineering
- Downside-risk label generation
- Logistic Regression baseline model
- Random Forest benchmark model
- Chronological train / validation / test framework
- Validation-based threshold optimization
- Daily risk prediction generation
- PostgreSQL prediction storage
- Initial Streamlit dashboard prototype

### In Development

- Stable modular Streamlit application
- Historical risk-score tracking
- Model-derived risk drivers
- AI-generated explanations
- Automated daily pipeline execution
- Cloud deployment

---

## Latest Results

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression V2 | **0.6130** | **0.2028** |
| Random Forest V2 | 0.6120 | 0.1987 |

### Production Model

**Logistic Regression V2**

Current universe:

- Approximately 500 stocks
- Approximately 1.24 million training observations
- 14 engineered features
- 10-trading-day prediction horizon

The model generates a probability-based downside-risk score for each stock and ranks securities relative to the rest of the tracked universe.

---

## Why I Built This

Most stock dashboards are descriptive.

They tell you what already happened.

I wanted to build something that behaves more like a risk intelligence system:

- Ingest market data
- Engineer predictive features
- Identify elevated downside-risk conditions
- Rank stocks relative to the broader market
- Explain why a stock is being flagged

RiskAtlas started as a machine-learning project but has evolved into a full-stack analytics platform combining data engineering, predictive modeling, explainability, and AI-assisted interpretation.

---

## Business Problem

Most stock dashboards are descriptive.

They show:

- Returns
- Volatility
- Moving averages
- Technical indicators

But they do not answer:

> Which stocks currently appear most vulnerable?

RiskAtlas approaches this as a predictive modeling problem.

Historical market behavior is transformed into engineered features and used to estimate whether a stock is currently exhibiting conditions associated with elevated future downside risk.

Potential use cases include:

- Portfolio monitoring
- Equity risk screening
- Investment research
- Market surveillance
- Early identification of deteriorating conditions

This project is intended for educational and analytical purposes only and should not be considered investment advice.

---

## System Architecture

```text
S&P 500 Constituents
        ↓
Historical Market Data
        ↓
Raw PostgreSQL Tables
        ↓
SQL Cleaning & Transformation
        ↓
Feature Engineering
        ↓
Downside-Risk Labels
        ↓
Machine Learning Models
        ↓
Risk Probabilities
        ↓
Prediction Storage
        ↓
Application Layer
        ↓
AI Explanations
```

---

## How RiskAtlas Works

### 1. Data Ingestion

Python retrieves:

- Current S&P 500 constituents
- Historical stock-price data

Market data is loaded into PostgreSQL for downstream processing.

---

### 2. Feature Engineering

Raw prices are transformed into predictive market indicators including:

- Daily returns
- 5-day momentum
- 20-day momentum
- 60-day momentum
- Rolling volatility
- Downside volatility
- Moving-average relationships
- Drawdowns
- Negative-return frequency
- Worst trailing return

---

### 3. Label Generation

Future returns are evaluated over a 10-trading-day horizon.

A binary downside-risk label is generated and used as the machine-learning target.

---

### 4. Model Training

Current models:

- Logistic Regression
- Random Forest

Training uses chronological train, validation, and test splits to reduce data leakage.

---

### 5. Prediction Generation

The production model generates:

- Risk probabilities
- Binary classifications
- Risk percentiles
- Risk-level classifications

Predictions are written back into PostgreSQL for application consumption.

---

## Streamlit Application

The current dashboard prototype is being rebuilt into a stable, modular application with four planned workflows.

### Overview

Provides a market-wide summary including:

- Stocks tracked
- Critical-risk stocks
- High-risk stocks
- Risk-score distribution
- Highest-risk stocks
- Model status

### Stock Lookup

Allows users to inspect individual securities.

Planned information includes:

- Risk score
- Risk percentile
- Risk level
- Latest price
- Historical risk trends
- AI explanation

### Top Risk Stocks

Ranks securities by current modeled risk.

Planned functionality includes:

- Filtering by risk level
- Comparing risk scores
- Identifying high-priority names for further analysis

### Model Insights

Explains:

- Model performance
- Feature set
- Methodology
- Prediction framework

---

## AI Explanation Layer

A planned LLM-powered explanation layer will translate structured model evidence into readable risk summaries.

The machine-learning model will remain responsible for generating predictions.

The AI layer will be responsible for interpretation.

Rather than showing only a score, the system will explain:

- Why a stock is considered risky
- Which signals contributed most
- How the stock compares with peers
- What conditions are driving the signal

Example:

```text
Ticker: NVDA

Risk Level: High Risk
Risk Score: 72.4

AI Summary:

NVDA currently exhibits elevated modeled downside risk.
The signal is associated with increased volatility,
weakening medium-term momentum, and a larger-than-normal
drawdown relative to recent highs.

Its current score ranks above 88.7% of stocks tracked by
RiskAtlas.
```

The AI does not make the prediction.

The machine-learning model generates the score.

The AI layer explains the evidence.

---

## RiskAtlas V3

The biggest lesson from V2 was that the model already understands what a stock is doing.

The next challenge is teaching the model context.

Today, if NVDA falls 5%, the model sees:

```text
NVDA is down 5%
```

But it does not know:

```text
SPY is down 10%
Semiconductors are down 15%
```

In reality, context matters.

A stock falling less than its peers may actually be demonstrating strength.

### V3 Goal

Move the model from:

```text
What is this stock doing?
```

to:

```text
What is this stock doing relative to
the market,
its sector,
and the rest of the stock universe?
```

### Planned V3 Features

#### Relative Strength

- Return relative to SPY
- Volatility relative to SPY
- Drawdown relative to SPY

#### Cross-Sectional Rankings

- Momentum percentile
- Volatility percentile
- Drawdown percentile
- Risk percentile

#### Sector Context

- Return relative to sector
- Volatility relative to sector
- Sector-relative rankings

#### Market Breadth

- Percentage of stocks above MA50
- Percentage of stocks above MA200
- Positive-return breadth
- Market participation measures

#### Market Regimes

- Bull market
- Neutral market
- Bear market
- High-volatility market

#### Beta and Market Sensitivity

- Rolling beta
- Rolling correlation
- Down-market beta

#### Volatility Regimes

- Short-term versus long-term volatility
- Volatility acceleration
- Volatility breakouts

#### Persistence Features

- Days below MA50
- Days below MA200
- Days since recent highs
- Duration of drawdowns

---

## Future Modeling

Once V3 features are complete, RiskAtlas will benchmark:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

The project philosophy is:

> Better features before more complex models.

A new model will only replace the current production model if it demonstrates superior out-of-sample performance.

---

## Tech Stack

### Programming

- Python
- pandas
- NumPy

### Data Engineering

- PostgreSQL
- SQL
- SQL window functions

### Machine Learning

- scikit-learn
- Logistic Regression
- Random Forest

### Visualization

- Streamlit
- Plotly

### AI

- OpenAI API *(planned)*
- Structured LLM outputs *(planned)*
- Model-aware explanations *(planned)*

### Development

- Git
- GitHub

---

## Project Structure

The application is currently being reorganized toward the following modular structure:

```text
RiskAtlas/
│
├── README.md
├── .gitignore
├── requirements.txt
├── run_pipeline.py
│
├── data/
│   ├── output/
│   └── processed/
│
├── models/
│   ├── logistic_risk_model.joblib
│   ├── logistic_risk_model_v2.joblib
│   ├── random_forest_risk_model.joblib
│   └── random_forest_risk_model_v2.joblib
│
├── src/
│   │
│   ├── ai/
│   │   └── ai_explanations.py
│   │
│   ├── app/
│   │   ├── app.py
│   │   ├── data_access.py
│   │   ├── components.py
│   │   │
│   │   └── pages/
│   │       ├── overview.py
│   │       ├── stock_lookup.py
│   │       ├── top_risk.py
│   │       └── model_insights.py
│   │
│   ├── data/
│   │   └── stock_load.py
│   │
│   ├── models/
│   │   ├── prediction.py
│   │   ├── model_training_logistic.py
│   │   ├── model_training_logistic_v2.py
│   │   ├── model_training_rf.py
│   │   └── model_training_rf_v2.py
│   │
│   └── __init__.py
│
└── tests/
    └── test_database.py
```

---

## How To Run

### Clone Repository

```bash
git clone https://github.com/nathanho27/RiskAtlas.git
cd RiskAtlas
```

### Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure PostgreSQL

```bash
export DATABASE_URL="postgresql://username:password@localhost:5432/risk_atlas"
```

### Run Pipeline

```bash
python run_pipeline.py
```

### Launch Dashboard

```bash
streamlit run src/app/app.py
```

---

## Future Development

### Application

- Historical risk tracking
- Watchlists
- Advanced filtering
- Export functionality

### Explainability

- Feature-level risk drivers
- Logistic-regression contribution analysis
- Model-aware AI explanations

### Modeling

- RiskAtlas V3 contextual features
- Tree-based model benchmarking
- Probability calibration
- Model monitoring

### Infrastructure

- Automated data ingestion
- Daily prediction generation
- Cloud deployment
- Monitoring and alerting

---

## Notes

RiskAtlas is designed as a market intelligence and analytics platform, not a trading system.

The model attempts to identify stocks exhibiting conditions historically associated with elevated downside risk, but markets are noisy, adaptive, and often unpredictable.

A high RiskAtlas score does not mean a stock will fall.

Likewise, a low score does not guarantee positive future performance.

The purpose of the platform is to surface potentially interesting risk signals, provide context around those signals, and support deeper research.

Think of RiskAtlas as a starting point for investigation rather than a source of investment recommendations.