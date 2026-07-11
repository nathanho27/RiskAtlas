# Risk Atlas

## Market Risk Modeling System for Equity Markets

Risk Atlas is a market risk modeling system designed to identify S&P 500 stocks with elevated downside risk.

The project combines Python data pipelines, PostgreSQL, SQL-based feature engineering, machine learning, and a planned AI explanation layer to transform historical market data into forward-looking stock-level risk predictions.

---

## Table of Contents

- [Project Status](#project-status)
- [Business Problem](#business-problem)
- [System Overview](#system-overview)
- [Analytical Objectives](#analytical-objectives)
- [Data Sources](#data-sources)
- [Methodology](#methodology)
- [Pipeline Architecture](#pipeline-architecture)
- [Machine Learning](#machine-learning)
- [AI Explanation Layer](#ai-explanation-layer)
- [Planned Application](#planned-application)
- [Tools and Technologies](#tools-and-technologies)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)
- [Outcome](#outcome)

---

## Project Status

The core data engineering and machine learning pipeline has been implemented.

### Completed

- S&P 500 constituent ingestion
- Historical market data ingestion
- PostgreSQL database architecture
- SQL-based data cleaning and transformation
- Financial feature engineering
- Downside-risk label generation
- Logistic regression model
- Random forest model
- Risk probability generation
- Prediction storage in PostgreSQL

### In Development

- Streamlit application
- AI-generated risk explanations
- Automated pipeline execution
- Cloud deployment
- Model monitoring and retraining

---

## Business Problem

Investors and analysts monitor hundreds of securities while attempting to identify which stocks may be vulnerable to future price declines.

Traditional market dashboards are descriptive. They summarize what has already happened through returns, volatility, and trend metrics.

Risk Atlas approaches the problem as a predictive modeling task. Historical market data is transformed into engineered features and used to estimate the probability that a stock will experience elevated downside risk over a future time horizon.

The resulting risk scores can support:

- Equity risk screening
- Portfolio monitoring
- Investment research
- Early identification of elevated downside conditions
- Prioritization of securities requiring deeper analysis

This project is intended for educational and analytical purposes only and does not constitute investment advice.

---

## System Overview

Risk Atlas follows an end-to-end market analytics workflow:

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
Application Layer
        ↓
AI Explanations
```

The project integrates data engineering, analytics engineering, machine learning, and future AI-powered interpretation into a single workflow.

---

## Analytical Objectives

- Identify stocks with elevated downside risk
- Model relationships between volatility, trend, and future declines
- Generate stock-level risk probabilities
- Compare linear and nonlinear machine learning approaches
- Create interpretable risk signals
- Build a production-style analytics workflow
- Develop a foundation for automated market monitoring

---

## Data Sources

### S&P 500 Constituents

Ticker universe collected from publicly available S&P 500 constituent data.

### Historical Market Prices

Historical stock-price data retrieved through Yahoo Finance using the `yfinance` package.

### Derived Datasets

Generated internally through SQL transformations and machine learning workflows.

Examples include:

- Daily returns
- Volatility measures
- Moving averages
- Downside-risk labels
- Model datasets
- Risk predictions

---

## Methodology

### 1. Data Ingestion

Python retrieves the current S&P 500 universe and downloads historical market-price data.

The resulting data is loaded into PostgreSQL.

### 2. Data Cleaning

SQL transformations standardize ticker symbols and prepare clean datasets for downstream analysis.

### 3. Feature Engineering

Historical prices are transformed into predictive market indicators including:

- Daily returns
- Rolling volatility
- Moving averages
- Trend indicators
- Relative price measures

### 4. Label Generation

Future returns are evaluated over a defined prediction horizon.

A binary downside-risk label is created to identify future negative price events.

### 5. Model Training

The project currently evaluates:

- Logistic Regression
- Random Forest

Time-based train/test splits are used to reduce data leakage.

### 6. Prediction Generation

Trained models generate:

- Risk probabilities
- Binary classifications
- Stock-level risk scores

Predictions are written back into PostgreSQL for downstream use.

---

## Pipeline Architecture

### Raw Layer

#### raw_market_prices

Stores historical daily stock-price observations.

### Staging Layer

#### stg_market_prices

Contains cleaned and standardized market data.

### Feature Layer

#### price_features

Contains engineered financial indicators such as:

- Daily returns
- Rolling volatility
- Moving averages
- Trend measures
- Relative price metrics

### Label Layer

#### labels

Contains:

- Future returns
- Downside-event labels

### Modeling Layer

#### model_dataset

Combines engineered features and labels into a machine-learning-ready dataset.

### Prediction Layer

#### predictions

Stores:

- Prediction date
- Ticker
- Risk probability
- Binary classification

---

## Machine Learning

### Logistic Regression

Provides an interpretable baseline model for downside-risk prediction.

### Random Forest

Captures nonlinear relationships between volatility, trend, momentum, and future downside events.

### Modeling Considerations

- Time-based train/test splits
- Class imbalance handling
- Probability-based predictions
- Focus on downside-event detection
- Comparison of baseline and nonlinear models

The primary objective is identifying downside-risk events rather than maximizing classification accuracy.

---

## AI Explanation Layer

A future AI explanation layer will translate model outputs into natural-language summaries.

Rather than displaying only a risk score, the system will explain:

- Volatility conditions
- Trend behavior
- Momentum characteristics
- Model-generated risk classifications

Example planned output:

```text
Ticker: AMD

Risk Probability: 0.64
Risk Classification: Elevated Risk

AI Risk Summary:

AMD currently exhibits elevated modeled downside risk.
The signal is associated with increased volatility,
negative recent momentum, and price weakness relative
to longer-term trend indicators.
```

The machine-learning model generates the prediction.

The AI layer provides interpretation of the underlying signals.

---

## Planned Application

The Streamlit application is expected to provide:

- Risk-monitoring dashboard
- Stock lookup functionality
- Historical risk-score tracking
- Risk classifications
- Supporting feature metrics
- AI-generated explanations

---

## Tools and Technologies

### Programming

- Python
- pandas
- NumPy

### Data Engineering

- PostgreSQL
- SQL
- SQL Window Functions

### Machine Learning

- scikit-learn
- Logistic Regression
- Random Forest

### Data Acquisition

- yfinance

### Development

- Git
- GitHub

### Planned Technologies

- Streamlit
- OpenAI API / LLM Integration

---

## Project Structure

```text
RiskAtlas/
├── data/
│   ├── output/
│   └── processed/
│
├── sql/
│   ├── schema/
│   │   └── raw_market_prices.sql
│   │
│   ├── staging/
│   │   └── stg_market_prices.sql
│   │
│   ├── features/
│   │   ├── price_features.sql
│   │   ├── label_generation.sql
│   │   └── model_dataset.sql
│   │
│   ├── analytics/
│   │   ├── risk_signals.sql
│   │   └── market_metrics.sql
│   │
│   └── marts/
│       └── market_summary.sql
│
├── src/
│   ├── ai/
│   │   └── ai_explanations.py
│   │
│   ├── app/
│   │   └── app.py
│   │
│   ├── data/
│   │   └── stock_load.py
│   │
│   ├── models/
│   │   ├── model_training_logistic.py
│   │   ├── model_training_rf.py
│   │   └── prediction.py
│   │
│   └── __init__.py
│
├── run_pipeline.py
├── README.md
└── .gitignore
```

---

## How to Run

### Clone Repository

```bash
git clone https://github.com/nathanho27/RiskAtlas.git
cd RiskAtlas
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install pandas numpy scikit-learn psycopg2-binary yfinance sqlalchemy
```

### Create PostgreSQL Database

```sql
CREATE DATABASE risk_atlas;
```

### Configure Database Connection

Update database credentials in:

```text
src/data/stock_load.py
```

### Load Market Data

```bash
python src/data/stock_load.py
```

### Build SQL Layers

Execute SQL scripts:

```text
sql/staging/stg_market_prices.sql
sql/features/price_features.sql
sql/features/label_generation.sql
sql/features/model_dataset.sql
```

### Train Models

```bash
python src/models/model_training_logistic.py
python src/models/model_training_rf.py
```

### Generate Predictions

```bash
python src/models/prediction.py
```

### Run Entire Pipeline

```bash
python run_pipeline.py
```

### Launch Application

```bash
streamlit run src/app/app.py
```

---

## Future Improvements

### Application Layer

- Interactive risk dashboard
- Stock search functionality
- Historical risk visualizations
- Supporting feature displays

### AI Layer

- Natural-language risk explanations
- Feature-aware model interpretation
- Structured explanation templates
- Explanation caching

### Pipeline Automation

- Scheduled market-data ingestion
- Automated retraining
- Daily prediction generation
- Pipeline monitoring
- Data-quality checks

### Model Development

- Hyperparameter tuning
- Additional classification models
- Feature importance analysis
- Model calibration
- Performance monitoring

### Cloud Deployment

- Managed PostgreSQL deployment
- Cloud-hosted application
- Automated pipeline orchestration
- Secure credential management

---

## Outcome

Risk Atlas demonstrates an end-to-end market risk modeling workflow that combines data engineering, SQL analytics, machine learning, and planned AI functionality.

The system transforms raw S&P 500 market data into forward-looking risk predictions and establishes the foundation for a production-style financial analytics platform capable of monitoring stock-level downside risk.