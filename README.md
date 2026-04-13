# Risk Atlas

## Market Risk Modeling System for Equity Markets

Risk Atlas is an end-to-end financial risk modeling system designed to identify stocks with elevated downside risk using historical S&P 500 data.

The project combines data engineering, SQL-based analytics, and machine learning to transform raw market data into forward-looking risk signals. These signals are designed to support portfolio risk monitoring and market condition analysis through a structured, system-oriented workflow.

---

## Table of Contents
- [Status](#status)
- [Overview](#overview)
- [Analytical Objectives](#analytical-objectives)
- [Data Sources](#data-sources)
- [Methodology](#methodology)
- [System Architecture](#system-architecture)
- [Machine Learning](#machine-learning)
- [Tools & Technologies](#tools--technologies)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Outcome](#outcome)

---

## Status

**In Progress**

The core data pipeline, feature engineering layer, and machine learning models have been implemented.

The system currently ingests historical market data, builds a SQL-based analytics layer, trains predictive models, and generates risk scores stored in a PostgreSQL database.

Planned next steps include building an interactive application layer, automating the pipeline, and deploying the system to the cloud.

---

## Overview

Risk Atlas is a financial analytics and modeling project focused on understanding and predicting downside risk in equity markets.

Rather than analyzing past performance alone, the project frames market risk as a forward-looking problem. Using historical price data, the system engineers features capturing volatility, trend, and market structure, and applies machine learning models to estimate the probability of significant short-term drawdowns.

The result is a structured pipeline that produces interpretable risk signals at both the individual stock and market level.

---

## Analytical Objectives

- Identify stocks with elevated downside risk over short-term horizons  
- Model relationships between volatility, trend, and future drawdowns  
- Build a structured data pipeline integrating Python, SQL, and machine learning  
- Generate interpretable risk signals that can be used for monitoring and decision-making  
- Develop a scalable system that can be extended to real-time or automated workflows  

---

## Data Sources

- Historical stock price data from Yahoo Finance  
- S&P 500 constituent data from public sources  
- Derived datasets generated through SQL transformations  

---

## Methodology

- Pull historical price data using Python ingestion scripts  
- Store raw data in PostgreSQL  
- Apply SQL transformations to clean, standardize, and structure the data  
- Compute derived metrics such as daily returns, rolling volatility, and moving averages using SQL window functions  
- Generate forward-looking labels based on future returns  
- Train machine learning models to estimate downside risk probabilities  
- Store model predictions in a database for downstream use  

---

## System Architecture

The project follows a layered data architecture:

### Raw Layer
- `raw_market_prices`
- Historical daily price data for S&P 500 equities  

### Staging Layer
- `stg_market_prices`
- Cleaned and standardized price data  

### Feature Layer
- `price_features`
- Daily returns  
- Rolling volatility (20, 30, 60 day)  
- Moving averages (20, 50, 200 day)  
- Price relative to trend  

### Label Layer
- `labels`
- Forward returns and risk event definitions  

### Modeling Layer
- `model_dataset`
- Combined feature and label dataset used for training  

### Prediction Layer
- `predictions`
- Model-generated risk probabilities and classifications  

---

## Machine Learning

The project includes multiple models for predicting downside risk:

- Logistic Regression (baseline model)  
- Random Forest (nonlinear model capturing feature interactions)  

Key modeling considerations:

- Class imbalance handled using class weighting  
- Time-based train/test split to preserve chronological structure  
- Evaluation focused on recall for rare risk events  

The final output of the modeling layer is a probability representing the likelihood of a stock experiencing a significant drawdown over a 10-day horizon.

---

## Tools & Technologies

- Python  
- PostgreSQL  
- SQL  
- pandas  
- scikit-learn  
- Git & GitHub  

---

## Project Structure
RiskAtlas/
├── src/
│   ├── data/
│   │   └── stock_load.py
│   ├── models/
│   │   ├── model_training.py
│   │   └── model_training_rf.py
│   └── app/ (planned)
│
├── sql/
│   ├── staging/
│   ├── features/
│   ├── analytics/
│   └── marts/
│
└── README.md

---

## Future Improvements

- Streamlit application for interactive risk exploration  
- Automated pipeline execution (scheduled data and model updates)  
- Cloud deployment of database and application layer  
- API layer for model inference  
- AI-generated explanations for risk signals  

---

## Outcome

Designed and built an end-to-end risk modeling system that transforms raw market data into predictive risk signals. The project integrates data engineering, SQL-based analytics, and machine learning into a structured pipeline capable of generating forward-looking insights for equity market risk.
