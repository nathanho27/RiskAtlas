# RiskAtlas

Current Production Model:
Random Forest V3

Universe:
500 U.S. Equities

Prediction Horizon:
10 Trading Days

Latest Test Performance:
ROC-AUC 0.6349 | PR-AUC 0.2180

## Context-Aware Stock Risk Intelligence Platform

RiskAtlas is an end-to-end market risk intelligence system designed to identify stocks exhibiting elevated downside-risk conditions.

Rather than simply describing what has already happened in the market, RiskAtlas attempts to answer a more useful question:

> Which stocks currently show the highest modeled downside risk, how unusual is that risk, and what signals are driving it?

The project combines data engineering, SQL analytics, feature engineering, machine learning, risk scoring, visualization, and AI-powered explainability into a single production-style workflow.

What started as a simple stock-risk prediction model evolved into a context-aware intelligence platform that incorporates market regimes, market breadth, sector-relative performance, and market sensitivity to better understand downside risk.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Current Status](#current-status)
- [Latest Results](#latest-results)
- [Key Findings](#key-findings)
- [Why I Built This](#why-i-built-this)
- [Business Problem](#business-problem)
- [System Architecture](#system-architecture)
- [How RiskAtlas Works](#how-riskatlas-works)
- [Context-Aware Modeling (V3)](#context-aware-modeling-v3)
- [Streamlit Application](#streamlit-application)
- [AI Explanation Layer](#ai-explanation-layer)
- [Future Modeling](#future-modeling)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How To Run](#how-to-run)
- [Future Development](#future-development)

---

## Project Overview

Most stock dashboards are descriptive.

They tell you:

- What a stock returned
- How volatile it was
- Whether it is above or below a moving average
- What happened yesterday

But they do not answer:

> Which stocks currently appear most vulnerable?

RiskAtlas approaches this as a predictive modeling problem.

Historical market behavior is transformed into engineered features and used to estimate whether a stock is currently exhibiting conditions associated with elevated future downside risk.

The goal is not to predict exact returns.

The goal is to identify unusually risky conditions, rank stocks relative to the broader market, and surface signals that may warrant further investigation.

Potential use cases include:

- Portfolio monitoring
- Risk screening
- Equity research
- Market surveillance
- Early identification of deteriorating conditions

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
- Streamlit dashboard prototype
- Context-aware V3 feature engineering
- Market regime modeling
- Market breadth analytics
- Sector-relative feature framework
- XGBoost benchmarking
- LightGBM benchmarking
- Random Forest hyperparameter tuning
- Stable modular Streamlit application
- Historical risk-score tracking
- Model-derived risk drivers
- AI-generated explanations

### In Development

- Automated daily pipeline execution
- Cloud deployment

---

## Latest Results

### Production Model

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Random Forest V3 | **0.6349** | **0.2180** |

RiskAtlas currently serves predictions using Random Forest V3.

The model was selected after outperforming all benchmarked alternatives on a fully held-out test set while maintaining stable live behavior across the production universe.

Dataset:

- Approximately 500 S&P 500 stocks
- Approximately 1.83 million observations
- Historical coverage from 2010–2026
- 10-trading-day prediction horizon

### Model Benchmarking

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression V2 | 0.6130 | 0.2028 |
| Logistic Regression V3 | 0.5864 | 0.1907 |
| Random Forest V3 | **0.6349** | **0.2180** |
| XGBoost V3 | 0.6260 | 0.1947 |
| LightGBM V3 | 0.6169 | 0.1891 |

Random Forest V3 currently represents the strongest out-of-sample performance achieved within the RiskAtlas framework.

---

## Key Findings

The biggest lesson from RiskAtlas V3 was that market context matters.

The strongest predictors of downside risk were not stock-specific indicators.

The most important features included:

- SPY 60-day volatility
- Percentage of stocks above MA200
- Stock 60-day volatility
- SPY 60-day returns
- SPY 20-day volatility
- Percentage of stocks above MA50

These findings suggest that market regime and market participation are more predictive of future downside-risk events than many traditional stock-level indicators.

This insight motivated the transition from stock-centric modeling in V2 to the context-aware architecture used in V3.

---

## Why I Built This

Most stock dashboards are descriptive.

They tell you what already happened.

I wanted to build something that behaves more like a risk intelligence system.

Instead of simply reporting what happened yesterday, RiskAtlas attempts to identify stocks exhibiting characteristics historically associated with elevated future downside risk.

The goal is not to predict exact returns.

The goal is to identify unusually risky conditions, rank stocks relative to the broader market, and surface signals that may warrant further investigation.

RiskAtlas started as a machine-learning experiment but evolved into a full-stack analytics platform combining data engineering, predictive modeling, explainability, and AI-assisted interpretation.

---

## Business Problem

Most stock dashboards focus on historical performance.

They show:

- Returns
- Volatility
- Moving averages
- Technical indicators
- Price charts

While useful, these metrics are primarily descriptive.

They help explain what has already happened but provide little insight into which stocks may currently be exhibiting elevated downside-risk conditions.

RiskAtlas approaches this challenge as a predictive modeling problem.

Historical market behavior is transformed into engineered features and used to estimate whether a stock is currently exhibiting characteristics associated with future downside-risk events.

The platform is designed to support:

- Portfolio monitoring
- Equity risk screening
- Investment research
- Market surveillance
- Early identification of deteriorating conditions

---

## System Architecture

```text
S&P 500 Universe
        ↓
Historical Market Data
        ↓
PostgreSQL Database
        ↓
SQL Cleaning & Transformation
        ↓
Feature Engineering
        ↓
Downside-Risk Labels
        ↓
Machine Learning Models
        ↓
Random Forest V3
        ↓
current_risk_predictions_v3
        ↓
inference_dataset_v3
        ↓
Risk Driver Engine
        ↓
Risk Brief Generation
        ↓
Streamlit Dashboard
```

---

## How RiskAtlas Works

### 1. Data Ingestion

RiskAtlas begins by collecting:

- Current S&P 500 constituents
- Historical market data
- Daily price history
- Volume data

Python ingestion pipelines retrieve and load the data into PostgreSQL for downstream processing.

---

### 2. SQL Transformation Layer

Raw market data is transformed through a series of SQL pipelines.

The transformation layer is responsible for:

- Data cleaning
- Missing-value handling
- Standardization
- Rolling calculations
- Dataset preparation

This creates a consistent analytical foundation for feature engineering and modeling.

---

### 3. Feature Engineering

Raw prices are transformed into predictive signals.

Core features include:

- Momentum
- Volatility
- Downside volatility
- Drawdowns
- Distance from highs
- Moving-average relationships

RiskAtlas V3 expands the feature framework by introducing market context.

Additional feature groups include:

- Market regime indicators
- Market breadth metrics
- Sector-relative performance
- Market sensitivity measures
- Cross-sectional rankings

---

### 4. Label Generation

Future returns are evaluated over a 10-trading-day horizon.

A downside-risk event is defined and converted into a binary classification target.

This target becomes the supervised learning label used during model training.

---

### 5. Model Training

RiskAtlas currently benchmarks multiple machine-learning models:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

All models use chronological train, validation, and test splits to better simulate real-world deployment and reduce look-ahead bias.

---

### 6. Prediction Generation

The production pipeline generates:

- Risk probabilities
- Risk classifications
- Risk percentiles
- Relative rankings

Predictions are written back into PostgreSQL where they become available to the application layer.

---

### 7. Application Layer

The Streamlit dashboard serves as the primary interface for interacting with model outputs.

Users can:

- Explore current risk conditions
- View high-risk securities
- Search individual stocks
- Inspect model results
- Review risk distributions

Future versions will incorporate historical tracking and AI-powered explanations.

---

## Context-Aware Modeling (V3)

The biggest lesson from RiskAtlas V2 was that the model already understood what a stock was doing.

The next challenge was teaching the model context.

For example:

```text
NVDA is down 5%
```

By itself, that information is incomplete.

The model should also understand:

```text
SPY is down 10%
Semiconductors are down 15%
Most stocks are trading below trend
```

A stock falling less than its peers may actually be demonstrating relative strength.

Likewise, a stock falling alongside a broad market selloff may not be exhibiting unusual risk at all.

RiskAtlas V3 was built around this idea.

The goal was to move from:

```text
What is this stock doing?
```

to:

```text
What is this stock doing relative to:

- The market
- Its sector
- The broader stock universe
```

### V3 Feature Groups

#### Stock-Level Features

- 20-day return
- 60-day return
- 20-day volatility
- 60-day volatility
- Downside volatility
- Drawdowns
- Distance from highs
- Moving-average relationships

#### Market Regime Features

- SPY 20-day return
- SPY 60-day return
- SPY 20-day volatility
- SPY 60-day volatility
- SPY drawdowns

#### Market Breadth Features

- Percentage of stocks above MA50
- Percentage of stocks above MA200
- Positive-return breadth

#### Sector Context Features

- Return relative to sector
- Volatility relative to sector

#### Market Sensitivity Features

- Rolling beta
- Rolling market correlation

#### Cross-Sectional Ranking Features

- Return percentiles
- Volatility percentiles
- Drawdown percentiles
- Sector-relative percentiles

### Dataset

- Approximately 1.83 million observations
- Approximately 500 S&P 500 stocks
- Historical coverage from 2010–2026
- Chronological train / validation / test framework

### Model Benchmarking

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression V3 | 0.5864 | 0.1907 |
| Random Forest V3 | **0.6349** | **0.2180** |
| XGBoost V3 | 0.6260 | 0.1947 |
| LightGBM V3 | 0.6169 | 0.1891 |

### Outcome

RiskAtlas V3 established a new performance benchmark within the project.

```text
Production Logistic Regression V2

ROC-AUC: 0.6130
PR-AUC : 0.2028

Context-Aware Random Forest V3

ROC-AUC: 0.6349
PR-AUC : 0.2180
```

Performance Improvements:

```text
ROC-AUC: +0.0219
PR-AUC : +0.0152
```

The results validate the importance of context-aware feature engineering and demonstrate that broader market conditions materially improve downside-risk prediction.

---

## Streamlit Application

The dashboard currently tracks 500 stocks and surfaces:

- Risk scores
- Risk percentiles
- Risk classifications
- Binary risk alerts
- Current market-wide risk conditions

### Overview

Provides a market-wide summary including:

- Stocks tracked
- Risk-score distribution
- Highest-risk stocks
- Model status
- Current market risk landscape

### Stock Lookup

Allows users to inspect individual securities.

Current functionality includes:

- Risk score
- Risk percentile
- Risk level
- Latest price
- Historical risk tracking
- Risk-driver analysis
- Risk briefs
- Model explanations

### Top Risk Stocks

Ranks securities by current modeled downside risk.

Planned functionality includes:

- Risk-level filtering
- Relative risk comparisons
- Identification of high-priority names for further research

### Model Insights

Explains:

- Model performance
- Feature framework
- Methodology
- Prediction workflow
- Model evolution


---

## Explainability Layer

RiskAtlas includes a model-aware explainability framework that converts engineered features into interpretable risk drivers.

Rather than exposing only a risk score, the platform identifies:

- Risk-increasing factors
- Protective factors
- Relative performance signals
- Market sensitivity signals
- Trend conditions

These drivers are combined into stock-level risk briefs that provide context around each prediction.

Example:

```text
Ticker: NVDA

Risk Level: High Risk
Risk Score: 72.4

AI Summary:

NVDA currently exhibits elevated modeled downside risk.

The signal is associated with increased volatility,
weakening medium-term momentum,
and a larger-than-normal drawdown relative to recent highs.

Its current score ranks in the 88.7th percentile of stocks tracked by RiskAtlas.
```

The AI does not make the prediction.

The machine-learning model generates the score.

The AI layer explains the evidence.

---

## Future Modeling

The primary lesson from V3 was that feature engineering generated larger gains than model complexity.

Adding market-regime awareness, breadth indicators, sector-relative performance metrics, and market-sensitivity features produced meaningful improvements across multiple model families.

Future research directions include:

- VIX integration
- Treasury-yield features
- Credit-spread indicators
- Earnings-event proximity
- Probability calibration
- Regime-specific models
- Alternative labeling frameworks
- Explainability systems
- Automated model monitoring

RiskAtlas will continue benchmarking new models, but a model will only replace the current leader if it demonstrates superior out-of-sample performance on a fully held-out test set.

Project philosophy:

> Better information beats more complexity.

---

## Tech Stack

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
- XGBoost
- LightGBM

### Visualization

- Streamlit
- Plotly

### AI

- OpenAI API *(planned)*
- Structured LLM Outputs *(planned)*
- Model-Aware Explanations *(planned)*

### Development

- Git
- GitHub

---

## Project Structure

RiskAtlas/
│
├── README.md
├── .gitignore
├── requirements.txt
├── run_pipeline.py
├── test_database.py
│
├── data/
│
├── models/
│   ├── logistic_risk_model.joblib
│   ├── logistic_risk_model_v2.joblib
│   ├── random_forest_risk_model.joblib
│   ├── random_forest_risk_model_v2.joblib
│   ├── best_random_forest_v3.joblib
│   ├── best_risk_model_v3.joblib
│   ├── v3_benchmark_results.csv
│   └── v3_rf_tuning_results.csv
│
├── sql/
│   │
│   ├── schema/
│   ├── staging/
│   ├── marts/
│   ├── analytics/
│   │
│   ├── features/
│   │   ├── price_features.sql
│   │   ├── model_dataset.sql
│   │   ├── label_generation.sql
│   │   ├── feature_engineering_v3.sql
│   │   └── inference_engineering_v3.sql
│   │
│   └── models/
│
├── src/
│   │
│   ├── ai/
│   │
│   ├── app/
│   │   ├── app.py
│   │   ├── data_access.py
│   │   └── views/
│   │       ├── overview.py
│   │       ├── stock_lookup.py
│   │       └── model_insights.py
│   │
│   ├── data/
│   │   ├── stock_load.py
│   │   └── context_load.py
│   │
│   ├── models/
│   │   ├── prediction.py
│   │   ├── prediction_v3.py
│   │   ├── model_training_logistic.py
│   │   ├── model_training_logistic_v2.py
│   │   ├── model_training_rf.py
│   │   ├── model_training_rf_v2.py
│   │   ├── model_training_v3.py
│   │   └── random_forest_tuning_v3.py
│   │
│   └── __init__.py
│
└── .vscode/

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
Update the PostgreSQL connection settings inside:

src/app/data_access.py
src/models/prediction_v3.py

to match your local PostgreSQL environment.
```

### Build Feature Tables

Run the SQL feature engineering pipeline:

```sql
sql/features/price_features.sql
sql/features/model_dataset.sql
sql/features/label_generation.sql
sql/features/feature_engineering_v3.sql
sql/features/inference_engineering_v3.sql
```

### Train Models

Production Logistic Regression:

```bash
python src/models/model_training_logistic_v2.py
```

V3 Benchmarking:

```bash
python src/models/model_training_v3.py
```

V3 Random Forest Tuning:

```bash
python src/models/random_forest_tuning_v3.py
```

### Generate Predictions

```bash
python src/models/prediction_v3.py
```

### Launch Dashboard

```bash
streamlit run src/app/app.py
```

---

## Future Development

### Application

- Historical risk-score tracking
- Watchlists
- Saved stock monitoring
- Advanced filtering
- Export functionality
- Historical prediction review

### Explainability

- Model contribution analysis
- SHAP explanations
- Natural-language narrative generation
- LLM-enhanced risk commentary

### Modeling

- VIX integration
- Treasury-yield features
- Credit-spread indicators
- Earnings-event proximity
- Probability calibration
- Regime-specific models
- Alternative labeling frameworks
- Ensemble modeling
- Model monitoring

### Infrastructure

- Automated data ingestion
- Automated feature generation
- Daily prediction generation
- Cloud deployment
- Monitoring and alerting
- Scheduled retraining pipeline

---

## What I Learned

The biggest lesson from this project was that better features often matter more than more sophisticated models.

My initial instinct was to improve performance by trying increasingly complex machine-learning algorithms.

However, the largest gains came from improving the information available to the model.

Moving from stock-centric features to context-aware features produced a larger improvement than switching model families.

The V3 experiments showed that market regime, market breadth, and relative positioning contain meaningful predictive information that traditional stock-level indicators often miss.

This project reinforced the importance of:

- Feature engineering
- Proper train / validation / test methodology
- Out-of-sample evaluation
- Context-aware modeling
- Building complete end-to-end systems rather than isolated models

RiskAtlas ultimately became much more than a machine-learning experiment.

It evolved into a full-stack data science project combining data engineering, analytics, machine learning, application development, and AI-assisted interpretation into a single workflow.