# RiskAtlas

Live Demo: https://riskatlas.streamlit.app/

**Current Production Model:**  
Random Forest V3

**Universe:**  
S&P 500 Universe (503 Constituents, ~499 Eligible Daily Predictions)

**Prediction Horizon:**  
10 Trading Days

**Latest Test Performance:**  
ROC-AUC 0.6349 | PR-AUC 0.2180

**Production Pipeline Runtime:**  
Approximately 1.3 Minutes

## Context-Aware Stock Risk Intelligence Platform

RiskAtlas is an end-to-end market risk intelligence system designed to identify stocks exhibiting elevated downside-risk conditions.

Rather than simply describing what has already happened in the market, RiskAtlas attempts to answer a more useful question:

> Which stocks currently show the highest modeled downside risk, how unusual is that risk, and what signals are driving it?

The project combines data engineering, cloud infrastructure, SQL analytics, feature engineering, machine learning, risk scoring, visualization, and AI-powered explainability into a single production-style workflow.

What started as a simple stock-risk prediction model evolved into a context-aware intelligence platform that incorporates market regimes, market breadth, sector-relative performance, and market sensitivity to better understand downside risk.

- 500 stocks
- 1.83 million observations
- 16 years of history
- Random Forest production model
- ROC-AUC 0.6349
- 1.3-minute production pipeline

---

## Table of Contents

- [Project Overview](#project-overview)
- [Application Preview](#application-preview)
- [Current Status](#current-status)
- [Latest Results](#latest-results)
- [Key Findings](#key-findings)
- [Why I Built This](#why-i-built-this)
- [Business Problem](#business-problem)
- [System Architecture](#system-architecture)
- [How RiskAtlas Works](#how-riskatlas-works)
- [Production Pipeline](#production-pipeline)
- [Context-Aware Modeling V3](#context-aware-modeling-v3)
- [Streamlit Application](#streamlit-application)
- [Explainability Layer](#explainability-layer)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How To Run](#how-to-run)
- [What I Learned](#what-i-learned)

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

## Application Preview

### Overview Dashboard

![Overview Dashboard](images/overview.png)

---

### Risk Percentile Analysis

![Risk Percentile](images/risk_percentile.png)

---

### Key Risk Drivers

![Key Risk Drivers](images/key_risk_drivers.png)

---

### AI Risk Brief

![Risk Brief](images/risk_brief.png)

---

### Ask RiskAtlas

![Ask RiskAtlas](images/ask_risk_atlas.png)

---

### AI Assistant

![AI Assistant](images/ai_assistant.png)

---

### Top Risk Stocks

![Top Risk Stocks](images/riskiest_stocks.png)

---

### Model Insights

![Model Insights](images/model_insights.png)

---

## Current Status

RiskAtlas is functionally complete as a local production-style application.

Completed components include:

- S&P 500 universe ingestion
- Historical market data ingestion
- Incremental daily market updates
- PostgreSQL database architecture
- Google Cloud SQL production warehouse
- Cloud database migration from Neon to GCP
- Production PostgreSQL deployment
- SQL cleaning and transformation layers
- Financial feature engineering
- Downside-risk label generation
- Logistic Regression baseline modeling
- Random Forest modeling
- XGBoost benchmarking
- LightGBM benchmarking
- Chronological train, validation, and test framework
- Validation-based threshold optimization
- Context-aware V3 feature engineering
- Market regime modeling
- Market breadth analytics
- Sector-relative features
- Market sensitivity features
- Cross-sectional rankings
- Random Forest hyperparameter tuning
- Daily risk prediction generation
- Current prediction storage
- Historical prediction tracking
- Stable modular Streamlit application
- Model-derived risk drivers
- Gemini-powered risk explanations
- Conversational RiskAtlas assistant
- Unified production pipeline
- Optimized latest-date inference workflow

The core platform is complete. Remaining work is limited to public application deployment and scheduled daily pipeline execution.

---

## Latest Results

### Production Model

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Random Forest V3 | **0.6349** | **0.2180** |

RiskAtlas currently serves predictions using Random Forest V3.

The model was selected after outperforming all benchmarked alternatives on a fully held-out test set while maintaining stable behavior across the production universe.

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

Random Forest V3 represents the strongest out-of-sample performance achieved within the RiskAtlas framework.

---

## Key Findings

The biggest lesson from RiskAtlas V3 was that market context matters.

The strongest predictors of downside risk were not exclusively stock-specific indicators.

The most important features included:

- SPY 60-day volatility
- Percentage of stocks above MA200
- Stock 60-day volatility
- SPY 60-day returns
- SPY 20-day volatility
- Percentage of stocks above MA50

These findings suggest that market regime and market participation contain meaningful predictive information beyond traditional stock-level indicators.

This insight motivated the transition from the stock-centric modeling approach used in V2 to the context-aware architecture used in V3.

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
SQL Cleaning and Transformation
        ↓
Feature Engineering
        ↓
Downside-Risk Labels
        ↓
Machine Learning Models
        ↓
Random Forest V3
        ↓
inference_dataset_v3
        ↓
Current Risk Predictions
        ↓
Historical Prediction Storage
        ↓
Risk Driver Engine
        ↓
Gemini Explanation Layer
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
- SPY market context
- Company and sector metadata

Python ingestion pipelines retrieve and load the data into PostgreSQL for downstream processing.

The production workflow uses incremental updates so that only recent market data is downloaded during daily execution rather than reloading the full historical dataset.

---

### 2. SQL Transformation Layer

Raw market data is transformed through a series of SQL pipelines.

The transformation layer is responsible for:

- Data cleaning
- Missing-value handling
- Standardization
- Rolling calculations
- Dataset preparation
- Index creation
- Production inference preparation

This creates a consistent analytical foundation for feature engineering and modeling.

---

### 3. Feature Engineering

Raw market prices are transformed into predictive signals.

Core features include:

- Momentum
- Volatility
- Downside volatility
- Worst recent returns
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

RiskAtlas benchmarked multiple machine-learning models:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

All models use chronological train, validation, and test splits to better simulate real-world deployment and reduce look-ahead bias.

The production model is a tuned Random Forest using 27 engineered features.

---

### 6. Prediction Generation

The production pipeline generates:

- Risk probabilities
- Binary risk classifications
- Risk percentiles
- Relative rankings
- Risk levels

Predictions are written back into PostgreSQL where they become available to the application layer.

Current predictions are stored separately from historical prediction records so that the application can display both the latest market state and changes in modeled risk over time.

---

### 7. Application Layer

The Streamlit dashboard serves as the primary interface for interacting with model outputs.

Users can:

- Explore current market-wide risk conditions
- View the highest-risk securities
- Search individual stocks
- Inspect model results
- Review risk distributions
- Track historical risk scores
- Analyze model-derived risk drivers
- Generate AI-powered risk explanations
- Ask questions through the RiskAtlas assistant

---

## Production Pipeline

RiskAtlas uses a single production entry point:

```bash
python run_pipeline.py
```

The pipeline performs the complete daily workflow:

1. Refreshes market price data
2. Downloads only recent incremental updates
3. Refreshes SPY market context
4. Refreshes company and sector metadata
5. Rebuilds the staging market-price table
6. Rebuilds rolling price features
7. Creates the latest inference dataset
8. Loads the Random Forest V3 model artifact
9. Generates current risk predictions
10. Updates current and historical prediction tables

A recent production execution processed updates for 503 S&P 500 constituents, generated predictions for 499 eligible securities, wrote results to a Google Cloud SQL warehouse, and completed successfully in approximately 1.3 minutes.

### Pipeline Optimization

The original production workflow took approximately 10 minutes to complete.

The optimized pipeline completes in approximately 1.3 minutes, representing an estimated 87% reduction in runtime.

The primary improvements included:

- Replacing full historical market reloads with incremental updates
- Downloading only a recent overlapping date window
- Limiting production inference to the latest available trading date
- Reducing the inference dataset from approximately 1.83 million rows to 499 current stock observations
- Preserving identical prediction outputs while reducing database and compute workload

The feature-engineering layer still maintains the historical data required for rolling calculations, while the final inference table contains only the latest eligible observation for each stock.

---

## Context-Aware Modeling V3

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

Likewise, a stock falling alongside a broad market selloff may not be exhibiting unusual company-specific risk.

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
- Worst recent return
- Drawdown from the 60-day high
- Distance from the 52-week high
- Price relative to the 200-day moving average

#### Market Regime Features

- SPY 20-day return
- SPY 60-day return
- SPY 20-day volatility
- SPY 60-day volatility
- SPY drawdown from the 60-day high

#### Market Breadth Features

- Percentage of stocks with positive 20-day returns
- Percentage of stocks above MA50
- Percentage of stocks above MA200

#### Sector Context Features

- 20-day return relative to sector
- 60-day return relative to sector
- 20-day volatility relative to sector

#### Market Sensitivity Features

- Rolling 60-day beta
- Rolling 60-day market correlation

#### Cross-Sectional Ranking Features

- 20-day return percentile
- 20-day volatility percentile
- Drawdown percentile
- Sector return percentile
- Sector volatility percentile

### Dataset

- Approximately 1.83 million observations
- Approximately 500 S&P 500 stocks
- Historical coverage from 2010–2026
- Chronological train, validation, and test framework

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

Performance improvements:

```text
ROC-AUC: +0.0219
PR-AUC : +0.0152
```

The results validate the importance of context-aware feature engineering and demonstrate that broader market conditions materially improve downside-risk prediction.

---

## Streamlit Application

The dashboard tracks approximately 500 stocks and surfaces:

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

Functionality includes:

- Risk score
- Risk percentile
- Risk level
- Latest price
- Historical risk tracking
- Risk-driver analysis
- AI-generated risk briefs
- Model explanations

### Top Risk Stocks

Ranks securities by current modeled downside risk.

Functionality includes:

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

RiskAtlas also includes a Gemini-powered conversational assistant that allows users to ask questions about current stock-risk conditions using model outputs and engineered risk drivers.

---

## Tech Stack

### Programming

- Python
- pandas
- NumPy

### Data Engineering

- PostgreSQL
- Google Cloud SQL
- SQL
- SQL window functions
- Incremental data ingestion

### Cloud Infrastructure

- Google Cloud Platform (GCP)
- Cloud SQL
- Cloud Storage
- Cloud Database Migration

### Machine Learning

- scikit-learn
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- joblib

### Visualization

- Streamlit
- Plotly

### AI

- Google Gemini
- Context-aware RiskAtlas assistant
- LLM-powered risk explanations
- Conversational risk analysis
- Model-aware risk briefs

### Development

- Git
- GitHub
- Python virtual environments

---

## Project Structure

```text
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
│   ├── schema/
│   ├── staging/
│   ├── marts/
│   ├── analytics/
│   ├── features/
│   │   ├── price_features.sql
│   │   ├── model_dataset.sql
│   │   ├── label_generation.sql
│   │   ├── feature_engineering_v3.sql
│   │   └── inference_engineering_v3.sql
│   └── models/
│
├── src/
│   ├── ai/
│   │   ├── ai_explanations.py
│   │   └── risk_drivers.py
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
```

---

## How To Run

### Clone the Repository

```bash
git clone https://github.com/nathanho27/RiskAtlas.git
cd RiskAtlas
```

### Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure PostgreSQL

Configure the database connection through the `DATABASE_URL` environment variable.

```bash
export DATABASE_URL="postgresql://username:password@host:5432/database"
```

The Gemini explanation layer requires a Gemini API key:

```bash
export GEMINI_API_KEY="your-api-key"
```

### Run the Production Pipeline

```bash
python run_pipeline.py
```

This command:

- Refreshes market data
- Refreshes market and company context
- Executes the SQL transformation pipeline
- Rebuilds the latest inference dataset
- Loads the Random Forest V3 model
- Generates current risk predictions
- Updates current and historical prediction tables

### Launch the Dashboard

```bash
streamlit run src/app/app.py
```

### Run Model Scripts Independently

V3 model benchmarking:

```bash
python src/models/model_training_v3.py
```

V3 Random Forest tuning:

```bash
python src/models/random_forest_tuning_v3.py
```

V3 prediction generation:

```bash
python src/models/prediction_v3.py
```

---

## What I Learned

The biggest lesson from this project was that better features often matter more than more sophisticated models.

My initial instinct was to improve performance by trying increasingly complex machine-learning algorithms.

However, the largest gains came from improving the information available to the model.

Moving from stock-centric features to context-aware features produced a larger improvement than switching model families.

The V3 experiments showed that market regime, market breadth, and relative positioning contain meaningful predictive information that traditional stock-level indicators often miss.

The production optimization process also reinforced the importance of designing efficient data workflows rather than repeatedly processing an entire historical dataset.

By introducing incremental updates and latest-date inference, RiskAtlas reduced production runtime from approximately 10 minutes to 1.3 minutes while preserving identical prediction outputs.

This project reinforced the importance of:

- Feature engineering
- Proper train, validation, and test methodology
- Out-of-sample evaluation
- Context-aware modeling
- Incremental data processing
- Efficient production inference
- Modular application development
- Building complete end-to-end systems rather than isolated models

The project also provided hands-on experience with cloud database migration, production PostgreSQL infrastructure, and deploying analytics systems on Google Cloud Platform.

RiskAtlas ultimately became much more than a machine-learning experiment.

It evolved into a full-stack data science project combining data engineering, analytics, machine learning, application development, production optimization, and AI-assisted interpretation into a single workflow.
