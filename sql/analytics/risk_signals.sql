-- risk_signals.sql
-- builds stock-level risk indicators from engineered price features

DROP TABLE IF EXISTS risk_signals;

CREATE TABLE risk_signals AS

-- calculate each stock's highest closing price up to the current date
WITH base AS (
    SELECT
        date,
        ticker,
        adj_close,
        daily_return,
        vol_20,
        vol_30,
        vol_60,
        ma_50,
        ma_200,
        price_to_ma50,
        price_to_ma200,

        MAX(adj_close) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS rolling_peak

    FROM price_features
),

-- calculate drawdown from the historical peak
drawdowns AS (
    SELECT
        date,
        ticker,
        adj_close,
        daily_return,
        vol_20,
        vol_30,
        vol_60,
        ma_50,
        ma_200,
        price_to_ma50,
        price_to_ma200,
        rolling_peak,

        adj_close/rolling_peak-1 AS drawdown

    FROM base
),

-- classify volatility and long-term trend conditions
regimes AS (
    SELECT
        *,

        CASE
            WHEN vol_30 IS NULL THEN NULL
            WHEN vol_30>0.03 THEN 'high_vol'
            WHEN vol_30<0.015 THEN 'low_vol'
            ELSE 'normal_vol'
        END AS vol_regime,

        CASE
            WHEN price_to_ma200 IS NULL THEN NULL
            WHEN price_to_ma200>1 THEN 'uptrend'
            ELSE 'downtrend'
        END AS trend_regime

    FROM drawdowns
)

SELECT
    date,
    ticker,
    adj_close,
    daily_return,
    vol_20,
    vol_30,
    vol_60,
    ma_50,
    ma_200,
    price_to_ma50,
    price_to_ma200,
    rolling_peak,
    drawdown,
    vol_regime,
    trend_regime,

    -- combine volatility and trend into a simple risk signal
    CASE
        WHEN vol_30 IS NULL OR price_to_ma200 IS NULL THEN NULL
        WHEN vol_30>0.03 AND price_to_ma200<1 THEN 'risk_off'
        WHEN vol_30<0.015 AND price_to_ma200>1 THEN 'risk_on'
        ELSE 'neutral'
    END AS risk_signal

FROM regimes;

-- enforce one row per stock and trading day
ALTER TABLE risk_signals
ADD PRIMARY KEY(date,ticker);

-- improve performance for ticker-level history and app queries
CREATE INDEX idx_risk_signals_ticker_date
ON risk_signals(ticker,date);