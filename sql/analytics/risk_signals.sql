-- risk_signals.sql
-- builds market risk indicators from engineered features

DROP TABLE IF EXISTS risk_signals;

CREATE TABLE risk_signals AS

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

        -- rolling peak for drawdown
        MAX(adj_close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rolling_peak

    FROM price_features
),

drawdowns AS (
    SELECT
        *,

        -- drawdown from peak
        (adj_close / rolling_peak) - 1 AS drawdown

    FROM base
),

regimes AS (
    SELECT
        *,

        -- volatility regime
        CASE
            WHEN vol_30 > 0.03 THEN 'high_vol'
            WHEN vol_30 < 0.015 THEN 'low_vol'
            ELSE 'normal_vol'
        END AS vol_regime,

        -- trend regime
        CASE
            WHEN price_to_ma200 > 1 THEN 'uptrend'
            ELSE 'downtrend'
        END AS trend_regime

    FROM drawdowns
)

SELECT
    *,

    -- combined risk signal
    CASE
        WHEN vol_30 > 0.03 AND price_to_ma200 < 1 THEN 'risk_off'
        WHEN vol_30 < 0.015 AND price_to_ma200 > 1 THEN 'risk_on'
        ELSE 'neutral'
    END AS risk_signal

FROM regimes;