-- price_features.sql
-- builds financial features from staged market data

DROP TABLE IF EXISTS price_features;

CREATE TABLE price_features AS

WITH base AS (
    SELECT
        date,
        ticker,
        adj_close,

        adj_close / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS daily_return

    FROM stg_market_prices
),

volatility AS (
    SELECT
        *,

        STDDEV(daily_return) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS vol_20,
        STDDEV(daily_return) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS vol_30,
        STDDEV(daily_return) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS vol_60

    FROM base
),

moving_avg AS (
    SELECT
        *,

        AVG(adj_close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma_20,
        AVG(adj_close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS ma_50,
        AVG(adj_close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS ma_200

    FROM volatility
)

SELECT
    *,

    adj_close / ma_50 AS price_to_ma50,
    adj_close / ma_200 AS price_to_ma200

FROM moving_avg;