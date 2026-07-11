-- market_metrics.sql
-- aggregates stock-level features into market-wide indicators

DROP TABLE IF EXISTS market_metrics;

CREATE TABLE market_metrics AS

SELECT
    date,

    -- average return across all stocks
    AVG(daily_return) AS avg_return,

    -- cross-sectional dispersion of returns
    STDDEV(daily_return) AS cross_sectional_vol,

    -- average rolling volatility across the market
    AVG(vol_30) AS avg_vol_30,

    -- percentage of stocks trading above their 200-day moving average
    AVG(
        CASE
            WHEN price_to_ma200 IS NULL THEN NULL
            WHEN price_to_ma200 > 1 THEN 1
            ELSE 0
        END
    ) AS pct_above_ma200,

    -- percentage of stocks trading above their 50-day moving average
    AVG(
        CASE
            WHEN price_to_ma50 IS NULL THEN NULL
            WHEN price_to_ma50 > 1 THEN 1
            ELSE 0
        END
    ) AS pct_above_ma50

FROM price_features

GROUP BY date;

-- one row per trading day
ALTER TABLE market_metrics
ADD PRIMARY KEY(date);