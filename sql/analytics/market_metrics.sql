-- market_metrics.sql
-- aggregates ticker-level features into market-wide metrics

DROP TABLE IF EXISTS market_metrics;

CREATE TABLE market_metrics AS

SELECT
    date,

    -- average return across all stocks
    AVG(daily_return) AS avg_return,

    -- cross-sectional volatility (dispersion)
    STDDEV(daily_return) AS cross_sectional_vol,

    -- average rolling volatility
    AVG(vol_30) AS avg_vol_30,

    -- trend breadth: % of stocks above MA200
    AVG(CASE WHEN price_to_ma200 > 1 THEN 1 ELSE 0 END) AS pct_above_ma200,

    -- short-term trend
    AVG(CASE WHEN price_to_ma50 > 1 THEN 1 ELSE 0 END) AS pct_above_ma50

FROM price_features
GROUP BY date;