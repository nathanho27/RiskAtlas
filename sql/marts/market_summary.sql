-- market_summary.sql
-- combines market-wide metrics into a final reporting mart
-- used for dashboards and the future Streamlit application

DROP TABLE IF EXISTS market_summary;

CREATE TABLE market_summary AS

SELECT
    date,
    avg_return,
    cross_sectional_vol,
    avg_vol_30,
    pct_above_ma200,
    pct_above_ma50,

    -- classify the overall market risk environment
    CASE
        WHEN avg_vol_30 IS NULL OR pct_above_ma200 IS NULL THEN NULL
        WHEN avg_vol_30>0.025 AND pct_above_ma200<0.4 THEN 'high_risk'
        WHEN avg_vol_30<0.015 AND pct_above_ma200>0.6 THEN 'low_risk'
        ELSE 'moderate_risk'
    END AS market_regime,

    -- classify the market's short-term directional bias
    CASE
        WHEN avg_return IS NULL OR pct_above_ma50 IS NULL THEN NULL
        WHEN avg_return>0 AND pct_above_ma50>0.5 THEN 'bullish'
        WHEN avg_return<0 AND pct_above_ma50<0.4 THEN 'bearish'
        ELSE 'neutral'
    END AS market_trend

FROM market_metrics;

-- one summary row per trading day
ALTER TABLE market_summary
ADD PRIMARY KEY(date);