-- market_summary.sql
-- combines market metrics with overall risk classification

DROP TABLE IF EXISTS market_summary;

CREATE TABLE market_summary AS

SELECT
    m.*,

    -- classify overall market regime
    CASE
        WHEN avg_vol_30 > 0.025 AND pct_above_ma200 < 0.4 THEN 'high_risk'
        WHEN avg_vol_30 < 0.015 AND pct_above_ma200 > 0.6 THEN 'low_risk'
        ELSE 'moderate_risk'
    END AS market_regime,

    -- directional bias
    CASE
        WHEN avg_return > 0 AND pct_above_ma50 > 0.5 THEN 'bullish'
        WHEN avg_return < 0 AND pct_above_ma50 < 0.4 THEN 'bearish'
        ELSE 'neutral'
    END AS market_trend

FROM market_metrics m;