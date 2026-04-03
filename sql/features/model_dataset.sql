-- model_dataset.sql
-- joins features with labels for ML

DROP TABLE IF EXISTS model_dataset;

CREATE TABLE model_dataset AS

SELECT
    f.*,
    l.risk_event

FROM price_features f
JOIN labels l
ON f.date=l.date AND f.ticker=l.ticker

WHERE l.future_return_10d IS NOT NULL;