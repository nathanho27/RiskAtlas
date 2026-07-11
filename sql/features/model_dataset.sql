-- model_dataset.sql
-- joins historical features with the forward-looking risk label
-- only keeps rows with complete feature windows and known outcomes

DROP TABLE IF EXISTS model_dataset;

CREATE TABLE model_dataset AS

SELECT
    f.date,
    f.ticker,
    f.adj_close,
    f.daily_return,
    f.vol_20,
    f.vol_30,
    f.vol_60,
    f.ma_20,
    f.ma_50,
    f.ma_200,
    f.price_to_ma50,
    f.price_to_ma200,
    l.risk_event

FROM price_features f

JOIN labels l
    ON f.date=l.date
    AND f.ticker=l.ticker

-- remove rows without a known future outcome
WHERE l.risk_event IS NOT NULL

-- remove early rows that do not have complete rolling windows
AND f.daily_return IS NOT NULL
AND f.vol_20 IS NOT NULL
AND f.vol_30 IS NOT NULL
AND f.vol_60 IS NOT NULL
AND f.ma_20 IS NOT NULL
AND f.ma_50 IS NOT NULL
AND f.ma_200 IS NOT NULL
AND f.price_to_ma50 IS NOT NULL
AND f.price_to_ma200 IS NOT NULL;

-- enforce one observation per stock and trading day
ALTER TABLE model_dataset
ADD PRIMARY KEY(date,ticker);

-- improve performance when loading and splitting data by date
CREATE INDEX idx_model_dataset_date
ON model_dataset(date);

-- improve performance for ticker-level analysis
CREATE INDEX idx_model_dataset_ticker_date
ON model_dataset(ticker,date);