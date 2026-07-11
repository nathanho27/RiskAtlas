DROP TABLE IF EXISTS model_dataset_v2;

CREATE TABLE model_dataset_v2 AS

SELECT
    pf.date,
    pf.ticker,
    pf.adj_close,

    -- Baseline features
    pf.daily_return,
    pf.vol_20,
    pf.vol_30,
    pf.vol_60,
    pf.price_to_ma50,
    pf.price_to_ma200,

    -- New momentum features
    pf.return_5d,
    pf.return_20d,
    pf.return_60d,

    -- New downside-risk features
    pf.downside_vol_20,
    pf.negative_return_count_20,
    pf.worst_return_20,
    pf.drawdown_from_60d_high,
    pf.distance_from_52w_high,

    -- Existing broader drawdown feature
    rs.drawdown,

    -- Target data
    l.future_return_10d,
    l.risk_event

FROM price_features pf

INNER JOIN labels l
    ON pf.date = l.date
    AND pf.ticker = l.ticker

LEFT JOIN risk_signals rs
    ON pf.date = rs.date
    AND pf.ticker = rs.ticker

WHERE l.risk_event IS NOT NULL;

ALTER TABLE model_dataset_v2
ADD PRIMARY KEY(date, ticker);

CREATE INDEX idx_model_dataset_v2_ticker_date
ON model_dataset_v2(ticker, date);