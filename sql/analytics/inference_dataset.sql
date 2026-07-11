DROP TABLE IF EXISTS inference_dataset;

CREATE TABLE inference_dataset AS
SELECT
    pf.date,
    pf.ticker,
    pf.adj_close,

    pf.daily_return,
    pf.vol_20,
    pf.vol_30,
    pf.vol_60,
    pf.price_to_ma50,
    pf.price_to_ma200,

    pf.return_5d,
    pf.return_20d,
    pf.return_60d,

    pf.downside_vol_20,
    pf.negative_return_count_20,
    pf.worst_return_20,

    pf.drawdown_from_60d_high,
    pf.distance_from_52w_high,

    rs.drawdown

FROM price_features pf
INNER JOIN risk_signals rs
    ON pf.date = rs.date
   AND pf.ticker = rs.ticker;

ALTER TABLE inference_dataset
ADD PRIMARY KEY (date, ticker);

CREATE INDEX idx_inference_dataset_ticker_date
ON inference_dataset (ticker, date);