-- label_generation.sql
-- creates the forward-looking target used by the risk models
-- risk_event = 1 when a stock falls more than 5%
-- over the next 10 trading days

DROP TABLE IF EXISTS labels;

CREATE TABLE labels AS

WITH future_returns AS (
    SELECT
        date,
        ticker,
        adj_close,

        LEAD(adj_close, 10) OVER (
            PARTITION BY ticker
            ORDER BY date
        ) AS future_price_10d

    FROM stg_market_prices
)

SELECT
    date,
    ticker,
    adj_close,

    future_price_10d / NULLIF(adj_close, 0) - 1
        AS future_return_10d,

    CASE
        WHEN future_price_10d IS NULL THEN NULL
        WHEN future_price_10d / NULLIF(adj_close, 0) - 1 < -0.05 THEN 1
        ELSE 0
    END AS risk_event

FROM future_returns;

ALTER TABLE labels
ADD PRIMARY KEY (date, ticker);

CREATE INDEX idx_labels_ticker_date
ON labels (ticker, date);