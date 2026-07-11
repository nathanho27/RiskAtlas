-- label_generation.sql
-- creates the forward-looking target used by the risk models
-- risk_event=1 when a stock falls more than 5% over the next 10 trading days

DROP TABLE IF EXISTS labels;

CREATE TABLE labels AS

-- calculate the price and return 10 trading days into the future
WITH future_returns AS (
    SELECT
        date,
        ticker,
        adj_close,

        LEAD(adj_close,10) OVER (
            PARTITION BY ticker
            ORDER BY date
        ) AS future_price_10d

    FROM stg_market_prices
)

SELECT
    date,
    ticker,
    adj_close,

    future_price_10d/adj_close-1 AS future_return_10d,

    -- leave the label null when future price data is unavailable
    CASE
        WHEN future_price_10d IS NULL THEN NULL
        WHEN future_price_10d/adj_close-1<-0.05 THEN 1
        ELSE 0
    END AS risk_event

FROM future_returns;

-- enforce one label per stock and trading day
ALTER TABLE labels
ADD PRIMARY KEY(date,ticker);

-- improve performance when joining labels to model features
CREATE INDEX idx_labels_ticker_date
ON labels(ticker,date);