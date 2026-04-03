-- label_generation.sql
-- creates forward-looking risk event label

DROP TABLE IF EXISTS labels;

CREATE TABLE labels AS

SELECT
    date,
    ticker,
    adj_close,

    LEAD(adj_close,10) OVER (PARTITION BY ticker ORDER BY date)/adj_close-1 AS future_return_10d,

    CASE
        WHEN LEAD(adj_close,10) OVER (PARTITION BY ticker ORDER BY date)/adj_close-1<-0.05 THEN 1
        ELSE 0
    END AS risk_event

FROM stg_market_prices;