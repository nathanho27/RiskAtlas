-- stg_market_prices.sql
-- cleans and standardizes raw market data

DROP TABLE IF EXISTS stg_market_prices;

CREATE TABLE stg_market_prices AS

SELECT DISTINCT
    date,
    UPPER(TRIM(ticker)) AS ticker,
    adj_close

FROM raw_market_prices

WHERE adj_close IS NOT NULL
AND adj_close>0;

ALTER TABLE stg_market_prices
ADD PRIMARY KEY(date,ticker);

CREATE INDEX idx_stg_ticker_date
ON stg_market_prices(ticker,date);