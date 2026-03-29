-- stg_market_prices.sql
-- cleans and standardizes raw market data

DROP TABLE IF EXISTS stg_market_prices;

CREATE TABLE stg_market_prices AS

SELECT
    date,
    UPPER(ticker) AS ticker,
    adj_close

FROM raw_market_prices
WHERE adj_close IS NOT NULL;