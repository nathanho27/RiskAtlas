-- raw_market_prices.sql
-- defines the raw market prices table structure

DROP TABLE IF EXISTS raw_market_prices;

CREATE TABLE raw_market_prices(
    date DATE,
    ticker TEXT,
    adj_close DOUBLE PRECISION
);