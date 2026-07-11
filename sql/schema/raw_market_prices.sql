-- raw_market_prices.sql
-- defines the raw market prices table structure

DROP TABLE IF EXISTS raw_market_prices;

CREATE TABLE raw_market_prices(
    date DATE NOT NULL,
    ticker TEXT NOT NULL,
    adj_close DOUBLE PRECISION NOT NULL,
    PRIMARY KEY(date,ticker)
);