-- price_features.sql
-- builds rolling return, volatility, and trend features
-- used as model inputs for risk prediction

DROP TABLE IF EXISTS price_features;

CREATE TABLE price_features AS

-- calculate daily returns for each stock
WITH base AS (
    SELECT
        date,
        ticker,
        adj_close,

        adj_close / LAG(adj_close) OVER (
            PARTITION BY ticker
            ORDER BY date
        ) - 1 AS daily_return

    FROM stg_market_prices
),

-- build rolling statistics used for feature engineering
rolling_features AS (
    SELECT
        date,
        ticker,
        adj_close,
        daily_return,

        -- count observations so we only keep complete windows
        COUNT(daily_return) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS return_count_20,

        COUNT(daily_return) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS return_count_30,

        COUNT(daily_return) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
        ) AS return_count_60,

        COUNT(adj_close) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS price_count_20,

        COUNT(adj_close) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
        ) AS price_count_50,

        COUNT(adj_close) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 199 PRECEDING AND CURRENT ROW
        ) AS price_count_200,

        -- rolling volatility measures
        STDDEV(daily_return) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS raw_vol_20,

        STDDEV(daily_return) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS raw_vol_30,

        STDDEV(daily_return) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
        ) AS raw_vol_60,

        -- moving averages used to capture trend
        AVG(adj_close) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS raw_ma_20,

        AVG(adj_close) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
        ) AS raw_ma_50,

        AVG(adj_close) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 199 PRECEDING AND CURRENT ROW
        ) AS raw_ma_200

    FROM base
),

-- remove partial windows to improve feature quality
validated_features AS (
    SELECT
        date,
        ticker,
        adj_close,
        daily_return,

        CASE WHEN return_count_20 = 20 THEN raw_vol_20 END AS vol_20,
        CASE WHEN return_count_30 = 30 THEN raw_vol_30 END AS vol_30,
        CASE WHEN return_count_60 = 60 THEN raw_vol_60 END AS vol_60,

        CASE WHEN price_count_20 = 20 THEN raw_ma_20 END AS ma_20,
        CASE WHEN price_count_50 = 50 THEN raw_ma_50 END AS ma_50,
        CASE WHEN price_count_200 = 200 THEN raw_ma_200 END AS ma_200

    FROM rolling_features
)

-- final feature table used for model development
SELECT
    date,
    ticker,
    adj_close,
    daily_return,
    vol_20,
    vol_30,
    vol_60,
    ma_20,
    ma_50,
    ma_200,

    adj_close / ma_50 AS price_to_ma50,
    adj_close / ma_200 AS price_to_ma200

FROM validated_features;

-- enforce uniqueness for each stock and trading day
ALTER TABLE price_features
ADD PRIMARY KEY(date,ticker);

-- improve performance for downstream queries
CREATE INDEX idx_price_features_ticker_date
ON price_features(ticker,date);