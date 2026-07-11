-- price_features.sql
-- builds momentum, volatility, trend, and downside-risk features
-- used as model inputs for risk prediction

DROP TABLE IF EXISTS price_features CASCADE;

CREATE TABLE price_features AS

WITH base AS (
    SELECT
        date,
        ticker,
        adj_close,

        adj_close
        / NULLIF(
            LAG(adj_close, 1) OVER (
                PARTITION BY ticker
                ORDER BY date
            ),
            0
        ) - 1 AS daily_return,

        adj_close
        / NULLIF(
            LAG(adj_close, 5) OVER (
                PARTITION BY ticker
                ORDER BY date
            ),
            0
        ) - 1 AS return_5d,

        adj_close
        / NULLIF(
            LAG(adj_close, 20) OVER (
                PARTITION BY ticker
                ORDER BY date
            ),
            0
        ) - 1 AS return_20d,

        adj_close
        / NULLIF(
            LAG(adj_close, 60) OVER (
                PARTITION BY ticker
                ORDER BY date
            ),
            0
        ) - 1 AS return_60d

    FROM stg_market_prices
),

rolling_features AS (
    SELECT
        date,
        ticker,
        adj_close,
        daily_return,
        return_5d,
        return_20d,
        return_60d,

        -- Complete-window validation
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
            ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
        ) AS price_count_60,

        COUNT(adj_close) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 199 PRECEDING AND CURRENT ROW
        ) AS price_count_200,

        COUNT(adj_close) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
        ) AS price_count_252,

        -- Rolling volatility
        STDDEV_SAMP(daily_return) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS raw_vol_20,

        STDDEV_SAMP(daily_return) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS raw_vol_30,

        STDDEV_SAMP(daily_return) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
        ) AS raw_vol_60,

        -- Volatility using only negative-return days
        STDDEV_SAMP(
            CASE
                WHEN daily_return < 0 THEN daily_return
            END
        ) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS raw_downside_vol_20,

        -- Number of negative sessions in trailing 20 days
        SUM(
            CASE
                WHEN daily_return < 0 THEN 1
                ELSE 0
            END
        ) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS raw_negative_return_count_20,

        -- Worst daily return in trailing 20 days
        MIN(daily_return) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS raw_worst_return_20,

        -- Moving averages
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
        ) AS raw_ma_200,

        -- Rolling price highs
        MAX(adj_close) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
        ) AS raw_high_60,

        MAX(adj_close) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
        ) AS raw_high_252

    FROM base
),

validated_features AS (
    SELECT
        date,
        ticker,
        adj_close,
        daily_return,
        return_5d,
        return_20d,
        return_60d,

        CASE
            WHEN return_count_20 = 20 THEN raw_vol_20
        END AS vol_20,

        CASE
            WHEN return_count_30 = 30 THEN raw_vol_30
        END AS vol_30,

        CASE
            WHEN return_count_60 = 60 THEN raw_vol_60
        END AS vol_60,

        CASE
            WHEN return_count_20 = 20 THEN raw_downside_vol_20
        END AS downside_vol_20,

        CASE
            WHEN return_count_20 = 20 THEN raw_negative_return_count_20
        END AS negative_return_count_20,

        CASE
            WHEN return_count_20 = 20 THEN raw_worst_return_20
        END AS worst_return_20,

        CASE
            WHEN price_count_20 = 20 THEN raw_ma_20
        END AS ma_20,

        CASE
            WHEN price_count_50 = 50 THEN raw_ma_50
        END AS ma_50,

        CASE
            WHEN price_count_200 = 200 THEN raw_ma_200
        END AS ma_200,

        CASE
            WHEN price_count_60 = 60 THEN raw_high_60
        END AS high_60,

        CASE
            WHEN price_count_252 = 252 THEN raw_high_252
        END AS high_252

    FROM rolling_features
)

SELECT
    date,
    ticker,
    adj_close,

    -- Return and momentum features
    daily_return,
    return_5d,
    return_20d,
    return_60d,

    -- Volatility features
    vol_20,
    vol_30,
    vol_60,
    downside_vol_20,

    -- Downside behavior
    negative_return_count_20,
    worst_return_20,

    -- Trend features
    ma_20,
    ma_50,
    ma_200,
    adj_close / NULLIF(ma_50, 0) AS price_to_ma50,
    adj_close / NULLIF(ma_200, 0) AS price_to_ma200,

    -- Drawdown and proximity-to-high features
    adj_close / NULLIF(high_60, 0) - 1 AS drawdown_from_60d_high,
    adj_close / NULLIF(high_252, 0) - 1 AS distance_from_52w_high

FROM validated_features;

ALTER TABLE price_features
ADD PRIMARY KEY(date, ticker);

CREATE INDEX idx_price_features_ticker_date
ON price_features(ticker, date);