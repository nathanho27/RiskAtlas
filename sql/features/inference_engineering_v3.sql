-- inference_engineering_v3.sql
-- builds the context-aware V3 inference dataset
-- includes market, breadth, sector, beta, and correlation features
-- excludes labels so the newest available market date can be scored

DROP TABLE IF EXISTS inference_dataset_v3;

CREATE TABLE inference_dataset_v3 AS

WITH spy_returns AS (

    SELECT
        date,
        adj_close AS spy_adj_close,

        adj_close
            / NULLIF(LAG(adj_close,1) OVER (
                ORDER BY date
            ),0) - 1
            AS spy_daily_return,

        adj_close
            / NULLIF(LAG(adj_close,5) OVER (
                ORDER BY date
            ),0) - 1
            AS spy_return_5d,

        adj_close
            / NULLIF(LAG(adj_close,20) OVER (
                ORDER BY date
            ),0) - 1
            AS spy_return_20d,

        adj_close
            / NULLIF(LAG(adj_close,60) OVER (
                ORDER BY date
            ),0) - 1
            AS spy_return_60d

    FROM market_benchmark_prices

    WHERE ticker = 'SPY'
),

spy_features AS (

    SELECT
        date,
        spy_adj_close,
        spy_daily_return,
        spy_return_5d,
        spy_return_20d,
        spy_return_60d,

        STDDEV_SAMP(spy_daily_return) OVER (
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS spy_vol_20,

        STDDEV_SAMP(spy_daily_return) OVER (
            ORDER BY date
            ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
        ) AS spy_vol_60,

        spy_adj_close
            / NULLIF(MAX(spy_adj_close) OVER (
                ORDER BY date
                ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
            ),0) - 1
            AS spy_drawdown_from_60d_high

    FROM spy_returns
),

market_breadth AS (

    SELECT
        date,

        AVG(
            CASE
                WHEN daily_return > 0 THEN 1.0
                ELSE 0.0
            END
        ) AS pct_positive_daily,

        AVG(
            CASE
                WHEN return_20d > 0 THEN 1.0
                ELSE 0.0
            END
        ) AS pct_positive_20d,

        AVG(
            CASE
                WHEN adj_close > ma_50 THEN 1.0
                ELSE 0.0
            END
        ) AS pct_above_ma50,

        AVG(
            CASE
                WHEN adj_close > ma_200 THEN 1.0
                ELSE 0.0
            END
        ) AS pct_above_ma200,

        AVG(return_20d) AS market_avg_return_20d,
        AVG(return_60d) AS market_avg_return_60d,
        AVG(vol_20) AS market_avg_vol_20

    FROM price_features

    GROUP BY date
),

sector_features AS (

    SELECT
        f.date,
        m.sector,

        AVG(f.return_5d) AS sector_avg_return_5d,
        AVG(f.return_20d) AS sector_avg_return_20d,
        AVG(f.return_60d) AS sector_avg_return_60d,

        AVG(f.vol_20) AS sector_avg_vol_20,
        AVG(f.downside_vol_20) AS sector_avg_downside_vol_20,

        AVG(
            CASE
                WHEN f.return_20d > 0 THEN 1.0
                ELSE 0.0
            END
        ) AS sector_pct_positive_20d,

        AVG(
            CASE
                WHEN f.adj_close > f.ma_50 THEN 1.0
                ELSE 0.0
            END
        ) AS sector_pct_above_ma50

    FROM price_features f

    JOIN ticker_metadata m
        ON f.ticker = m.ticker

    GROUP BY
        f.date,
        m.sector
),

stock_market_history AS (

    SELECT
        f.date,
        f.ticker,
        f.daily_return,
        s.spy_daily_return

    FROM price_features f

    JOIN spy_features s
        ON f.date = s.date
),

stock_market_relationships AS (

    SELECT
        date,
        ticker,

        COVAR_SAMP(
            daily_return,
            spy_daily_return
        ) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        )
        / NULLIF(
            VAR_SAMP(spy_daily_return) OVER (
                PARTITION BY ticker
                ORDER BY date
                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            ),
            0
        ) AS beta_20,

        CORR(
            daily_return,
            spy_daily_return
        ) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS correlation_20,

        COVAR_SAMP(
            daily_return,
            spy_daily_return
        ) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
        )
        / NULLIF(
            VAR_SAMP(spy_daily_return) OVER (
                PARTITION BY ticker
                ORDER BY date
                ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
            ),
            0
        ) AS beta_60,

        CORR(
            daily_return,
            spy_daily_return
        ) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
        ) AS correlation_60

    FROM stock_market_history
),

engineered_features AS (

    SELECT
        f.date,
        f.ticker,
        m.company_name,
        m.sector,
        m.sub_industry,
        f.adj_close,

        -- stock return features
        f.daily_return,
        f.return_5d,
        f.return_20d,
        f.return_60d,

        -- stock volatility features
        f.vol_20,
        f.vol_30,
        f.vol_60,
        f.downside_vol_20,
        f.negative_return_count_20,
        f.worst_return_20,

        -- stock trend features
        f.ma_20,
        f.ma_50,
        f.ma_200,
        f.price_to_ma50,
        f.price_to_ma200,

        -- stock drawdown features
        f.drawdown_from_60d_high,
        f.distance_from_52w_high,

        -- stock volatility regime
        f.vol_20 / NULLIF(f.vol_60,0)
            AS vol_20_to_vol_60_ratio,

        f.vol_20 - f.vol_60
            AS vol_20_minus_vol_60,

        -- stock momentum acceleration
        f.return_5d - f.return_20d
            AS short_term_acceleration,

        f.return_20d - f.return_60d
            AS medium_term_acceleration,

        -- SPY features
        s.spy_daily_return,
        s.spy_return_5d,
        s.spy_return_20d,
        s.spy_return_60d,
        s.spy_vol_20,
        s.spy_vol_60,
        s.spy_drawdown_from_60d_high,

        -- performance relative to SPY
        f.daily_return - s.spy_daily_return
            AS excess_return_daily,

        f.return_5d - s.spy_return_5d
            AS excess_return_5d,

        f.return_20d - s.spy_return_20d
            AS excess_return_20d,

        f.return_60d - s.spy_return_60d
            AS excess_return_60d,

        -- volatility relative to SPY
        f.vol_20 / NULLIF(s.spy_vol_20,0)
            AS relative_vol_20,

        f.vol_60 / NULLIF(s.spy_vol_60,0)
            AS relative_vol_60,

        f.vol_20 - s.spy_vol_20
            AS excess_vol_20,

        -- market breadth features
        b.pct_positive_daily,
        b.pct_positive_20d,
        b.pct_above_ma50,
        b.pct_above_ma200,
        b.market_avg_return_20d,
        b.market_avg_return_60d,
        b.market_avg_vol_20,

        -- performance relative to the daily universe
        f.return_20d - b.market_avg_return_20d
            AS return_20d_vs_market,

        f.return_60d - b.market_avg_return_60d
            AS return_60d_vs_market,

        f.vol_20 / NULLIF(b.market_avg_vol_20,0)
            AS vol_20_vs_market,

        -- sector features
        sec.sector_avg_return_5d,
        sec.sector_avg_return_20d,
        sec.sector_avg_return_60d,
        sec.sector_avg_vol_20,
        sec.sector_avg_downside_vol_20,
        sec.sector_pct_positive_20d,
        sec.sector_pct_above_ma50,

        -- performance relative to sector
        f.return_5d - sec.sector_avg_return_5d
            AS return_5d_vs_sector,

        f.return_20d - sec.sector_avg_return_20d
            AS return_20d_vs_sector,

        f.return_60d - sec.sector_avg_return_60d
            AS return_60d_vs_sector,

        f.vol_20 / NULLIF(sec.sector_avg_vol_20,0)
            AS vol_20_vs_sector,

        f.downside_vol_20
            / NULLIF(sec.sector_avg_downside_vol_20,0)
            AS downside_vol_20_vs_sector,

        -- rolling market sensitivity
        rel.beta_20,
        rel.beta_60,
        rel.correlation_20,
        rel.correlation_60

    FROM price_features f

    JOIN ticker_metadata m
        ON f.ticker = m.ticker

    JOIN spy_features s
        ON f.date = s.date

    JOIN market_breadth b
        ON f.date = b.date

    JOIN sector_features sec
        ON f.date = sec.date
        AND m.sector = sec.sector

    JOIN stock_market_relationships rel
        ON f.date = rel.date
        AND f.ticker = rel.ticker
),

ranked_features AS (

    SELECT
        e.*,

        -- daily universe rankings
        PERCENT_RANK() OVER (
            PARTITION BY e.date
            ORDER BY e.return_20d
        ) AS return_20d_percentile,

        PERCENT_RANK() OVER (
            PARTITION BY e.date
            ORDER BY e.return_60d
        ) AS return_60d_percentile,

        PERCENT_RANK() OVER (
            PARTITION BY e.date
            ORDER BY e.vol_20
        ) AS vol_20_percentile,

        PERCENT_RANK() OVER (
            PARTITION BY e.date
            ORDER BY e.downside_vol_20
        ) AS downside_vol_20_percentile,

        PERCENT_RANK() OVER (
            PARTITION BY e.date
            ORDER BY e.drawdown_from_60d_high
        ) AS drawdown_60d_percentile,

        PERCENT_RANK() OVER (
            PARTITION BY e.date
            ORDER BY e.excess_return_20d
        ) AS excess_return_20d_percentile,

        PERCENT_RANK() OVER (
            PARTITION BY e.date
            ORDER BY e.relative_vol_20
        ) AS relative_vol_20_percentile,

        -- sector rankings
        PERCENT_RANK() OVER (
            PARTITION BY e.date, e.sector
            ORDER BY e.return_20d
        ) AS sector_return_20d_percentile,

        PERCENT_RANK() OVER (
            PARTITION BY e.date, e.sector
            ORDER BY e.vol_20
        ) AS sector_vol_20_percentile,

        PERCENT_RANK() OVER (
            PARTITION BY e.date, e.sector
            ORDER BY e.drawdown_from_60d_high
        ) AS sector_drawdown_60d_percentile

    FROM engineered_features e
)

SELECT
    r.*

FROM ranked_features r

-- require complete stock history
WHERE r.daily_return IS NOT NULL
AND r.return_5d IS NOT NULL
AND r.return_20d IS NOT NULL
AND r.return_60d IS NOT NULL
AND r.vol_20 IS NOT NULL
AND r.vol_30 IS NOT NULL
AND r.vol_60 IS NOT NULL
AND r.downside_vol_20 IS NOT NULL
AND r.worst_return_20 IS NOT NULL
AND r.ma_50 IS NOT NULL
AND r.ma_200 IS NOT NULL
AND r.price_to_ma200 IS NOT NULL
AND r.drawdown_from_60d_high IS NOT NULL
AND r.distance_from_52w_high IS NOT NULL

-- require complete market context
AND r.spy_return_5d IS NOT NULL
AND r.spy_return_20d IS NOT NULL
AND r.spy_return_60d IS NOT NULL
AND r.spy_vol_20 IS NOT NULL
AND r.spy_vol_60 IS NOT NULL
AND r.spy_drawdown_from_60d_high IS NOT NULL

-- require complete relative features
AND r.excess_return_5d IS NOT NULL
AND r.excess_return_20d IS NOT NULL
AND r.excess_return_60d IS NOT NULL
AND r.relative_vol_20 IS NOT NULL
AND r.relative_vol_60 IS NOT NULL

-- require complete breadth and sector context
AND r.pct_positive_20d IS NOT NULL
AND r.pct_above_ma50 IS NOT NULL
AND r.pct_above_ma200 IS NOT NULL
AND r.sector_avg_return_20d IS NOT NULL
AND r.sector_avg_vol_20 IS NOT NULL
AND r.return_20d_vs_sector IS NOT NULL
AND r.return_60d_vs_sector IS NOT NULL
AND r.vol_20_vs_sector IS NOT NULL

-- require complete market relationships
AND r.beta_20 IS NOT NULL
AND r.beta_60 IS NOT NULL
AND r.correlation_20 IS NOT NULL
AND r.correlation_60 IS NOT NULL

-- require every feature used by the Random Forest V3 artifact
AND r.return_20d_percentile IS NOT NULL
AND r.vol_20_percentile IS NOT NULL
AND r.drawdown_60d_percentile IS NOT NULL
AND r.sector_return_20d_percentile IS NOT NULL
AND r.sector_vol_20_percentile IS NOT NULL;

-- enforce one observation per stock and trading day
ALTER TABLE inference_dataset_v3
ADD PRIMARY KEY (date, ticker);

-- improve latest-row lookup performance
CREATE INDEX idx_inference_dataset_v3_date
ON inference_dataset_v3 (date);

CREATE INDEX idx_inference_dataset_v3_ticker_date
ON inference_dataset_v3 (ticker, date);

CREATE INDEX idx_inference_dataset_v3_sector_date
ON inference_dataset_v3 (sector, date);

ANALYZE inference_dataset_v3;