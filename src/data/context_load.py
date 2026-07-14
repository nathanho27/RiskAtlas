# context_load.py
# loads SPY benchmark prices and S&P 500 sector metadata
# keeps all context data separate from the production stock universe

from io import StringIO

import os
import pandas as pd
import psycopg2
import requests
import yfinance as yf
from psycopg2.extras import execute_values


DB_NAME = "risk_atlas"
DB_USER = "nathanho"
DB_HOST = "localhost"
DB_PORT = "5432"


def get_connection():
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        return psycopg2.connect(
            database_url,
            sslmode="require",
        )

    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host=DB_HOST,
        port=DB_PORT,
    )

def get_benchmark_prices(start_date="2010-01-01"):
    raw = yf.download(
        "SPY",
        start=start_date,
        progress=False,
        threads=False,
        auto_adjust=True,
    )

    if raw.empty:
        raise ValueError("No SPY benchmark data downloaded")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()

        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
    else:
        prices = raw["Close"].copy()

    benchmark_df = prices.reset_index()
    benchmark_df.columns = ["date", "adj_close"]

    benchmark_df["date"] = pd.to_datetime(
        benchmark_df["date"]
    ).dt.date

    benchmark_df["ticker"] = "SPY"

    benchmark_df = benchmark_df[
        ["date", "ticker", "adj_close"]
    ].dropna()

    benchmark_df["adj_close"] = benchmark_df[
        "adj_close"
    ].astype(float)

    return benchmark_df


def get_ticker_metadata():
    url = (
        "https://en.wikipedia.org/wiki/"
        "List_of_S%26P_500_companies"
    )

    headers = {
        "User-Agent": "Mozilla/5.0",
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    tables = pd.read_html(
        StringIO(response.text),
        match="Symbol",
    )

    table = tables[0]

    metadata_df = table[
        [
            "Symbol",
            "Security",
            "GICS Sector",
            "GICS Sub-Industry",
        ]
    ].copy()

    metadata_df = metadata_df.rename(
        columns={
            "Symbol": "ticker",
            "Security": "company_name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
        }
    )

    metadata_df["ticker"] = (
        metadata_df["ticker"]
        .astype(str)
        .str.replace(".", "-", regex=False)
        .str.upper()
    )

    metadata_df = metadata_df.dropna(
        subset=[
            "ticker",
            "company_name",
            "sector",
            "sub_industry",
        ]
    )

    metadata_df = metadata_df.drop_duplicates(
        subset=["ticker"]
    )

    return metadata_df


def save_benchmark_prices(df):
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DROP TABLE IF EXISTS market_benchmark_prices;

                CREATE TABLE market_benchmark_prices (
                    date DATE NOT NULL,
                    ticker TEXT NOT NULL,
                    adj_close DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY (date, ticker)
                );
                """
            )

            records = list(
                df[
                    [
                        "date",
                        "ticker",
                        "adj_close",
                    ]
                ].itertuples(index=False, name=None)
            )

            execute_values(
                cursor,
                """
                INSERT INTO market_benchmark_prices (
                    date,
                    ticker,
                    adj_close
                )
                VALUES %s
                """,
                records,
                page_size=1000,
            )

            cursor.execute(
                """
                CREATE INDEX idx_market_benchmark_date
                ON market_benchmark_prices(date);
                """
            )

        connection.commit()


def save_ticker_metadata(df):
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DROP TABLE IF EXISTS ticker_metadata;

                CREATE TABLE ticker_metadata (
                    ticker TEXT PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    sector TEXT NOT NULL,
                    sub_industry TEXT NOT NULL
                );
                """
            )

            records = list(
                df[
                    [
                        "ticker",
                        "company_name",
                        "sector",
                        "sub_industry",
                    ]
                ].itertuples(index=False, name=None)
            )

            execute_values(
                cursor,
                """
                INSERT INTO ticker_metadata (
                    ticker,
                    company_name,
                    sector,
                    sub_industry
                )
                VALUES %s
                """,
                records,
                page_size=1000,
            )

            cursor.execute(
                """
                CREATE INDEX idx_ticker_metadata_sector
                ON ticker_metadata(sector);
                """
            )

        connection.commit()


def main():
    benchmark_df = get_benchmark_prices()
    metadata_df = get_ticker_metadata()

    print(f"SPY rows: {len(benchmark_df):,}")
    print(
        "SPY period: "
        f"{benchmark_df['date'].min()} "
        f"to {benchmark_df['date'].max()}"
    )

    print(f"Metadata rows: {len(metadata_df):,}")
    print(
        "Sectors: "
        f"{metadata_df['sector'].nunique()}"
    )

    save_benchmark_prices(benchmark_df)
    save_ticker_metadata(metadata_df)

    print("Context tables saved successfully.")


if __name__ == "__main__":
    main()