import os

import pandas as pd
import psycopg2
import streamlit as st


@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def load_predictions() -> pd.DataFrame:
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set."
        )

    query = """
        SELECT
            date,
            ticker,
            adj_close,
            risk_score,
            risk_pred,
            risk_level,
            model_name,
            generated_at,
            risk_percentile
        FROM current_risk_predictions_v3
        ORDER BY
            risk_score DESC,
            ticker ASC;
    """

    connection = None

    try:
        connection = psycopg2.connect(
            database_url,
            connect_timeout=10,
        )

        predictions = pd.read_sql_query(
            query,
            connection,
        )

    finally:
        if connection is not None:
            connection.close()

    if predictions.empty:
        return predictions

    predictions["ticker"] = (
        predictions["ticker"]
        .map(
            lambda value: str(value).strip().upper()
        )
        .astype(object)
    )

    predictions["risk_level"] = (
        predictions["risk_level"]
        .map(
            lambda value: str(value).strip().title()
        )
        .astype(object)
    )

    predictions["model_name"] = (
        predictions["model_name"]
        .map(
            lambda value: str(value).strip()
        )
        .astype(object)
    )

    predictions["date"] = pd.to_datetime(
        predictions["date"],
        errors="coerce",
    )

    predictions["generated_at"] = pd.to_datetime(
        predictions["generated_at"],
        errors="coerce",
    )

    numeric_columns = [
        "adj_close",
        "risk_score",
        "risk_pred",
        "risk_percentile",
    ]

    for column in numeric_columns:
        predictions[column] = (
            pd.to_numeric(
                predictions[column],
                errors="coerce",
            )
            .astype("float64")
        )

    predictions = predictions.dropna(
        subset=[
            "date",
            "ticker",
            "adj_close",
            "risk_score",
            "risk_level",
            "risk_percentile",
        ]
    )

    predictions = (
        predictions
        .sort_values(
            by=[
                "ticker",
                "generated_at",
                "date",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .drop_duplicates(
            subset=["ticker"],
            keep="first",
        )
        .sort_values(
            by="risk_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return predictions


def clear_prediction_cache() -> None:
    load_predictions.clear()