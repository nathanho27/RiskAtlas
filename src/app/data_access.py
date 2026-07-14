import pandas as pd
import psycopg2
import streamlit as st


DB_NAME = "risk_atlas"
DB_USER = "nathanho"
DB_HOST = "localhost"
DB_PORT = "5432"


def get_database_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host=DB_HOST,
        port=DB_PORT,
        connect_timeout=10,
    )


@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def load_predictions() -> pd.DataFrame:
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
        connection = get_database_connection()

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


@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def load_risk_history(
    ticker: str,
    days: int = 90,
) -> pd.DataFrame:
    normalized_ticker = str(ticker).strip().upper()

    if not normalized_ticker:
        return pd.DataFrame()

    query = """
        SELECT
            date,
            ticker,
            adj_close,
            risk_score,
            risk_percentile,
            risk_pred,
            risk_level,
            model_name,
            generated_at
        FROM risk_prediction_history_v3
        WHERE ticker = %s
          AND date >= CURRENT_DATE - %s
        ORDER BY
            date ASC,
            generated_at ASC;
    """

    connection = None

    try:
        connection = get_database_connection()

        history = pd.read_sql_query(
            query,
            connection,
            params=(
                normalized_ticker,
                int(days),
            ),
        )

    finally:
        if connection is not None:
            connection.close()

    if history.empty:
        return history

    history["ticker"] = (
        history["ticker"]
        .map(
            lambda value: str(value).strip().upper()
        )
        .astype(object)
    )

    history["risk_level"] = (
        history["risk_level"]
        .map(
            lambda value: str(value).strip().title()
        )
        .astype(object)
    )

    history["model_name"] = (
        history["model_name"]
        .map(
            lambda value: str(value).strip()
        )
        .astype(object)
    )

    history["date"] = pd.to_datetime(
        history["date"],
        errors="coerce",
    )

    history["generated_at"] = pd.to_datetime(
        history["generated_at"],
        errors="coerce",
    )

    numeric_columns = [
        "adj_close",
        "risk_score",
        "risk_percentile",
        "risk_pred",
    ]

    for column in numeric_columns:
        history[column] = (
            pd.to_numeric(
                history[column],
                errors="coerce",
            )
            .astype("float64")
        )

    history = history.dropna(
        subset=[
            "date",
            "ticker",
            "adj_close",
            "risk_score",
            "risk_percentile",
            "risk_level",
        ]
    )

    history = (
        history
        .sort_values(
            by=[
                "date",
                "generated_at",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .drop_duplicates(
            subset=[
                "date",
                "ticker",
                "model_name",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return history


@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def load_stock_features(
    ticker: str,
) -> pd.DataFrame:
    normalized_ticker = str(ticker).strip().upper()

    if not normalized_ticker:
        return pd.DataFrame()

    query = """
        SELECT *
        FROM inference_dataset_v3
        WHERE ticker = %s
        ORDER BY date DESC
        LIMIT 1;
    """

    connection = None

    try:
        connection = get_database_connection()

        features = pd.read_sql_query(
            query,
            connection,
            params=(normalized_ticker,),
        )

    finally:
        if connection is not None:
            connection.close()

    if features.empty:
        return features

    features["date"] = pd.to_datetime(
        features["date"],
        errors="coerce",
    )

    features["ticker"] = (
        features["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return features


def clear_prediction_cache() -> None:
    load_predictions.clear()
    load_risk_history.clear()
    load_stock_features.clear()