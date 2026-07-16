# prediction_v3.py
# Loads the production Random Forest V3 model, scores the latest complete row
# for each ticker from inference_dataset_v3, and stores both the current
# prediction snapshot and historical prediction records in PostgreSQL.

import os
from io import StringIO

import joblib
import pandas as pd
import psycopg2
from sqlalchemy import create_engine


DB_NAME = "risk_atlas"
DB_USER = "nathanho"
DB_HOST = "localhost"
DB_PORT = "5432"

MODEL_PATH = "models/best_random_forest_v3.joblib"
SOURCE_TABLE = "inference_dataset_v3"
OUTPUT_TABLE = "current_risk_predictions_v3"
HISTORY_TABLE = "risk_prediction_history_v3"


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

def get_engine():
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        return create_engine(
            database_url,
            pool_pre_ping=True,
        )

    return create_engine(
        f"postgresql+psycopg2://"
        f"{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        pool_pre_ping=True,
    )

def load_model_artifact():
    """Load the saved Random Forest V3 model and its metadata."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    if not isinstance(artifact, dict):
        raise TypeError(
            "The model artifact must be a dictionary."
        )

    required_keys = {
        "model",
        "features",
        "threshold",
        "model_name",
    }

    missing_keys = required_keys.difference(artifact)

    if missing_keys:
        raise KeyError(
            f"Model artifact is missing keys: "
            f"{sorted(missing_keys)}"
        )

    model = artifact["model"]
    features = artifact["features"]

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "The saved model does not support predict_proba()."
        )

    if not isinstance(features, (list, tuple)):
        raise TypeError(
            "Artifact features must be stored as a list or tuple."
        )

    if len(features) == 0:
        raise ValueError(
            "The model artifact contains no feature names."
        )

    print("\nModel Artifact Loaded")
    print("-" * 50)
    print(f"Path: {MODEL_PATH}")
    print(f"Model type: {type(model).__name__}")
    print(f"Model name: {artifact['model_name']}")
    print(f"Feature count: {len(features)}")
    print(f"Threshold: {float(artifact['threshold']):.4f}")

    return artifact


def print_feature_list(features):
    """Print the exact feature order expected by the model."""
    print("\nV3 Feature Order")
    print("-" * 50)

    for index, feature in enumerate(features, start=1):
        print(f"{index:>2}. {feature}")


def load_latest_features(features):
    """
    Load the most recent complete row for each ticker.

    Rows missing any required V3 feature are excluded before the newest row
    is selected.
    """
    feature_sql = ",\n            ".join(features)

    completeness_filter = " AND ".join(
        f"{feature} IS NOT NULL"
        for feature in features
    )

    query = f"""
    WITH complete_rows AS (
        SELECT
            date,
            ticker,
            adj_close,
            {feature_sql}
        FROM {SOURCE_TABLE}
        WHERE {completeness_filter}
    ),
    ranked_rows AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY ticker
                ORDER BY date DESC
            ) AS row_number
        FROM complete_rows
    )
    SELECT
        date,
        ticker,
        adj_close,
        {feature_sql}
    FROM ranked_rows
    WHERE row_number = 1
    ORDER BY ticker;
    """

    engine = get_engine()

    try:
        with engine.connect() as connection:
            df = pd.read_sql(
                query,
                connection,
            )
    finally:
        engine.dispose()

    if df.empty:
        raise ValueError(
            f"No complete feature rows were found in {SOURCE_TABLE}."
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    duplicate_count = int(
        df["ticker"].duplicated().sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Prediction input contains {duplicate_count} "
            "duplicate ticker rows."
        )

    print("\nInference Data Loaded")
    print("-" * 50)
    print(f"Source table: {SOURCE_TABLE}")
    print(f"Stocks available: {len(df):,}")
    print(
        "Prediction date range: "
        f"{df['date'].min().date()} "
        f"to {df['date'].max().date()}"
    )

    return df


def build_predictions(df, artifact):
    """
    Generate model risk scores, binary predictions, percentiles,
    and categorical risk levels.
    """
    model = artifact["model"]
    features = list(artifact["features"])
    threshold = float(artifact["threshold"])
    model_name = str(artifact["model_name"])

    missing_columns = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Prediction data is missing features: "
            f"{missing_columns}"
        )

    X = df.loc[:, features].copy()

    non_numeric_columns = [
        column
        for column in X.columns
        if not pd.api.types.is_numeric_dtype(X[column])
    ]

    if non_numeric_columns:
        raise TypeError(
            "The following prediction features are not numeric: "
            f"{non_numeric_columns}"
        )

    missing_value_count = int(
        X.isna().sum().sum()
    )

    if missing_value_count > 0:
        raise ValueError(
            f"Prediction features contain "
            f"{missing_value_count:,} missing values."
        )

    probability_matrix = model.predict_proba(X)

    if probability_matrix.ndim != 2:
        raise ValueError(
            "predict_proba() returned an unexpected output shape."
        )

    if probability_matrix.shape[1] < 2:
        raise ValueError(
            "predict_proba() did not return a positive-class "
            "probability column."
        )

    risk_scores = probability_matrix[:, 1]

    risk_predictions = (
        risk_scores >= threshold
    ).astype(int)

    pred_df = pd.DataFrame(
        {
            "date": df["date"].dt.date,
            "ticker": df["ticker"].astype(str),
            "adj_close": pd.to_numeric(
                df["adj_close"],
                errors="raise",
            ),
            "risk_score": risk_scores,
            "risk_pred": risk_predictions,
            "model_name": model_name,
        }
    )

    pred_df["risk_percentile"] = (
        pred_df["risk_score"]
        .rank(
            pct=True,
            method="max",
        )
    )

    pred_df["risk_level"] = pd.cut(
        pred_df["risk_percentile"],
        bins=[
            0.0,
            0.50,
            0.80,
            0.95,
            1.0,
        ],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Critical",
        ],
        include_lowest=True,
    )

    if pred_df["risk_level"].isna().any():
        raise ValueError(
            "One or more predictions could not be assigned "
            "a risk level."
        )

    pred_df["risk_level"] = (
        pred_df["risk_level"]
        .astype(str)
    )

    pred_df = pred_df.sort_values(
        [
            "risk_score",
            "ticker",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    return pred_df


def save_predictions(pred_df):
    """
    Save the latest snapshot and preserve historical predictions.

    The current table is replaced on every run.

    The history table uses an upsert so rerunning the pipeline for the same
    date, ticker, and model does not create duplicate rows.
    """
    connection = get_connection()
    cursor = connection.cursor()

    export_columns = [
        "date",
        "ticker",
        "adj_close",
        "risk_score",
        "risk_percentile",
        "risk_pred",
        "risk_level",
        "model_name",
    ]

    buffer = StringIO()

    pred_df[export_columns].to_csv(
        buffer,
        index=False,
        header=False,
    )

    buffer.seek(0)

    try:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
                date DATE NOT NULL,
                ticker TEXT NOT NULL,
                adj_close DOUBLE PRECISION NOT NULL,
                risk_score DOUBLE PRECISION NOT NULL,
                risk_percentile DOUBLE PRECISION NOT NULL,
                risk_pred INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                model_name TEXT NOT NULL,
                generated_at TIMESTAMP NOT NULL
                    DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (
                    date,
                    ticker,
                    model_name
                )
            );
            """
        )

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {HISTORY_TABLE} (
                date DATE NOT NULL,
                ticker TEXT NOT NULL,
                adj_close DOUBLE PRECISION NOT NULL,
                risk_score DOUBLE PRECISION NOT NULL,
                risk_percentile DOUBLE PRECISION NOT NULL,
                risk_pred INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                model_name TEXT NOT NULL,
                generated_at TIMESTAMP NOT NULL
                    DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (
                    date,
                    ticker,
                    model_name
                )
            );
            """
        )

        cursor.execute(
            """
            CREATE TEMP TABLE prediction_batch (
                date DATE NOT NULL,
                ticker TEXT NOT NULL,
                adj_close DOUBLE PRECISION NOT NULL,
                risk_score DOUBLE PRECISION NOT NULL,
                risk_percentile DOUBLE PRECISION NOT NULL,
                risk_pred INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                model_name TEXT NOT NULL
            ) ON COMMIT DROP;
            """
        )

        cursor.copy_expert(
            """
            COPY prediction_batch (
                date,
                ticker,
                adj_close,
                risk_score,
                risk_percentile,
                risk_pred,
                risk_level,
                model_name
            )
            FROM STDIN WITH CSV;
            """,
            buffer,
        )

        cursor.execute(
            f"""
            TRUNCATE TABLE {OUTPUT_TABLE};
            """
        )

        cursor.execute(
            f"""
            INSERT INTO {OUTPUT_TABLE} (
                date,
                ticker,
                adj_close,
                risk_score,
                risk_percentile,
                risk_pred,
                risk_level,
                model_name
            )
            SELECT
                date,
                ticker,
                adj_close,
                risk_score,
                risk_percentile,
                risk_pred,
                risk_level,
                model_name
            FROM prediction_batch;
            """
        )

        cursor.execute(
            f"""
            INSERT INTO {HISTORY_TABLE} (
                date,
                ticker,
                adj_close,
                risk_score,
                risk_percentile,
                risk_pred,
                risk_level,
                model_name
            )
            SELECT
                date,
                ticker,
                adj_close,
                risk_score,
                risk_percentile,
                risk_pred,
                risk_level,
                model_name
            FROM prediction_batch
            ON CONFLICT (
                date,
                ticker,
                model_name
            )
            DO UPDATE SET
                adj_close = EXCLUDED.adj_close,
                risk_score = EXCLUDED.risk_score,
                risk_percentile = EXCLUDED.risk_percentile,
                risk_pred = EXCLUDED.risk_pred,
                risk_level = EXCLUDED.risk_level,
                generated_at = CURRENT_TIMESTAMP;
            """
        )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def print_summary(pred_df, artifact):
    """Print a compact prediction summary."""
    threshold = float(artifact["threshold"])

    print("\nCurrent V3 Risk Prediction Summary")
    print("-" * 50)
    print(f"Model: {artifact['model_name']}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Stocks scored: {len(pred_df):,}")

    print(
        "Prediction date range: "
        f"{pred_df['date'].min()} "
        f"to {pred_df['date'].max()}"
    )

    print(
        "Risk score range: "
        f"{pred_df['risk_score'].min():.4f} "
        f"to {pred_df['risk_score'].max():.4f}"
    )

    positive_predictions = int(
        pred_df["risk_pred"].sum()
    )

    positive_rate = (
        positive_predictions / len(pred_df)
    )

    print(
        f"Positive predictions: "
        f"{positive_predictions:,} "
        f"({positive_rate:.2%})"
    )

    print("\nRisk Level Counts")

    risk_counts = (
        pred_df["risk_level"]
        .value_counts()
        .reindex(
            [
                "Critical",
                "High",
                "Moderate",
                "Low",
            ],
            fill_value=0,
        )
    )

    print(risk_counts.to_string())

    print("\nHighest-Risk Stocks")

    summary_columns = [
        "ticker",
        "date",
        "adj_close",
        "risk_score",
        "risk_percentile",
        "risk_pred",
        "risk_level",
    ]

    print(
        pred_df[summary_columns]
        .head(15)
        .to_string(
            index=False,
            formatters={
                "adj_close": "{:.2f}".format,
                "risk_score": "{:.4f}".format,
                "risk_percentile": "{:.2%}".format,
            },
        )
    )


def main():
    """Run the production Random Forest V3 prediction pipeline."""
    print("\nStarting Random Forest V3 Prediction Pipeline")
    print("=" * 50)

    artifact = load_model_artifact()

    print_feature_list(
        artifact["features"]
    )

    latest_features = load_latest_features(
        artifact["features"]
    )

    predictions = build_predictions(
        latest_features,
        artifact,
    )

    save_predictions(
        predictions
    )

    print_summary(
        predictions,
        artifact,
    )

    print(
        "\nCurrent V3 predictions saved to PostgreSQL table: "
        f"{OUTPUT_TABLE}"
    )

    print(
        "Historical V3 predictions saved to PostgreSQL table: "
        f"{HISTORY_TABLE}"
    )

    print(
        "\nPredictions were generated from "
        f"'{SOURCE_TABLE}', which includes the latest "
        "available unlabeled market data."
    )


if __name__ == "__main__":
    main()