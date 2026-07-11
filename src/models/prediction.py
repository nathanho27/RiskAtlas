# prediction.py
# Loads the production model, scores the latest complete row for each ticker,
# and writes current stock-level risk predictions to PostgreSQL.

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

MODEL_PATH = "models/logistic_risk_model_v2.joblib"
OUTPUT_TABLE = "current_risk_predictions"


def get_connection():
    """Create a PostgreSQL connection for prediction writes."""
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host=DB_HOST,
        port=DB_PORT,
    )


def get_engine():
    """Create a SQLAlchemy engine for loading feature data."""
    return create_engine(
        f"postgresql+psycopg2://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


def load_model_artifact():
    """Load the saved production model and its metadata."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    required_keys = {
        "model",
        "features",
        "threshold",
        "model_name",
    }

    missing_keys = required_keys.difference(artifact)

    if missing_keys:
        raise KeyError(
            f"Model artifact is missing keys: {sorted(missing_keys)}"
        )

    return artifact


def load_latest_features(features):
    """
    Load the most recent complete inference row for each ticker.

    Rows with missing production-model features are excluded before the
    most recent row is selected.
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
        FROM inference_dataset
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
            df = pd.read_sql(query, connection)
    finally:
        engine.dispose()

    if df.empty:
        raise ValueError(
            "No complete feature rows were found for prediction."
        )

    df["date"] = pd.to_datetime(df["date"])

    return df


def build_predictions(df, artifact):
    """
    Generate model scores, binary predictions, percentiles, and risk levels.
    """
    model = artifact["model"]
    features = artifact["features"]
    threshold = float(artifact["threshold"])
    model_name = artifact["model_name"]

    missing_columns = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Prediction data is missing features: {missing_columns}"
        )

    X = df[features]

    risk_scores = model.predict_proba(X)[:, 1]
    risk_predictions = (
        risk_scores >= threshold
    ).astype(int)

    pred_df = pd.DataFrame(
        {
            "date": df["date"].dt.date,
            "ticker": df["ticker"],
            "adj_close": df["adj_close"],
            "risk_score": risk_scores,
            "risk_pred": risk_predictions,
            "model_name": model_name,
        }
    )

    pred_df["risk_percentile"] = pred_df["risk_score"].rank(
        pct=True,
        method="max",
    )

    pred_df["risk_level"] = pd.cut(
        pred_df["risk_percentile"],
        bins=[0.0, 0.50, 0.80, 0.95, 1.0],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Critical",
        ],
        include_lowest=True,
    )

    pred_df["risk_level"] = (
        pred_df["risk_level"].astype(str)
    )

    pred_df = pred_df.sort_values(
        ["risk_score", "ticker"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return pred_df


def save_current_predictions(pred_df):
    """Replace the current prediction snapshot in PostgreSQL."""
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
                date DATE NOT NULL,
                ticker TEXT NOT NULL,
                adj_close DOUBLE PRECISION NOT NULL,
                risk_score DOUBLE PRECISION NOT NULL,
                risk_percentile DOUBLE PRECISION,
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
            ALTER TABLE {OUTPUT_TABLE}
            ADD COLUMN IF NOT EXISTS
                risk_percentile DOUBLE PRECISION;
            """
        )

        cursor.execute(
            f"""
            TRUNCATE TABLE {OUTPUT_TABLE};
            """
        )

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

        cursor.copy_expert(
            f"""
            COPY {OUTPUT_TABLE} (
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
            ALTER TABLE {OUTPUT_TABLE}
            ALTER COLUMN risk_percentile SET NOT NULL;
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
    """Print a compact scoring summary."""
    threshold = float(artifact["threshold"])

    print("\nCurrent Risk Prediction Summary")
    print("-" * 50)
    print(f"Model: {artifact['model_name']}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Stocks scored: {len(pred_df):,}")

    print(
        "Prediction date range: "
        f"{pred_df['date'].min()} "
        f"to {pred_df['date'].max()}"
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
    artifact = load_model_artifact()

    latest_features = load_latest_features(
        artifact["features"]
    )

    predictions = build_predictions(
        latest_features,
        artifact,
    )

    save_current_predictions(predictions)
    print_summary(predictions, artifact)

    print(
        "\nCurrent predictions saved to PostgreSQL table: "
        f"{OUTPUT_TABLE}"
    )


if __name__ == "__main__":
    main()