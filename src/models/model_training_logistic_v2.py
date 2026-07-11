# model_training_logistic_v2.py
# trains logistic regression with expanded stock-specific risk features

import os
from io import StringIO

import joblib
import numpy as np
import pandas as pd
import psycopg2

from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DB_NAME = "risk_atlas"
DB_USER = "nathanho"
DB_HOST = "localhost"
DB_PORT = "5432"

MODEL_NAME = "logistic_regression_v2"
MODEL_PATH = "models/logistic_risk_model_v2.joblib"


def get_connection():
    """Create a PostgreSQL connection for prediction writes."""
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host=DB_HOST,
        port=DB_PORT,
    )


def get_engine():
    """Create a SQLAlchemy engine for loading model data."""
    return create_engine(
        f"postgresql+psycopg2://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


def load_data():
    """Load the expanded V2 modeling dataset in chronological order."""
    query = """
    SELECT
        date,
        ticker,
        daily_return,
        vol_20,
        vol_30,
        vol_60,
        price_to_ma50,
        price_to_ma200,
        return_5d,
        return_20d,
        return_60d,
        downside_vol_20,
        negative_return_count_20,
        worst_return_20,
        drawdown_from_60d_high,
        distance_from_52w_high,
        drawdown,
        risk_event
    FROM model_dataset_v2
    ORDER BY date, ticker;
    """

    engine = get_engine()

    try:
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
    finally:
        engine.dispose()

    df["date"] = pd.to_datetime(df["date"])

    return df


def chronological_split(df):
    """Split data using fixed dates for fair comparison with the baseline."""
    train_end = pd.Timestamp("2021-09-22")
    validation_start = pd.Timestamp("2021-10-06")
    validation_end = pd.Timestamp("2024-01-31")
    test_start = pd.Timestamp("2024-02-14")

    train_df = df[df["date"] <= train_end].copy()

    validation_df = df[
        (df["date"] >= validation_start)
        & (df["date"] <= validation_end)
    ].copy()

    test_df = df[df["date"] >= test_start].copy()

    return train_df, validation_df, test_df


def select_threshold(y_true, y_score):
    """Select the validation threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = (
        2 * precision * recall
        / (precision + recall + 1e-10)
    )

    best_index = np.argmax(f1_scores)

    return {
        "threshold": float(thresholds[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(f1_scores[best_index]),
    }


def evaluate_model(name, y_true, y_score, threshold):
    """Print test metrics and return binary predictions."""
    y_pred = (y_score >= threshold).astype(int)

    print(f"\n{name}")
    print("-" * 50)
    print(f"Threshold: {threshold:.4f}")
    print(f"Risk-event prevalence: {y_true.mean():.4f}")

    print("\nClassification Report")
    print(
        classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    return y_pred


def save_predictions(pred_df):
    """Replace saved predictions for this model only."""
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                date DATE NOT NULL,
                ticker TEXT NOT NULL,
                actual_risk_event INT NOT NULL,
                risk_score DOUBLE PRECISION NOT NULL,
                risk_pred INT NOT NULL,
                model_name TEXT NOT NULL,
                PRIMARY KEY (date, ticker, model_name)
            );
            """
        )

        cursor.execute(
            """
            DELETE FROM predictions
            WHERE model_name = %s;
            """,
            (MODEL_NAME,),
        )

        buffer = StringIO()
        pred_df.to_csv(
            buffer,
            index=False,
            header=False,
        )
        buffer.seek(0)

        cursor.copy_expert(
            """
            COPY predictions (
                date,
                ticker,
                actual_risk_event,
                risk_score,
                risk_pred,
                model_name
            )
            FROM STDIN WITH CSV
            """,
            buffer,
        )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def main():
    features = [
        "daily_return",
        "vol_20",
        "vol_30",
        "vol_60",
        "price_to_ma50",
        "price_to_ma200",
        "return_5d",
        "return_20d",
        "return_60d",
        "downside_vol_20",
        "negative_return_count_20",
        "worst_return_20",
        "drawdown_from_60d_high",
        "distance_from_52w_high",
        "drawdown",
    ]

    df = load_data()

    required_columns = [
        "date",
        "ticker",
        "risk_event",
        *features,
    ]

    df = df.dropna(subset=required_columns).copy()

    train_df, validation_df, test_df = chronological_split(df)

    if train_df.empty:
        raise ValueError("Training dataset is empty.")

    if validation_df.empty:
        raise ValueError("Validation dataset is empty.")

    if test_df.empty:
        raise ValueError("Testing dataset is empty.")

    X_train = train_df[features]
    y_train = train_df["risk_event"].astype(int)

    X_validation = validation_df[features]
    y_validation = validation_df["risk_event"].astype(int)

    X_test = test_df[features]
    y_test = test_df["risk_event"].astype(int)

    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Testing rows: {len(test_df):,}")

    print(
        "Training period: "
        f"{train_df['date'].min().date()} "
        f"to {train_df['date'].max().date()}"
    )

    print(
        "Validation period: "
        f"{validation_df['date'].min().date()} "
        f"to {validation_df['date'].max().date()}"
    )

    print(
        "Testing period: "
        f"{test_df['date'].min().date()} "
        f"to {test_df['date'].max().date()}"
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    validation_score = model.predict_proba(X_validation)[:, 1]

    threshold_results = select_threshold(
        y_validation,
        validation_score,
    )

    selected_threshold = threshold_results["threshold"]

    print("\nValidation Threshold Selection")
    print("-" * 50)
    print(f"Selected threshold: {selected_threshold:.4f}")
    print(
        "Validation precision: "
        f"{threshold_results['precision']:.4f}"
    )
    print(
        "Validation recall: "
        f"{threshold_results['recall']:.4f}"
    )
    print(
        "Validation F1: "
        f"{threshold_results['f1']:.4f}"
    )

    test_score = model.predict_proba(X_test)[:, 1]

    test_pred = evaluate_model(
        "Logistic Regression V2 Test Results",
        y_test,
        test_score,
        selected_threshold,
    )

    coefficients = pd.DataFrame(
        {
            "feature": features,
            "coefficient": model.named_steps[
                "logistic"
            ].coef_[0],
        }
    ).sort_values(
        "coefficient",
        ascending=False,
    )

    print("\nFeature Coefficients")
    print("-" * 50)
    print(coefficients.to_string(index=False))

    pred_df = pd.DataFrame(
        {
            "date": test_df["date"].dt.date.values,
            "ticker": test_df["ticker"].values,
            "actual_risk_event": y_test.values,
            "risk_score": test_score,
            "risk_pred": test_pred,
            "model_name": MODEL_NAME,
        }
    )

    save_predictions(pred_df)

    model_artifact = {
        "model": model,
        "features": features,
        "threshold": selected_threshold,
        "model_name": MODEL_NAME,
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(model_artifact, MODEL_PATH)

    print("\nV2 predictions saved")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()