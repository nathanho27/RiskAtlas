# model_training_v3.py
# benchmarks context-aware V3 models behind production
# does not modify the current production model or dashboard predictions

import os
import time

import joblib
import numpy as np
import pandas as pd

from sqlalchemy import create_engine

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


DB_NAME = "risk_atlas"
DB_USER = "nathanho"
DB_HOST = "localhost"
DB_PORT = "5432"

RESULTS_PATH = "models/v3_benchmark_results.csv"
BEST_MODEL_PATH = "models/best_risk_model_v3.joblib"

V2_ROC_AUC = 0.6130
V2_PR_AUC = 0.2028

MIN_VALIDATION_PRECISION = 0.25

# Train-only clipping prevents extreme values from destabilizing models.
LOWER_CLIP_QUANTILE = 0.001
UPPER_CLIP_QUANTILE = 0.999


FEATURES = [
    # Stock condition
    "return_20d",
    "return_60d",
    "vol_20",
    "vol_60",
    "downside_vol_20",
    "worst_return_20",
    "price_to_ma200",
    "drawdown_from_60d_high",
    "distance_from_52w_high",

    # Market regime
    "spy_return_20d",
    "spy_return_60d",
    "spy_vol_20",
    "spy_vol_60",
    "spy_drawdown_from_60d_high",

    # Market breadth
    "pct_positive_20d",
    "pct_above_ma50",
    "pct_above_ma200",

    # Sector-relative behavior
    "return_20d_vs_sector",
    "return_60d_vs_sector",
    "vol_20_vs_sector",

    # Market sensitivity
    "beta_60",
    "correlation_60",

    # Daily universe rankings
    "return_20d_percentile",
    "vol_20_percentile",
    "drawdown_60d_percentile",

    # Sector rankings
    "sector_return_20d_percentile",
    "sector_vol_20_percentile",
]


def get_engine():
    return create_engine(
        f"postgresql+psycopg2://"
        f"{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


def load_data():
    selected_columns = [
        "date",
        "ticker",
        *FEATURES,
        "risk_event",
    ]

    query = f"""
    SELECT
        {", ".join(selected_columns)}
    FROM model_dataset_v3
    ORDER BY date, ticker;
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

    df["date"] = pd.to_datetime(df["date"])

    return df


def chronological_split(df):
    # Same fixed dates as V2 for a fair comparison.
    train_end = pd.Timestamp("2021-09-22")
    validation_start = pd.Timestamp("2021-10-06")
    validation_end = pd.Timestamp("2024-01-31")
    test_start = pd.Timestamp("2024-02-14")

    train_df = df[
        df["date"] <= train_end
    ].copy()

    validation_df = df[
        (df["date"] >= validation_start)
        & (df["date"] <= validation_end)
    ].copy()

    test_df = df[
        df["date"] >= test_start
    ].copy()

    return train_df, validation_df, test_df


def calculate_clip_bounds(X_train):
    # Bounds come only from training data to avoid leakage.
    lower_bounds = X_train.quantile(
        LOWER_CLIP_QUANTILE
    )

    upper_bounds = X_train.quantile(
        UPPER_CLIP_QUANTILE
    )

    return lower_bounds, upper_bounds


def clip_features(
    X,
    lower_bounds,
    upper_bounds,
):
    return X.clip(
        lower=lower_bounds,
        upper=upper_bounds,
        axis="columns",
    )


def print_clipping_summary(
    X_train,
    lower_bounds,
    upper_bounds,
):
    clipped_low = X_train.lt(
        lower_bounds,
        axis="columns",
    ).sum()

    clipped_high = X_train.gt(
        upper_bounds,
        axis="columns",
    ).sum()

    summary_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "clipped_low": clipped_low.values,
            "clipped_high": clipped_high.values,
        }
    )

    summary_df["total_clipped"] = (
        summary_df["clipped_low"]
        + summary_df["clipped_high"]
    )

    summary_df = summary_df.sort_values(
        "total_clipped",
        ascending=False,
    )

    print("\nFeature Clipping Summary")
    print("-" * 60)

    print(
        summary_df
        .head(10)
        .to_string(index=False)
    )


def select_threshold(
    y_true,
    y_score,
    min_precision=MIN_VALIDATION_PRECISION,
):
    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            y_score,
        )
    )

    # Precision and recall contain one extra endpoint.
    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = (
        2 * precision * recall
        / (precision + recall + 1e-10)
    )

    valid_mask = precision >= min_precision

    if valid_mask.any():
        valid_indices = np.flatnonzero(
            valid_mask
        )

        best_local_index = np.argmax(
            f1_scores[valid_mask]
        )

        best_index = valid_indices[
            best_local_index
        ]

        precision_constraint_met = True

    else:
        # Fall back to unrestricted F1 rather than crashing.
        best_index = int(
            np.argmax(f1_scores)
        )

        precision_constraint_met = False

    return {
        "threshold": float(
            thresholds[best_index]
        ),
        "precision": float(
            precision[best_index]
        ),
        "recall": float(
            recall[best_index]
        ),
        "f1": float(
            f1_scores[best_index]
        ),
        "precision_constraint_met": (
            precision_constraint_met
        ),
    }


def evaluate_model(
    model_name,
    y_true,
    y_score,
    threshold,
    training_seconds,
):
    y_pred = (
        y_score >= threshold
    ).astype(int)

    roc_auc = roc_auc_score(
        y_true,
        y_score,
    )

    pr_auc = average_precision_score(
        y_true,
        y_score,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    predicted_positive_rate = (
        y_pred.mean()
    )

    print(f"\n{model_name} Test Results")
    print("-" * 60)
    print(f"Threshold: {threshold:.4f}")

    print(
        "Risk-event prevalence: "
        f"{y_true.mean():.4f}"
    )

    print(
        "Predicted positive rate: "
        f"{predicted_positive_rate:.4f}"
    )

    print(
        "Training time: "
        f"{training_seconds:.2f} seconds"
    )

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

    print(
        confusion_matrix(
            y_true,
            y_pred,
        )
    )

    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    return {
        "model": model_name,
        "threshold": threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": (
            predicted_positive_rate
        ),
        "training_seconds": training_seconds,
    }


def print_feature_importance(
    model_name,
    model,
):
    if model_name == "Logistic Regression":
        importance_values = (
            model.named_steps[
                "logistic"
            ].coef_[0]
        )

        importance_name = "coefficient"

    else:
        importance_values = (
            model.feature_importances_
        )

        importance_name = "importance"

    importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            importance_name: importance_values,
        }
    )

    if model_name == "Logistic Regression":
        importance_df["absolute_coefficient"] = (
            importance_df[
                "coefficient"
            ].abs()
        )

        importance_df = (
            importance_df
            .sort_values(
                "absolute_coefficient",
                ascending=False,
            )
            .drop(
                columns="absolute_coefficient"
            )
        )

    else:
        importance_df = (
            importance_df
            .sort_values(
                "importance",
                ascending=False,
            )
        )

    print(
        f"\n{model_name} Feature Importance"
    )
    print("-" * 60)

    print(
        importance_df
        .head(25)
        .to_string(index=False)
    )


def build_models(scale_pos_weight):
    return {
        "Logistic Regression": Pipeline(
            steps=[
                (
                    "scaler",
                    RobustScaler(),
                ),
                (
                    "logistic",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),

        # Sampling keeps the forest practical on 1M+ rows.
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=50,
            max_features="sqrt",
            max_samples=0.35,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
            verbose=1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=10,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.10,
            reg_lambda=1.00,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=100,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.80,
            reg_alpha=0.10,
            reg_lambda=1.00,
            scale_pos_weight=scale_pos_weight,
            objective="binary",
            n_jobs=-1,
            random_state=42,
            verbosity=1,
        ),
    }


def validate_split(
    split_name,
    split_df,
):
    if split_df.empty:
        raise ValueError(
            f"{split_name} dataset is empty"
        )

    if split_df["risk_event"].nunique() < 2:
        raise ValueError(
            f"{split_name} dataset does not "
            "contain both target classes"
        )


def main():
    df = load_data()

    required_columns = [
        "date",
        "ticker",
        "risk_event",
        *FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    # Remove values that cannot be used safely.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=required_columns
    ).copy()

    train_df, validation_df, test_df = (
        chronological_split(df)
    )

    validate_split(
        "Training",
        train_df,
    )

    validate_split(
        "Validation",
        validation_df,
    )

    validate_split(
        "Testing",
        test_df,
    )

    X_train = (
        train_df[FEATURES]
        .astype("float64")
        .copy()
    )

    X_validation = (
        validation_df[FEATURES]
        .astype("float64")
        .copy()
    )

    X_test = (
        test_df[FEATURES]
        .astype("float64")
        .copy()
    )

    y_train = train_df[
        "risk_event"
    ].astype(int)

    y_validation = validation_df[
        "risk_event"
    ].astype(int)

    y_test = test_df[
        "risk_event"
    ].astype(int)

    # Clip all splits using bounds learned from training only.
    lower_bounds, upper_bounds = (
        calculate_clip_bounds(
            X_train
        )
    )

    print_clipping_summary(
        X_train,
        lower_bounds,
        upper_bounds,
    )

    X_train = clip_features(
        X_train,
        lower_bounds,
        upper_bounds,
    )

    X_validation = clip_features(
        X_validation,
        lower_bounds,
        upper_bounds,
    )

    X_test = clip_features(
        X_test,
        lower_bounds,
        upper_bounds,
    )

    negative_count = int(
        (y_train == 0).sum()
    )

    positive_count = int(
        (y_train == 1).sum()
    )

    scale_pos_weight = (
        negative_count / positive_count
    )

    print(f"\nFeatures: {len(FEATURES)}")

    print(
        f"Total usable rows: {len(df):,}"
    )

    print(
        f"Training rows: {len(train_df):,}"
    )

    print(
        "Validation rows: "
        f"{len(validation_df):,}"
    )

    print(
        f"Testing rows: {len(test_df):,}"
    )

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

    print(
        "Training risk-event prevalence: "
        f"{y_train.mean():.4f}"
    )

    print(
        "Validation risk-event prevalence: "
        f"{y_validation.mean():.4f}"
    )

    print(
        "Testing risk-event prevalence: "
        f"{y_test.mean():.4f}"
    )

    print(
        "Scale positive weight: "
        f"{scale_pos_weight:.4f}"
    )

    print(
        "Minimum validation precision: "
        f"{MIN_VALIDATION_PRECISION:.2f}"
    )

    models = build_models(
        scale_pos_weight
    )

    results = []
    best_pr_auc = -1.0
    best_model_name = None

    os.makedirs(
        "models",
        exist_ok=True,
    )

    for model_name, model in models.items():
        print("\n")
        print("=" * 70)

        print(
            f"TRAINING {model_name.upper()}"
        )

        print("=" * 70)

        start_time = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        training_seconds = (
            time.perf_counter()
            - start_time
        )

        validation_score = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        threshold_results = select_threshold(
            y_validation,
            validation_score,
        )

        selected_threshold = (
            threshold_results["threshold"]
        )

        print(
            "\nValidation Threshold Selection"
        )

        print("-" * 60)

        if threshold_results[
            "precision_constraint_met"
        ]:
            print(
                "Precision constraint: met"
            )
        else:
            print(
                "Precision constraint: not met; "
                "using unrestricted best F1"
            )

        print(
            "Selected threshold: "
            f"{selected_threshold:.4f}"
        )

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

        test_score = model.predict_proba(
            X_test
        )[:, 1]

        model_results = evaluate_model(
            model_name=model_name,
            y_true=y_test,
            y_score=test_score,
            threshold=selected_threshold,
            training_seconds=training_seconds,
        )

        model_results[
            "validation_precision"
        ] = threshold_results[
            "precision"
        ]

        model_results[
            "validation_recall"
        ] = threshold_results[
            "recall"
        ]

        model_results[
            "validation_f1"
        ] = threshold_results[
            "f1"
        ]

        model_results[
            "precision_constraint_met"
        ] = threshold_results[
            "precision_constraint_met"
        ]

        results.append(
            model_results
        )

        print_feature_importance(
            model_name,
            model,
        )

        # Save the experimental leader only.
        if model_results["pr_auc"] > best_pr_auc:
            best_pr_auc = (
                model_results["pr_auc"]
            )

            best_model_name = model_name

            model_artifact = {
                "model": model,
                "features": FEATURES,
                "threshold": selected_threshold,
                "model_name": model_name,
                "dataset": "model_dataset_v3",
                "clip_lower_bounds": (
                    lower_bounds.to_dict()
                ),
                "clip_upper_bounds": (
                    upper_bounds.to_dict()
                ),
                "minimum_validation_precision": (
                    MIN_VALIDATION_PRECISION
                ),
                "test_roc_auc": (
                    model_results["roc_auc"]
                ),
                "test_pr_auc": (
                    model_results["pr_auc"]
                ),
            }

            joblib.dump(
                model_artifact,
                BEST_MODEL_PATH,
            )

            print(
                "\nNew leading model saved to "
                f"{BEST_MODEL_PATH}"
            )

    results_df = pd.DataFrame(
        results
    ).sort_values(
        [
            "pr_auc",
            "roc_auc",
        ],
        ascending=False,
    )

    results_df[
        "roc_auc_change_vs_v2"
    ] = (
        results_df["roc_auc"]
        - V2_ROC_AUC
    )

    results_df[
        "pr_auc_change_vs_v2"
    ] = (
        results_df["pr_auc"]
        - V2_PR_AUC
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print("\n")
    print("=" * 70)
    print("V3 MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.4f}"
            ),
        )
    )

    print("\nV2 Baseline")
    print("-" * 60)
    print(f"ROC-AUC: {V2_ROC_AUC:.4f}")
    print(f"PR-AUC: {V2_PR_AUC:.4f}")

    print("\nBest Experimental V3 Model")
    print("-" * 60)
    print(f"Model: {best_model_name}")

    print(
        f"Model saved to: {BEST_MODEL_PATH}"
    )

    print(
        f"Results saved to: {RESULTS_PATH}"
    )

    print(
        "\nNo production prediction tables "
        "or dashboard files were changed."
    )


if __name__ == "__main__":
    main()