# random_forest_tuning_v3.py
# tunes the strongest V3 Random Forest configuration
# validation selects the model; test is evaluated only once

import gc
import os
import time

import joblib
import numpy as np
import pandas as pd

from sqlalchemy import create_engine

from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import ParameterGrid


DB_NAME = "risk_atlas"
DB_USER = "nathanho"
DB_HOST = "localhost"
DB_PORT = "5432"

RESULTS_PATH = "models/v3_rf_tuning_results.csv"
BEST_MODEL_PATH = "models/best_random_forest_v3.joblib"

V2_ROC_AUC = 0.6130
V2_PR_AUC = 0.2028

MIN_VALIDATION_PRECISION = 0.25

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


PARAM_GRID = {
    "n_estimators": [200],
    "max_depth": [10, 14, 18],
    "min_samples_leaf": [25, 50, 100],
    "max_features": ["sqrt", 0.5],
}


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

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


def chronological_split(df):
    # Fixed V2 dates keep comparisons fair.
    train_end = pd.Timestamp(
        "2021-09-22"
    )

    validation_start = pd.Timestamp(
        "2021-10-06"
    )

    validation_end = pd.Timestamp(
        "2024-01-31"
    )

    test_start = pd.Timestamp(
        "2024-02-14"
    )

    train_df = df[
        df["date"] <= train_end
    ].copy()

    validation_df = df[
        (
            df["date"]
            >= validation_start
        )
        & (
            df["date"]
            <= validation_end
        )
    ].copy()

    test_df = df[
        df["date"] >= test_start
    ].copy()

    return (
        train_df,
        validation_df,
        test_df,
    )


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


def calculate_clip_bounds(X_train):
    # Learn clipping bounds from training only.
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


def select_threshold(
    y_true,
    y_score,
):
    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            y_score,
        )
    )

    # The final PR endpoint has no threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = (
        2 * precision * recall
        / (
            precision
            + recall
            + 1e-10
        )
    )

    valid_indices = np.flatnonzero(
        precision
        >= MIN_VALIDATION_PRECISION
    )

    if len(valid_indices) > 0:
        # Maximize recall while keeping precision useful.
        valid_recalls = recall[
            valid_indices
        ]

        highest_recall = valid_recalls.max()

        recall_candidates = valid_indices[
            np.isclose(
                valid_recalls,
                highest_recall,
            )
        ]

        # Use F1 to break ties.
        best_index = recall_candidates[
            np.argmax(
                f1_scores[
                    recall_candidates
                ]
            )
        ]

        constraint_met = True

    else:
        # Safe fallback if 25% precision is unreachable.
        best_index = int(
            np.argmax(f1_scores)
        )

        constraint_met = False

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
            constraint_met
        ),
    }


def build_random_forest(
    parameters,
):
    return RandomForestClassifier(
        n_estimators=parameters[
            "n_estimators"
        ],
        max_depth=parameters[
            "max_depth"
        ],
        min_samples_leaf=parameters[
            "min_samples_leaf"
        ],
        max_features=parameters[
            "max_features"
        ],

        # Keep each run practical on 1M+ rows.
        max_samples=0.35,
        class_weight=(
            "balanced_subsample"
        ),
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )


def evaluate_test_set(
    model,
    X_test,
    y_test,
    threshold,
):
    test_score = model.predict_proba(
        X_test
    )[:, 1]

    test_prediction = (
        test_score >= threshold
    ).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        test_score,
    )

    pr_auc = average_precision_score(
        y_test,
        test_score,
    )

    precision = precision_score(
        y_test,
        test_prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        test_prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        test_prediction,
        zero_division=0,
    )

    predicted_positive_rate = (
        test_prediction.mean()
    )

    print("\nFinal Test Results")
    print("-" * 60)

    print(
        f"Threshold: {threshold:.4f}"
    )

    print(
        "Risk-event prevalence: "
        f"{y_test.mean():.4f}"
    )

    print(
        "Predicted positive rate: "
        f"{predicted_positive_rate:.4f}"
    )

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            test_prediction,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion Matrix")

    print(
        confusion_matrix(
            y_test,
            test_prediction,
        )
    )

    print(
        f"\nROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC: {pr_auc:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1: {f1:.4f}"
    )

    return {
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_predicted_positive_rate": (
            predicted_positive_rate
        ),
    }


def print_feature_importance(
    model,
):
    importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": (
                model.feature_importances_
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print(
        "\nWinning Feature Importance"
    )

    print("-" * 60)

    print(
        importance_df
        .head(25)
        .to_string(index=False)
    )


def main():
    os.makedirs(
        "models",
        exist_ok=True,
    )

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

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=required_columns
    ).copy()

    (
        train_df,
        validation_df,
        test_df,
    ) = chronological_split(df)

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
        .astype("float32")
        .copy()
    )

    X_validation = (
        validation_df[FEATURES]
        .astype("float32")
        .copy()
    )

    X_test = (
        test_df[FEATURES]
        .astype("float32")
        .copy()
    )

    y_train = train_df[
        "risk_event"
    ].astype("int8")

    y_validation = validation_df[
        "risk_event"
    ].astype("int8")

    y_test = test_df[
        "risk_event"
    ].astype("int8")

    lower_bounds, upper_bounds = (
        calculate_clip_bounds(
            X_train
        )
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

    parameter_combinations = list(
        ParameterGrid(PARAM_GRID)
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    print(
        "Parameter combinations: "
        f"{len(parameter_combinations)}"
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
        "Training prevalence: "
        f"{y_train.mean():.4f}"
    )

    print(
        "Validation prevalence: "
        f"{y_validation.mean():.4f}"
    )

    results = []

    best_model = None
    best_parameters = None
    best_validation_pr_auc = -1.0
    best_validation_roc_auc = -1.0
    best_training_seconds = None

    total_runs = len(
        parameter_combinations
    )

    for run_number, parameters in enumerate(
        parameter_combinations,
        start=1,
    ):
        print("\n")
        print("=" * 70)

        print(
            f"RANDOM FOREST RUN "
            f"{run_number}/{total_runs}"
        )

        print("=" * 70)

        print(
            f"Parameters: {parameters}"
        )

        model = build_random_forest(
            parameters
        )

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

        validation_roc_auc = (
            roc_auc_score(
                y_validation,
                validation_score,
            )
        )

        validation_pr_auc = (
            average_precision_score(
                y_validation,
                validation_score,
            )
        )

        run_result = {
            "run": run_number,
            **parameters,
            "validation_roc_auc": (
                validation_roc_auc
            ),
            "validation_pr_auc": (
                validation_pr_auc
            ),
            "training_seconds": (
                training_seconds
            ),
        }

        results.append(
            run_result
        )

        # Save progress after every configuration.
        pd.DataFrame(
            results
        ).sort_values(
            [
                "validation_pr_auc",
                "validation_roc_auc",
            ],
            ascending=False,
        ).to_csv(
            RESULTS_PATH,
            index=False,
        )

        print(
            "Validation ROC-AUC: "
            f"{validation_roc_auc:.4f}"
        )

        print(
            "Validation PR-AUC: "
            f"{validation_pr_auc:.4f}"
        )

        print(
            "Training time: "
            f"{training_seconds:.2f} seconds"
        )

        is_better = (
            validation_pr_auc
            > best_validation_pr_auc
        )

        is_tied_pr = np.isclose(
            validation_pr_auc,
            best_validation_pr_auc,
        )

        is_better_roc = (
            validation_roc_auc
            > best_validation_roc_auc
        )

        if is_better or (
            is_tied_pr
            and is_better_roc
        ):
            # Keep only the current leader in memory.
            if best_model is not None:
                del best_model
                gc.collect()

            best_model = model
            best_parameters = (
                parameters.copy()
            )

            best_validation_pr_auc = (
                validation_pr_auc
            )

            best_validation_roc_auc = (
                validation_roc_auc
            )

            best_training_seconds = (
                training_seconds
            )

            print(
                "New validation leader"
            )

        else:
            del model
            gc.collect()

    results_df = pd.DataFrame(
        results
    ).sort_values(
        [
            "validation_pr_auc",
            "validation_roc_auc",
        ],
        ascending=False,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print("\n")
    print("=" * 70)
    print("RANDOM FOREST TUNING RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.4f}"
            ),
        )
    )

    print("\nWinning Parameters")
    print("-" * 60)

    print(best_parameters)

    print(
        "Validation ROC-AUC: "
        f"{best_validation_roc_auc:.4f}"
    )

    print(
        "Validation PR-AUC: "
        f"{best_validation_pr_auc:.4f}"
    )

    # Select the operating threshold only after tuning.
    winning_validation_score = (
        best_model.predict_proba(
            X_validation
        )[:, 1]
    )

    threshold_results = select_threshold(
        y_validation,
        winning_validation_score,
    )

    selected_threshold = (
        threshold_results["threshold"]
    )

    print(
        "\nWinning Validation Threshold"
    )

    print("-" * 60)

    print(
        "Precision constraint met: "
        f"{threshold_results['precision_constraint_met']}"
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

    # Test is touched once, after model selection.
    test_results = evaluate_test_set(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        threshold=selected_threshold,
    )

    print_feature_importance(
        best_model
    )

    model_artifact = {
        "model": best_model,
        "features": FEATURES,
        "threshold": selected_threshold,
        "model_name": (
            "Random Forest V3 Tuned"
        ),
        "dataset": "model_dataset_v3",
        "parameters": best_parameters,
        "clip_lower_bounds": (
            lower_bounds.to_dict()
        ),
        "clip_upper_bounds": (
            upper_bounds.to_dict()
        ),
        "minimum_validation_precision": (
            MIN_VALIDATION_PRECISION
        ),
        "validation_roc_auc": (
            best_validation_roc_auc
        ),
        "validation_pr_auc": (
            best_validation_pr_auc
        ),
        "validation_precision": (
            threshold_results["precision"]
        ),
        "validation_recall": (
            threshold_results["recall"]
        ),
        "validation_f1": (
            threshold_results["f1"]
        ),
        "training_seconds": (
            best_training_seconds
        ),
        **test_results,
    }

    joblib.dump(
        model_artifact,
        BEST_MODEL_PATH,
    )

    print("\n")
    print("=" * 70)
    print("FINAL V3 RANDOM FOREST")
    print("=" * 70)

    print(
        f"Test ROC-AUC: "
        f"{test_results['test_roc_auc']:.4f}"
    )

    print(
        f"Change vs V2: "
        f"{test_results['test_roc_auc'] - V2_ROC_AUC:+.4f}"
    )

    print(
        f"Test PR-AUC: "
        f"{test_results['test_pr_auc']:.4f}"
    )

    print(
        f"Change vs V2: "
        f"{test_results['test_pr_auc'] - V2_PR_AUC:+.4f}"
    )

    print(
        f"\nModel saved to: "
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"Results saved to: "
        f"{RESULTS_PATH}"
    )

    print(
        "\nNo production prediction tables "
        "or dashboard files were changed."
    )


if __name__ == "__main__":
    main()