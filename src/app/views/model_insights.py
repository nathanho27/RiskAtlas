from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


ROC_AUC = 0.6349
PR_AUC = 0.2180
FEATURE_COUNT = 27
MODEL_TYPE = "Random Forest"
MODEL_VERSION = "V3"
MODEL_NAME = "Random Forest V3 Tuned"
PREDICTION_HORIZON = "10 trading days"
RANKING_METHOD = "Cross-sectional percentile"
TRAINING_DATASET = "model_dataset_v3"
INFERENCE_DATASET = "inference_dataset_v3"
DECISION_THRESHOLD = 0.6897

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_risk_model_v3.joblib"
)


def format_timestamp(timestamp) -> str:
    if pd.isna(timestamp):
        return "Unavailable"

    return timestamp.strftime("%Y-%m-%d %H:%M")


@st.cache_data(
    show_spinner=False,
)
def load_feature_importance() -> pd.DataFrame:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact was not found at {MODEL_PATH}."
        )

    artifact = joblib.load(MODEL_PATH)

    model = artifact

    if isinstance(artifact, dict):
        for key in [
            "model",
            "best_model",
            "estimator",
            "pipeline",
        ]:
            if key in artifact:
                model = artifact[key]
                break

    if hasattr(model, "named_steps"):
        for step_name in [
            "model",
            "classifier",
            "random_forest",
            "rf",
        ]:
            if step_name in model.named_steps:
                estimator = model.named_steps[step_name]
                break
        else:
            estimator = list(
                model.named_steps.values()
            )[-1]
    else:
        estimator = model

    if not hasattr(estimator, "feature_importances_"):
        raise AttributeError(
            "The production model does not expose "
            "feature_importances_."
        )

    importances = estimator.feature_importances_

    feature_names = None

    if hasattr(estimator, "feature_names_in_"):
        feature_names = list(
            estimator.feature_names_in_
        )

    elif hasattr(model, "feature_names_in_"):
        feature_names = list(
            model.feature_names_in_
        )

    elif isinstance(artifact, dict):
        for key in [
            "feature_names",
            "features",
            "feature_columns",
        ]:
            if key in artifact:
                feature_names = list(
                    artifact[key]
                )
                break

    if (
        feature_names is None
        or len(feature_names) != len(importances)
    ):
        feature_names = [
            f"Feature {index + 1}"
            for index in range(len(importances))
        ]

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    )

    importance_df["Importance Percent"] = (
        importance_df["Importance"] * 100
    )

    return (
        importance_df
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def render_feature_importance() -> None:
    st.subheader("Global Feature Importance")

    try:
        importance_df = load_feature_importance()

    except Exception as error:
        st.warning(
            "Feature importance could not be loaded."
        )
        st.caption(str(error))
        return

    top_features = (
        importance_df
        .head(15)
        .sort_values(
            "Importance Percent",
            ascending=True,
        )
    )

    figure = px.bar(
        top_features,
        x="Importance Percent",
        y="Feature",
        orientation="h",
        labels={
            "Importance Percent": "Importance",
            "Feature": "",
        },
    )

    figure.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Importance: %{x:.2f}%"
            "<extra></extra>"
        )
    )

    figure.update_layout(
        height=520,
        margin={
            "l": 20,
            "r": 20,
            "t": 20,
            "b": 20,
        },
        xaxis_ticksuffix="%",
        showlegend=False,
    )

    st.plotly_chart(
        figure,
        width="stretch",
        key="global_feature_importance",
    )

    st.caption(
        "Feature importance shows each variable's overall "
        "contribution to the Random Forest model. It does not "
        "explain an individual stock prediction."
    )


def render_model_insights(df: pd.DataFrame) -> None:
    prediction_date = df["date"].max()
    last_updated = df["generated_at"].max()

    table_model_name = (
        df["model_name"].dropna().iloc[0]
        if (
            "model_name" in df.columns
            and not df["model_name"].dropna().empty
        )
        else MODEL_NAME
    )

    st.subheader("Model Insights")

    st.caption(
        f"Production model: {table_model_name}"
    )

    col1, col2, col3 = st.columns(
        3,
        gap="medium",
    )

    with col1:
        st.metric(
            "ROC-AUC",
            f"{ROC_AUC:.4f}",
        )

    with col2:
        st.metric(
            "PR-AUC",
            f"{PR_AUC:.4f}",
        )

    with col3:
        st.metric(
            "Features",
            str(FEATURE_COUNT),
        )

    render_feature_importance()

    st.subheader("Model Configuration")

    model_info = pd.DataFrame(
        {
            "Field": [
                "Model Type",
                "Model Version",
                "Model Name",
                "Prediction Horizon",
                "Decision Threshold",
                "Ranking Method",
                "Training Dataset",
                "Inference Dataset",
                "Latest Prediction Date",
                "Last Updated",
            ],
            "Value": [
                MODEL_TYPE,
                MODEL_VERSION,
                table_model_name,
                PREDICTION_HORIZON,
                f"{DECISION_THRESHOLD:.4f}",
                RANKING_METHOD,
                TRAINING_DATASET,
                INFERENCE_DATASET,
                prediction_date.strftime(
                    "%Y-%m-%d"
                ),
                format_timestamp(last_updated),
            ],
        }
    )

    st.dataframe(
        model_info,
        width="stretch",
        hide_index=True,
    )