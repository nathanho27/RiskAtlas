import pandas as pd
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


def format_timestamp(timestamp) -> str:
    if pd.isna(timestamp):
        return "Unavailable"

    return timestamp.strftime("%Y-%m-%d %H:%M")


def render_model_insights(df: pd.DataFrame) -> None:
    prediction_date = df["date"].max()
    last_updated = df["generated_at"].max()

    table_model_name = (
        df["model_name"].dropna().iloc[0]
        if "model_name" in df.columns and not df["model_name"].dropna().empty
        else MODEL_NAME
    )

    st.subheader("Model Insights")

    st.caption(f"Production model: {table_model_name}")

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
                prediction_date.strftime("%Y-%m-%d"),
                format_timestamp(last_updated),
            ],
        }
    )

    st.dataframe(
        model_info,
        width="stretch",
        hide_index=True,
    )