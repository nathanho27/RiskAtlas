import pandas as pd
import streamlit as st


def format_timestamp(timestamp) -> str:
    if pd.isna(timestamp):
        return "Unavailable"

    return timestamp.strftime("%Y-%m-%d %H:%M")


def render_model_insights(df: pd.DataFrame) -> None:
    prediction_date = df["date"].max()
    last_updated = df["generated_at"].max()

    st.subheader("Model Insights")

    col1, col2, col3 = st.columns(
        3,
        gap="medium",
    )

    with col1:
        st.metric(
            "ROC-AUC",
            "0.6130",
        )

    with col2:
        st.metric(
            "PR-AUC",
            "0.2028",
        )

    with col3:
        st.metric(
            "Features",
            "14",
        )

    model_info = pd.DataFrame(
        {
            "Field": [
                "Model Type",
                "Model Version",
                "Prediction Horizon",
                "Ranking Method",
                "Training Dataset",
                "Latest Prediction Date",
                "Last Updated",
            ],
            "Value": [
                "Logistic Regression",
                "V2",
                "10 trading days",
                "Cross-sectional percentile",
                "model_dataset_v2",
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