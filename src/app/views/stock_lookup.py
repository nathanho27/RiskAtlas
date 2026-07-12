import pandas as pd
import streamlit as st


def get_risk_explanation(risk_level: str) -> str:
    explanations = {
        "Critical": (
            "This stock currently ranks among the highest-risk "
            "stocks tracked by RiskAtlas."
        ),
        "High": (
            "This stock currently shows elevated downside-risk "
            "conditions relative to most tracked stocks."
        ),
        "Moderate": (
            "This stock shows some elevated risk, but it is not "
            "currently among the highest-risk stocks."
        ),
        "Low": (
            "This stock currently shows relatively limited "
            "downside-risk pressure."
        ),
    }

    return explanations.get(
        risk_level,
        "Risk conditions are currently unavailable.",
    )


def render_stock_lookup(df: pd.DataFrame) -> None:
    ticker_options = sorted(
        df["ticker"]
        .dropna()
        .tolist()
    )

    if not ticker_options:
        st.warning("No valid tickers were found.")
        return

    st.subheader("Stock Lookup")

    default_index = (
        ticker_options.index("NVDA")
        if "NVDA" in ticker_options
        else 0
    )

    selected_ticker = st.selectbox(
        "Select a ticker",
        options=ticker_options,
        index=default_index,
    )

    stock_rows = df.loc[
        df["ticker"].eq(selected_ticker)
    ]

    if stock_rows.empty:
        st.warning(
            "The selected ticker is unavailable."
        )
        return

    stock = stock_rows.iloc[0]

    col1, col2, col3, col4 = st.columns(
        4,
        gap="medium",
    )

    with col1:
        st.metric(
            "Risk Score",
            f"{stock['risk_score'] * 100:.1f}",
        )

    with col2:
        st.metric(
            "Risk Percentile",
            f"{stock['risk_percentile'] * 100:.1f}%",
        )

    with col3:
        st.metric(
            "Risk Level",
            stock["risk_level"],
        )

    with col4:
        st.metric(
            "Latest Price",
            f"${stock['adj_close']:,.2f}",
        )

    st.subheader("Risk Explanation")

    explanation = get_risk_explanation(
        stock["risk_level"]
    )

    st.info(
        f"{selected_ticker} is currently classified as "
        f"{stock['risk_level']} Risk with a risk score of "
        f"{stock['risk_score'] * 100:.1f}.\n\n"
        f"{explanation}\n\n"
        f"Its score ranks above "
        f"{stock['risk_percentile'] * 100:.1f}% "
        "of stocks tracked by RiskAtlas."
    )

    st.caption(
        "This is a relative risk signal, not a prediction "
        "that the stock will decline with certainty."
    )