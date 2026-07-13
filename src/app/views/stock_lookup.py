import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_access import load_risk_history


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


def get_alert_status(risk_pred: int) -> str:
    return (
        "Triggered"
        if risk_pred == 1
        else "Below Threshold"
    )


def render_risk_history(
    ticker: str,
) -> None:
    st.subheader("Historical Risk Percentile")

    try:
        history = load_risk_history(
            ticker=ticker,
            days=90,
        )

    except Exception as error:
        st.warning(
            "Historical risk data could not be loaded."
        )
        st.caption(str(error))
        return

    if history.empty:
        st.info(
            "No historical risk observations are available "
            f"for {ticker} yet."
        )
        return
    
    if len(history) == 1:
        st.info(
        "Historical tracking has started. Additional dates "
        "will appear automatically as new predictions are generated."
        )

        return

    chart_data = history.copy()

    chart_data["risk_percentile_pct"] = (
        chart_data["risk_percentile"] * 100
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=chart_data["date"],
            y=chart_data["risk_percentile_pct"],
            mode="lines+markers",
            name="Risk Percentile",
            customdata=chart_data[
                [
                    "risk_score",
                    "risk_level",
                    "adj_close",
                ]
            ],
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                "Risk Percentile: %{y:.1f}%<br>"
                "Risk Score: %{customdata[0]:.3f}<br>"
                "Risk Level: %{customdata[1]}<br>"
                "Price: $%{customdata[2]:,.2f}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_hline(
        y=50,
        line_dash="dot",
        annotation_text="Moderate",
        annotation_position="bottom right",
    )

    figure.add_hline(
        y=80,
        line_dash="dot",
        annotation_text="High",
        annotation_position="bottom right",
    )

    figure.add_hline(
        y=95,
        line_dash="dot",
        annotation_text="Critical",
        annotation_position="bottom right",
    )

    figure.update_layout(
        xaxis_title="Prediction Date",
        yaxis_title="Risk Percentile",
        yaxis=dict(
            range=[0, 100],
            ticksuffix="%",
        ),
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
        height=420,
        showlegend=False,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        key=f"risk_history_{ticker}",
    )

    observation_count = len(chart_data)
    first_date = chart_data["date"].min()
    latest_date = chart_data["date"].max()

    if observation_count == 1:
        st.caption(
            "Historical tracking currently contains one observation. "
            "The chart will build automatically as future pipeline "
            "runs add new prediction dates."
        )

    else:
        st.caption(
            f"Showing {observation_count} observations from "
            f"{first_date:%b %d, %Y} through "
            f"{latest_date:%b %d, %Y}."
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

    risk_score = stock["risk_score"] * 100
    risk_percentile = stock["risk_percentile"] * 100
    risk_pred = int(stock["risk_pred"])
    alert_status = get_alert_status(risk_pred)

    col1, col2, col3, col4, col5 = st.columns(
        5,
        gap="medium",
    )

    with col1:
        st.metric(
            "Risk Score",
            f"{risk_score:.1f}",
        )

    with col2:
        st.metric(
            "Risk Percentile",
            f"{risk_percentile:.1f}%",
        )

    with col3:
        st.metric(
            "Risk Level",
            stock["risk_level"],
        )

    with col4:
        st.metric(
            "Alert Status",
            alert_status,
        )

    with col5:
        st.metric(
            "Latest Price",
            f"${stock['adj_close']:,.2f}",
        )

    render_risk_history(
        ticker=selected_ticker,
    )

    st.subheader("Risk Explanation")

    explanation = get_risk_explanation(
        stock["risk_level"]
    )

    alert_explanation = (
        "The model's estimated downside-risk probability "
        "exceeds the production alert threshold."
        if risk_pred == 1
        else (
            "The stock is ranked relative to the current market, "
            "but its estimated downside-risk probability remains "
            "below the production alert threshold."
        )
    )

    st.info(
        f"{selected_ticker} is currently classified as "
        f"{stock['risk_level']} Risk with a risk score of "
        f"{risk_score:.1f}.\n\n"
        f"{explanation}\n\n"
        f"It ranks in the {risk_percentile:.1f}th percentile "
        "of stocks tracked by RiskAtlas.\n\n"
        f"{alert_explanation}"
    )

    st.caption(
        "RiskAtlas estimates relative downside risk over the next "
        "10 trading days. This signal does not guarantee that the "
        "stock will decline."
    )