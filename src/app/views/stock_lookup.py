import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


SRC_DIR = Path(__file__).resolve().parents[2]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from ai.ai_explanations import (
    ask_riskatlas,
    build_stock_context,
)
from ai.risk_drivers import generate_risk_drivers
from data_access import (
    load_risk_history,
    load_stock_features,
)


def get_alert_status(risk_pred: int) -> str:
    return (
        "Triggered"
        if risk_pred == 1
        else "No Alert"
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

    st.caption(
        f"Showing {observation_count} observations from "
        f"{first_date:%b %d, %Y} through "
        f"{latest_date:%b %d, %Y}."
    )


def render_risk_drivers(
    ticker: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    st.subheader("Key Risk Drivers")

    try:
        feature_data = load_stock_features(
            ticker=ticker,
        )

    except Exception as error:
        st.warning(
            "Risk drivers could not be loaded."
        )
        st.caption(str(error))
        return [], {}

    if feature_data.empty:
        st.info(
            "No feature data is currently available "
            f"for {ticker}."
        )
        return [], {}

    stock_features = feature_data.iloc[0]

    try:
        drivers = generate_risk_drivers(
            stock_features=stock_features,
            max_drivers=5,
        )

    except Exception as error:
        st.warning(
            "Risk drivers could not be generated."
        )
        st.caption(str(error))
        return [], stock_features.to_dict()

    if not drivers:
        st.info(
            "No major risk drivers were identified "
            f"for {ticker} using the current thresholds."
        )

        return [], stock_features.to_dict()

    for index, driver in enumerate(
        drivers,
        start=1,
    ):
        direction = driver["direction"]
        severity = int(driver["severity"])

        status_label = (
            "Protective"
            if direction == "protective"
            else "Risk Increasing"
        )

        with st.container(border=True):
            title_col, status_col = st.columns(
                [3, 1],
                gap="medium",
            )

            with title_col:
                st.markdown(
                    f"#### {index}. {driver['title']}"
                )

                st.caption(
                    driver["category"]
                )

            with status_col:
                st.markdown(
                    f"**{status_label}**"
                )

                st.caption(
                    f"Severity: {severity}/5"
                )

            st.write(
                driver["explanation"]
            )

    st.caption(
        "Drivers are generated from the latest stock, "
        "market, sector, volatility and breadth features "
        "used by the RiskAtlas V3 model."
    )

    return drivers, stock_features.to_dict()


def render_risk_brief(
    ticker: str,
    risk_level: str,
    risk_score: float,
    risk_percentile: float,
    risk_pred: int,
    drivers: list[dict[str, Any]],
) -> None:
    st.subheader("Risk Brief")

    risk_drivers = [
        driver
        for driver in drivers
        if driver["direction"] == "risk"
    ]

    protective_drivers = [
        driver
        for driver in drivers
        if driver["direction"] == "protective"
    ]

    brief_parts = [
        (
            f"{ticker} is currently classified as "
            f"{risk_level} Risk with a risk score of "
            f"{risk_score:.1f}."
        ),
        (
            f"It ranks in the {risk_percentile:.1f}th percentile "
            "of stocks tracked by RiskAtlas."
        ),
    ]

    if risk_drivers:
        risk_titles = [
            driver["title"].lower()
            for driver in risk_drivers[:3]
        ]

        brief_parts.append(
            "The primary risk signals are "
            f"{', '.join(risk_titles)}."
        )

    else:
        brief_parts.append(
            "No major risk-increasing drivers exceeded "
            "the current explanation thresholds."
        )

    if protective_drivers:
        protective_titles = [
            driver["title"].lower()
            for driver in protective_drivers[:2]
        ]

        brief_parts.append(
            "Offsetting factors include "
            f"{', '.join(protective_titles)}."
        )

    if risk_pred == 1:
        brief_parts.append(
            "The model's estimated downside-risk signal "
            "exceeds the production alert threshold."
        )

    else:
        brief_parts.append(
            "The model's estimated downside-risk signal "
            "remains below the production alert threshold."
        )

    st.info(
        "\n\n".join(brief_parts)
    )

    st.caption(
        "RiskAtlas estimates relative downside risk over the next "
        "10 trading days. This signal does not guarantee that the "
        "stock will decline."
    )


def render_ask_riskatlas(
    ticker: str,
    company_name: str,
    sector: str,
    sub_industry: str,
    risk_level: str,
    risk_score: float,
    risk_percentile: float,
    risk_pred: int,
    drivers: list[dict[str, Any]],
    stock_features: dict[str, Any],
) -> None:
    st.subheader("Ask RiskAtlas")

    st.caption(
        "Ask questions about the model's risk assessment, "
        "risk drivers, trends, and market signals."
    )

    ticker_state_key = "riskatlas_chat_ticker"
    messages_state_key = "riskatlas_chat_messages"
    pending_question_key = "riskatlas_pending_question"

    if st.session_state.get(ticker_state_key) != ticker:
        st.session_state[ticker_state_key] = ticker
        st.session_state[messages_state_key] = []
        st.session_state[pending_question_key] = None

    if messages_state_key not in st.session_state:
        st.session_state[messages_state_key] = []

    if pending_question_key not in st.session_state:
        st.session_state[pending_question_key] = None

    stock_context = build_stock_context(
        ticker=ticker,
        company_name=company_name,
        sector=sector,
        sub_industry=sub_industry,
        risk_level=risk_level,
        risk_score=risk_score,
        risk_percentile=risk_percentile,
        risk_pred=risk_pred,
        drivers=drivers,
        stock_features=stock_features,
    )

    suggested_questions = [
        f"Summarize why {ticker} is currently {risk_level.lower()} risk.",
        "What are the three strongest risk-increasing signals?",
        "What is helping offset the current risk?",
        "Explain this like I'm new to investing.",
    ]

    has_messages = bool(
        st.session_state[messages_state_key]
    )

    with st.expander(
        "Suggested questions",
        expanded=not has_messages,
    ):
        suggestion_columns = st.columns(
            2,
            gap="small",
        )

        for index, suggestion in enumerate(
            suggested_questions
        ):
            column = suggestion_columns[index % 2]

            if column.button(
                suggestion,
                key=f"{ticker}_suggestion_{index}",
                use_container_width=True,
            ):
                st.session_state[
                    pending_question_key
                ] = suggestion

    for message in st.session_state[
        messages_state_key
    ]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed_question = st.chat_input(
        f"Ask about {ticker}'s risk assessment"
    )

    question = (
        typed_question
        or st.session_state.get(
            pending_question_key
        )
    )

    if question:
        st.session_state[
            pending_question_key
        ] = None

        previous_messages = list(
            st.session_state[messages_state_key]
        )

        st.session_state[
            messages_state_key
        ].append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        answer = None

        with st.chat_message("assistant"):
            with st.spinner(
                "Analyzing RiskAtlas signals..."
            ):
                try:
                    answer = ask_riskatlas(
                        question=question,
                        stock_context=stock_context,
                        conversation_history=(
                            previous_messages
                        ),
                    )

                except Exception as error:
                    st.error(
                        "Ask RiskAtlas could not generate "
                        "a response."
                    )
                    st.caption(str(error))

                else:
                    st.markdown(answer)

        if answer:
            st.session_state[
                messages_state_key
            ].append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

    if st.session_state[messages_state_key]:
        if st.button(
            "Clear conversation",
            key=f"clear_chat_{ticker}",
        ):
            st.session_state[
                messages_state_key
            ] = []

            st.session_state[
                pending_question_key
            ] = None

            st.rerun()

    st.caption(
        "Ask RiskAtlas explains model-generated evidence. "
        "It does not provide investment advice or generate "
        "the underlying prediction."
    )

def render_stock_lookup(
    df: pd.DataFrame,
) -> None:
    ticker_options = sorted(
        df["ticker"]
        .dropna()
        .unique()
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

    risk_score = float(
        stock["risk_score"]
    ) * 100

    risk_percentile = float(
        stock["risk_percentile"]
    ) * 100

    risk_pred = int(
        stock["risk_pred"]
    )

    alert_status = get_alert_status(
        risk_pred
    )

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
            str(stock["risk_level"]),
        )

    with col4:
        st.metric(
            "Alert Status",
            alert_status,
        )

    with col5:
        st.metric(
            "Latest Price",
            f"${float(stock['adj_close']):,.2f}",
        )

    drivers, stock_features = render_risk_drivers(
        ticker=selected_ticker,
    )

    if not stock_features:
        stock_features = stock.to_dict()

    company_name = str(
        stock_features.get(
            "company_name",
            selected_ticker,
        )
    )

    sector = str(
        stock_features.get(
            "sector",
            "Unknown",
        )
    )

    sub_industry = str(
        stock_features.get(
            "sub_industry",
            "Unknown",
        )
    )
    st.markdown(f"### {company_name}")
    st.caption(f"{sector} • {sub_industry}")

    render_risk_history(
        ticker=selected_ticker,
    )

    render_risk_brief(
        ticker=selected_ticker,
        risk_level=str(
            stock["risk_level"]
        ),
        risk_score=risk_score,
        risk_percentile=risk_percentile,
        risk_pred=risk_pred,
        drivers=drivers,
    )

    render_ask_riskatlas(
        ticker=selected_ticker,
        company_name=company_name,
        sector=sector,
        sub_industry=sub_industry,
        risk_level=str(
            stock["risk_level"]
        ),
        risk_score=risk_score,
        risk_percentile=risk_percentile,
        risk_pred=risk_pred,
        drivers=drivers,
        stock_features=stock_features,
    )