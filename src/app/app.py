import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine


st.set_page_config(
    page_title="RiskAtlas",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background-color: #070b12;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background-color: #080d16;
        border-right: 1px solid #1b2433;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.03em;
    }

    .page-title {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.15rem;
    }

    .page-subtitle {
        color: #8b98aa;
        font-size: 0.95rem;
        margin-bottom: 1.75rem;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.1rem;
        font-weight: 650;
        margin-bottom: 0.85rem;
    }

    .card {
        background:
            linear-gradient(
                145deg,
                rgba(16, 23, 37, 0.98),
                rgba(10, 15, 25, 0.98)
            );
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 1.15rem 1.25rem;
        min-height: 124px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
    }

    .card-label {
        color: #8b98aa;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .card-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.45rem;
        line-height: 1;
    }

    .card-note {
        color: #718096;
        font-size: 0.8rem;
        margin-top: 0.65rem;
    }

    .critical-value {
        color: #ef4444;
    }

    .high-value {
        color: #f97316;
    }

    .status-row {
        display: flex;
        gap: 0.65rem;
        justify-content: flex-end;
        margin-bottom: 1rem;
    }

    .status-chip {
        background-color: #101725;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 0.55rem 0.8rem;
        color: #cbd5e1;
        font-size: 0.8rem;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #22c55e;
        margin-right: 0.45rem;
    }

    .stock-header {
        background-color: #0e1522;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 1.25rem 1.35rem;
        margin-bottom: 1rem;
    }

    .stock-ticker {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
    }

    .stock-description {
        color: #8b98aa;
        font-size: 0.9rem;
        margin-top: 0.2rem;
    }

    .risk-badge {
        display: inline-block;
        padding: 0.35rem 0.65rem;
        border-radius: 7px;
        font-size: 0.8rem;
        font-weight: 650;
        margin-top: 0.7rem;
    }

    .badge-critical {
        background-color: rgba(239, 68, 68, 0.16);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .badge-high {
        background-color: rgba(249, 115, 22, 0.16);
        color: #fb923c;
        border: 1px solid rgba(249, 115, 22, 0.3);
    }

    .badge-moderate {
        background-color: rgba(234, 179, 8, 0.16);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }

    .badge-low {
        background-color: rgba(34, 197, 94, 0.16);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #1e293b;
        border-radius: 14px;
        overflow: hidden;
    }

    div[data-testid="stSelectbox"] > div {
        border-radius: 10px;
    }

    .explanation-box {
        background-color: #0e1522;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 1.25rem 1.35rem;
        color: #cbd5e1;
        line-height: 1.65;
    }

    .model-row {
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid #1e293b;
        padding: 0.65rem 0;
    }

    .model-row:last-child {
        border-bottom: none;
    }

    .model-label {
        color: #8b98aa;
    }

    .model-value {
        color: #f8fafc;
        font-weight: 600;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is not set in the current terminal."
        )

    return create_engine(
        database_url,
        pool_pre_ping=True,
    )


@st.cache_data(ttl=300)
def load_predictions():
    query = """
        SELECT
            date,
            ticker,
            adj_close,
            risk_score,
            risk_percentile,
            risk_pred,
            risk_level,
            model_name,
            generated_at
        FROM current_risk_predictions
        ORDER BY risk_score DESC;
    """

    predictions = pd.read_sql(
        query,
        get_engine(),
    )

    predictions["date"] = pd.to_datetime(
        predictions["date"]
    )

    predictions["generated_at"] = pd.to_datetime(
        predictions["generated_at"]
    )

    numeric_columns = [
        "adj_close",
        "risk_score",
        "risk_percentile",
    ]

    for column in numeric_columns:
        predictions[column] = pd.to_numeric(
            predictions[column],
            errors="coerce",
        )

    return predictions


def show_metric_card(
    label,
    value,
    note,
    value_class="",
):
    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">{label}</div>
            <div class="card-value {value_class}">
                {value}
            </div>
            <div class="card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_badge_class(risk_level):
    badge_classes = {
        "Critical": "badge-critical",
        "High": "badge-high",
        "Moderate": "badge-moderate",
        "Low": "badge-low",
    }

    return badge_classes.get(
        risk_level,
        "badge-low",
    )


def build_risk_distribution(predictions):
    risk_scores = predictions["risk_score"] * 100

    figure = go.Figure()

    figure.add_trace(
        go.Histogram(
            x=risk_scores,
            nbinsx=28,
            marker=dict(
                color="#4f7cff",
                line=dict(
                    color="#7596ff",
                    width=0.4,
                ),
            ),
            hovertemplate=(
                "Risk Score: %{x:.1f}<br>"
                "Stocks: %{y}<extra></extra>"
            ),
        )
    )

    figure.add_vline(
        x=50,
        line_width=1,
        line_dash="dot",
        line_color="#eab308",
    )

    figure.add_vline(
        x=80,
        line_width=1,
        line_dash="dot",
        line_color="#f97316",
    )

    figure.add_vline(
        x=95,
        line_width=1,
        line_dash="dot",
        line_color="#ef4444",
    )

    figure.update_layout(
        height=390,
        margin=dict(
            l=15,
            r=15,
            t=10,
            b=15,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.04,
        font=dict(
            color="#94a3b8",
            size=12,
        ),
        xaxis=dict(
            title="Risk Score",
            range=[0, 100],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Number of Stocks",
            gridcolor="rgba(148,163,184,0.12)",
            zeroline=False,
        ),
        showlegend=False,
    )

    return figure


def prepare_risk_table(predictions, limit=10):
    table = predictions[
        [
            "ticker",
            "adj_close",
            "risk_score",
            "risk_percentile",
            "risk_level",
        ]
    ].head(limit).copy()

    table.insert(
        0,
        "rank",
        range(1, len(table) + 1),
    )

    table["adj_close"] = table["adj_close"].map(
        lambda value: f"${value:,.2f}"
    )

    table["risk_score"] = (
        table["risk_score"] * 100
    ).round(1)

    table["risk_percentile"] = (
        table["risk_percentile"] * 100
    ).map(
        lambda value: f"{value:.1f}%"
    )

    table = table.rename(
        columns={
            "rank": "Rank",
            "ticker": "Ticker",
            "adj_close": "Price",
            "risk_score": "Risk Score",
            "risk_percentile": "Percentile",
            "risk_level": "Risk Level",
        }
    )

    return table


def build_risk_explanation(stock):
    ticker = stock["ticker"]
    risk_level = stock["risk_level"]
    percentile = stock["risk_percentile"] * 100
    risk_score = stock["risk_score"] * 100

    if risk_level == "Critical":
        condition = (
            "The model is flagging this stock as one of the "
            "highest-risk securities in the current universe."
        )
    elif risk_level == "High":
        condition = (
            "The stock currently shows elevated downside-risk "
            "conditions compared with most tracked companies."
        )
    elif risk_level == "Moderate":
        condition = (
            "The stock shows some elevated risk signals, but it "
            "is not currently among the highest-risk names."
        )
    else:
        condition = (
            "The stock currently shows relatively limited "
            "downside-risk pressure."
        )

    return (
        f"<strong>{ticker}</strong> is currently classified as "
        f"<strong>{risk_level} Risk</strong> with a risk score of "
        f"<strong>{risk_score:.1f}</strong>.<br><br>"
        f"{condition}<br><br>"
        f"Its score ranks above <strong>{percentile:.1f}%</strong> "
        f"of stocks tracked by RiskAtlas.<br><br>"
        "This score is a relative risk signal, not a prediction "
        "that the stock will crash or decline with certainty."
    )


try:
    predictions = load_predictions()
except Exception as error:
    st.error(f"Unable to load RiskAtlas data: {error}")
    st.stop()


if predictions.empty:
    st.warning("No predictions were found.")
    st.stop()


stocks_tracked = predictions["ticker"].nunique()

critical_count = (
    predictions["risk_level"]
    .eq("Critical")
    .sum()
)

high_count = (
    predictions["risk_level"]
    .eq("High")
    .sum()
)

moderate_count = (
    predictions["risk_level"]
    .eq("Moderate")
    .sum()
)

low_count = (
    predictions["risk_level"]
    .eq("Low")
    .sum()
)

prediction_date = predictions["date"].max()

last_updated = predictions["generated_at"].max()


with st.sidebar:
    st.markdown("## RiskAtlas")
    st.caption("Stock Risk Intelligence Platform")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Stock Lookup",
            "Top Risk Stocks",
            "Model Insights",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.caption("Production Model")
    st.write("Logistic Regression V2")

    st.caption("Universe")
    st.write(f"{stocks_tracked:,} stocks")

    st.caption("Last Updated")
    st.write(
        last_updated.strftime(
            "%Y-%m-%d %H:%M"
        )
    )


if page == "Overview":
    header_col, status_col = st.columns(
        [2.3, 1]
    )

    with header_col:
        st.markdown(
            '<div class="page-title">Overview</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="page-subtitle">
                Daily stock risk intelligence across the tracked universe
            </div>
            """,
            unsafe_allow_html=True,
        )

    with status_col:
        st.markdown(
            f"""
            <div class="status-row">
                <div class="status-chip">
                    Market Date: {prediction_date:%Y-%m-%d}
                </div>
                <div class="status-chip">
                    <span class="status-dot"></span>
                    Data Current
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    metric_col1, metric_col2, metric_col3, metric_col4 = (
        st.columns(4)
    )

    with metric_col1:
        show_metric_card(
            "Stocks Tracked",
            f"{stocks_tracked:,}",
            "Current stock universe",
        )

    with metric_col2:
        show_metric_card(
            "Critical Risk",
            f"{critical_count:,}",
            f"{critical_count / stocks_tracked:.1%} of universe",
            "critical-value",
        )

    with metric_col3:
        show_metric_card(
            "High Risk",
            f"{high_count:,}",
            f"{high_count / stocks_tracked:.1%} of universe",
            "high-value",
        )

    with metric_col4:
        show_metric_card(
            "Prediction Date",
            prediction_date.strftime("%Y-%m-%d"),
            "Latest market close",
        )

    st.write("")

    chart_col, table_col = st.columns(
        [1.2, 1]
    )

    with chart_col:
        st.markdown(
            '<div class="section-title">'
            "Risk Score Distribution"
            "</div>",
            unsafe_allow_html=True,
        )

        risk_distribution = build_risk_distribution(
            predictions
        )

        st.plotly_chart(
            risk_distribution,
            width="stretch",
            config={
                "displayModeBar": False,
            },
        )

    with table_col:
        st.markdown(
            '<div class="section-title">'
            "Highest Risk Stocks"
            "</div>",
            unsafe_allow_html=True,
        )

        top_risk_table = prepare_risk_table(
            predictions,
            limit=10,
        )

        st.dataframe(
            top_risk_table,
            width="stretch",
            height=390,
            hide_index=True,
        )

    st.write("")

    distribution_col1, distribution_col2 = st.columns(
        [1.15, 1]
    )

    with distribution_col1:
        st.markdown(
            '<div class="section-title">'
            "Risk Level Breakdown"
            "</div>",
            unsafe_allow_html=True,
        )

        level_data = pd.DataFrame(
            {
                "Risk Level": [
                    "Critical",
                    "High",
                    "Moderate",
                    "Low",
                ],
                "Stocks": [
                    critical_count,
                    high_count,
                    moderate_count,
                    low_count,
                ],
            }
        )

        st.bar_chart(
            level_data,
            x="Risk Level",
            y="Stocks",
            horizontal=True,
            color="#4f7cff",
        )

    with distribution_col2:
        st.markdown(
            '<div class="section-title">'
            "Model Summary"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="card">
                <div class="model-row">
                    <span class="model-label">Model</span>
                    <span class="model-value">
                        Logistic Regression V2
                    </span>
                </div>
                <div class="model-row">
                    <span class="model-label">ROC-AUC</span>
                    <span class="model-value">0.6130</span>
                </div>
                <div class="model-row">
                    <span class="model-label">PR-AUC</span>
                    <span class="model-value">0.2028</span>
                </div>
                <div class="model-row">
                    <span class="model-label">
                        Prediction Horizon
                    </span>
                    <span class="model-value">
                        10 trading days
                    </span>
                </div>
                <div class="model-row">
                    <span class="model-label">
                        Ranking Method
                    </span>
                    <span class="model-value">
                        Cross-sectional percentile
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


elif page == "Stock Lookup":
    st.markdown(
        '<div class="page-title">Stock Lookup</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="page-subtitle">
            Explore the latest RiskAtlas signal for an individual stock
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_ticker = st.selectbox(
        "Select a ticker",
        sorted(
            predictions["ticker"].unique()
        ),
    )

    selected_stock = predictions.loc[
        predictions["ticker"].eq(
            selected_ticker
        )
    ].iloc[0]

    risk_level = selected_stock["risk_level"]
    badge_class = get_badge_class(risk_level)

    st.markdown(
        f"""
        <div class="stock-header">
            <div class="stock-ticker">
                {selected_ticker}
            </div>
            <div class="stock-description">
                Latest RiskAtlas market signal
            </div>
            <div class="risk-badge {badge_class}">
                {risk_level} Risk
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stock_col1, stock_col2, stock_col3, stock_col4 = (
        st.columns(4)
    )

    with stock_col1:
        show_metric_card(
            "Risk Score",
            f"{selected_stock['risk_score'] * 100:.1f}",
            "Model probability score",
        )

    with stock_col2:
        show_metric_card(
            "Risk Percentile",
            (
                f"{selected_stock['risk_percentile'] * 100:.1f}%"
            ),
            "Relative to tracked stocks",
        )

    with stock_col3:
        show_metric_card(
            "Risk Level",
            selected_stock["risk_level"],
            "Percentile classification",
        )

    with stock_col4:
        show_metric_card(
            "Latest Price",
            f"${selected_stock['adj_close']:,.2f}",
            prediction_date.strftime("%Y-%m-%d"),
        )

    st.write("")

    explanation_col, score_col = st.columns(
        [1.35, 1]
    )

    with explanation_col:
        st.markdown(
            '<div class="section-title">'
            "Risk Explanation"
            "</div>",
            unsafe_allow_html=True,
        )

        explanation = build_risk_explanation(
            selected_stock
        )

        st.markdown(
            f"""
            <div class="explanation-box">
                {explanation}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with score_col:
        st.markdown(
            '<div class="section-title">'
            "Relative Risk Position"
            "</div>",
            unsafe_allow_html=True,
        )

        percentile = (
            selected_stock["risk_percentile"] * 100
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=percentile,
                number={
                    "suffix": "%",
                    "font": {
                        "color": "#f8fafc",
                    },
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#64748b",
                    },
                    "bar": {
                        "color": "#4f7cff",
                    },
                    "bgcolor": "#101725",
                    "bordercolor": "#1e293b",
                    "steps": [
                        {
                            "range": [0, 50],
                            "color": "#12251d",
                        },
                        {
                            "range": [50, 80],
                            "color": "#2a2412",
                        },
                        {
                            "range": [80, 95],
                            "color": "#2c1b10",
                        },
                        {
                            "range": [95, 100],
                            "color": "#2b1318",
                        },
                    ],
                },
            )
        )

        gauge.update_layout(
            height=290,
            margin=dict(
                l=30,
                r=30,
                t=20,
                b=20,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#94a3b8",
            ),
        )

        st.plotly_chart(
            gauge,
            width="stretch",
            config={
                "displayModeBar": False,
            },
        )


elif page == "Top Risk Stocks":
    st.markdown(
        '<div class="page-title">Top Risk Stocks</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="page-subtitle">
            Rank and filter stocks by current RiskAtlas signals
        </div>
        """,
        unsafe_allow_html=True,
    )

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        selected_level = st.selectbox(
            "Risk level",
            [
                "All",
                "Critical",
                "High",
                "Moderate",
                "Low",
            ],
        )

    with filter_col2:
        row_limit = st.selectbox(
            "Rows",
            [
                10,
                25,
                50,
                100,
                500,
            ],
            index=1,
        )

    filtered_predictions = predictions.copy()

    if selected_level != "All":
        filtered_predictions = filtered_predictions.loc[
            filtered_predictions["risk_level"].eq(
                selected_level
            )
        ]

    full_table = prepare_risk_table(
        filtered_predictions,
        limit=row_limit,
    )

    st.dataframe(
        full_table,
        width="stretch",
        hide_index=True,
        height=620,
    )


elif page == "Model Insights":
    st.markdown(
        '<div class="page-title">Model Insights</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="page-subtitle">
            Production model performance and methodology
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3 = (
        st.columns(3)
    )

    with metric_col1:
        show_metric_card(
            "ROC-AUC",
            "0.6130",
            "Discrimination performance",
        )

    with metric_col2:
        show_metric_card(
            "PR-AUC",
            "0.2028",
            "Positive-class precision and recall",
        )

    with metric_col3:
        show_metric_card(
            "Features",
            "14",
            "Production input variables",
        )

    st.write("")

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        st.markdown(
            '<div class="section-title">'
            "Model Information"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="card">
                <div class="model-row">
                    <span class="model-label">Model Type</span>
                    <span class="model-value">
                        Logistic Regression
                    </span>
                </div>
                <div class="model-row">
                    <span class="model-label">Model Version</span>
                    <span class="model-value">V2</span>
                </div>
                <div class="model-row">
                    <span class="model-label">Target</span>
                    <span class="model-value">
                        10-day downside event
                    </span>
                </div>
                <div class="model-row">
                    <span class="model-label">
                        Risk Classification
                    </span>
                    <span class="model-value">
                        Percentile buckets
                    </span>
                </div>
                <div class="model-row">
                    <span class="model-label">
                        Training Dataset
                    </span>
                    <span class="model-value">
                        model_dataset_v2
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with detail_col2:
        st.markdown(
            '<div class="section-title">'
            "Production Features"
            "</div>",
            unsafe_allow_html=True,
        )

        feature_data = pd.DataFrame(
            {
                "Feature": [
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
                    "drawdown",
                ]
            }
        )

        st.dataframe(
            feature_data,
            width="stretch",
            hide_index=True,
            height=430,
        )