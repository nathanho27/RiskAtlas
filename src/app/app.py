import os

import pandas as pd
import psycopg2
import streamlit as st


st.set_page_config(
    page_title="RiskAtlas",
    page_icon="📊",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background-color: #080c14;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 3rem !important;
        font-weight: 750 !important;
        letter-spacing: -0.04em;
        margin-bottom: 0 !important;
    }

    h2,
    h3 {
        letter-spacing: -0.025em;
    }

    [data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #263247;
        border-radius: 14px;
        padding: 1rem 1.15rem;
        min-height: 120px;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 650;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 700;
    }

    [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid #263247;
    }

    [data-baseweb="tab"] {
        height: 48px;
        padding-left: 0;
        padding-right: 0;
        font-weight: 600;
    }

    [data-baseweb="select"] > div {
        background-color: #111827;
        border-color: #263247;
        border-radius: 10px;
        min-height: 46px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #263247;
        border-radius: 14px;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    [data-testid="stButton"] button {
        border-radius: 10px;
        min-height: 44px;
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


@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def load_predictions():
    database_url = os.getenv(
        "DATABASE_URL"
    )

    if not database_url:
        raise ValueError(
            "DATABASE_URL is not set."
        )

    connection = None

    query = """
        SELECT
            date,
            ticker,
            adj_close,
            risk_score,
            risk_pred,
            risk_level,
            model_name,
            generated_at,
            risk_percentile
        FROM current_risk_predictions
        ORDER BY
            risk_score DESC,
            ticker ASC;
    """

    try:
        connection = psycopg2.connect(
            database_url,
            connect_timeout=10,
        )

        predictions = pd.read_sql_query(
            query,
            connection,
        )

    finally:
        if connection is not None:
            connection.close()

    if predictions.empty:
        return predictions

    predictions["ticker"] = (
        predictions["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    predictions["risk_level"] = (
        predictions["risk_level"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    predictions["date"] = pd.to_datetime(
        predictions["date"],
        errors="coerce",
    )

    predictions["generated_at"] = (
        pd.to_datetime(
            predictions["generated_at"],
            errors="coerce",
        )
    )

    numeric_columns = [
        "adj_close",
        "risk_score",
        "risk_pred",
        "risk_percentile",
    ]

    for column in numeric_columns:
        predictions[column] = (
            pd.to_numeric(
                predictions[column],
                errors="coerce",
            )
        )

    predictions = predictions.dropna(
        subset=[
            "date",
            "ticker",
            "adj_close",
            "risk_score",
            "risk_level",
            "risk_percentile",
        ]
    )

    predictions = (
        predictions
        .sort_values(
            by=[
                "ticker",
                "generated_at",
                "date",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .drop_duplicates(
            subset=["ticker"],
            keep="first",
        )
        .sort_values(
            by="risk_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return predictions


def get_risk_explanation(stock):
    risk_level = stock["risk_level"]

    if risk_level == "Critical":
        return (
            "This stock currently ranks among the "
            "highest-risk stocks tracked by RiskAtlas."
        )

    if risk_level == "High":
        return (
            "This stock currently shows elevated "
            "downside-risk conditions relative to "
            "most tracked stocks."
        )

    if risk_level == "Moderate":
        return (
            "This stock shows some elevated risk, "
            "but it is not currently among the "
            "highest-risk stocks."
        )

    return (
        "This stock currently shows relatively "
        "limited downside-risk pressure."
    )


def format_timestamp(timestamp):
    if pd.isna(timestamp):
        return "Unavailable"

    return timestamp.strftime(
        "%Y-%m-%d %H:%M"
    )


def initialize_predictions():
    if "predictions" in st.session_state:
        return

    with st.spinner(
        "Loading RiskAtlas data..."
    ):
        st.session_state.predictions = (
            load_predictions()
        )


def validate_ticker_state(
    ticker_options,
):
    if not ticker_options:
        return

    default_ticker = (
        "NVDA"
        if "NVDA" in ticker_options
        else ticker_options[0]
    )

    current_ticker = (
        st.session_state.get(
            "ticker_selector"
        )
    )

    if current_ticker not in ticker_options:
        st.session_state.ticker_selector = (
            default_ticker
        )


def refresh_predictions():
    try:
        load_predictions.clear()

        with st.spinner(
            "Refreshing predictions..."
        ):
            refreshed_predictions = (
                load_predictions()
            )

        if refreshed_predictions.empty:
            st.warning(
                "Refresh returned no predictions. "
                "The existing data was preserved."
            )
            return

        st.session_state.predictions = (
            refreshed_predictions
        )

        refreshed_tickers = (
            refreshed_predictions[
                "ticker"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        current_ticker = (
            st.session_state.get(
                "ticker_selector"
            )
        )

        if current_ticker not in refreshed_tickers:
            st.session_state.ticker_selector = (
                "NVDA"
                if "NVDA" in refreshed_tickers
                else refreshed_tickers[0]
            )

        st.rerun()

    except Exception as error:
        st.error(
            "Unable to refresh RiskAtlas data. "
            f"The existing data was preserved: {error}"
        )


try:
    initialize_predictions()

except Exception as error:
    st.error(
        f"Unable to load RiskAtlas data: {error}"
    )
    st.stop()


df = st.session_state.predictions


if df.empty:
    st.warning(
        "No predictions were found."
    )
    st.stop()


ticker_options = sorted(
    df["ticker"]
    .dropna()
    .unique()
    .tolist()
)


if not ticker_options:
    st.warning(
        "No valid tickers were found."
    )
    st.stop()


validate_ticker_state(
    ticker_options
)


stocks_tracked = df["ticker"].nunique()

critical_count = (
    df["risk_level"]
    .eq("Critical")
    .sum()
)

high_count = (
    df["risk_level"]
    .eq("High")
    .sum()
)

prediction_date = df["date"].max()

last_updated = df["generated_at"].max()


header_col, refresh_col = st.columns(
    [5, 1]
)


with header_col:
    st.title("RiskAtlas")

    st.caption(
        "AI-Powered Stock Risk Intelligence"
    )


with refresh_col:
    st.write("")

    st.button(
        "Refresh Data",
        use_container_width=True,
        type="secondary",
        on_click=refresh_predictions,
    )


overview_tab, lookup_tab, model_tab = (
    st.tabs(
        [
            "Overview",
            "Stock Lookup",
            "Model Insights",
        ]
    )
)


with overview_tab:
    st.subheader(
        "Market Overview"
    )

    col1, col2, col3, col4 = (
        st.columns(
            4,
            gap="medium",
        )
    )

    with col1:
        st.metric(
            "Stocks Tracked",
            f"{stocks_tracked:,}",
        )

    with col2:
        st.metric(
            "Critical Risk",
            f"{critical_count:,}",
        )

    with col3:
        st.metric(
            "High Risk",
            f"{high_count:,}",
        )

    with col4:
        st.metric(
            "Prediction Date",
            prediction_date.strftime(
                "%Y-%m-%d"
            ),
        )

    st.subheader(
        "Top Risk Stocks"
    )

    top_risk = (
        df[
            [
                "ticker",
                "adj_close",
                "risk_score",
                "risk_percentile",
                "risk_level",
            ]
        ]
        .head(15)
        .copy()
    )

    top_risk["adj_close"] = (
        top_risk["adj_close"]
        .map(
            lambda value: (
                f"${value:,.2f}"
            )
        )
    )

    top_risk["risk_score"] = (
        top_risk["risk_score"]
        .mul(100)
        .round(1)
    )

    top_risk["risk_percentile"] = (
        top_risk[
            "risk_percentile"
        ]
        .mul(100)
        .map(
            lambda value: (
                f"{value:.1f}%"
            )
        )
    )

    top_risk = top_risk.rename(
        columns={
            "ticker": "Ticker",
            "adj_close": "Price",
            "risk_score": "Risk Score",
            "risk_percentile": (
                "Percentile"
            ),
            "risk_level": "Risk Level",
        }
    )

    st.dataframe(
        top_risk,
        width="stretch",
        hide_index=True,
        height=560,
    )


with lookup_tab:
    st.subheader(
        "Stock Lookup"
    )

    st.selectbox(
        "Select a ticker",
        options=ticker_options,
        key="ticker_selector",
    )

    selected_ticker = (
        st.session_state[
            "ticker_selector"
        ]
    )

    stock_rows = df.loc[
        df["ticker"].eq(
            selected_ticker
        )
    ]

    if stock_rows.empty:
        st.warning(
            "The selected ticker is no longer "
            "available. Select another ticker."
        )

    else:
        stock = stock_rows.iloc[0]

        col1, col2, col3, col4 = (
            st.columns(
                4,
                gap="medium",
            )
        )

        with col1:
            st.metric(
                "Risk Score",
                (
                    f"{stock['risk_score'] * 100:.1f}"
                ),
            )

        with col2:
            st.metric(
                "Risk Percentile",
                (
                    f"{stock['risk_percentile'] * 100:.1f}%"
                ),
            )

        with col3:
            st.metric(
                "Risk Level",
                stock["risk_level"],
            )

        with col4:
            st.metric(
                "Latest Price",
                (
                    f"${stock['adj_close']:,.2f}"
                ),
            )

        st.subheader(
            "Risk Explanation"
        )

        explanation = (
            get_risk_explanation(
                stock
            )
        )

        st.info(
            f"{selected_ticker} is currently "
            f"classified as "
            f"{stock['risk_level']} Risk with "
            f"a risk score of "
            f"{stock['risk_score'] * 100:.1f}."
            "\n\n"
            f"{explanation}"
            "\n\n"
            "Its score ranks above "
            f"{stock['risk_percentile'] * 100:.1f}% "
            "of stocks tracked by RiskAtlas."
        )

        st.caption(
            "This is a relative risk signal, "
            "not a prediction that the stock "
            "will decline with certainty."
        )


with model_tab:
    st.subheader(
        "Model Insights"
    )

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
                (
                    "Cross-sectional "
                    "percentile"
                ),
                "model_dataset_v2",
                prediction_date.strftime(
                    "%Y-%m-%d"
                ),
                format_timestamp(
                    last_updated
                ),
            ],
        }
    )

    st.dataframe(
        model_info,
        width="stretch",
        hide_index=True,
    )