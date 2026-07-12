import streamlit as st

from data_access import (
    clear_prediction_cache,
    load_predictions,
)
from views.model_insights import render_model_insights
from views.overview import render_overview
from views.stock_lookup import render_stock_lookup


st.set_page_config(
    page_title="RiskAtlas",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --background: #07111f;
        --surface: #0d1828;
        --surface-light: #132238;
        --border: #20324a;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --accent: #38bdf8;
    }

    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(56, 189, 248, 0.08),
                transparent 30%
            ),
            #07111f;
        color: var(--text-primary);
    }

    .block-container {
        max-width: 1420px;
        padding-top: 5rem;
        padding-bottom: 4rem;
    }

    h1 {
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        line-height: 1.1 !important;
        letter-spacing: -0.05em;
        margin-top: 0 !important;
        margin-bottom: 0.25rem !important;
        padding-bottom: 0.15rem !important;
        color: #f8fafc !important;
    }

    h2 {
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em;
        color: #f8fafc !important;
        margin-top: 1.5rem !important;
    }

    h3 {
        color: #e2e8f0 !important;
        letter-spacing: -0.02em;
    }

    p,
    label,
    .stCaption {
        color: var(--text-secondary);
    }

    [data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(19, 34, 56, 0.95),
                rgba(13, 24, 40, 0.95)
            );
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.15rem 1.25rem;
        min-height: 125px;
        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.18),
            inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }

    [data-testid="stMetric"]:hover {
        border-color: rgba(56, 189, 248, 0.45);
        transform: translateY(-1px);
        transition: 0.2s ease;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 750;
        letter-spacing: -0.03em;
    }

    [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1rem;
    }

    [data-baseweb="tab"] {
        height: 48px;
        padding-left: 0;
        padding-right: 0;
        font-weight: 650;
        color: #94a3b8;
    }

    [aria-selected="true"][data-baseweb="tab"] {
        color: #38bdf8 !important;
    }

    [data-baseweb="select"] > div {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        min-height: 48px;
    }

    [data-baseweb="select"] > div:hover {
        border-color: rgba(56, 189, 248, 0.5);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 16px;
        overflow: hidden;
        background-color: var(--surface);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.14);
    }

    [data-testid="stAlert"] {
        background-color: rgba(19, 34, 56, 0.92);
        border: 1px solid var(--border);
        border-radius: 14px;
    }

    [data-testid="stButton"] button {
        border-radius: 12px;
        min-height: 44px;
        font-weight: 650;
        border: 1px solid var(--border);
        background-color: var(--surface-light);
    }

    [data-testid="stButton"] button:hover {
        border-color: #38bdf8;
        color: #38bdf8;
    }

    .riskatlas-eyebrow {
        color: #38bdf8;
        font-size: 0.75rem;
        font-weight: 750;
        line-height: 1.4;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .riskatlas-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.35rem;
    }

    .risk-badge {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 0.04em;
    }

    .risk-low {
        color: #86efac;
        background-color: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    .risk-moderate {
        color: #fde68a;
        background-color: rgba(234, 179, 8, 0.12);
        border: 1px solid rgba(234, 179, 8, 0.3);
    }

    .risk-high {
        color: #fdba74;
        background-color: rgba(249, 115, 22, 0.12);
        border: 1px solid rgba(249, 115, 22, 0.3);
    }

    .risk-critical {
        color: #fca5a5;
        background-color: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.3);
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


header_col, refresh_col = st.columns([5, 1])

with header_col:
    st.markdown(
        """
        <div class="riskatlas-eyebrow">
            Market Risk Intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title("RiskAtlas")

    st.markdown(
        """
        <div class="riskatlas-subtitle">
            Forward-looking downside risk signals across the S&P 500
        </div>
        """,
        unsafe_allow_html=True,
    )

with refresh_col:
    st.write("")

    refresh_clicked = st.button(
        "Refresh Data",
        use_container_width=True,
        type="secondary",
    )


if refresh_clicked:
    clear_prediction_cache()


try:
    with st.spinner("Loading RiskAtlas data..."):
        predictions = load_predictions()

except Exception as error:
    st.error("Unable to load RiskAtlas data.")
    st.code(str(error), language="text")
    st.stop()


if predictions.empty:
    st.warning(
        "The database query completed, but no predictions were returned."
    )
    st.stop()


overview_tab, lookup_tab, model_tab = st.tabs(
    [
        "Overview",
        "Stock Lookup",
        "Model Insights",
    ]
)


with overview_tab:
    render_overview(predictions)


with lookup_tab:
    render_stock_lookup(predictions)


with model_tab:
    render_model_insights(predictions)