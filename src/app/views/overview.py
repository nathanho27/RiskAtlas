import pandas as pd
import plotly.express as px
import streamlit as st


def render_overview(df: pd.DataFrame) -> None:
    stocks_tracked = len(df)

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

    average_risk_score = (
        df["risk_score"].mean() * 100
    )

    prediction_date = df["date"].max()

    st.subheader("Market Overview")

    col1, col2, col3, col4 = st.columns(
        4,
        gap="medium",
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
            "Average Risk Score",
            f"{average_risk_score:.1f}",
        )

    st.caption(
        "Latest prediction date: "
        f"{prediction_date.strftime('%Y-%m-%d')}"
    )

    st.subheader("Market Risk Breakdown")

    chart_col1, chart_col2 = st.columns(
        [1, 1.6],
        gap="large",
    )

    risk_order = [
        "Low",
        "Moderate",
        "High",
        "Critical",
    ]

    risk_colors = {
        "Low": "#22c55e",
        "Moderate": "#eab308",
        "High": "#f97316",
        "Critical": "#ef4444",
    }

    risk_distribution = (
        df["risk_level"]
        .value_counts()
        .reindex(
            risk_order,
            fill_value=0,
        )
        .rename_axis("Risk Level")
        .reset_index(name="Stocks")
    )

    with chart_col1:
        distribution_figure = px.bar(
            risk_distribution,
            x="Risk Level",
            y="Stocks",
            color="Risk Level",
            color_discrete_map=risk_colors,
            category_orders={
                "Risk Level": risk_order,
            },
            text="Stocks",
        )

        distribution_figure.update_traces(
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Stocks: %{y}<extra></extra>"
            ),
        )

        distribution_figure.update_layout(
            title="Stocks by Risk Level",
            height=390,
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis_title=None,
            yaxis_title="Number of Stocks",
            font=dict(
                color="#cbd5e1",
            ),
        )

        distribution_figure.update_xaxes(
            showgrid=False,
        )

        distribution_figure.update_yaxes(
            gridcolor="rgba(148,163,184,0.12)",
            zeroline=False,
        )

        st.plotly_chart(
            distribution_figure,
            use_container_width=True,
            config={
                "displayModeBar": False,
            },
        )

    top_risk_chart = (
        df[
            [
                "ticker",
                "risk_score",
                "risk_level",
            ]
        ]
        .head(10)
        .copy()
    )

    top_risk_chart["risk_score"] = (
        top_risk_chart["risk_score"] * 100
    )

    top_risk_chart = top_risk_chart.sort_values(
        by="risk_score",
        ascending=True,
    )

    with chart_col2:
        ranking_figure = px.bar(
            top_risk_chart,
            x="risk_score",
            y="ticker",
            orientation="h",
            color="risk_level",
            color_discrete_map=risk_colors,
            text="risk_score",
            labels={
                "risk_score": "Risk Score",
                "ticker": "Ticker",
                "risk_level": "Risk Level",
            },
        )

        ranking_figure.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Risk Score: %{x:.1f}<extra></extra>"
            ),
        )

        ranking_figure.update_layout(
            title="Top 10 Highest-Risk Stocks",
            height=390,
            margin=dict(
                l=20,
                r=35,
                t=55,
                b=20,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis_title="Risk Score",
            yaxis_title=None,
            font=dict(
                color="#cbd5e1",
            ),
        )

        ranking_figure.update_xaxes(
            range=[0, 105],
            gridcolor="rgba(148,163,184,0.12)",
            zeroline=False,
        )

        ranking_figure.update_yaxes(
            showgrid=False,
        )

        st.plotly_chart(
            ranking_figure,
            use_container_width=True,
            config={
                "displayModeBar": False,
            },
        )

    st.subheader("Top Risk Stocks")

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
            lambda value: f"${value:,.2f}"
        )
    )

    top_risk["risk_score"] = (
        top_risk["risk_score"]
        .mul(100)
        .round(1)
    )

    top_risk["risk_percentile"] = (
        top_risk["risk_percentile"]
        .mul(100)
        .map(
            lambda value: f"{value:.1f}%"
        )
    )

    top_risk = top_risk.rename(
        columns={
            "ticker": "Ticker",
            "adj_close": "Price",
            "risk_score": "Risk Score",
            "risk_percentile": "Percentile",
            "risk_level": "Risk Level",
        }
    )

    st.dataframe(
        top_risk,
        use_container_width=True,
        hide_index=True,
        height=560,
    )