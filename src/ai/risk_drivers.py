from __future__ import annotations

from typing import Any

import pandas as pd


def _to_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add_driver(
    drivers: list[dict[str, Any]],
    *,
    category: str,
    title: str,
    explanation: str,
    direction: str,
    severity: int,
    feature: str,
    value: float,
) -> None:
    drivers.append(
        {
            "category": category,
            "title": title,
            "explanation": explanation,
            "direction": direction,
            "severity": severity,
            "feature": feature,
            "value": value,
        }
    )


def generate_risk_drivers(
    stock_features: pd.Series | dict[str, Any],
    max_drivers: int = 5,
) -> list[dict[str, Any]]:
    """
    Generate rule-based explanations for a stock's current risk profile.

    Parameters
    ----------
    stock_features:
        A pandas Series or dictionary containing the latest row from
        inference_dataset_v3.

    max_drivers:
        Maximum number of explanations returned.

    Returns
    -------
    list[dict[str, Any]]
        Sorted explanations, with the strongest risk drivers first.
    """

    if isinstance(stock_features, pd.Series):
        features = stock_features.to_dict()
    else:
        features = dict(stock_features)

    drivers: list[dict[str, Any]] = []

    ticker = str(
        features.get("ticker", "This stock")
    ).strip().upper()

    return_20d = _to_float(
        features.get("return_20d")
    )
    return_60d = _to_float(
        features.get("return_60d")
    )
    vol_20 = _to_float(
        features.get("vol_20")
    )
    vol_60 = _to_float(
        features.get("vol_60")
    )
    downside_vol_20 = _to_float(
        features.get("downside_vol_20")
    )
    worst_return_20 = _to_float(
        features.get("worst_return_20")
    )
    price_to_ma200 = _to_float(
        features.get("price_to_ma200")
    )
    drawdown_60d = _to_float(
        features.get("drawdown_from_60d_high")
    )
    distance_52w_high = _to_float(
        features.get("distance_from_52w_high")
    )

    spy_return_20d = _to_float(
        features.get("spy_return_20d")
    )
    spy_return_60d = _to_float(
        features.get("spy_return_60d")
    )
    spy_vol_20 = _to_float(
        features.get("spy_vol_20")
    )
    spy_vol_60 = _to_float(
        features.get("spy_vol_60")
    )
    spy_drawdown_60d = _to_float(
        features.get(
            "spy_drawdown_from_60d_high"
        )
    )

    pct_positive_20d = _to_float(
        features.get("pct_positive_20d")
    )
    pct_above_ma50 = _to_float(
        features.get("pct_above_ma50")
    )
    pct_above_ma200 = _to_float(
        features.get("pct_above_ma200")
    )

    return_20d_vs_sector = _to_float(
        features.get("return_20d_vs_sector")
    )
    return_60d_vs_sector = _to_float(
        features.get("return_60d_vs_sector")
    )
    vol_20_vs_sector = _to_float(
        features.get("vol_20_vs_sector")
    )

    beta_60 = _to_float(
        features.get("beta_60")
    )
    correlation_60 = _to_float(
        features.get("correlation_60")
    )

    return_percentile = _to_float(
        features.get("return_20d_percentile")
    )
    vol_percentile = _to_float(
        features.get("vol_20_percentile")
    )
    drawdown_percentile = _to_float(
        features.get("drawdown_60d_percentile")
    )
    sector_return_percentile = _to_float(
        features.get(
            "sector_return_20d_percentile"
        )
    )
    sector_vol_percentile = _to_float(
        features.get(
            "sector_vol_20_percentile"
        )
    )

    if vol_percentile is not None:
        if vol_percentile >= 0.90:
            _add_driver(
                drivers,
                category="Volatility",
                title="Extreme short-term volatility",
                explanation=(
                    f"{ticker}'s 20-day volatility is higher "
                    "than at least 90% of the stock universe."
                ),
                direction="risk",
                severity=5,
                feature="vol_20_percentile",
                value=vol_percentile,
            )

        elif vol_percentile >= 0.75:
            _add_driver(
                drivers,
                category="Volatility",
                title="Elevated short-term volatility",
                explanation=(
                    f"{ticker}'s recent volatility is elevated "
                    "relative to most stocks."
                ),
                direction="risk",
                severity=4,
                feature="vol_20_percentile",
                value=vol_percentile,
            )

    if downside_vol_20 is not None:
        if downside_vol_20 >= 0.40:
            _add_driver(
                drivers,
                category="Volatility",
                title="High downside volatility",
                explanation=(
                    f"{ticker} has experienced large and unstable "
                    "negative price movements over the past 20 days."
                ),
                direction="risk",
                severity=5,
                feature="downside_vol_20",
                value=downside_vol_20,
            )

        elif downside_vol_20 >= 0.25:
            _add_driver(
                drivers,
                category="Volatility",
                title="Elevated downside volatility",
                explanation=(
                    f"{ticker}'s negative returns have been more "
                    "volatile than normal recently."
                ),
                direction="risk",
                severity=4,
                feature="downside_vol_20",
                value=downside_vol_20,
            )

    if worst_return_20 is not None:
        if worst_return_20 <= -0.10:
            _add_driver(
                drivers,
                category="Tail Risk",
                title="Severe recent daily loss",
                explanation=(
                    f"{ticker}'s worst single-day return during the "
                    f"past 20 trading days was {worst_return_20:.1%}."
                ),
                direction="risk",
                severity=5,
                feature="worst_return_20",
                value=worst_return_20,
            )

        elif worst_return_20 <= -0.06:
            _add_driver(
                drivers,
                category="Tail Risk",
                title="Large recent daily loss",
                explanation=(
                    f"{ticker} recently experienced a one-day decline "
                    f"of {abs(worst_return_20):.1%}."
                ),
                direction="risk",
                severity=4,
                feature="worst_return_20",
                value=worst_return_20,
            )

    if drawdown_percentile is not None:
        if drawdown_percentile >= 0.90:
            _add_driver(
                drivers,
                category="Drawdown",
                title="Extreme drawdown pressure",
                explanation=(
                    f"{ticker}'s decline from its 60-day high ranks "
                    "among the most severe in the stock universe."
                ),
                direction="risk",
                severity=5,
                feature="drawdown_60d_percentile",
                value=drawdown_percentile,
            )

        elif drawdown_percentile >= 0.75:
            _add_driver(
                drivers,
                category="Drawdown",
                title="Elevated drawdown pressure",
                explanation=(
                    f"{ticker} is trading meaningfully below its "
                    "recent high."
                ),
                direction="risk",
                severity=4,
                feature="drawdown_60d_percentile",
                value=drawdown_percentile,
            )

    if drawdown_60d is not None:
        if drawdown_60d <= -0.20:
            _add_driver(
                drivers,
                category="Drawdown",
                title="Deep 60-day drawdown",
                explanation=(
                    f"{ticker} is {abs(drawdown_60d):.1%} below its "
                    "60-day high, indicating significant price damage."
                ),
                direction="risk",
                severity=5,
                feature="drawdown_from_60d_high",
                value=drawdown_60d,
            )

        elif drawdown_60d <= -0.10:
            _add_driver(
                drivers,
                category="Drawdown",
                title="Material 60-day drawdown",
                explanation=(
                    f"{ticker} is {abs(drawdown_60d):.1%} below its "
                    "recent 60-day high."
                ),
                direction="risk",
                severity=4,
                feature="drawdown_from_60d_high",
                value=drawdown_60d,
            )

    if distance_52w_high is not None:
        if distance_52w_high <= -0.30:
            _add_driver(
                drivers,
                category="Trend",
                title="Far below 52-week high",
                explanation=(
                    f"{ticker} is {abs(distance_52w_high):.1%} below "
                    "its 52-week high, reflecting prolonged weakness."
                ),
                direction="risk",
                severity=4,
                feature="distance_from_52w_high",
                value=distance_52w_high,
            )

    if price_to_ma200 is not None:
        if price_to_ma200 <= 0.80:
            _add_driver(
                drivers,
                category="Trend",
                title="Severe long-term trend weakness",
                explanation=(
                    f"{ticker} is trading approximately "
                    f"{(1 - price_to_ma200):.1%} below its 200-day "
                    "moving average."
                ),
                direction="risk",
                severity=5,
                feature="price_to_ma200",
                value=price_to_ma200,
            )

        elif price_to_ma200 <= 0.95:
            _add_driver(
                drivers,
                category="Trend",
                title="Below long-term trend",
                explanation=(
                    f"{ticker} is trading below its 200-day moving "
                    "average, signaling weaker long-term momentum."
                ),
                direction="risk",
                severity=4,
                feature="price_to_ma200",
                value=price_to_ma200,
            )

        elif price_to_ma200 >= 1.10:
            _add_driver(
                drivers,
                category="Trend",
                title="Strong long-term price trend",
                explanation=(
                    f"{ticker} is trading approximately "
                    f"{(price_to_ma200 - 1):.1%} above its 200-day "
                    "moving average."
                ),
                direction="protective",
                severity=2,
                feature="price_to_ma200",
                value=price_to_ma200,
            )

    if return_20d is not None:
        if return_20d <= -0.15:
            _add_driver(
                drivers,
                category="Momentum",
                title="Sharp short-term decline",
                explanation=(
                    f"{ticker} has fallen {abs(return_20d):.1%} over "
                    "the past 20 trading days."
                ),
                direction="risk",
                severity=5,
                feature="return_20d",
                value=return_20d,
            )

        elif return_20d <= -0.07:
            _add_driver(
                drivers,
                category="Momentum",
                title="Negative short-term momentum",
                explanation=(
                    f"{ticker} has declined {abs(return_20d):.1%} "
                    "during the past 20 trading days."
                ),
                direction="risk",
                severity=4,
                feature="return_20d",
                value=return_20d,
            )

        elif return_20d >= 0.10:
            _add_driver(
                drivers,
                category="Momentum",
                title="Strong short-term momentum",
                explanation=(
                    f"{ticker} has gained {return_20d:.1%} over the "
                    "past 20 trading days."
                ),
                direction="protective",
                severity=2,
                feature="return_20d",
                value=return_20d,
            )

    if return_60d is not None:
        if return_60d <= -0.20:
            _add_driver(
                drivers,
                category="Momentum",
                title="Persistent medium-term weakness",
                explanation=(
                    f"{ticker} has declined {abs(return_60d):.1%} "
                    "over the past 60 trading days."
                ),
                direction="risk",
                severity=5,
                feature="return_60d",
                value=return_60d,
            )

        elif return_60d <= -0.10:
            _add_driver(
                drivers,
                category="Momentum",
                title="Weak 60-day performance",
                explanation=(
                    f"{ticker}'s medium-term return remains negative "
                    f"at {return_60d:.1%}."
                ),
                direction="risk",
                severity=4,
                feature="return_60d",
                value=return_60d,
            )

    if return_percentile is not None:
        if return_percentile <= 0.10:
            _add_driver(
                drivers,
                category="Relative Performance",
                title="Bottom-decile recent performance",
                explanation=(
                    f"{ticker}'s 20-day return ranks in the bottom "
                    "10% of the stock universe."
                ),
                direction="risk",
                severity=5,
                feature="return_20d_percentile",
                value=return_percentile,
            )

        elif return_percentile <= 0.25:
            _add_driver(
                drivers,
                category="Relative Performance",
                title="Weak relative performance",
                explanation=(
                    f"{ticker}'s recent return trails at least 75% "
                    "of the stock universe."
                ),
                direction="risk",
                severity=4,
                feature="return_20d_percentile",
                value=return_percentile,
            )

    if return_20d_vs_sector is not None:
        if return_20d_vs_sector <= -0.10:
            _add_driver(
                drivers,
                category="Sector",
                title="Severe sector underperformance",
                explanation=(
                    f"{ticker} has underperformed its sector by "
                    f"{abs(return_20d_vs_sector):.1%} over 20 days."
                ),
                direction="risk",
                severity=5,
                feature="return_20d_vs_sector",
                value=return_20d_vs_sector,
            )

        elif return_20d_vs_sector <= -0.05:
            _add_driver(
                drivers,
                category="Sector",
                title="Recent sector underperformance",
                explanation=(
                    f"{ticker} has lagged its sector by "
                    f"{abs(return_20d_vs_sector):.1%} over 20 days."
                ),
                direction="risk",
                severity=4,
                feature="return_20d_vs_sector",
                value=return_20d_vs_sector,
            )

    if return_60d_vs_sector <= -0.15:
        if return_60d is not None and return_60d > 0:
            title = "Lagging a strong sector"
            explanation = (
                f"{ticker} gained {return_60d:.1%} over 60 days, "
                f"but trailed its sector benchmark by "
                f"{abs(return_60d_vs_sector):.1%}."
            )
        else:
            title = "Persistent sector underperformance"
            explanation = (
                f"{ticker} underperformed its sector benchmark by "
                f"{abs(return_60d_vs_sector):.1%} over 60 days."
            )

    _add_driver(
        drivers,
        category="Sector",
        title=title,
        explanation=explanation,
        direction="risk",
        severity=5,
        feature="return_60d_vs_sector",
        value=return_60d_vs_sector,
    )

    if vol_20_vs_sector is not None:
        if vol_20_vs_sector >= 1.50:
            _add_driver(
                drivers,
                category="Sector",
                title="Much more volatile than sector",
                explanation=(
                    f"{ticker}'s volatility is approximately "
                    f"{vol_20_vs_sector:.1f} times its sector level."
                ),
                direction="risk",
                severity=5,
                feature="vol_20_vs_sector",
                value=vol_20_vs_sector,
            )

        elif vol_20_vs_sector >= 1.20:
            _add_driver(
                drivers,
                category="Sector",
                title="More volatile than sector",
                explanation=(
                    f"{ticker} is displaying greater volatility "
                    "than its sector peers."
                ),
                direction="risk",
                severity=4,
                feature="vol_20_vs_sector",
                value=vol_20_vs_sector,
            )

    if sector_return_percentile is not None:
        if sector_return_percentile <= 0.15:
            _add_driver(
                drivers,
                category="Sector",
                title="Weak sector-relative return",
                explanation=(
                    f"{ticker}'s recent performance ranks near the "
                    "bottom of its sector."
                ),
                direction="risk",
                severity=4,
                feature="sector_return_20d_percentile",
                value=sector_return_percentile,
            )

    if sector_vol_percentile is not None:
        if sector_vol_percentile >= 0.85:
            _add_driver(
                drivers,
                category="Sector",
                title="High sector-relative volatility",
                explanation=(
                    f"{ticker}'s volatility ranks near the top of "
                    "its sector."
                ),
                direction="risk",
                severity=4,
                feature="sector_vol_20_percentile",
                value=sector_vol_percentile,
            )

    if spy_vol_20 is not None:
        if spy_vol_20 >= 0.30:
            _add_driver(
                drivers,
                category="Market Regime",
                title="Highly volatile market environment",
                explanation=(
                    "Broad-market volatility is elevated, increasing "
                    "the probability of sharp stock-level losses."
                ),
                direction="risk",
                severity=5,
                feature="spy_vol_20",
                value=spy_vol_20,
            )

        elif spy_vol_20 >= 0.20:
            _add_driver(
                drivers,
                category="Market Regime",
                title="Elevated market volatility",
                explanation=(
                    "The broader market is experiencing elevated "
                    "short-term volatility."
                ),
                direction="risk",
                severity=4,
                feature="spy_vol_20",
                value=spy_vol_20,
            )

    if spy_vol_60 is not None and spy_vol_20 is not None:
        if spy_vol_20 >= spy_vol_60 * 1.25:
            _add_driver(
                drivers,
                category="Market Regime",
                title="Market volatility is accelerating",
                explanation=(
                    "Short-term market volatility is substantially "
                    "above its 60-day level."
                ),
                direction="risk",
                severity=4,
                feature="spy_vol_20",
                value=spy_vol_20,
            )

    if spy_return_20d is not None:
        if spy_return_20d <= -0.08:
            _add_driver(
                drivers,
                category="Market Regime",
                title="Broad-market selloff",
                explanation=(
                    f"The S&P 500 has declined "
                    f"{abs(spy_return_20d):.1%} over 20 trading days."
                ),
                direction="risk",
                severity=5,
                feature="spy_return_20d",
                value=spy_return_20d,
            )

        elif spy_return_20d <= -0.04:
            _add_driver(
                drivers,
                category="Market Regime",
                title="Weak broad-market momentum",
                explanation=(
                    "The broader market has declined recently, "
                    "creating a less supportive environment."
                ),
                direction="risk",
                severity=4,
                feature="spy_return_20d",
                value=spy_return_20d,
            )

    if spy_return_60d is not None:
        if spy_return_60d <= -0.10:
            _add_driver(
                drivers,
                category="Market Regime",
                title="Persistent market weakness",
                explanation=(
                    f"The S&P 500 has declined "
                    f"{abs(spy_return_60d):.1%} over 60 trading days."
                ),
                direction="risk",
                severity=5,
                feature="spy_return_60d",
                value=spy_return_60d,
            )

    if spy_drawdown_60d is not None:
        if spy_drawdown_60d <= -0.10:
            _add_driver(
                drivers,
                category="Market Regime",
                title="Broad-market drawdown",
                explanation=(
                    f"The S&P 500 is "
                    f"{abs(spy_drawdown_60d):.1%} below its 60-day "
                    "high."
                ),
                direction="risk",
                severity=4,
                feature="spy_drawdown_from_60d_high",
                value=spy_drawdown_60d,
            )

    if pct_above_ma200 is not None:
        if pct_above_ma200 <= 0.30:
            _add_driver(
                drivers,
                category="Market Breadth",
                title="Severely weak market breadth",
                explanation=(
                    "Fewer than 30% of stocks are trading above their "
                    "200-day moving averages."
                ),
                direction="risk",
                severity=5,
                feature="pct_above_ma200",
                value=pct_above_ma200,
            )

        elif pct_above_ma200 <= 0.45:
            _add_driver(
                drivers,
                category="Market Breadth",
                title="Weak long-term market breadth",
                explanation=(
                    "A minority of stocks are trading above their "
                    "200-day moving averages."
                ),
                direction="risk",
                severity=4,
                feature="pct_above_ma200",
                value=pct_above_ma200,
            )

        elif pct_above_ma200 >= 0.70:
            _add_driver(
                drivers,
                category="Market Breadth",
                title="Strong long-term market breadth",
                explanation=(
                    "Most stocks are trading above their 200-day "
                    "moving averages."
                ),
                direction="protective",
                severity=2,
                feature="pct_above_ma200",
                value=pct_above_ma200,
            )

    if pct_above_ma50 is not None:
        if pct_above_ma50 <= 0.30:
            _add_driver(
                drivers,
                category="Market Breadth",
                title="Weak short-term market breadth",
                explanation=(
                    "Fewer than 30% of stocks are trading above their "
                    "50-day moving averages."
                ),
                direction="risk",
                severity=4,
                feature="pct_above_ma50",
                value=pct_above_ma50,
            )

    if pct_positive_20d is not None:
        if pct_positive_20d <= 0.30:
            _add_driver(
                drivers,
                category="Market Breadth",
                title="Few stocks have positive momentum",
                explanation=(
                    "Less than 30% of stocks have produced positive "
                    "20-day returns."
                ),
                direction="risk",
                severity=4,
                feature="pct_positive_20d",
                value=pct_positive_20d,
            )

    if beta_60 is not None:
        if beta_60 >= 1.60:
            _add_driver(
                drivers,
                category="Market Sensitivity",
                title="Very high market sensitivity",
                explanation=(
                    f"{ticker}'s 60-day beta is {beta_60:.2f}, "
                    "indicating amplified moves relative to the market."
                ),
                direction="risk",
                severity=4,
                feature="beta_60",
                value=beta_60,
            )

        elif beta_60 >= 1.30:
            _add_driver(
                drivers,
                category="Market Sensitivity",
                title="High market sensitivity",
                explanation=(
                    f"{ticker}'s 60-day beta is {beta_60:.2f}, making "
                    "it more sensitive to broad-market movements."
                ),
                direction="risk",
                severity=3,
                feature="beta_60",
                value=beta_60,
            )

        elif beta_60 <= 0.70:
            _add_driver(
                drivers,
                category="Market Sensitivity",
                title="Lower market sensitivity",
                explanation=(
                    f"{ticker}'s 60-day beta is {beta_60:.2f}, "
                    "indicating less sensitivity to market swings."
                ),
                direction="protective",
                severity=2,
                feature="beta_60",
                value=beta_60,
            )

    if correlation_60 is not None:
        if correlation_60 >= 0.80 and spy_return_20d is not None:
            if spy_return_20d < 0:
                _add_driver(
                    drivers,
                    category="Market Sensitivity",
                    title="Strong exposure to a falling market",
                    explanation=(
                        f"{ticker} is highly correlated with the "
                        "broader market while market momentum is weak."
                    ),
                    direction="risk",
                    severity=4,
                    feature="correlation_60",
                    value=correlation_60,
                )

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

    risk_drivers = sorted(
        risk_drivers,
        key=lambda driver: driver["severity"],
        reverse=True,
    )

    protective_drivers = sorted(
        protective_drivers,
        key=lambda driver: driver["severity"],
        reverse=True,
    )

    selected_drivers = risk_drivers[:max_drivers]

    if len(selected_drivers) < max_drivers:
        remaining_slots = (
            max_drivers - len(selected_drivers)
        )

        selected_drivers.extend(
            protective_drivers[:remaining_slots]
        )

    return selected_drivers