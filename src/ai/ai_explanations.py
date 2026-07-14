from __future__ import annotations

import json
import os
from typing import Any

from google import genai


MODEL_NAME = "gemini-3.1-flash-lite"


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Add it to your environment before using Ask RiskAtlas."
        )

    return genai.Client(api_key=api_key)


def build_stock_context(
    ticker: str,
    company_name: str,
    sector: str,
    risk_level: str,
    risk_score: float,
    risk_percentile: float,
    risk_pred: int,
    drivers: list[dict[str, Any]],
    stock_features: dict[str, Any] | None = None,
) -> str:
    """
    Convert RiskAtlas model output into structured Gemini context.
    """

    context = {
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        "risk_level": risk_level,
        "risk_score": round(float(risk_score), 4),
        "risk_percentile": round(float(risk_percentile), 2),
        "production_alert_triggered": bool(risk_pred),
        "risk_drivers": drivers,
        "latest_model_features": stock_features or {},
    }

    return json.dumps(
        context,
        indent=2,
        default=str,
    )


def ask_riskatlas(
    question: str,
    stock_context: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Answer a user's question using only supplied RiskAtlas evidence.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    history = conversation_history or []

    recent_history = history[-10:]

    history_text = "\n".join(
        (
            f"{message.get('role', 'user').upper()}: "
            f"{message.get('content', '')}"
        )
        for message in recent_history
    )

    prompt = f"""
You are Ask RiskAtlas, an AI assistant embedded in a stock-risk dashboard.

Your job is to answer the user's specific question using only the supplied
RiskAtlas model output.

Rules:
- Answer the exact question asked.
- Begin answers naturally, as if speaking to a user.
- Avoid sounding like a database query.
- Prefer conversational explanations over formal reports.
- Do not repeat the entire risk assessment when answering follow-up questions.
- Do not restate the entire stock assessment unless the user requests a summary.
- Use only the RiskAtlas context below.
- Do not invent news, earnings results, analyst opinions, economic events,
  company events, or statistics.
- Clearly say when the available data is insufficient.
- Do not provide personalized investment advice.
- Do not tell the user to buy, sell, or hold.
- Do not describe the risk score as a guaranteed probability of decline.
- Separate risk-increasing and protective signals when relevant.
- Prefer concise answers between 60 and 140 words.
- Use bullets for questions asking about multiple signals.
- Use a short paragraph for conceptual or follow-up questions.
- Do not repeat numbers already stated earlier in the conversation unless
  they are needed to answer the current question.
- Do not add generic investing advice or disclaimers inside every response.
- Refer to the selected stock by ticker or company name.
- Never claim that the listed drivers are the exact mathematical weights used
  by the model. Describe them as explanatory signals derived from model features.

RISKATLAS CONTEXT:
{stock_context}

RECENT CONVERSATION:
{history_text or "No previous conversation."}

USER QUESTION:
{question}
"""

    client = _get_client()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    answer = response.text

    if not answer or not answer.strip():
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return answer.strip()