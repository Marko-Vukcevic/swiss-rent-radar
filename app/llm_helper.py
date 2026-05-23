"""
LLM-powered market commentary using Together.ai.

Together.ai exposes an OpenAI-compatible API and offers a free tier that
is plenty for this project. The key is read from the ``TOGETHER_API_KEY``
environment variable; if it is missing the helper returns a clearly
labelled fallback string so the rest of the pipeline still works.

The LLM is fed an aggregated view of the listings (not raw rows), which
keeps prompts cheap and reproducible.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

API_URL = "https://api.together.xyz/v1/chat/completions"
# Together.ai retired Mistral-7B-Instruct-v0.2 from the serverless tier.
# Llama-3.3-70B-Instruct-Turbo is currently serverless on Together.ai.
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
TIMEOUT_SECONDS = 30


def _build_prompt(canton_summary: pd.DataFrame, room_summary: pd.DataFrame) -> str:
    """Turn the aggregated tables into a structured prompt."""
    canton_lines: list[str] = []
    for _, row in canton_summary.iterrows():
        canton_lines.append(
            f"- {row['canton_name']} ({row['canton']}): "
            f"{row['n_listings']} listings, "
            f"avg CHF {row['avg_rent_chf']:,.0f}/month, "
            f"CHF {row['avg_price_per_m2']:.2f}/m²"
        )
    room_lines: list[str] = []
    for _, row in room_summary.iterrows():
        room_lines.append(
            f"- {row['rooms']} rooms: {row['n_listings']} listings, "
            f"avg CHF {row['avg_rent_chf']:,.0f}, "
            f"CHF {row['avg_price_per_m2']:.2f}/m²"
        )

    prompt = (
        "You are an analyst writing a short, neutral market commentary on "
        "the current Swiss rental market based on the data below. "
        "Do not invent figures. Be concrete and brief.\n\n"
        "AVERAGE RENTS BY CANTON\n"
        + "\n".join(canton_lines)
        + "\n\nAVERAGE RENTS BY ROOM COUNT\n"
        + "\n".join(room_lines)
        + "\n\nWrite 4–6 sentences highlighting (1) the most expensive and "
          "cheapest cantons by price per m², (2) the relationship between "
          "rooms and rent, (3) any notable urban-vs-rural gap. End with one "
          "sentence on what a renter should take away."
    )
    return prompt


def market_commentary(
    canton_summary: pd.DataFrame,
    room_summary: pd.DataFrame,
    api_key: Optional[str] = None,
) -> str:
    """Return a short LLM-generated market commentary or a fallback."""
    api_key = api_key or os.getenv("TOGETHER_API_KEY")
    if not api_key:
        logger.warning("No TOGETHER_API_KEY set — returning offline summary.")
        return _offline_fallback(canton_summary, room_summary)

    prompt = _build_prompt(canton_summary, room_summary)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 350,
        "temperature": 0.4,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload,
                                 timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        logger.error("LLM network error: %s", exc)
        return _offline_fallback(canton_summary, room_summary)

    if not response.ok:
        # Surface the upstream error so misconfigurations (wrong model,
        # missing credits, bad key) are visible instead of silently
        # returning the offline summary.
        try:
            err = response.json().get("error", {}).get("message", response.text)
        except ValueError:
            err = response.text
        logger.error("Together.ai %s: %s", response.status_code, err)
        return (
            f"[LLM call failed: HTTP {response.status_code} — {err}]\n\n"
            + _offline_fallback(canton_summary, room_summary)
        )

    try:
        return response.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError) as exc:
        logger.error("Unexpected LLM response shape: %s", exc)
        return _offline_fallback(canton_summary, room_summary)


def _offline_fallback(
    canton_summary: pd.DataFrame,
    room_summary: pd.DataFrame,
) -> str:
    """Produce a deterministic summary when the LLM is unavailable.

    Useful for the project video — guarantees reproducible output even
    if the lecturer runs the code without an API key.
    """
    if canton_summary.empty or room_summary.empty:
        return "No data available for commentary."

    most_exp = canton_summary.iloc[0]
    least_exp = canton_summary.iloc[-1]
    smallest = room_summary["rooms"].min()
    largest = room_summary["rooms"].max()

    return (
        "[Offline summary — set TOGETHER_API_KEY for an LLM-generated version.]\n\n"
        f"The most expensive canton by price per m² is "
        f"{most_exp['canton_name']} at CHF {most_exp['avg_price_per_m2']:.2f}/m², "
        f"while the cheapest is {least_exp['canton_name']} at "
        f"CHF {least_exp['avg_price_per_m2']:.2f}/m². "
        f"Listings span from {smallest:.1f}-room to {largest:.1f}-room "
        "apartments, with rent rising broadly in line with size. "
        "Urban cantons (ZH, GE, BS, BE, VD) command a clear premium per m² "
        "compared to rural cantons. Renters should compare the listing price "
        "against the BFS canton baseline to spot above-market asking rents."
    )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    demo_canton = pd.DataFrame({
        "canton": ["ZH", "JU"],
        "canton_name": ["Zürich", "Jura"],
        "n_listings": [50, 10],
        "avg_rent_chf": [2300, 1100],
        "avg_price_per_m2": [25.0, 12.0],
    })
    demo_rooms = pd.DataFrame({
        "rooms": [2.0, 4.0],
        "n_listings": [40, 30],
        "avg_rent_chf": [1500, 2400],
        "avg_price_per_m2": [22.0, 18.0],
    })
    print(market_commentary(demo_canton, demo_rooms))
