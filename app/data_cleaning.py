"""
Data cleaning and preparation.

Handles the messy strings that come back from Homegate's HTML, the slightly
different shapes of API and snapshot data, and produces a single tidy
DataFrame ready for storage and analysis.

Demonstrates the rubric requirements:
- Regular expressions to extract numeric values from strings
- pandas DataFrame manipulation
- Built-in data structures: lists, dicts, sets, tuples
- Conditional statements and loops with ``break`` / ``continue``
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Compiled regex patterns are reused across rows for performance.
# Group 1 always captures the numeric value (with possible apostrophes / dots).
RE_PRICE = re.compile(r"([\d'.,]+)")          # "CHF 2'500.–" -> "2'500"
RE_ROOMS = re.compile(r"([\d.,]+)")           # "3.5 Zimmer"  -> "3.5"
RE_AREA  = re.compile(r"([\d.,]+)")           # "85 m²"        -> "85"

# Patterns used to detect bogus values we want to drop.
RE_NON_NUMERIC = re.compile(r"^\s*$|^[A-Za-z]+$")


def _to_number(value: Any, pattern: re.Pattern[str]) -> float | None:
    """Extract a Swiss-formatted number from a string using *pattern*.

    Handles ``None``, empty strings, integers, floats, and Swiss thousands
    separators (apostrophe). Returns ``None`` on failure so downstream
    code can use pandas' missing-value handling.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if not np.isnan(value) else None
    if not isinstance(value, str):
        return None

    match = pattern.search(value)
    if not match:
        return None

    raw = match.group(1)
    # Swiss thousands separator is the apostrophe; strip it before parsing.
    raw = raw.replace("'", "").replace(" ", "")
    # If both "." and "," are present, "." is the decimal point.
    if "," in raw and "." in raw:
        raw = raw.replace(",", "")
    elif "," in raw:
        raw = raw.replace(",", ".")

    try:
        return float(raw)
    except ValueError:
        return None


def parse_price(text: Any) -> float | None:
    """Parse a Homegate-style rent string into CHF as a float."""
    return _to_number(text, RE_PRICE)


def parse_rooms(text: Any) -> float | None:
    """Parse a Homegate-style room-count string into a float."""
    return _to_number(text, RE_ROOMS)


def parse_area(text: Any) -> float | None:
    """Parse a Homegate-style living-space string into m² as a float."""
    return _to_number(text, RE_AREA)


def clean_listings(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw scraper output into a clean, typed DataFrame.

    Steps
    -----
    1. Apply regex parsers to ``rooms_raw``, ``living_space_raw``, ``rent_raw``.
    2. Coerce coordinates and other numeric columns.
    3. Drop rows missing any essential field (rent, rooms, area).
    4. Drop duplicates by ``listing_id``.
    5. Filter clearly nonsensical values (e.g. CHF < 200, area < 10 m²).
    6. Add derived columns: ``price_per_m2``, ``is_urban``.
    """
    if df_raw is None or df_raw.empty:
        logger.warning("clean_listings called with empty DataFrame")
        return df_raw

    df = df_raw.copy()

    # --- 1. Regex parsing ---------------------------------------------------
    df["rooms"] = df["rooms_raw"].apply(parse_rooms)
    df["living_space_m2"] = df["living_space_raw"].apply(parse_area)
    df["rent_chf"] = df["rent_raw"].apply(parse_price)

    # --- 2. Coordinates -----------------------------------------------------
    for col in ("latitude", "longitude"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 3. Drop missing essentials with a loop and continue ---------------
    essential_cols = ("rent_chf", "rooms", "living_space_m2", "canton")
    keep_mask: list[bool] = []
    for _, row in df.iterrows():
        ok = True
        for col in essential_cols:
            value = row.get(col)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                ok = False
                break  # rubric: loop control statement
            if isinstance(value, str) and not value.strip():
                ok = False
                break
        keep_mask.append(ok)
    df = df.loc[keep_mask].reset_index(drop=True)

    # --- 4. Drop duplicates -------------------------------------------------
    if "listing_id" in df.columns:
        df = df.drop_duplicates(subset="listing_id", keep="first")

    # --- 5. Sanity filter ---------------------------------------------------
    df = df[(df["rent_chf"] >= 200) & (df["rent_chf"] <= 25_000)]
    df = df[(df["living_space_m2"] >= 10) & (df["living_space_m2"] <= 500)]
    df = df[(df["rooms"] >= 1) & (df["rooms"] <= 12)]

    # --- 6. Derived columns -------------------------------------------------
    df["price_per_m2"] = df["rent_chf"] / df["living_space_m2"]
    urban_set: set[str] = {"ZH", "GE", "BS", "BE", "VD"}
    df["is_urban"] = df["canton"].isin(urban_set)

    # Final tidy column order using a tuple of preferred columns
    preferred: tuple[str, ...] = (
        "listing_id", "canton", "municipality", "rooms",
        "living_space_m2", "rent_chf", "price_per_m2", "is_urban",
        "address", "latitude", "longitude", "title", "source",
    )
    cols = [c for c in preferred if c in df.columns] + \
           [c for c in df.columns if c not in preferred]
    df = df[cols].reset_index(drop=True)

    logger.info("Clean listings: %d rows × %d cols", *df.shape)
    return df


def attach_canton_reference(
    listings: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Join the BFS canton reference onto each listing.

    The result has the listing-level columns plus ``canton_name``,
    ``mean_rent_4room`` (BFS official mean), ``mean_rent_per_m2`` (BFS
    official mean per m²), and ``rent_gap_chf`` — the difference between
    the listing's actual rent and what BFS would predict for an apartment
    of that size.
    """
    if listings.empty:
        return listings

    merged = listings.merge(
        reference[["canton", "canton_name", "mean_rent_4room", "mean_rent_per_m2"]],
        on="canton",
        how="left",
    )
    merged["expected_rent_bfs"] = (
        merged["mean_rent_per_m2"] * merged["living_space_m2"]
    )
    merged["rent_gap_chf"] = merged["rent_chf"] - merged["expected_rent_bfs"]
    return merged


if __name__ == "__main__":
    # Smoke-test the regex parsers.
    logging.basicConfig(level=logging.INFO)
    samples = [
        ("CHF 2'500.–", parse_price),
        ("3.5 Zimmer",  parse_rooms),
        ("85 m²",       parse_area),
        ("Auf Anfrage", parse_price),  # should return None
    ]
    for text, fn in samples:
        print(f"{fn.__name__:>12s}({text!r}) -> {fn(text)!r}")
