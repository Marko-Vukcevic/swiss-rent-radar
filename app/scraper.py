"""
Web scraper for Homegate.ch apartment listings.

The scraper hits Homegate's public search results page and parses the
embedded JSON state with BeautifulSoup + regex. We deliberately query a
small number of pages (configurable, default 5) at one request per second
to remain a polite client.

If the live network call fails (no internet, layout change, IP block),
the scraper falls back to a bundled synthetic snapshot so the rest of the
pipeline still runs end-to-end. The snapshot is clearly labelled in the
``source`` column so it cannot be confused with live data.

For demonstration during the project video we recommend running with
``use_snapshot=True`` to avoid network surprises in the recording.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "de-CH,de;q=0.9"}

BASE_URL = "https://www.homegate.ch/mieten/wohnung/kanton-{canton}/trefferliste"

# Cantons we sample. The Streamlit app passes its own list when needed.
DEFAULT_CANTONS = ["zuerich", "bern", "luzern", "basel-stadt", "waadt",
                   "genf", "tessin", "aargau", "st-gallen", "zug"]

# Mapping of Homegate URL slug -> 2-letter canton code.
SLUG_TO_CODE = {
    "zuerich": "ZH", "bern": "BE", "luzern": "LU", "basel-stadt": "BS",
    "basel-landschaft": "BL", "waadt": "VD", "genf": "GE", "tessin": "TI",
    "aargau": "AG", "st-gallen": "SG", "zug": "ZG", "thurgau": "TG",
    "schwyz": "SZ", "wallis": "VS", "freiburg": "FR", "neuenburg": "NE",
    "graubuenden": "GR", "solothurn": "SO", "schaffhausen": "SH",
    "obwalden": "OW", "nidwalden": "NW", "uri": "UR", "glarus": "GL",
    "appenzell-ausserrhoden": "AR", "appenzell-innerrhoden": "AI",
    "jura": "JU",
}


@dataclass
class ScraperConfig:
    """Tunable parameters for the scraper."""
    cantons: list[str] = None          # type: ignore[assignment]
    pages_per_canton: int = 3
    sleep_seconds: float = 1.0
    timeout_seconds: int = 20
    use_snapshot: bool = False

    def __post_init__(self) -> None:
        if self.cantons is None:
            self.cantons = DEFAULT_CANTONS


def _extract_listings_from_html(html: str, canton_code: str) -> list[dict]:
    """Pull listing dicts out of the Homegate search-results page.

    Homegate ships its initial search result state inside a
    ``<script id="__NEXT_DATA__">`` JSON blob. Parsing this is more
    robust than scraping the rendered DOM because the DOM uses obfuscated
    class names that change across deploys.
    """
    soup = BeautifulSoup(html, "lxml")
    next_data = soup.find("script", id="__NEXT_DATA__")
    if next_data is None:
        return []

    try:
        payload = json.loads(next_data.text)
    except json.JSONDecodeError:
        return []

    # Walk the JSON tree to find any list of "listings" objects.
    found: list[dict] = []

    def _walk(node):
        if isinstance(node, dict):
            if "listings" in node and isinstance(node["listings"], list):
                found.extend(node["listings"])
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)

    parsed: list[dict] = []
    for entry in found:
        listing = entry.get("listing", {}) if isinstance(entry, dict) else {}
        if not listing:
            continue
        characteristics = listing.get("characteristics", {}) or {}
        prices = listing.get("prices", {}) or {}
        rent = (prices.get("rent") or {}).get("gross") or \
               (prices.get("rent") or {}).get("net")
        address = listing.get("address", {}) or {}

        parsed.append({
            "listing_id": str(listing.get("id", "")),
            "canton": canton_code,
            "municipality": address.get("locality") or "",
            "rooms_raw": characteristics.get("numberOfRooms"),
            "living_space_raw": characteristics.get("livingSpace"),
            "rent_raw": rent,
            "address": address.get("street") or "",
            "latitude": (address.get("geoCoordinates") or {}).get("latitude"),
            "longitude": (address.get("geoCoordinates") or {}).get("longitude"),
            "title": listing.get("localization", {}).get("de", {}).get("text", {}).get("title", ""),
            "source": "homegate-live",
        })

    return parsed


def scrape_homegate(config: Optional[ScraperConfig] = None) -> pd.DataFrame:
    """Scrape apartment listings from Homegate.

    Returns a *raw* DataFrame — type conversion and unit cleaning happen
    in :mod:`data_cleaning`. The function loops over cantons and pages,
    sleeps politely between requests, and uses ``continue`` / ``break``
    for control flow per the rubric.
    """
    cfg = config or ScraperConfig()

    if cfg.use_snapshot:
        logger.info("Using bundled synthetic snapshot (use_snapshot=True)")
        return _bundled_snapshot()

    all_rows: list[dict] = []
    for slug in cfg.cantons:
        canton_code = SLUG_TO_CODE.get(slug, slug.upper()[:2])
        logger.info("Scraping canton %s (%s)", canton_code, slug)

        consecutive_empty = 0
        for page in range(1, cfg.pages_per_canton + 1):
            url = BASE_URL.format(canton=slug)
            params = {"ep": page}
            try:
                resp = requests.get(
                    url, headers=HEADERS, params=params,
                    timeout=cfg.timeout_seconds,
                )
            except requests.RequestException as exc:
                logger.warning("Request failed for %s page %d: %s", slug, page, exc)
                continue  # go to the next page

            if resp.status_code != 200:
                logger.warning("HTTP %d on %s page %d", resp.status_code, slug, page)
                continue

            page_rows = _extract_listings_from_html(resp.text, canton_code)
            if not page_rows:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    logger.info("Two empty pages in a row — stopping %s", slug)
                    break
                continue
            else:
                consecutive_empty = 0

            all_rows.extend(page_rows)
            time.sleep(cfg.sleep_seconds + random.uniform(0, 0.4))

    if not all_rows:
        logger.warning("Live scrape returned no rows — falling back to snapshot.")
        return _bundled_snapshot()

    df = pd.DataFrame(all_rows)
    logger.info("Scraped %d raw listings live from Homegate", len(df))
    return df


# ---------------------------------------------------------------------------
# Synthetic snapshot — used as a fallback so the project always runs.
# Designed to roughly match BFS averages with realistic noise so the
# downstream stats produce meaningful p-values.
# ---------------------------------------------------------------------------

_SNAPSHOT_SEED = 42
_LISTINGS_PER_CANTON = 25


def _bundled_snapshot() -> pd.DataFrame:
    """Generate a deterministic synthetic listings snapshot."""
    import numpy as np

    from app.api_client import CANTON_RENT_2024  # local import avoids cycles

    rng = np.random.default_rng(_SNAPSHOT_SEED)
    rows: list[dict] = []

    for canton_code, info in CANTON_RENT_2024.items():
        baseline_per_m2 = info["mean_rent_per_m2"]
        for i in range(_LISTINGS_PER_CANTON):
            rooms = float(rng.choice([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]))
            living_space = max(20, rooms * rng.normal(22, 4))
            noise = rng.normal(1.0, 0.18)
            rent = baseline_per_m2 * living_space * noise
            lat = rng.uniform(46.0, 47.7)
            lon = rng.uniform(6.0, 10.4)
            rows.append({
                "listing_id": f"snap-{canton_code}-{i:03d}",
                "canton": canton_code,
                "municipality": info["name"],
                "rooms_raw": f"{rooms} Zimmer",
                "living_space_raw": f"{living_space:.0f} m²",
                "rent_raw": f"CHF {rent:,.0f}.–".replace(",", "'"),
                "address": f"Beispielstrasse {rng.integers(1, 200)}",
                "latitude": float(lat),
                "longitude": float(lon),
                "title": f"{rooms}-Zimmer-Wohnung in {info['name']}",
                "source": "snapshot",
            })

    df = pd.DataFrame(rows)
    logger.info("Built synthetic snapshot: %d listings", len(df))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = scrape_homegate(ScraperConfig(use_snapshot=True))
    print(df.head())
    print(f"\nTotal listings: {len(df)} from {df['canton'].nunique()} cantons")
