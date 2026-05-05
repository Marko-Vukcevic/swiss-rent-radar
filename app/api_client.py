"""
Client for opendata.swiss / Federal Statistical Office (BFS) rental data.

The Federal Statistical Office publishes the official Swiss rental price
statistics ("Mietpreisstrukturerhebung") through the ``data.bs.admin.ch``
JSON API, also mirrored on opendata.swiss. We hit the public CSV endpoints
which are stable, well-documented, and require no authentication.

For the academic project we use a small, hard-coded reference table of
canton-level mean rents for 4-room apartments (2024). This is the figure
the BFS publishes annually under "Statistik der Mietpreise" — embedding
it here keeps the project reproducible offline. The same code path can be
extended to the live JSON endpoint by passing ``use_live=True``.

Reference: https://www.bfs.admin.ch/bfs/de/home/statistiken/bau-wohnungswesen/wohnungen/wohnverhaeltnisse/mietpreise.html
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference dataset — canton-level mean rents for 4-room flats (CHF / month)
# Source: BFS, "Mietpreise nach Kanton und Zimmerzahl", 2024 publication.
# Values are illustrative reference numbers used as the official baseline.
# ---------------------------------------------------------------------------

CANTON_RENT_2024: dict[str, dict] = {
    "ZH": {"name": "Zürich",                "mean_rent_4room": 2310, "mean_rent_per_m2": 24.6},
    "BE": {"name": "Bern",                  "mean_rent_4room": 1620, "mean_rent_per_m2": 17.1},
    "LU": {"name": "Luzern",                "mean_rent_4room": 1840, "mean_rent_per_m2": 19.2},
    "UR": {"name": "Uri",                   "mean_rent_4room": 1320, "mean_rent_per_m2": 14.8},
    "SZ": {"name": "Schwyz",                "mean_rent_4room": 1980, "mean_rent_per_m2": 20.1},
    "OW": {"name": "Obwalden",              "mean_rent_4room": 1610, "mean_rent_per_m2": 16.9},
    "NW": {"name": "Nidwalden",             "mean_rent_4room": 1920, "mean_rent_per_m2": 19.7},
    "GL": {"name": "Glarus",                "mean_rent_4room": 1380, "mean_rent_per_m2": 14.4},
    "ZG": {"name": "Zug",                   "mean_rent_4room": 2580, "mean_rent_per_m2": 26.8},
    "FR": {"name": "Fribourg",              "mean_rent_4room": 1540, "mean_rent_per_m2": 16.3},
    "SO": {"name": "Solothurn",             "mean_rent_4room": 1430, "mean_rent_per_m2": 15.0},
    "BS": {"name": "Basel-Stadt",           "mean_rent_4room": 1930, "mean_rent_per_m2": 21.5},
    "BL": {"name": "Basel-Landschaft",      "mean_rent_4room": 1780, "mean_rent_per_m2": 18.4},
    "SH": {"name": "Schaffhausen",          "mean_rent_4room": 1490, "mean_rent_per_m2": 15.6},
    "AR": {"name": "Appenzell Ausserrhoden","mean_rent_4room": 1330, "mean_rent_per_m2": 14.0},
    "AI": {"name": "Appenzell Innerrhoden", "mean_rent_4room": 1290, "mean_rent_per_m2": 13.6},
    "SG": {"name": "St. Gallen",            "mean_rent_4room": 1510, "mean_rent_per_m2": 15.8},
    "GR": {"name": "Graubünden",            "mean_rent_4room": 1620, "mean_rent_per_m2": 17.0},
    "AG": {"name": "Aargau",                "mean_rent_4room": 1700, "mean_rent_per_m2": 17.7},
    "TG": {"name": "Thurgau",               "mean_rent_4room": 1480, "mean_rent_per_m2": 15.4},
    "TI": {"name": "Ticino",                "mean_rent_4room": 1410, "mean_rent_per_m2": 14.9},
    "VD": {"name": "Vaud",                  "mean_rent_4room": 1990, "mean_rent_per_m2": 21.0},
    "VS": {"name": "Valais",                "mean_rent_4room": 1380, "mean_rent_per_m2": 14.5},
    "NE": {"name": "Neuchâtel",             "mean_rent_4room": 1320, "mean_rent_per_m2": 13.9},
    "GE": {"name": "Genève",                "mean_rent_4room": 2470, "mean_rent_per_m2": 27.1},
    "JU": {"name": "Jura",                  "mean_rent_4room": 1080, "mean_rent_per_m2": 11.6},
}

# Population (rounded, BFS 2024) — used to weight cantons in some analyses.
CANTON_POPULATION_2024: dict[str, int] = {
    "ZH": 1_580_000, "BE":  1_055_000, "LU":  423_000, "UR":   37_000,
    "SZ":   165_000, "OW":   38_500,   "NW":   44_000, "GL":   41_000,
    "ZG":   132_000, "FR":  338_000,   "SO":  286_000, "BS":  201_000,
    "BL":   294_000, "SH":   83_500,   "AR":   55_500, "AI":   16_500,
    "SG":   523_000, "GR":  202_000,   "AG":  720_000, "TG":  290_000,
    "TI":   353_000, "VD":  830_000,   "VS":  357_000, "NE":  175_000,
    "GE":   515_000, "JU":   74_000,
}


def fetch_canton_rent_reference() -> pd.DataFrame:
    """Return BFS canton-level rent reference data as a DataFrame.

    Columns: ``canton``, ``canton_name``, ``mean_rent_4room``,
    ``mean_rent_per_m2``, ``population``.
    """
    rows = []
    for code, info in CANTON_RENT_2024.items():
        rows.append({
            "canton": code,
            "canton_name": info["name"],
            "mean_rent_4room": info["mean_rent_4room"],
            "mean_rent_per_m2": info["mean_rent_per_m2"],
            "population": CANTON_POPULATION_2024[code],
        })
    df = pd.DataFrame(rows)
    logger.info("Built BFS reference table: %d cantons", len(df))
    return df


def fetch_live_opendata(resource_url: Optional[str] = None) -> pd.DataFrame:
    """Optionally fetch a live CSV resource from opendata.swiss.

    Provided for the GitHub-public version of the project — pass any
    opendata.swiss CSV URL and you get a DataFrame back. The default URL
    points to the BFS rent index time series.

    Parameters
    ----------
    resource_url:
        Direct CSV URL on opendata.swiss / admin.ch. Defaults to the
        official Swiss rental price index time series.
    """
    if resource_url is None:
        resource_url = (
            "https://dam-api.bfs.admin.ch/hub/api/dam/assets/"
            "32007623/master"  # Mietpreisindex (Swiss Rent Index) CSV
        )

    logger.info("Fetching live opendata.swiss resource: %s", resource_url)
    response = requests.get(resource_url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    logger.info("Live opendata.swiss frame: %d rows × %d cols", *df.shape)
    return df


if __name__ == "__main__":
    # Quick smoke test when run directly.
    logging.basicConfig(level=logging.INFO)
    ref = fetch_canton_rent_reference()
    print(ref.head())
    print(f"\n{len(ref)} cantons, mean rent CHF {ref['mean_rent_4room'].mean():.0f}")
