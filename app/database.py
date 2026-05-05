"""
SQLite persistence layer.

The database file ``data/swiss_rent.db`` holds two tables:

- ``listings``       — one row per scraped apartment listing
- ``canton_stats``   — one row per canton, BFS reference statistics

The module exposes simple write helpers and a small library of
analytical SQL queries used by the Streamlit app and notebooks.

Demonstrates the rubric requirements:
- Use of a database (SQLite)
- SQL queries embedded in Python (``GROUP BY``, ``JOIN``, aggregations)
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "swiss_rent.db"


@contextmanager
def connect(db_path: Path | str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with row factory and FK pragma set."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Create the database schema if it does not already exist."""
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS canton_stats (
                canton TEXT PRIMARY KEY,
                canton_name TEXT NOT NULL,
                mean_rent_4room REAL,
                mean_rent_per_m2 REAL,
                population INTEGER
            );

            CREATE TABLE IF NOT EXISTS listings (
                listing_id TEXT PRIMARY KEY,
                canton TEXT NOT NULL,
                municipality TEXT,
                rooms REAL,
                living_space_m2 REAL,
                rent_chf REAL,
                price_per_m2 REAL,
                is_urban INTEGER,
                address TEXT,
                latitude REAL,
                longitude REAL,
                title TEXT,
                source TEXT,
                FOREIGN KEY (canton) REFERENCES canton_stats(canton)
            );

            CREATE INDEX IF NOT EXISTS idx_listings_canton  ON listings(canton);
            CREATE INDEX IF NOT EXISTS idx_listings_rooms   ON listings(rooms);
            """
        )
    logger.info("Database schema initialised at %s", db_path)


def write_canton_stats(df: pd.DataFrame, db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Replace the ``canton_stats`` table with the supplied DataFrame."""
    cols = ["canton", "canton_name", "mean_rent_4room", "mean_rent_per_m2", "population"]
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"canton_stats DataFrame missing columns: {missing}")

    with connect(db_path) as conn:
        df[cols].to_sql("canton_stats", conn, if_exists="replace", index=False)
    logger.info("Wrote %d canton rows", len(df))


def write_listings(df: pd.DataFrame, db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Replace the ``listings`` table with the supplied DataFrame."""
    cols = [
        "listing_id", "canton", "municipality", "rooms", "living_space_m2",
        "rent_chf", "price_per_m2", "is_urban", "address",
        "latitude", "longitude", "title", "source",
    ]
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"listings DataFrame missing columns: {missing}")

    df_to_write = df[cols].copy()
    df_to_write["is_urban"] = df_to_write["is_urban"].astype(int)

    with connect(db_path) as conn:
        df_to_write.to_sql("listings", conn, if_exists="replace", index=False)
    logger.info("Wrote %d listings to SQLite", len(df_to_write))


# ---------------------------------------------------------------------------
# Analytical SQL queries — used by the Streamlit app and the notebooks.
# Keeping them here means they live next to the schema and are reusable.
# ---------------------------------------------------------------------------

def query_avg_rent_by_canton(db_path: Path | str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Average rent and price per m² per canton, joined with population."""
    sql = """
        SELECT
            l.canton,
            cs.canton_name,
            COUNT(*)                          AS n_listings,
            ROUND(AVG(l.rent_chf), 0)         AS avg_rent_chf,
            ROUND(AVG(l.price_per_m2), 2)     AS avg_price_per_m2,
            cs.population
        FROM listings AS l
        LEFT JOIN canton_stats AS cs
               ON cs.canton = l.canton
        GROUP BY l.canton, cs.canton_name, cs.population
        ORDER BY avg_price_per_m2 DESC;
    """
    with connect(db_path) as conn:
        return pd.read_sql(sql, conn)


def query_top_municipalities(
    n: int = 10,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Top-N most expensive municipalities by average price per m²."""
    sql = """
        SELECT municipality, canton, COUNT(*) AS n_listings,
               ROUND(AVG(price_per_m2), 2) AS avg_price_per_m2
        FROM listings
        WHERE municipality IS NOT NULL AND municipality != ''
        GROUP BY municipality, canton
        HAVING COUNT(*) >= 2
        ORDER BY avg_price_per_m2 DESC
        LIMIT ?;
    """
    with connect(db_path) as conn:
        return pd.read_sql(sql, conn, params=(n,))


def query_room_distribution(db_path: Path | str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Distribution of listings by room count + median rent per bucket."""
    sql = """
        SELECT rooms,
               COUNT(*) AS n_listings,
               ROUND(AVG(rent_chf), 0)        AS avg_rent_chf,
               ROUND(AVG(price_per_m2), 2)    AS avg_price_per_m2
        FROM listings
        GROUP BY rooms
        ORDER BY rooms;
    """
    with connect(db_path) as conn:
        return pd.read_sql(sql, conn)


def query_urban_vs_rural(db_path: Path | str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Compare urban (5 large cantons) and rural cantons via SQL."""
    sql = """
        SELECT CASE WHEN is_urban = 1 THEN 'urban' ELSE 'rural' END AS area_type,
               COUNT(*)                          AS n_listings,
               ROUND(AVG(rent_chf), 0)           AS avg_rent_chf,
               ROUND(AVG(price_per_m2), 2)       AS avg_price_per_m2
        FROM listings
        GROUP BY is_urban
        ORDER BY area_type;
    """
    with connect(db_path) as conn:
        return pd.read_sql(sql, conn)


def load_listings(db_path: Path | str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Read all listings back as a DataFrame (for the Streamlit app)."""
    with connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM listings;", conn)
    return df


def load_canton_stats(db_path: Path | str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Read all canton stats back as a DataFrame."""
    with connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM canton_stats;", conn)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_schema()
    print("Schema initialised at", DEFAULT_DB_PATH)
