"""
End-to-end orchestrator: collect → clean → store → analyse.

Run this once before opening the Streamlit app or the notebooks::

    python scripts/run_pipeline.py

Use ``--live`` to attempt a real Homegate scrape; without the flag the
pipeline uses the bundled deterministic snapshot (recommended for the
project video to avoid network surprises).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the ``app`` package importable.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import api_client, scraper, data_cleaning, database, analysis  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true",
        help="Hit Homegate live (may fail without internet). "
             "Default: use bundled snapshot.",
    )
    parser.add_argument(
        "--pages", type=int, default=2,
        help="Pages per canton when --live (default: 2)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("pipeline")

    log.info("=" * 60)
    log.info("swiss-rent-radar — end-to-end pipeline starting")
    log.info("=" * 60)

    # ---- 1. Fetch BFS canton reference -----------------------------------
    log.info("Step 1/5: fetch canton reference from opendata.swiss / BFS")
    canton_df = api_client.fetch_canton_rent_reference()

    # ---- 2. Scrape Homegate (or snapshot) --------------------------------
    log.info("Step 2/5: collect Homegate listings (live=%s)", args.live)
    cfg = scraper.ScraperConfig(
        pages_per_canton=args.pages,
        use_snapshot=not args.live,
    )
    raw_df = scraper.scrape_homegate(cfg)
    log.info("  → %d raw rows", len(raw_df))

    # ---- 3. Clean and enrich --------------------------------------------
    log.info("Step 3/5: clean + enrich with regex and pandas")
    clean_df = data_cleaning.clean_listings(raw_df)
    enriched_df = data_cleaning.attach_canton_reference(clean_df, canton_df)
    log.info("  → %d clean listings", len(enriched_df))

    # ---- 4. Persist into SQLite -----------------------------------------
    log.info("Step 4/5: persist to SQLite")
    database.init_schema()
    database.write_canton_stats(canton_df)
    database.write_listings(enriched_df)

    # ---- 5. Run statistical tests as a sanity check ---------------------
    log.info("Step 5/5: run statistical tests")
    results = analysis.run_all(enriched_df)
    print()
    print(analysis.results_to_dataframe(results).to_string(index=False))
    print()

    summary_sql = database.query_avg_rent_by_canton()
    print("\nAverage rent by canton (top 5 from SQL):")
    print(summary_sql.head().to_string(index=False))

    log.info("Pipeline complete. Database at %s", database.DEFAULT_DB_PATH)


if __name__ == "__main__":
    main()
