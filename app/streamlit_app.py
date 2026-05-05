"""
Streamlit dashboard for swiss-rent-radar.

Run from the repository root with:

    streamlit run app/streamlit_app.py

The app reads from the SQLite database produced by ``run_pipeline.py``
and exposes filtering, summary tables, charts, an interactive map, and
an LLM-powered market commentary button.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the ``app`` package importable when Streamlit launches the script
# directly (i.e. ``streamlit run app/streamlit_app.py``).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from app import analysis, database, llm_helper, visualization
from app.models import market_from_dataframe


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Swiss Rent Radar",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏠 Swiss Rent Radar")
st.caption(
    "Scientific Programming project (FS2026) — Marko Vukcevic et al. "
    "Live Homegate listings + official BFS reference, all in one dashboard."
)

# ---------------------------------------------------------------------------
# Data loading (cached for snappy interactions)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load listings and canton stats from the SQLite DB, cached."""
    listings = database.load_listings()
    canton_stats = database.load_canton_stats()
    return listings, canton_stats


# Load and surface a friendly error if the pipeline has not been run yet.
try:
    listings_df, canton_df = load_data()
except Exception as exc:  # pragma: no cover - UI safety
    st.error(
        "Could not read the SQLite database. "
        "Have you run `python scripts/run_pipeline.py` first?"
    )
    st.exception(exc)
    st.stop()

if listings_df.empty:
    st.warning("Database is empty. Run `python scripts/run_pipeline.py` first.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")

cantons = sorted(listings_df["canton"].unique().tolist())
selected_cantons = st.sidebar.multiselect(
    "Cantons", cantons, default=cantons,
    help="Hold ⌘/Ctrl to select multiple.",
)

room_min, room_max = float(listings_df["rooms"].min()), float(listings_df["rooms"].max())
rooms_range = st.sidebar.slider(
    "Number of rooms",
    min_value=room_min, max_value=room_max,
    value=(room_min, room_max), step=0.5,
)

rent_min, rent_max = float(listings_df["rent_chf"].min()), float(listings_df["rent_chf"].max())
rent_range = st.sidebar.slider(
    "Monthly rent (CHF)",
    min_value=int(rent_min), max_value=int(rent_max),
    value=(int(rent_min), int(rent_max)), step=100,
)

source_filter = st.sidebar.radio(
    "Data source",
    options=("Both", "homegate-live", "snapshot"),
    index=0,
)

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

mask = (
    listings_df["canton"].isin(selected_cantons) &
    listings_df["rooms"].between(*rooms_range) &
    listings_df["rent_chf"].between(*rent_range)
)
if source_filter != "Both":
    mask &= listings_df["source"] == source_filter

filtered = listings_df[mask].copy()

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------

market = market_from_dataframe(filtered) if not filtered.empty else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Listings",          f"{len(filtered):,}")
c2.metric("Avg rent (CHF)",    f"{filtered['rent_chf'].mean():,.0f}" if len(filtered) else "—")
c3.metric("Avg price / m²",    f"{filtered['price_per_m2'].mean():.1f}" if len(filtered) else "—")
c4.metric("Cantons covered",   f"{filtered['canton'].nunique()}")

# ---------------------------------------------------------------------------
# Tabs: Overview · Statistics · Map · LLM
# ---------------------------------------------------------------------------

tab_overview, tab_stats, tab_map, tab_llm = st.tabs(
    ["📊 Overview", "📐 Statistics", "🗺️ Map", "🤖 Market commentary"]
)

# ----- Overview ------------------------------------------------------------
with tab_overview:
    if filtered.empty:
        st.info("No listings match the current filters.")
    else:
        st.subheader("Top 10 most expensive municipalities")
        top_muni = (filtered.groupby(["municipality", "canton"], as_index=False)
                            .agg(n_listings=("listing_id", "count"),
                                 avg_price_per_m2=("price_per_m2", "mean"))
                            .sort_values("avg_price_per_m2", ascending=False)
                            .head(10))
        st.dataframe(top_muni.round(2), use_container_width=True)

        st.subheader("Rent vs. living space")
        st.pyplot(visualization.plot_rent_vs_size(filtered))

        st.subheader("Price per m² across cantons")
        st.pyplot(visualization.plot_price_per_m2_by_canton(filtered))

        st.subheader("SQL query — average rent by canton (live from DB)")
        sql_df = database.query_avg_rent_by_canton()
        st.dataframe(sql_df, use_container_width=True)

# ----- Statistics ----------------------------------------------------------
with tab_stats:
    if filtered.empty:
        st.info("No listings to analyse.")
    else:
        st.subheader("Statistical tests (all report a p-value)")
        results = analysis.run_all(filtered)
        st.dataframe(analysis.results_to_dataframe(results),
                     use_container_width=True, hide_index=True)

        st.subheader("Urban vs. rural — distribution")
        st.pyplot(visualization.plot_urban_vs_rural(filtered))

        st.subheader("Distribution of room counts")
        st.pyplot(visualization.plot_room_distribution(filtered))

        st.markdown(
            "> **Reading the table:** a p-value below 0.05 means we reject "
            "the null hypothesis at the conventional significance level. "
            "Effect sizes (Pearson r, Cohen's d, eta²) tell you how big the "
            "difference is, not just whether it is statistically detectable."
        )

# ----- Map -----------------------------------------------------------------
with tab_map:
    if filtered.empty:
        st.info("Add listings to see them on the map.")
    else:
        st.subheader(f"{len(filtered):,} listings on a map")
        out_path = visualization.build_listing_map(filtered)
        # Embed the saved HTML directly so users see the same map as in /outputs.
        with open(out_path, encoding="utf-8") as f:
            map_html = f.read()
        st.components.v1.html(map_html, height=600, scrolling=False)

# ----- LLM ----------------------------------------------------------------
with tab_llm:
    st.subheader("Market commentary")
    st.caption(
        "Click the button to ask Mistral 7B (via Together.ai) for a short "
        "natural-language summary of the *currently filtered* market. "
        "Without an API key the app returns an offline deterministic summary."
    )

    if st.button("Generate commentary", type="primary"):
        canton_sql = database.query_avg_rent_by_canton()
        room_sql = database.query_room_distribution()
        with st.spinner("Calling the language model…"):
            commentary = llm_helper.market_commentary(canton_sql, room_sql)
        st.markdown(commentary)

st.divider()
st.caption(
    "Data: opendata.swiss / BFS (canton reference) + Homegate.ch (live listings). "
    "Code: github.com/<your-user>/swiss-rent-radar"
)
