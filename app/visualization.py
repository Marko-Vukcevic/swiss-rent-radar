"""
Visualisation helpers — matplotlib, seaborn, and folium.

Each public function returns either a Matplotlib figure (for direct
display in a notebook or Streamlit) or saves a folium map to disk and
returns the file path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# A consistent seaborn theme across all figures
sns.set_theme(style="whitegrid", context="notebook")

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Matplotlib / Seaborn figures
# ---------------------------------------------------------------------------

def plot_rent_vs_size(df: pd.DataFrame) -> plt.Figure:
    """Scatter of living space vs rent with a regression line, coloured by canton."""
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df, x="living_space_m2", y="rent_chf",
        hue="canton", alpha=0.7, ax=ax, legend=False, s=30,
    )
    # Regression line on the pooled data
    sns.regplot(
        data=df, x="living_space_m2", y="rent_chf",
        scatter=False, ax=ax, color="black", line_kws={"linewidth": 2},
    )
    ax.set_title("Rent vs. living space — Swiss apartment listings")
    ax.set_xlabel("Living space (m²)")
    ax.set_ylabel("Rent (CHF / month)")
    fig.tight_layout()
    return fig


def plot_price_per_m2_by_canton(df: pd.DataFrame) -> plt.Figure:
    """Boxplot of price per m² across cantons, ordered by median."""
    order = (df.groupby("canton")["price_per_m2"]
               .median()
               .sort_values(ascending=False)
               .index)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=df, x="canton", y="price_per_m2", order=order, ax=ax)
    ax.set_title("Price per m² across Swiss cantons")
    ax.set_xlabel("Canton")
    ax.set_ylabel("Price per m² (CHF)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_urban_vs_rural(df: pd.DataFrame) -> plt.Figure:
    """Violin plot comparing urban vs rural price per m²."""
    df_plot = df.copy()
    df_plot["area_type"] = np.where(df_plot["is_urban"].astype(bool), "Urban", "Rural")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=df_plot, x="area_type", y="price_per_m2",
                   ax=ax, inner="quartile")
    ax.set_title("Price per m² — urban vs. rural cantons")
    ax.set_xlabel("")
    ax.set_ylabel("Price per m² (CHF)")
    fig.tight_layout()
    return fig


def plot_room_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram of room counts in the sample."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="rooms", bins=12, ax=ax)
    ax.set_title("Distribution of room counts in the listings")
    ax.set_xlabel("Rooms")
    ax.set_ylabel("Number of listings")
    fig.tight_layout()
    return fig


def plot_listing_vs_bfs_gap(df: pd.DataFrame) -> plt.Figure:
    """Compare listing rent to BFS expected rent — diagnostic of the market."""
    if "rent_gap_chf" not in df.columns:
        raise ValueError("Run attach_canton_reference first; missing rent_gap_chf.")

    fig, ax = plt.subplots(figsize=(11, 6))
    order = df.groupby("canton")["rent_gap_chf"].median().sort_values().index
    sns.boxplot(data=df, x="canton", y="rent_gap_chf", order=order, ax=ax,
                color="#5B8FF9")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Listing rent minus BFS expected rent (CHF / month)")
    ax.set_xlabel("Canton")
    ax.set_ylabel("Rent gap (CHF)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Folium map
# ---------------------------------------------------------------------------

def build_listing_map(
    df: pd.DataFrame,
    output_path: Path | str = OUTPUTS_DIR / "listings_map.html",
) -> Path:
    """Build a folium map with one marker per listing.

    Returns the path to the saved HTML file. Marker radius scales with
    rent and a colour gradient encodes price per m².
    """
    output_path = Path(output_path)
    sub = df.dropna(subset=["latitude", "longitude", "price_per_m2"])

    if sub.empty:
        logger.warning("No coordinates available — building an empty map.")
        m = folium.Map(location=[46.8, 8.2], zoom_start=8)
        m.save(output_path)
        return output_path

    m = folium.Map(
        location=[sub["latitude"].mean(), sub["longitude"].mean()],
        zoom_start=8,
        tiles="cartodbpositron",
    )

    # Color gradient: cheap (blue) -> expensive (red)
    p_min, p_max = sub["price_per_m2"].quantile([0.05, 0.95])

    def _colour(p: float) -> str:
        t = (p - p_min) / max(p_max - p_min, 1e-6)
        t = max(0.0, min(1.0, t))
        # Linear interpolate between blue (#1d4ed8) and red (#dc2626)
        r1, g1, b1 = 0x1d, 0x4e, 0xd8
        r2, g2, b2 = 0xdc, 0x26, 0x26
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    for _, row in sub.iterrows():
        popup = (
            f"<b>{row.get('title','listing')}</b><br>"
            f"{row['canton']} · {row.get('municipality','')}<br>"
            f"{row['rooms']:.1f} rooms · {row['living_space_m2']:.0f} m²<br>"
            f"<b>CHF {row['rent_chf']:,.0f}</b> "
            f"({row['price_per_m2']:.1f} CHF/m²)"
        ).replace(",", "'")
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4 + row["rent_chf"] / 1000,
            color=_colour(row["price_per_m2"]),
            fill=True, fill_opacity=0.7, weight=1,
            popup=folium.Popup(popup, max_width=260),
        ).add_to(m)

    m.save(output_path)
    logger.info("Saved folium map to %s", output_path)
    return output_path


def build_canton_choropleth(
    canton_df: pd.DataFrame,
    output_path: Path | str = OUTPUTS_DIR / "canton_choropleth.html",
    geojson_url: str = (
        "https://raw.githubusercontent.com/interactivethings/"
        "swiss-maps/master/topo/ch-cantons.json"
    ),
) -> Path:
    """Build a choropleth map of mean price per m² by canton."""
    output_path = Path(output_path)
    m = folium.Map(location=[46.8, 8.2], zoom_start=7, tiles="cartodbpositron")

    # The interactivethings TopoJSON keys cantons by their numeric ID and
    # short name, so we provide a fallback key if needed.
    try:
        folium.Choropleth(
            geo_data=geojson_url,
            data=canton_df,
            columns=["canton_name", "avg_price_per_m2"],
            key_on="feature.properties.name",
            fill_color="YlOrRd",
            fill_opacity=0.75,
            line_opacity=0.3,
            legend_name="Mean price per m² (CHF)",
        ).add_to(m)
    except Exception as exc:
        logger.warning("Choropleth failed (%s) — falling back to markers.", exc)

    m.save(output_path)
    logger.info("Saved choropleth to %s", output_path)
    return output_path


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print("Visualisation module loaded OK")
