"""
Object-oriented data model for the Swiss rental market.

Demonstrates the OOP requirement of the project rubric:
- ``Listing``        : a single apartment listing with parsed attributes
- ``RentalMarket``   : an aggregate of listings with summary methods

The procedural pipeline (``scripts/run_pipeline.py``) and the notebooks use
plain functions; the Streamlit app and a few analyses use these classes to
show both paradigms in the same project.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from statistics import mean, median
from typing import Iterable, Iterator


@dataclass
class Listing:
    """A single apartment listing.

    All attributes are already cleaned (numeric where appropriate). The
    cleaning lives in ``data_cleaning.py``; this class only carries the
    structured result.
    """

    listing_id: str
    canton: str
    municipality: str
    rooms: float
    living_space_m2: float
    rent_chf: float
    address: str = ""
    latitude: float | None = None
    longitude: float | None = None
    source: str = "homegate"

    # ----- derived properties ---------------------------------------------

    @property
    def price_per_m2(self) -> float | None:
        """Rent per square metre, or ``None`` if living space is missing."""
        if self.living_space_m2 and self.living_space_m2 > 0:
            return self.rent_chf / self.living_space_m2
        return None

    @property
    def is_urban(self) -> bool:
        """Heuristic: the five biggest urban cantons are flagged as urban."""
        urban_cantons = {"ZH", "GE", "BS", "BE", "VD"}
        return self.canton in urban_cantons

    # ----- conversion helpers --------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain ``dict`` for serialisation (DB / DataFrame)."""
        d = asdict(self)
        d["price_per_m2"] = self.price_per_m2
        d["is_urban"] = self.is_urban
        return d

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<Listing {self.listing_id} | {self.canton} {self.municipality} "
            f"| {self.rooms} rooms | {self.living_space_m2:.0f} m² "
            f"| CHF {self.rent_chf:,.0f}>"
        )


@dataclass
class RentalMarket:
    """A collection of :class:`Listing` objects with aggregate methods."""

    listings: list[Listing] = field(default_factory=list)

    # ----- container protocol --------------------------------------------

    def __len__(self) -> int:
        return len(self.listings)

    def __iter__(self) -> Iterator[Listing]:
        return iter(self.listings)

    def add(self, listing: Listing) -> None:
        """Append a single listing to the market."""
        self.listings.append(listing)

    def extend(self, listings: Iterable[Listing]) -> None:
        """Append many listings at once."""
        self.listings.extend(listings)

    # ----- aggregations ---------------------------------------------------

    def average_rent(self) -> float:
        """Mean rent across all listings (CHF)."""
        if not self.listings:
            return 0.0
        return mean(l.rent_chf for l in self.listings)

    def median_rent(self) -> float:
        """Median rent — robust to outlier luxury listings."""
        if not self.listings:
            return 0.0
        return median(l.rent_chf for l in self.listings)

    def average_price_per_m2(self) -> float:
        """Mean price per square metre, ignoring listings with missing area."""
        values = [l.price_per_m2 for l in self.listings if l.price_per_m2]
        return mean(values) if values else 0.0

    def by_canton(self) -> dict[str, "RentalMarket"]:
        """Group listings into one :class:`RentalMarket` per canton."""
        groups: dict[str, RentalMarket] = {}
        for l in self.listings:
            groups.setdefault(l.canton, RentalMarket()).add(l)
        return groups

    def cantons(self) -> set[str]:
        """Distinct canton codes present in the market (a ``set``)."""
        return {l.canton for l in self.listings}

    def filter_by_rooms(self, min_rooms: float, max_rooms: float) -> "RentalMarket":
        """Return a sub-market within an inclusive room-count range."""
        return RentalMarket(
            [l for l in self.listings if min_rooms <= l.rooms <= max_rooms]
        )

    def summary(self) -> dict:
        """Return a dictionary of headline statistics for reports."""
        return {
            "n_listings": len(self),
            "n_cantons": len(self.cantons()),
            "avg_rent_chf": round(self.average_rent(), 2),
            "median_rent_chf": round(self.median_rent(), 2),
            "avg_price_per_m2": round(self.average_price_per_m2(), 2),
        }


# ---------------------------------------------------------------------------
# Helper to build a RentalMarket directly from a pandas DataFrame.
# Kept here so the OOP model can travel between notebooks and the app.
# ---------------------------------------------------------------------------

def market_from_dataframe(df) -> RentalMarket:
    """Build a :class:`RentalMarket` from a cleaned listings DataFrame."""
    market = RentalMarket()
    required_cols = {"listing_id", "canton", "municipality", "rooms",
                     "living_space_m2", "rent_chf"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    for _, row in df.iterrows():
        listing = Listing(
            listing_id=str(row["listing_id"]),
            canton=str(row["canton"]),
            municipality=str(row["municipality"]),
            rooms=float(row["rooms"]),
            living_space_m2=float(row["living_space_m2"]),
            rent_chf=float(row["rent_chf"]),
            address=str(row.get("address", "")),
            latitude=row.get("latitude"),
            longitude=row.get("longitude"),
            source=str(row.get("source", "homegate")),
        )
        market.add(listing)
    return market
