"""
Statistical analysis of the cleaned rental data.

Runs three independent tests, each producing a p-value (rubric requirement
#7). Together they answer different facets of the research question:

1. **Pearson correlation** — does rent increase linearly with living space?
2. **Welch t-test** — do urban cantons have higher price per m² than rural?
3. **One-way ANOVA** — do mean prices per m² differ significantly across
   cantons?

A small helper interprets each p-value into a human-readable verdict so
the Streamlit app and notebooks can reuse it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

ALPHA = 0.05  # Conventional significance threshold


@dataclass
class TestResult:
    """A single statistical test's outcome with metadata."""
    name: str
    statistic: float
    p_value: float
    effect: float | None = None
    note: str = ""

    @property
    def significant(self) -> bool:
        return self.p_value < ALPHA

    def verdict(self) -> str:
        """One-sentence interpretation suitable for slides / video."""
        verb = "rejects" if self.significant else "fails to reject"
        return (
            f"{self.name}: p = {self.p_value:.4g} — {verb} the null hypothesis "
            f"at α = {ALPHA}."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "effect": None if self.effect is None else float(self.effect),
            "significant": self.significant,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Test 1: Pearson correlation between living space and rent
# ---------------------------------------------------------------------------

def correlation_rent_vs_size(df: pd.DataFrame) -> TestResult:
    """Pearson correlation of ``living_space_m2`` and ``rent_chf``."""
    sub = df[["living_space_m2", "rent_chf"]].dropna()
    if len(sub) < 3:
        return TestResult("Pearson rent~size", float("nan"), float("nan"),
                          note="Not enough data")

    r, p = stats.pearsonr(sub["living_space_m2"], sub["rent_chf"])
    return TestResult(
        name="Pearson correlation: rent vs. living space",
        statistic=float(r),
        p_value=float(p),
        effect=float(r),
        note=f"n = {len(sub):,}",
    )


# ---------------------------------------------------------------------------
# Test 2: Welch t-test, urban vs rural price per m²
# ---------------------------------------------------------------------------

def ttest_urban_vs_rural(df: pd.DataFrame) -> TestResult:
    """Welch's two-sample t-test on price per m² (urban vs. rural)."""
    urban = df.loc[df["is_urban"] == 1, "price_per_m2"].dropna()
    rural = df.loc[df["is_urban"] == 0, "price_per_m2"].dropna()

    # Some pipelines pass booleans; handle both.
    if len(urban) == 0 and "is_urban" in df.columns:
        urban = df.loc[df["is_urban"].astype(bool), "price_per_m2"].dropna()
        rural = df.loc[~df["is_urban"].astype(bool), "price_per_m2"].dropna()

    if len(urban) < 2 or len(rural) < 2:
        return TestResult("Welch t-test urban~rural", float("nan"), float("nan"),
                          note="Not enough data in one of the groups")

    t, p = stats.ttest_ind(urban, rural, equal_var=False)
    # Cohen's d as effect size — rough but useful in slides.
    pooled_sd = np.sqrt((urban.var(ddof=1) + rural.var(ddof=1)) / 2)
    cohens_d = (urban.mean() - rural.mean()) / pooled_sd if pooled_sd else float("nan")

    return TestResult(
        name="Welch's t-test: price/m² urban vs. rural",
        statistic=float(t),
        p_value=float(p),
        effect=float(cohens_d),
        note=(f"urban mean = CHF {urban.mean():.2f}/m²  |  "
              f"rural mean = CHF {rural.mean():.2f}/m²  |  "
              f"n_urban = {len(urban):,}, n_rural = {len(rural):,}"),
    )


# ---------------------------------------------------------------------------
# Test 3: One-way ANOVA across cantons
# ---------------------------------------------------------------------------

def anova_price_by_canton(df: pd.DataFrame, min_per_group: int = 5) -> TestResult:
    """One-way ANOVA: does mean price per m² differ across cantons?

    Cantons with fewer than ``min_per_group`` listings are dropped to keep
    the F-test well-conditioned.
    """
    valid = df.dropna(subset=["price_per_m2", "canton"])
    counts = valid["canton"].value_counts()
    cantons_used = counts[counts >= min_per_group].index.tolist()
    valid = valid[valid["canton"].isin(cantons_used)]

    groups = [valid.loc[valid["canton"] == c, "price_per_m2"].values
              for c in cantons_used]
    if len(groups) < 2:
        return TestResult("ANOVA price~canton", float("nan"), float("nan"),
                          note="Need at least 2 cantons with enough data")

    f, p = stats.f_oneway(*groups)
    # Eta-squared as an effect-size estimate.
    grand_mean = valid["price_per_m2"].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((valid["price_per_m2"] - grand_mean) ** 2).sum()
    eta_sq = ss_between / ss_total if ss_total else float("nan")

    return TestResult(
        name="One-way ANOVA: price/m² across cantons",
        statistic=float(f),
        p_value=float(p),
        effect=float(eta_sq),
        note=f"{len(cantons_used)} cantons, n = {len(valid):,}",
    )


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_all(df: pd.DataFrame) -> dict[str, TestResult]:
    """Run all three tests and return a dictionary of results."""
    results = {
        "correlation": correlation_rent_vs_size(df),
        "t_test":      ttest_urban_vs_rural(df),
        "anova":       anova_price_by_canton(df),
    }
    for name, r in results.items():
        logger.info("%-12s %s", name, r.verdict())
    return results


def results_to_dataframe(results: dict[str, TestResult]) -> pd.DataFrame:
    """Pretty-print all tests as a DataFrame for slides / app."""
    rows = []
    for r in results.values():
        rows.append({
            "Test":       r.name,
            "Statistic":  round(r.statistic, 4),
            "p-value":    f"{r.p_value:.4g}",
            "Effect":     "—" if r.effect is None else round(r.effect, 4),
            "Significant (α=0.05)": "✓" if r.significant else "✗",
            "Note":       r.note,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick check with random data
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    demo = pd.DataFrame({
        "living_space_m2": rng.uniform(30, 150, 500),
        "rent_chf":        rng.uniform(800, 5000, 500),
        "price_per_m2":    rng.uniform(15, 35, 500),
        "is_urban":        rng.choice([0, 1], 500),
        "canton":          rng.choice(["ZH", "BE", "GE", "VD", "TI"], 500),
    })
    res = run_all(demo)
    print(results_to_dataframe(res).to_string(index=False))
