"""
utils.py
--------
Helper functions used across the project.
Includes data loading, downloading, and date utilities.
"""

import os
import pandas as pd
import urllib.request


# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

RESULTS_CSV   = os.path.join(DATA_DIR, "results.csv")
SHOOTOUTS_CSV = os.path.join(DATA_DIR, "shootouts.csv")

# Source URLs (martj42/international_results on GitHub)
RESULTS_URL   = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SHOOTOUTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"


# ── Download ───────────────────────────────────────────────────────────────────
def download_data(force: bool = False) -> None:
    """Download results.csv and shootouts.csv from the martj42 repo."""
    os.makedirs(DATA_DIR, exist_ok=True)

    for url, path in [(RESULTS_URL, RESULTS_CSV), (SHOOTOUTS_URL, SHOOTOUTS_CSV)]:
        fname = os.path.basename(path)
        if os.path.exists(path) and not force:
            print(f"  ✓ {fname} already exists — skipping (use force=True to re-download)")
        else:
            print(f"  ↓ Downloading {fname} ...")
            urllib.request.urlretrieve(url, path)
            print(f"  ✓ Saved to {path}")


# ── Load ───────────────────────────────────────────────────────────────────────
def load_results(
    min_year: int = 1990,
    exclude_friendlies: bool = False
) -> pd.DataFrame:
    """
    Load and clean the results dataset.

    Parameters
    ----------
    min_year : int
        Only keep matches from this year onwards.
        Default 1990 — older data is less reliable for rating modern teams.
    exclude_friendlies : bool
        If True, drop friendly matches (lower competitive weight).

    Returns
    -------
    pd.DataFrame with columns:
        date, home_team, away_team, home_score, away_score,
        tournament, city, country, neutral, year
    """
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(
            f"Dataset not found at {RESULTS_CSV}.\n"
            "Run utils.download_data() first."
        )

    df = pd.read_csv(RESULTS_CSV, parse_dates=["date"])

    # Basic cleaning
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["year"] = df["date"].dt.year
    df["neutral"] = df["neutral"].astype(bool)

    # Filter by year
    df = df[df["year"] >= min_year].copy()

    # Optionally drop friendlies
    if exclude_friendlies:
        df = df[df["tournament"] != "Friendly"].copy()

    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_shootouts() -> pd.DataFrame:
    """Load the penalty shootout results."""
    if not os.path.exists(SHOOTOUTS_CSV):
        raise FileNotFoundError(
            f"Shootouts file not found at {SHOOTOUTS_CSV}.\n"
            "Run utils.download_data() first."
        )
    return pd.read_csv(SHOOTOUTS_CSV, parse_dates=["date"])


# ── Match result helpers ───────────────────────────────────────────────────────
def get_result(home_score: int, away_score: int) -> str:
    """Return 'H' (home win), 'D' (draw), or 'A' (away win)."""
    if home_score > away_score:
        return "H"
    elif home_score == away_score:
        return "D"
    else:
        return "A"


def tournament_weight(tournament: str) -> float:
    """
    Return a weight (0–1) for a tournament type.
    Used in Elo to scale the K-factor by match importance.
    """
    t = tournament.lower()
    if "world cup" in t and "qualif" not in t:
        return 1.0
    elif "continental championship" in t or "copa america" in t or "euro" in t or "afcon" in t:
        return 0.85
    elif "qualif" in t or "qualification" in t:
        return 0.65
    elif "friendly" in t or "friendlies" in t:
        return 0.3
    else:
        return 0.5


# ── CLI entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Football Match Predictore — utilities")
    parser.add_argument("--download", action="store_true", help="Download the dataset")
    parser.add_argument("--force",    action="store_true", help="Force re-download even if files exist")
    args = parser.parse_args()

    if args.download:
        print("Downloading dataset...")
        download_data(force=args.force)
        print("Done!")
