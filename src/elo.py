"""
elo.py
------
Elo rating system for international football teams.

Elo is an iterative rating system originally designed for chess.
Each team starts at 1500. After every match:
  - The winner gains points, the loser loses the same amount.
  - The magnitude depends on: how surprising the result was (K-factor × surprise).

Key formula:
  E_home = 1 / (1 + 10^((R_away - R_home - home_adv) / 400))
  ΔR = K × tournament_weight × (actual - E_home)

Where:
  actual = 1 (home win), 0.5 (draw), 0 (away win)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

from src.utils import tournament_weight


# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_RATING  = 1500   # starting Elo for every team
HOME_ADVANTAGE  = 100    # Elo points added to home team's effective rating
K_BASE          = 40     # base K-factor (sensitivity)


# ── Core Elo functions ─────────────────────────────────────────────────────────
def expected_result(rating_home: float, rating_away: float, home_advantage: float = HOME_ADVANTAGE) -> float:
    """
    Compute the expected result (probability) for the home team.

    This is the standard Elo expected-score formula.
    Returns a value between 0 and 1.
    """
    return 1.0 / (1.0 + 10 ** ((rating_away - rating_home - home_advantage) / 400))


def update_ratings(
    rating_home: float,
    rating_away: float,
    home_score: int,
    away_score: int,
    tournament: str,
    home_advantage: float = HOME_ADVANTAGE,
    k_base: float = K_BASE
) -> Tuple[float, float]:
    """
    Update Elo ratings after a single match.

    Parameters
    ----------
    rating_home, rating_away : float
        Current Elo ratings before the match.
    home_score, away_score : int
        Full-time scoreline.
    tournament : str
        Tournament name — used to scale the K-factor.
    home_advantage : float
        Elo bonus for the home team.
    k_base : float
        Base sensitivity factor.

    Returns
    -------
    (new_rating_home, new_rating_away) : Tuple[float, float]
    """
    # Actual result from home team's perspective
    if home_score > away_score:
        actual = 1.0      # home win
    elif home_score == away_score:
        actual = 0.5      # draw
    else:
        actual = 0.0      # away win

    # Scale K by tournament importance
    tw = tournament_weight(tournament)
    k  = k_base * tw

    # Expected result
    e_home = expected_result(rating_home, rating_away, home_advantage)

    # Rating change
    delta = k * (actual - e_home)

    return (rating_home + delta, rating_away - delta)


# ── Build ratings from the full dataset ───────────────────────────────────────
def build_elo_ratings(
    df: pd.DataFrame,
    default_rating: float = DEFAULT_RATING,
    home_advantage: float = HOME_ADVANTAGE,
    k_base: float = K_BASE
) -> Dict[str, float]:
    """
    Replay every match in the dataset (sorted by date) and compute
    final Elo ratings for all teams.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: date, home_team, away_team,
                           home_score, away_score, tournament.
    default_rating : float
        Starting rating for teams with no history.
    home_advantage : float
        Elo bonus for home team.
    k_base : float
        Base K-factor.

    Returns
    -------
    dict : {team_name: elo_rating}
    """
    ratings: Dict[str, float] = {}

    df_sorted = df.sort_values("date")

    for _, row in df_sorted.iterrows():
        ht = row["home_team"]
        at = row["away_team"]

        # Initialise new teams
        if ht not in ratings:
            ratings[ht] = default_rating
        if at not in ratings:
            ratings[at] = default_rating

        new_ht, new_at = update_ratings(
            rating_home=ratings[ht],
            rating_away=ratings[at],
            home_score=int(row["home_score"]),
            away_score=int(row["away_score"]),
            tournament=row["tournament"],
            home_advantage=home_advantage if not row.get("neutral", False) else 0,
            k_base=k_base
        )

        ratings[ht] = new_ht
        ratings[at] = new_at

    return ratings


def ratings_to_dataframe(ratings: Dict[str, float]) -> pd.DataFrame:
    """Convert the ratings dict to a sorted DataFrame."""
    df = (
        pd.DataFrame(list(ratings.items()), columns=["team", "elo_rating"])
        .sort_values("elo_rating", ascending=False)
        .reset_index(drop=True)
    )
    df["rank"] = df.index + 1
    return df


# ── History tracking (optional — for plotting how ratings changed) ─────────────
def build_elo_history(
    df: pd.DataFrame,
    teams_to_track: list,
    default_rating: float = DEFAULT_RATING,
    home_advantage: float = HOME_ADVANTAGE,
    k_base: float = K_BASE
) -> pd.DataFrame:
    """
    Like build_elo_ratings() but also records each team's rating over time.
    Useful for plotting how a team's strength evolved.

    Returns a DataFrame with columns: date, team, elo_rating
    """
    ratings: Dict[str, float] = {}
    history = []

    df_sorted = df.sort_values("date")

    for _, row in df_sorted.iterrows():
        ht = row["home_team"]
        at = row["away_team"]

        if ht not in ratings:
            ratings[ht] = default_rating
        if at not in ratings:
            ratings[at] = default_rating

        new_ht, new_at = update_ratings(
            rating_home=ratings[ht],
            rating_away=ratings[at],
            home_score=int(row["home_score"]),
            away_score=int(row["away_score"]),
            tournament=row["tournament"],
            home_advantage=home_advantage if not row.get("neutral", False) else 0,
            k_base=k_base
        )

        ratings[ht] = new_ht
        ratings[at] = new_at

        # Record only tracked teams
        for team in teams_to_track:
            if team in (ht, at):
                history.append({
                    "date":       row["date"],
                    "team":       team,
                    "elo_rating": ratings[team]
                })

    return pd.DataFrame(history)
