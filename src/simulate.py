"""
simulate.py
-----------
Monte Carlo simulation for football matches and full World Cup tournaments.

What is Monte Carlo simulation?
---------------------------------
Instead of computing probabilities analytically (hard for a whole tournament),
we run the tournament thousands of times, each time sampling random outcomes
from our Poisson model. Then we count how often each team wins the title.

After 10,000 simulations:
  P(team wins World Cup) ≈ (times they won) / 10,000

The more simulations, the more accurate the estimate.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from src.poisson import PoissonModel


# ── Single match simulation ────────────────────────────────────────────────────
def simulate_match(
    home_team: str,
    away_team: str,
    model: PoissonModel,
    neutral: bool = False,
    n_simulations: int = 10_000
) -> Dict[str, float]:
    """
    Simulate a single match N times via Monte Carlo.

    Each simulation:
      1. Sample home goals from Poisson(λ_home)
      2. Sample away goals from Poisson(λ_away)
      3. Record the result

    Parameters
    ----------
    home_team, away_team : str
    model : PoissonModel (must be fitted)
    neutral : bool
        True for neutral venue matches (no home advantage).
    n_simulations : int
        Number of Monte Carlo trials.

    Returns
    -------
    dict with keys: home_win, draw, away_win, lambda_home, lambda_away,
                    avg_home_goals, avg_away_goals
    """
    lh, la = model.predict_lambda(home_team, away_team, neutral)

    # Draw n_simulations scorelines at once (vectorised — fast!)
    home_goals = np.random.poisson(lh, size=n_simulations)
    away_goals = np.random.poisson(la, size=n_simulations)

    home_wins = np.sum(home_goals > away_goals)
    draws      = np.sum(home_goals == away_goals)
    away_wins  = np.sum(home_goals < away_goals)

    return {
        "home_win":       home_wins / n_simulations,
        "draw":           draws     / n_simulations,
        "away_win":       away_wins / n_simulations,
        "lambda_home":    lh,
        "lambda_away":    la,
        "avg_home_goals": home_goals.mean(),
        "avg_away_goals": away_goals.mean()
    }


def simulate_match_once(
    home_team: str,
    away_team: str,
    model: PoissonModel,
    neutral: bool = False
) -> Tuple[int, int]:
    """
    Simulate a single match ONCE — returns (home_goals, away_goals).
    Used inside the tournament simulator.
    """
    lh, la = model.predict_lambda(home_team, away_team, neutral)
    return int(np.random.poisson(lh)), int(np.random.poisson(la))


# ── Knockout match (with extra time / penalty decider) ────────────────────────
def simulate_knockout_match(
    team_a: str,
    team_b: str,
    model: PoissonModel
) -> str:
    """
    Simulate a knockout match. If it ends in a draw after 90 min,
    we go to a penalty shootout (50/50 coin flip).

    Returns the name of the winner.
    """
    hg, ag = simulate_match_once(team_a, team_b, model, neutral=True)

    if hg > ag:
        return team_a
    elif ag > hg:
        return team_b
    else:
        # Penalty shootout — 50/50
        return team_a if np.random.random() < 0.5 else team_b


# ── Group stage simulation ─────────────────────────────────────────────────────
def simulate_group(
    teams: List[str],
    model: PoissonModel
) -> pd.DataFrame:
    """
    Simulate a round-robin group stage for 4 teams.

    Returns a DataFrame sorted by points (then GD, then GF) with columns:
    team, points, gd (goal difference), gf (goals for), played
    """
    # Initialise standings
    standings = {t: {"points": 0, "gd": 0, "gf": 0, "played": 0} for t in teams}

    # Play every pair once (6 matches for 4 teams)
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            home, away = teams[i], teams[j]
            hg, ag = simulate_match_once(home, away, model, neutral=True)

            # Points
            if hg > ag:
                standings[home]["points"] += 3
            elif hg == ag:
                standings[home]["points"] += 1
                standings[away]["points"] += 1
            else:
                standings[away]["points"] += 3

            # Goal stats
            standings[home]["gd"] += hg - ag
            standings[away]["gd"] += ag - hg
            standings[home]["gf"] += hg
            standings[away]["gf"] += ag
            standings[home]["played"] += 1
            standings[away]["played"] += 1

    # Convert to sorted DataFrame
    rows = [{"team": t, **standings[t]} for t in teams]
    df = pd.DataFrame(rows)
    df = df.sort_values(["points", "gd", "gf"], ascending=[False, False, False])
    return df.reset_index(drop=True)


# ── Full World Cup simulation ──────────────────────────────────────────────────
def simulate_world_cup(
    groups: Dict[str, List[str]],
    model: PoissonModel,
    n_simulations: int = 10_000,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Simulate the entire World Cup N times and return each team's win probability.

    This uses a simplified 32-team bracket structure:
      - 8 groups of 4 teams
      - Top 2 from each group advance (16 teams)
      - Standard knockout: Round of 16 → Quarters → Semis → Final

    For the 2026 format (48 teams, 12 groups), use simulate_wc2026().

    Parameters
    ----------
    groups : dict
        {"Group A": ["Brazil", "Germany", ...], "Group B": [...], ...}
        Must have 8 groups of 4 teams each.
    model : PoissonModel
        Fitted Poisson model.
    n_simulations : int
        Number of tournament simulations.
    verbose : bool
        Show progress bar.

    Returns
    -------
    pd.DataFrame with columns: team, win_pct, semifinal_pct, final_pct
    """
    group_names = list(groups.keys())
    all_teams   = [t for g in groups.values() for t in g]

    # Track outcomes
    wins       = {t: 0 for t in all_teams}
    finals     = {t: 0 for t in all_teams}
    semifinals = {t: 0 for t in all_teams}

    iterator = tqdm(range(n_simulations), desc="Simulating WC") if verbose else range(n_simulations)

    for _ in iterator:
        # ── Group stage ──────────────────────────────────────────────────────
        qualifiers = []  # teams that advance (top 2 per group)

        for gname in group_names:
            table = simulate_group(groups[gname], model)
            qualifiers.append(table.iloc[0]["team"])  # 1st place
            qualifiers.append(table.iloc[1]["team"])  # 2nd place

        # ── Knockout bracket ──────────────────────────────────────────────────
        # Standard bracket: Group A 1st vs Group B 2nd, etc.
        # [A1,B2,C1,D2,E1,F2,G1,H2] vs [B1,A2,D1,C2,F1,E2,H1,G2]
        bracket = qualifiers  # 16 teams, already in order

        round_teams = bracket[:]

        # R16 → QF → SF → Final
        while len(round_teams) > 1:
            next_round = []
            for k in range(0, len(round_teams), 2):
                winner = simulate_knockout_match(round_teams[k], round_teams[k+1], model)
                next_round.append(winner)

            # Track milestone
            if len(round_teams) == 4:
                for t in round_teams:
                    semifinals[t] += 1
            if len(round_teams) == 2:
                for t in round_teams:
                    finals[t] += 1

            round_teams = next_round

        # Champion
        wins[round_teams[0]] += 1

    # ── Aggregate results ────────────────────────────────────────────────────
    rows = []
    for team in all_teams:
        rows.append({
            "team":           team,
            "win_pct":        wins[team]       / n_simulations * 100,
            "final_pct":      finals[team]     / n_simulations * 100,
            "semifinal_pct":  semifinals[team] / n_simulations * 100
        })

    return (
        pd.DataFrame(rows)
        .sort_values("win_pct", ascending=False)
        .reset_index(drop=True)
    )


# ── Pretty print match prediction ─────────────────────────────────────────────
def print_match_prediction(
    home_team: str,
    away_team: str,
    model: PoissonModel,
    neutral: bool = False,
    n_simulations: int = 10_000
) -> None:
    """
    Print a formatted match prediction to the console.
    """
    result = simulate_match(home_team, away_team, model, neutral, n_simulations)

    venue_str = "(Neutral)" if neutral else "(Home/Away)"
    print(f"\n{'='*50}")
    print(f"  {home_team}  vs  {away_team}  {venue_str}")
    print(f"{'='*50}")
    print(f"  Expected goals:  {result['lambda_home']:.2f}  –  {result['lambda_away']:.2f}")
    print(f"  {home_team:<20} win:  {result['home_win']*100:5.1f}%")
    print(f"  {'Draw':<20}      :  {result['draw']*100:5.1f}%")
    print(f"  {away_team:<20} win:  {result['away_win']*100:5.1f}%")
    print(f"{'='*50}\n")
