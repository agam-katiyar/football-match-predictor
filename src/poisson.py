"""
poisson.py
----------
Poisson regression model for predicting football match scorelines.

How it works
------------
1. Each team has two parameters:
     - attack  : how many goals they tend to score
     - defense : how many goals they tend to concede (lower = stronger defence)

2. Expected goals for each match:
     λ_home = exp(attack_home + defense_away + home_advantage + intercept)
     λ_away = exp(attack_away + defense_home + intercept)

   (We use exp() because goals must be ≥ 0 — this is the Poisson log-link)

3. We fit these parameters by Maximum Likelihood Estimation (MLE):
     - Assume actual goals follow a Poisson distribution
     - Find the attack/defense values that make the observed scorelines most probable

4. To predict a match:
     - Compute λ_home and λ_away
     - Build a score probability matrix P(home_goals=i, away_goals=j) for i,j ∈ 0..8
     - Sum the right cells for P(home win), P(draw), P(away win)

This is the simplest version — no Dixon-Coles low-score correction yet.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from typing import Dict, Tuple


# ── Model class ────────────────────────────────────────────────────────────────
class PoissonModel:
    """
    Poisson regression model for predicting football match outcomes.

    Attributes
    ----------
    teams : list[str]
        Sorted list of all teams in the training data.
    params : np.ndarray
        Fitted parameters [attack_0, ..., attack_n, defense_0, ..., defense_n,
                           home_advantage, intercept]
    attack : dict[str, float]
        Fitted attack strength per team (in log space — add intercept to use).
    defense : dict[str, float]
        Fitted defensive weakness per team (higher = weaker defence).
    home_adv : float
        Home advantage parameter (log scale).
    intercept : float
        Model intercept.
    is_fitted : bool
        Whether .fit() has been called.
    """

    def __init__(self):
        self.teams: list       = []
        self.params: np.ndarray = None
        self.attack: Dict[str, float]  = {}
        self.defense: Dict[str, float] = {}
        self.home_adv: float   = 0.0
        self.intercept: float  = 0.0
        self.is_fitted: bool   = False

    # ── Negative log-likelihood ────────────────────────────────────────────────
    def _neg_log_likelihood(self, params: np.ndarray, df: pd.DataFrame) -> float:
        """
        Compute the negative log-likelihood of the data given parameters.
        scipy.minimize minimises, so we return the negative of the log-likelihood.
        """
        n = len(self.teams)
        team_idx = {t: i for i, t in enumerate(self.teams)}

        attack_params  = params[:n]      # one per team
        defense_params = params[n:2*n]   # one per team
        home_adv       = params[2*n]
        intercept      = params[2*n + 1]

        log_lik = 0.0

        for _, row in df.iterrows():
            hi = team_idx[row["home_team"]]
            ai = team_idx[row["away_team"]]

            # Log of expected goals (Poisson mean)
            log_lambda_home = (attack_params[hi]
                               + defense_params[ai]
                               + home_adv
                               + intercept)
            log_lambda_away = (attack_params[ai]
                               + defense_params[hi]
                               + intercept)

            lambda_home = np.exp(log_lambda_home)
            lambda_away = np.exp(log_lambda_away)

            # Poisson log-likelihood: log P(k; λ) = k*log(λ) - λ - log(k!)
            home_goals = int(row["home_score"])
            away_goals = int(row["away_score"])

            log_lik += (home_goals * np.log(lambda_home + 1e-10) - lambda_home
                        + away_goals * np.log(lambda_away + 1e-10) - lambda_away)

        return -log_lik   # negative because we minimise

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, verbose: bool = True) -> "PoissonModel":
        """
        Fit the Poisson model to historical match data.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: home_team, away_team, home_score, away_score.
        verbose : bool
            Print training info.

        Returns
        -------
        self (for method chaining)
        """
        # Collect all teams
        home_teams = df["home_team"].unique().tolist()
        away_teams = df["away_team"].unique().tolist()
        self.teams = sorted(set(home_teams + away_teams))
        n = len(self.teams)

        if verbose:
            print(f"Fitting Poisson model on {len(df):,} matches, {n} teams...")

        # ── Initial parameter values ──────────────────────────────────────────
        # [attack × n, defense × n, home_advantage, intercept]
        # Small random noise helps the optimizer avoid flat starting points
        init_params = np.concatenate([
            np.random.normal(0, 0.1, n),   # attack params
            np.random.normal(0, 0.1, n),   # defense params
            [0.3],                          # home advantage (positive = helps home)
            [0.0]                           # intercept
        ])

        # ── Constraint: sum of attack params = 0 (identifiability) ───────────
        # Without this, we can freely shift all attack params up and all defense
        # params down by the same amount — infinitely many equivalent solutions.
        constraints = [
            {"type": "eq", "fun": lambda p: np.sum(p[:n])}   # Σ attack = 0
        ]

        # ── Optimise ──────────────────────────────────────────────────────────
        result = minimize(
            fun=self._neg_log_likelihood,
            x0=init_params,
            args=(df,),
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 200, "disp": verbose}
        )

        if verbose:
            print(f"  Optimisation status: {'Success ✓' if result.success else 'Failed ✗'}")
            print(f"  Final neg-log-likelihood: {result.fun:.2f}")

        # ── Store fitted parameters ───────────────────────────────────────────
        self.params    = result.x
        n_teams        = len(self.teams)
        attack_p       = result.x[:n_teams]
        defense_p      = result.x[n_teams:2*n_teams]
        self.home_adv  = result.x[2*n_teams]
        self.intercept = result.x[2*n_teams + 1]

        self.attack  = {t: attack_p[i]  for i, t in enumerate(self.teams)}
        self.defense = {t: defense_p[i] for i, t in enumerate(self.teams)}
        self.is_fitted = True

        return self

    # ── Predict lambdas ───────────────────────────────────────────────────────
    def predict_lambda(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False
    ) -> Tuple[float, float]:
        """
        Return expected goals (λ_home, λ_away) for a given matchup.

        Parameters
        ----------
        home_team, away_team : str
        neutral : bool
            If True, home advantage is not applied (neutral venue).

        Returns
        -------
        (lambda_home, lambda_away) : Tuple[float, float]
        """
        self._check_fitted()

        ha = self.home_adv if not neutral else 0.0

        log_lh = self.attack[home_team] + self.defense[away_team] + ha + self.intercept
        log_la = self.attack[away_team] + self.defense[home_team] + self.intercept

        return np.exp(log_lh), np.exp(log_la)

    # ── Score probability matrix ───────────────────────────────────────────────
    def score_matrix(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
        max_goals: int = 8
    ) -> np.ndarray:
        """
        Return a (max_goals+1) × (max_goals+1) matrix where
        entry [i, j] = P(home_goals=i, away_goals=j).

        Assumes goals are independent Poisson random variables.
        """
        lh, la = self.predict_lambda(home_team, away_team, neutral)

        home_probs = poisson.pmf(np.arange(max_goals + 1), lh)
        away_probs = poisson.pmf(np.arange(max_goals + 1), la)

        # Outer product: P(home=i, away=j) = P(home=i) × P(away=j)
        return np.outer(home_probs, away_probs)

    # ── Match outcome probabilities ────────────────────────────────────────────
    def predict_match(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
        max_goals: int = 8
    ) -> Dict[str, float]:
        """
        Predict win/draw/loss probabilities for a match.

        Returns
        -------
        dict with keys: 'home_win', 'draw', 'away_win', 'lambda_home', 'lambda_away'
        """
        matrix = self.score_matrix(home_team, away_team, neutral, max_goals)

        home_win = float(np.tril(matrix, k=-1).sum())   # i > j  (home scored more)
        draw     = float(np.trace(matrix))               # i == j
        away_win = float(np.triu(matrix, k=1).sum())     # j > i  (away scored more)

        lh, la = self.predict_lambda(home_team, away_team, neutral)

        return {
            "home_win":    home_win,
            "draw":        draw,
            "away_win":    away_win,
            "lambda_home": lh,
            "lambda_away": la
        }

    # ── Strength table ─────────────────────────────────────────────────────────
    def team_strengths(self) -> pd.DataFrame:
        """
        Return a DataFrame with each team's fitted attack and defense parameters.
        Sorted by overall strength (attack - defense).
        """
        self._check_fitted()
        rows = []
        for team in self.teams:
            rows.append({
                "team":    team,
                "attack":  self.attack[team],
                "defense": self.defense[team],
                "strength": self.attack[team] - self.defense[team]  # higher = stronger
            })
        return (
            pd.DataFrame(rows)
            .sort_values("strength", ascending=False)
            .reset_index(drop=True)
        )

    # ── Internal helpers ───────────────────────────────────────────────────────
    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call .fit(df) first.")
