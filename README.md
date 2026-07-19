# ⚽ Football Match Predictor

A statistical model that predicts international football matches and simulates the FIFA World Cup — built step by step as a personal learning project.

**Tech stack:** Python · Jupyter Notebooks · Pandas · Scipy · Matplotlib

---

## 🧠 How It Works

The model is built in three layers:

```
Historical Results (49,000 matches)
        ↓
   Elo Ratings         ← strength score for every team
        ↓
 Poisson Regression    ← predict expected goals per match
        ↓
 Monte Carlo Sim       ← simulate each match 10,000 times
        ↓
  Win / Draw / Loss Probabilities + World Cup Bracket Odds
```

---

## 📂 Project Structure

```
football-match-predictor/
│
├── data/
│   ├── results.csv          # 49,000+ international matches since 1872
│   └── shootouts.csv        # penalty shootout results
│
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory Data Analysis
│   ├── 02_elo_ratings.ipynb # Building the Elo rating system
│   ├── 03_poisson_model.ipynb  # Poisson regression for goal prediction
│   └── 04_simulation.ipynb  # Monte Carlo World Cup simulation
│
├── src/
│   ├── elo.py               # Elo rating engine
│   ├── poisson.py           # Poisson model
│   ├── simulate.py          # Monte Carlo simulator
│   └── utils.py             # Helper functions
│
├── outputs/
│   └── team_ratings.csv     # Latest computed Elo ratings
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

Data from [martj42/international_results](https://github.com/martj42/international_results) — ~49,000 official international matches from 1872 to present.

Columns: `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `city`, `country`, `neutral`

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/football-match-predictor.git
cd football-match-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# Place results.csv and shootouts.csv in the data/ folder
# Or run: python src/utils.py --download

# 4. Open notebooks in order
jupyter notebook notebooks/01_eda.ipynb
```

---

## 📈 Progress

- [x] Data collection & EDA
- [ ] Elo rating system
- [ ] Poisson regression model
- [ ] Monte Carlo simulation
- [ ] World Cup 2026 predictions

---

## 📚 What I'm Learning

| Module | Concepts |
|---|---|
| EDA | Pandas, data cleaning, visualisation |
| Elo | Iterative rating systems, K-factors |
| Poisson | Probability distributions, MLE |
| Monte Carlo | Sampling, simulation, uncertainty |

---

## References

- Dataset: [martj42/international_results](https://github.com/martj42/international_results)
- Dixon & Coles (1997) — *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*
- Elo, A.E. (1978) — *The Rating of Chess Players, Past and Present*
