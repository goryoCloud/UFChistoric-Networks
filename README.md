# UFC Historic Networks

This repository develops a **complex network approach to Ultimate Fighting Championship (UFC) history**. Fighters are represented as nodes and bouts as edges. The goal is to capture the structural, temporal, and dynamical properties of the UFC competition landscape over time.

The project provides:

* **Network construction**: scripts to transform bout lists into encounter networks (directed or undirected, full roster or ranked subsets).
* **Temporal metrics**: rolling and windowed measures such as Jaccard overlap of ranked sets, temporal degree distributions, and evolving network statistics.
* **Per-fighter analytics**: individual metrics (degree, clustering, centrality, lifetime statistics) tracked over time.
* **Ranking and champion analyses**: specialized modules for champions, divisions, and pound-for-pound (P4P) lists.
* **Simulations**: stochastic models of roster evolution and encounter dynamics to compare real-world properties with synthetic baselines.
* **Correlation with external signals**: scripts linking network metrics to Pay-Per-View (PPV) and Google Trends data, probing the relationship between network dynamics and popularity.
* **Figures and artifacts**: plotting scripts that reproduce the figures used in ongoing scientific analysis.

The repository serves both as an **academic exploration of complex networks in sports** and a **technical toolkit for combat sports analytics**.

---

## Repository Structure

* **FIGURES/** — exported plots of network metrics, Jaccard curves, degree distributions, etc.
* **SIMULATION RESULTS/** — raw outputs from simulation experiments (CSVs and DATs with averages and standard deviations).
* **fighterTrends/** — Google Trends pulls and cached JSON/CSV data for fighters.
* **paperImages/** — figure assets used in scientific manuscripts.
* **data files** (`data.csv`, `fights2.txt` … `fights6.txt`, `.dat` outputs) — bout lists, processed statistics, and simulation aggregates.
* **Python scripts** — the core of the project, grouped into categories below.

---

## Installation

Python 3.10 or later is recommended.

```
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -U pip
pip install networkx numpy pandas scipy matplotlib tqdm
pip install pytrends requests   # if running Google Trends correlation
```

---

## Workflow and Quickstart

1. **Build the base network**
   Run `2network.py` on `data.csv` to construct the encounter network.

   Example:
   `python 2network.py --input data.csv --out results/graph.parquet`

2. **Compute temporal metrics**
   Use `evolvingJaccard.py` or `rollingJaccard.py` to calculate rolling Jaccard overlaps of ranked sets, specifying window size and frequency.

   Example:
   `python evolvingJaccard.py --input data.csv --window 24 --freq M --out results/jaccard_2y_monthly.csv`

3. **Per-fighter metrics**
   Compute statistics per fighter:

   * `metricsPerFighter.py` for general metrics.
   * `plotMetricsPerFighter.py` to visualize trajectories.

   Example:
   `python metricsPerFighter.py --input data.csv --out results/metrics_per_fighter.csv`

4. **Ranking and champion subsets**
   Analyze only ranked fighters with `rankedNetwork.py` and `metricsRanked.py`, or champions with `Champions.py`.
   Division-specific subsets are available via `peleitasRankedByDivision.py` and `jaccardRankedSetsByDivision.py`.

5. **Run simulations**
   The `simulation.py` series (`simulation.py` through `simulationV6.py`) generates synthetic networks under different parameterizations.
   Later versions include refined features; `simulationV6.py` is the most complete.

   Example:
   `python simulationV6.py --config configs/sim_v6.yaml --out "SIMULATION RESULTS/sim_v6.csv"`

6. **Reproduce figures**
   Use plotting scripts (`plottingEvolvingJaccard.py`, `plotMetricsPerFighter.py`, etc.) to regenerate figures in `FIGURES/` or `paperImages/`.

---

## Data Overview

* **data.csv** and **fights*.txt*\*: Bout lists with fighter names, dates, outcomes, and sometimes weight categories.
* **jaccard\_results\_rolling2year\_monthly.csv**: Precomputed temporal overlap data used in Jaccard visualizations.
* **average\_and\_std\_degreeDist.dat**, **degreeDist.dat**: Degree distribution statistics from real networks and simulations.
* **averages\_and\_std\_devsV*.dat*\*: Aggregated simulation results across multiple runs.

Future work: a `DATA.md` describing schemas (column names, formats, examples).

---

## Script Reference

### Network construction and exports

* **2network.py** — Constructs the global encounter network.
* **peleita.py** — Utilities for single-fight data structures.
* **peleitasDirected.py** — Directed encounter network (winner → loser).
* **encounterNetworkEPS.py** — Exports encounter network visualization (EPS format).

### Temporal metrics and Jaccard analyses

* **evolvingJaccard.py** — Rolling Jaccard overlap of ranked sets over sliding windows.
* **plottingEvolvingJaccard.py** — Plots temporal Jaccard curves.
* **rollingJaccard.py** — Alternative rolling implementation.
* **jaccardRankedSets.py** — Jaccard overlap restricted to ranked subsets.
* **jaccardRankedSetsByDivision.py** — Division-specific Jaccard overlap.

### Per-fighter analyses

* **metricsPerFighter.py** — Degree, clustering, and other per-fighter metrics.
* **metricsPerFighterExtractions.py** — Extended per-fighter statistics.
* **plotMetricsPerFighter.py** — Visualization of per-fighter metrics over time.
* **fightersPerYear.py** — Counts active fighters by year.
* **fightersPerYearAverage.py** — Annual averages of fighter metrics.

### Rankings and champions

* **Champions.py** — Extracts champions per division and aligns with network metrics.
* **PFP.py** — Pound-for-pound metrics.
* **peleitasRanked.py** — Ranked-only network.
* **peleitasRankedByDivision.py** — Ranked, per-division networks.
* **rankedNetwork.py** — Network of ranked fighters.
* **rankedNetworkGoogle.py** — Ranked network integrated with Google Trends.
* **metricsRanked.py** — Metrics for ranked networks.
* **metricsRankedExtractions.py** — Extended metrics for ranked networks.
* **rankedAndChampions.py** — Combined ranked and champion analyses.

### Winners, losers, and weight divisions

* **networkFightWinners.py** — Directed networks built from winners → losers.
* **networkFightWinnersLoosers.py** — Variants including both winners and losers.
* **weight category.py** — Weight division analyses (note: file name contains a space).

### Simulations

* **simulation.py** through **simulationV6.py** — Stochastic simulation models of roster and encounter evolution.
* Early versions test parameter settings; `simulationV6.py` includes refined logic.
* Associated `.dat` files contain averages and standard deviations across runs.

### External signals and correlations

* **metricsPPVGoogleCorr.py** — Correlation between network metrics, PPV sales, and Google Trends.
* **google.py** — API interface to Google Trends.
* **trendsDownload.py** — Downloads Google Trends data.
* **metricsPerFighterWIKI.py** — Integrates Wikipedia page metrics with fighter data.

### Figures and outputs

* Scripts ending in `plot*.py` or `*EPS.py` generate figures used in the `FIGURES/` and `paperImages/` folders.

---

## Reproducibility Roadmap

1. **Input data preparation**
   Place raw bout lists in `data/`. Check formatting against `data.csv`.

2. **Network build**
   Run `2network.py` to generate base network objects.

3. **Metric extraction**
   Use `metricsPerFighter.py`, `metricsRanked.py`, or temporal scripts to extract network measures.

4. **Simulation runs**
   Run `simulationV6.py` with appropriate configuration. Collect `.dat` files.

5. **Figure generation**
   Run plotting scripts to regenerate time series, distributions, and EPS diagrams.

6. **Correlation analysis**
   Use `metricsPPVGoogleCorr.py` to test alignment with PPV/Trends.

---

## Author and Permissions

Author: **Maximiliano S. Castillo**

This codebase is released for **research and educational purposes only**. Redistribution, modification, or reuse is permitted provided that proper citation is given to the author. For any commercial or derivative use, please contact the author in advance to obtain explicit permission.


---

## How to Cite

If you use this repository or results derived from it in your research, please cite the associated preprint:

**BibTeX:**

```
@article{silva2025ufcnetworks,
  title     = {Network Dynamics in Mixed Martial Arts: A Complex Systems Approach to UFC Competition Insights},
  author    = {Silva Castillo, Maximiliano and Muñoz, Víctor and Calderón, Francisco and Velázquez, Luisberis and Fraczinet, Gastón},
  journal   = {arXiv preprint arXiv:2502.07020},
  year      = {2025},
  url       = {https://arxiv.org/abs/2502.07020}
}
```

**Plain text:**
Silva Castillo, M., Muñoz, V., Calderón, F., Velázquez, L., & Fraczinet, G. (2025). *Network Dynamics in Mixed Martial Arts: A Complex Systems Approach to UFC Competition Insights*. arXiv:2502.07020. [https://arxiv.org/abs/2502.07020](https://arxiv.org/abs/2502.07020)

---



