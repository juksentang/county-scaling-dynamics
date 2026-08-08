# Scaling exponents, escape dynamics and multifractality in county economies

Analysis code for *Deepening before dissolution: non-monotonic escape dynamics
in the scaling exponents of county economies*.

The study treats the effective scaling exponent β_eff of the power law
`Y ~ N^β` as an order parameter, estimates its crossover as a function of
urbanisation, and models county-level deviations from the scaling benchmark as
a one-dimensional Langevin process whose drift and diffusion are estimated
non-parametrically (Kramers–Moyal) and parametrically (Euler pseudo-likelihood).
Data cover roughly 2,300 Chinese counties, 2000–2023.

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Reproduce every reported statistic (~1 h; QUICK=1 shortens the bootstraps)
make numbers

# Regenerate the figures
make figures
```

`make numbers` writes to `results/tables/`; `make figures` writes to
`results/figures/`. Both require the analysis panel in `data_clean/` — see
**Data** below.

Individual stages:

| Target | Produces |
|---|---|
| `make core` | Scaling regressions, SAAI statistics, mixture models, descriptive tables |
| `make escape` | Transition-pair counts, drift/diffusion estimates, bootstrap intervals, grain-tonnage robustness, quasi-DiD |
| `make attribution` | State-conditioned regressions, placebo tests, endogenous break search, event-time coefficients |
| `make multifractal` | Per-year MFDFA, surrogate tests, fixed-sample-size resampling |
| `make io` | Sector interaction-intensity measures from input–output tables |
| `make urbanisation` | Boundary matching and the nightlights crossover replication |

---

## Notebooks

The notebooks walk through the paper section by section, print every reported
value next to its published counterpart, and call the same functions the
command-line pipeline uses. They ship executed, so the numbers can be read
without running anything.

| Notebook | Reproduces |
|---|---|
| `01_panel_construction.ipynb` | Panel assembly and coverage (Methods) |
| `02_scaling_and_crossover.ipynb` | Sectoral exponents, urbanisation crossover, satellite replications |
| `03_escape_dynamics.ipynb` | Drift, diffusion, escape times, dual-estimator inference |
| `04_attribution.ipynb` | Quasi-DiD, state-conditioned margins, break timing |
| `05_multifractal.ipynb` | Cross-sectional DFA and multifractal width |

Long-running steps (bootstraps, surrogate tests) are read from
`results/tables/` rather than recomputed; each such cell names the script that
produces the file.

---

## Layout

```
src/                     Analysis library
  config.py              Paths, period definitions, estimator constants
  data_loader.py         Source workbook readers
  data_cleaner.py        Type coercion, county-code standardisation
  deflator.py            Province-level GDP deflators
  data_merger.py         Panel assembly
  saai.py                Scaling-adjusted anomaly index (the state variable ξ)
  scaling.py             Scaling regressions, interaction models, sigmoid fits
  fokker_planck.py       Kramers–Moyal estimation, potentials, escape observables
  km_mle.py              Parametric Euler pseudo-likelihood estimator
  distribution.py        Mixture models, effective potentials
  landau.py              Landau parameterisation of annual cross-sections
  dfa.py                 Detrended fluctuation analysis and MFDFA
  robustness.py          Bootstraps, placebo tests, quasi-DiD
  spatial.py             Spatial autocorrelation and correlation lengths
  early_warning.py       Critical-slowing-down indicators
  io_linkage.py          Input–output linkage indicators
  plotting.py            Figure helpers

scripts/                 Command-line entry points (see the Makefile)
notebooks/               Executed, paper-aligned walkthroughs
audit/                   Verification record for every printed value
results/tables/          Derived statistics (CSV)
results/figures/         Generated figures
```

---

## Data

Primary data are **not** included: the county statistical database is licensed
commercially and several boundary layers are supplied under non-redistributable
terms. `DATA_SOURCES.md` lists every source, its licence, and how to obtain it,
together with the exact field mappings needed to rebuild the panel.

Analyses read `data_clean/panel_scaling.parquet` (the analysis panel, 19,392
county-years). `notebooks/01_panel_construction.ipynb` builds it from the source
workbooks.

Two conventions that matter when reading the code:

- The analysis panel, not the full panel, is the input to every reported result.
- The post-campaign window (2020–2023) is **not identifiable** from this panel:
  the state variable is standardised within annual cross-sections, so selective
  exit from the sample injects drift into the surviving counties. Estimates for
  that period are reported as such and no dynamical claim is based on them.

---

## Reproducibility

Every number in the paper is produced by `make numbers` from a fixed commit of
this repository; nothing is transcribed by hand.

`audit/` holds the verification record behind that statement: all 509 numeric
claims in the manuscript and supplement, the verdict for each against the
pipeline's output, and the list of values corrected during peer review.
See `audit/README.md`.

Random seeds are set explicitly in each script (root seed 20260806), and
bootstraps resample counties rather than county-years so that within-county
dependence is respected.

---

## Licence and citation

Code is MIT-licensed; derived tables and figures are CC BY 4.0 (`LICENSE`).
Input data retain the licences listed in `DATA_SOURCES.md`.

If you use this code, please cite the paper and the data sources it depends on.
