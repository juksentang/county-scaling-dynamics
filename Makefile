# One-command reproduction of every reported statistic and figure.
#
#   make numbers          regenerate all analysis tables  -> results/tables/
#   make figures          regenerate all figures          -> results/figures/
#   make check            figures carry no withdrawn or superseded value
#   QUICK=1 make numbers  reduced bootstrap sizes (smoke run)
#
# Inputs: the analysis panels in data_clean/ (built by
# notebooks/01_panel_construction.ipynb).  The urbanisation stage
# additionally needs the Earth Engine exports in data/gee_exports/
# (scripts/gee_export_urbanization.py, scripts/download_gee_urbanization.py).
# See DATA_SOURCES.md for how to obtain the primary data.

PY := python3
QUICKFLAG := $(if $(QUICK),--quick,)

.PHONY: all numbers figures check \
        core escape attribution multifractal io urbanisation textables

all: numbers figures

numbers: core escape attribution multifractal io urbanisation textables
	@echo "== all analysis tables regenerated in results/tables/ =="

# Scaling regressions, SAAI variants and group statistics, mixture models,
# crossover fits with bootstrap intervals, descriptive statistics.
core:
	$(PY) scripts/run_core_statistics.py

# Kramers-Moyal drift and diffusion estimation: transition-pair counts,
# escape observables with county-block bootstrap intervals, the
# physical-tonnage diffusion check, and the quasi difference-in-differences
# estimate.
escape:
	$(PY) scripts/run_escape_dynamics.py --steps counts,km,grain,did $(QUICKFLAG)
	$(PY) scripts/decompose_escape_time.py

# Attribution designs: state-conditioned regressions, pre-period placebo,
# endogenous break search, event-time coefficients.
attribution:
	$(PY) scripts/run_attribution_designs.py
	$(PY) scripts/run_attribution_extras.py

# Multifractal analysis: per-year MFDFA, phase-randomised surrogate tests,
# fixed-sample-size resampling.
multifractal:
	$(PY) scripts/run_mfdfa_surrogates.py
	$(PY) scripts/run_mfdfa_fixedn.py

# Sector interaction-intensity measures from national input-output tables.
io:
	@$(PY) -c "import sys; sys.path.insert(0,'.'); from src.io_linkage import IO_DIR; sys.exit(0 if IO_DIR.exists() else 1)" 2>/dev/null || \
	  { echo "SKIP io: input-output workbooks not found - see DATA_SOURCES.md"; exit 0; }
	$(PY) scripts/run_io_linkage.py

# GDP-independent urbanisation measure from night-time lights.
urbanisation:
	@test -e data/gee_exports/econp_ntl_chunk0.csv || \
	  { echo "SKIP urbanisation: data/gee_exports missing — see DATA_SOURCES.md"; exit 0; }
	$(PY) scripts/build_county_boundaries.py
	$(PY) scripts/run_crossover_nightlights.py

# LaTeX table bodies for the manuscript tables (results/tex_tables/*.tex).
textables:
	$(PY) scripts/compare_panels.py
	$(PY) scripts/make_headline_values.py
	$(PY) scripts/make_grouped_attractors.py
	$(PY) scripts/make_tex_tables.py

# Ownership is checked before and after. Two generators writing one figure is
# how a superseded figure survived a rebuild in an earlier version of this
# archive, so the build now refuses to complete when that happens.
figures:
	$(PY) scripts/check_figure_owners.py
	@date +%s > .figbuild.stamp
	$(PY) scripts/generate_figures.py
	$(PY) scripts/make_fig_kramers.py
	$(PY) scripts/make_si_figures.py
	$(PY) scripts/check_figure_owners.py --since $$(cat .figbuild.stamp)
	@rm -f .figbuild.stamp
	@echo "== figures regenerated in results/figures/ =="

# Figures must not carry a value the analysis has withdrawn or superseded.
check:
	$(PY) scripts/audit_figure_text.py
