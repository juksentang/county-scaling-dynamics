PY ?= python3

# LaTeX macro and table files of the numbers script go to build/tex/;
# the figures are written to results/figures/.
export PAPER_K1_TEX_DIR := build/tex
export PAPER_K1_FIG_DIR := results/figures

.PHONY: numbers figures verify test us brazil china-panel

## numbers: every printed value -> results/tables/paper_k1_values.csv (+ build/tex/)
numbers:
	$(PY) scripts/paper_k1_numbers.py

## figures: the thirteen figures -> results/figures/*.pdf
figures:
	PAPER_K1_FIG_OUTDIR=results/figures $(PY) scripts/paper_k1_figures.py

## verify: regenerate both, then compare every output with expected_outputs.sha256
verify: numbers figures
	$(PY) scripts/verify_outputs.py

## test: numbers and figure tests (regeneration vs expected_outputs.sha256) and the tests of the estimation code
test:
	$(PY) -m pytest -q tests

## us / brazil: download the public inputs into data/ and verify their sha256
us:
	$(PY) scripts/fetch_public_data.py us

brazil:
	$(PY) scripts/fetch_public_data.py brazil

## china-panel: data_clean/panel_full.parquet and panel_scaling.parquet from the CSMAR files under data/ (licensed)
china-panel:
	$(PY) -c "from src.data_merger import run_full_pipeline; run_full_pipeline()"
