# county-scaling-dynamics

Code, result tables and figures for

> Juk-Sen Tang, Jing Gong, Lingyi Wu and Junhong Chen,
> *Unit turnover and yearly standardization create spurious attractor shifts
> in Langevin inference on open panels* (version 2.0.0 of this archive).

When an index is standardized each year over the units observed that year,
units entering and leaving the panel force a composition term on every unit
that stays, and persistence turns that term into a spurious shift of the
estimated attractor. The paper derives the term exactly, shows in simulations
that restandardizing on the stayers (V1) removes it, and measures it on three
open panels: Chinese counties, US metropolitan areas and Brazilian
municipalities. A controlled-masking experiment on the complete US and
Brazilian panels keeps every trajectory fixed and changes only which units
are observed in which years: the change of observational composition alone
moves the uncorrected (V0) attractor, and the experiment tests how much of
that shift V1 removes and how well the frozen projection of the composition
term predicts the V0-V1 difference.

## Layout

| Path | Content |
|---|---|
| `results/tables/` | the result tables every number and figure of the paper is computed from (see `release_manifest.csv` for what each one holds and which script writes it) |
| `results/tables/paper_k1_values.csv` | every number printed in the paper and its Supplementary Information that comes from the analysis: one row per value, with the file, row and column it comes from and the transform applied |
| `results/figures/` | the thirteen figures as vector PDF |
| `results/records/` | where the estimation scripts write their run records (truth checks, prechecks, attempt logs); empty in the archive |
| `scripts/paper_k1_numbers.py` | writes `paper_k1_values.csv` (and LaTeX macro and table files under `build/tex/`) from `results/tables/` |
| `scripts/paper_k1_figures.py` | draws the thirteen figures from `results/tables/` |
| `scripts/fetch_public_data.py` | downloads the US and Brazilian inputs and checks them against the recorded sha256 |
| `scripts/k2_*.py`, `scripts/run_k2.py` | US analysis: coverage and lineages, V0/V1 contrasts, bootstrap, composition predictions |
| `scripts/k2br_*.py`, `scripts/run_k2br.py` | Brazilian analysis: minimum comparable areas, PIB reader, V0/V1 contrasts, bootstrap, optimiser-noise check |
| `scripts/run_k4_mask.py` | controlled masking of the complete US and Brazilian panels (pilot and confirmatory stages; `results/tables/k4_mask/`) |
| `scripts/k1_*.py`, `src/k1_*.py` | simulation study: truth checks, cells, aggregation, gates, exploratory driver; the projection of the composition term (`src/k1_theory.py`) |
| `scripts/run_attrition_diagnostic.py`, `run_coverage_breaks.py`, `run_expi_*.py` | Chinese panel: decomposition, coverage breaks, province-block intervals, paired estimands, identifiability (need CSMAR) |
| `src/config.py`, `data_loader.py`, `data_cleaner.py`, `deflator.py`, `data_merger.py` | construction of the Chinese county panel from the CSMAR files (need CSMAR) |
| `src/saai.py`, `src/km_mle.py`, `src/fokker_planck.py`, `src/scaling.py` | scaling regression and yearly standardization (V0, V1), transition pairs, likelihood fit of the Langevin model |
| `src/k1_sim_calibration.json`, `src/k1_design_processes.json` | calibration constants and registered parameters of the simulated processes |
| `scripts/verify_outputs.py` | `make verify`: compares every output of `make numbers` and `make figures` with `expected_outputs.sha256` |
| `src/paper_k1_style.py` | figure style and the figure lint (font sizes, widths, overlap, greyscale separability) |
| `tests/` | numbers and figure tests; tests of the estimation code on synthetic data, including the symbolic checks of the algebra (`tests/test_k1_algebra.py`) |
| `DATA_ACCESS.md` | where every input comes from, with checksums |
| `expected_outputs.sha256` | sha256 of every output of `make numbers` (including `build/tex/`) and `make figures` |
| `release_manifest.csv` | every file of this archive, its size, sha256 and why it is here (for each table: what it holds, its source, and which script reads it) |

## Figures

| File | In the paper |
|---|---|
| `fig1_mechanism.pdf` | Fig. 1, the mechanism (panels a-c are a labelled fixed-seed illustration drawn by the script, panel d the fitted V1 slopes) |
| `fig2_simulation.pdf` | Fig. 2, the registered simulation test (K1c) and the truth-check status grid |
| `fig3_dose_ladder.pdf` | Fig. 3, observed V0-V1 attractor differences against the composition dose in the three panels |
| `fig4_china_coverage.pdf` | Fig. 4, coverage of the Chinese county panel and the inference trap |
| `fig5_masking.pdf` | Fig. 5, controlled masking of the complete US and Brazilian panels: uncorrected shift, V1 recovery and the frozen projection |
| `figS1_k1a_exploratory.pdf` | Supplementary Fig. S1, exploratory formula test (K1a) |
| `figS2_k1b_controls.pdf` | Supplementary Fig. S2, exploratory null coverage (K1b) and report-only controls |
| `figS3_china_ci.pdf` | Supplementary Fig. S3, Chinese intervals, superseded and primary |
| `figS4_us_detail.pdf` | Supplementary Fig. S4, US contrasts per specification |
| `figS5_guarded.pdf` | Supplementary Fig. S5, guarded and unguarded pooled projections |
| `figS6_brazil_detail.pdf` | Supplementary Fig. S6, Brazilian contrasts per cell |
| `figS7_mask_cells.pdf` | Supplementary Fig. S7, controlled masking, every mask by cell |
| `figS8_mask_guards.pdf` | Supplementary Fig. S8, controlled masking, every guard |

Each figure is written by one function of `scripts/paper_k1_figures.py`
(named after the file). Every plotted value is read from one cell of a table
in `results/tables/` or is a difference, ratio or product of such cells done
in the script; the only simulation is the illustration in Fig. 1a-c.

## Numbers

Every number in the paper is a LaTeX macro. The 1,024 macros of the
analysis are rows of `results/tables/paper_k1_values.csv`: the columns
`source_file` and `source_row_or_column` give the table cell, `transform`
the operation (read, ratio, difference, count, minimum, ...). Nothing in the
numbers script is estimated; a missing file, column or row, or a row filter
that does not match exactly one row, stops it. The dates, commit identifiers
and event counts of the registration history (Supplementary Note S3 and the
chronology tables of the Supplementary Information) describe the authors'
private version-controlled record; they are written by a script of that
record, are not among the rows of `paper_k1_values.csv` and are not part of
this archive.

## Reproducing

Requirements: Python 3.12 and the pinned packages in `requirements.txt`.
The figure style needs the Arial font (`arial.ttf`, `arialbd.ttf`,
`ariali.ttf`, `arialbi.ttf`, not redistributed here): set
`PAPER_K1_ARIAL_DIR` to the directory that holds them (if it is not set,
an Arial known to matplotlib's font manager is used). The figure PDFs are reproduced
byte for byte with the pinned packages (notably matplotlib and fontTools,
which subsets the embedded font) and the Arial build used here:

| File | sha256 |
|---|---|
| `arial.ttf` | `b3658eadae55e682b5f69eb64c439c1ecc8f196c0bb8d4756d145d13bc86476a` |
| `arialbd.ttf` | `e8f4e3baf6cc35fed6fcce3a540e8b39e8f6cda1d22a28f2ec8f526fef7a43f5` |
| `ariali.ttf` | `86b32db9a06f9694e2a3760c42e5117bcdc5cc1255bb5186ca8ce0305e22f288` |
| `arialbi.ttf` | `267a411525980d648f10a7222a6ff1ebe4d4feeefbf9b3100cae2b5cfab5594a` |

Another Arial build or fontTools version changes only the embedded font
bytes of the PDFs, not the plotted content; `make verify` and the figure test
then report the PDFs as different.

```
pip install -r requirements.txt
make numbers    # results/tables/paper_k1_values.csv + build/tex/
make figures    # results/figures/*.pdf
make verify     # runs both and compares every output with expected_outputs.sha256
make test       # numbers, figure and estimation-code tests
```

`make numbers` and `make figures` overwrite the shipped files in place;
`make verify` shows that they reproduce them exactly.
`make test` skips two tests outside a git checkout and the tests that need
the CSMAR-derived presence mask when it is absent (see Tests below).

## Recomputing the result tables

The estimation and data-construction code ships with the archive: the code
that turned the inputs into the tables under `results/tables/`. The public
copy is the analysis code with comments and private-record integrity guards
removed; the build checks that the syntax tree of every public analysis file
equals that of the analysis code up to exactly the following edits (three
small files, `tests/conftest.py`, `src/__init__.py` and
`scripts/verify_outputs.py`, are written for the archive and are not
analysis code):

* comments are removed, and so are docstring paragraphs that refer to the
  authors' private record (docstrings are not part of the syntax-tree
  comparison; a module that reads its own `__doc__` keeps a docstring);
* each call of an integrity guard that checks the authors' private
  version-controlled record (the theory-freeze record, the registration
  files, the approval and signature lines of the authors' plan) is replaced
  by `public_guard_skipped(value)`, a no-op defined at the top of the module
  that returns the value the analysis continues with (for a registration:
  the seed, bootstrap size and input checksums that the script and the
  shipped tables fix; for the freeze of the controlled masking: the seed
  entropy and the frozen guards and criteria recorded in the shipped
  `results/tables/k4_mask/confirmatory/k4_meta.json`); `grep -rn public_guard_skipped` lists every call site;
* string constants that carried labels of the authors' plan (section numbers,
  item labels, amendment labels in Chinese, check codes) are rewritten in
  English, record paths point to `results/records/` and default data paths
  to `data/us/` and `data/brazil/`; where such a string is written into a
  shipped table, the shipped table carries the same wording;
* three identifiers of one cross-check are renamed, and the tests of the
  integrity guards and the tests that compare with files of the private
  record are left out (among the numbered tests of
  `tests/test_k1_algebra.py`: B19 and B21).

Run records that the scripts write (truth-check and precheck records,
attempt logs) go to `results/records/`. Most scripts refuse to write to
`results/tables/` from a working tree with uncommitted changes, the
truth-check scripts commit their records, and the `--dev` paths still read
`git status`: every script needs this archive as a git checkout. Use a git
clone of the repository; in a tree unpacked from the archive zip, run
`git init -q && git add -A && git commit -qm archive` first. Commit after
every step that writes into the repository, as in the blocks below.

Reruns reproduce the shipped CSV tables. The shipped meta JSON files are
reduced (run-specific keys such as the commit, start time, run time and the
checks of the private record are removed), so a rerun writes them with the
full set of keys and they are not expected to match byte for byte;
`results/tables/k2/k2_boot_draws.csv.gz` reproduces in content (its gzip
header carries the time it was written).

### United States and Brazil (public inputs)

```
make us                                       # data/us/: BEA and OMB files, sha256 checked
python3 scripts/k2_coverage.py --data data/us # k2_coverage_* (Panel A/B, lineages)
git add -A && git commit -qm "rerun: US coverage"
python3 scripts/run_k2.py --real --allow-rerun "public rerun" --jobs 8
git add -A && git commit -qm "rerun: US contrasts"
python3 scripts/k2_literal_noise_summary.py
git add -A && git commit -qm "rerun: US literal-draw summary"

make brazil                                   # data/brazil/: IBGE files, sha256 checked
python3 scripts/k2br_coverage.py --data data/brazil
python3 scripts/k2br_pib.py --existence --data data/brazil
git add -A && git commit -qm "rerun: Brazil coverage"
python3 scripts/run_k2br.py --real --workers 8
git add -A && git commit -qm "rerun: Brazil contrasts"
python3 scripts/k2br_optimizer_noise.py
git add -A && git commit -qm "rerun: Brazil optimiser noise"
python3 scripts/k1_guard_sensitivity.py --us-data data/us --br-data data/brazil
```

`run_k2.py` and `run_k2br.py` first rerun the coverage scripts and stop
unless they reproduce the shipped coverage tables byte for byte, and check
every input against the recorded sha256. With 1,000 bootstrap draws the US
run took about 50 minutes and the Brazilian run about 2.2 hours (`runtime_s`
in `results/tables/k2/k2_meta.json` and `results/tables/k2br_meta.json`).
`run_k2br.py --real` always draws 1,000 bootstrap samples and writes to
`results/tables/`; its `--n-boot` and `--out` options apply to `--synthetic`
only and are ignored with `--real`. `--synthetic` runs either pipeline on a
toy panel in under a minute.

### Controlled masking (public inputs)

`scripts/run_k4_mask.py` masks the complete US and Brazilian panels: it
needs `data/us/` and `data/brazil/` (`make us`, `make brazil`) and the
shipped tables it checks against (input sha256, complete-panel Panel A
rows); the US and Brazilian steps above need not be rerun first.

```
python3 scripts/run_k4_mask.py --dev --out ../k4_dev             # synthetic panels only, outside results/
python3 scripts/run_k4_mask.py --preflight pilot                 # stopping rules 1-4, no masks, no seeds
python3 scripts/run_k4_mask.py --pilot --workers 8               # US, 18 masks
git add -A && git commit -qm "rerun: masking pilot"
python3 scripts/run_k4_mask.py --preflight confirmatory
python3 scripts/run_k4_mask.py --confirmatory --workers 32       # US and Brazil, 720 masks
git add -A && git commit -qm "rerun: masking confirmatory"
```

A stage refuses an uncommitted tree and runs only after a passing preflight
of the same stage at the same commit in the same environment; the preflight
writes its receipt outside the repository (`$K4_BUILD_DIR/preflight`,
default `~/econp_mask_build/preflight`). Each stage is a single run: its
attempt is logged in `results/records/k4_mask_attempts.jsonl`, and a second
attempt of the same stage is refused while that entry is there. The pilot
uses a fixed entropy written in the script; the confirmatory stage reads the
entropy of its seeds and the frozen guards and criteria from the shipped
`results/tables/k4_mask/confirmatory/k4_meta.json` and redraws every mask
identical to one of the shipped pilot masks (`mask_hash` in
`results/tables/k4_mask/pilot/k4_masks.csv`), so a rerun draws the masks of
the shipped tables.

Stopping rule 2 requires the complete-panel cubic attractor to reproduce the
shipped Panel A row to 1e-12, which holds only on the numerical stack that
produced it. The confirmatory run was made with one thread per process (the
script sets `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to 1) and with

```
export OPENBLAS_CORETYPE=Haswell
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"
```

On another linear-algebra kernel or vector instruction set the rebuilt
attractor can differ by far less than the effects reported but by more than
the tolerance, and the preflight stops (Supplementary Note S7); a failed
preflight writes nothing under `results/` and does not use the run. The
pilot took about 11 minutes on 8 workers and the confirmatory run about 86
minutes on 32 workers (`seconds` and `workers` in the shipped `k4_meta.json`
files; the Brazilian panel dominates). `k4_draws.csv.gz` reproduces in
content (its gzip header carries the time it was written).

### China (needs the CSMAR database)

The county panel is built from the CSMAR China County Statistical Database
(`DATA_ACCESS.md`), which cannot be redistributed, and the list of the 832
national poverty counties with their exit years (`DATA_ACCESS.md`; used for
the poverty indicator columns of the panel). With these Excel files under
`data/` (file and sheet names in `src/config.py`):

```
make china-panel                              # data_clean/panel_full.parquet, panel_scaling.parquet
python3 scripts/run_attrition_diagnostic.py   # expi_attrition_*
python3 scripts/run_attrition_diagnostic.py --degenerate limit
python3 scripts/run_coverage_breaks.py        # k1_coverage_province_year, k1_leaver_provinces, data_clean/k1_mask.parquet
python3 scripts/run_expi_rerun_ci.py          # expi_rerun_ci_*
python3 scripts/run_expi_paired.py            # expi_paired_* (reads expi_rerun_ci_variants.csv)
python3 scripts/run_expi_identify.py          # expi_identify_* (rebuilds the draws of expi_paired_draws.csv.gz)
```

The province-block bootstraps (1,000 draws) of `run_expi_rerun_ci.py`,
`run_expi_paired.py` and `run_expi_identify.py` took about 9, 3 and 3
minutes on 8 workers (`seconds` and `workers` in their meta files). The
shipped `k1_leaver_provinces.csv` leaves out the column `mean_xi_leave`,
which the script writes (some cells hold one county). `scripts/k1_calibrate.py`
writes the calibration constants of the simulation
(`src/k1_sim_calibration.json`, shipped) from the same panel.

### Simulation study

The simulated panels have the county, province and window structure of the
Chinese panel: every cell reads the county-by-year presence mask
`data_clean/k1_mask.*`, which `scripts/run_coverage_breaks.py` writes from
the CSMAR panel and which is not shipped (it is derived from CSMAR). The
simulator refuses a mask whose content hash differs from the one recorded
in `src/k1_sim.py`. With the mask in place, the registered order is
enforced by the scripts: a truth check (S0) per process and pipeline, then
the coverage precheck on the cells without exit, then the exit cells.

```
python3 scripts/k1_s0.py start  --process cubic_shift --pipeline saai     # commits a 'started' entry (results/records/)
python3 scripts/k1_s0.py run    --process cubic_shift --pipeline saai --replicates 0:500 --out RAW --workers 8
python3 scripts/k1_s0.py record --process cubic_shift --pipeline saai --raw RAW   # writes and commits the S0 record
python3 scripts/k1_run_cells.py ntasks --cells 23-25
python3 scripts/k1_run_cells.py run --cells 23-25 --task I --out CHUNKS    # for I in 0..ntasks-1
python3 scripts/k1_run_cells.py aggregate --in CHUNKS --out results/tables/k1_k1c_f2
git add -A && git commit -qm "rerun: K1c aggregate"
python3 scripts/k1_gates.py --tables results/tables/k1_k1c_f2
python3 scripts/k1_exploratory.py ntasks; python3 scripts/k1_exploratory.py run --task I; python3 scripts/k1_exploratory.py aggregate
```

`--replicates r0:r1` is a half-open range of replicate indices (default
`0:500`, the registered R); a run can be split into chunks with the same
`--out`. `k1_s0.py --help` and the module docstrings give the other options
(the precheck cells are `--cells a`; `--dev` and `--non-registered` give
labelled runs of a few replicates).

Each step reads the records of the steps before it, and the archive ships
none of these records (they carry the commit identifiers of the authors'
private record): `k1_gates.py` reads the aggregate meta `k1_sim_meta.json`
that `k1_run_cells.py aggregate` writes next to the tables and the S0
records under `results/records/`; `k1_exploratory.py` also needs the
coverage-precheck aggregate (cells `a`, aggregated to
`results/records/k1_precheck_a_registered_f2/`). The shipped tables under
`results/tables/k1_k1c_f2/` and `results/tables/k1_exploratory/` can
therefore not be fed to `k1_gates.py` or the exploratory driver on their
own: those steps are reproduced by rerunning the chain from the S0 checks
(every process and pipeline the cells use), the precheck and the cells.

The simulation tables reproduce to the tolerance of the optimiser, not byte
for byte: a full rerun of the S0 truth checks of cubic/saai and cubic/direct
from this archive gave the same verdicts as the shipped `k1_gates.csv`, with
median biases within 1.3e-4 of the shipped ones (at most 0.04% of the
tolerance) and the closed-form lag-one autocorrelation agreeing to 1e-16.

Each cell has 500 replicates; the cells whose
intervals are evaluated use 200 bootstrap draws, the null and control cells
none. The registered K1c run (cells 23-25) took 50 core-hours, the
exploratory K1a/K1b run (20 cells) 304 core-hours and the report-only
controls 49 core-hours, as array jobs of five replicates per task
(`core_hours_wall` in the run records under `results/tables/k1_k1c_f2/` and
`results/tables/k1_exploratory/`). The neural estimator (a conditional
module that was not run) is not part of the archive.

### Tests

`make test` runs the numbers and figure tests and the tests of the
estimation code on synthetic data: the symbolic checks of the composition
algebra (`tests/test_k1_algebra.py`, sympy), the truth-check, aggregation,
gate and exploratory-driver tests, the US, Brazilian and Chinese
pipeline tests on synthetic panels, and the controlled-masking tests
(`tests/test_k4_mask.py`: masks, seeds, stopping rules, the projection API
and the evaluation, on synthetic panels). The simulator tests
(`tests/test_k1_sim.py`) simulate on the CSMAR-derived presence mask and are
collected only when `data_clean/k1_mask.*` exists; four truth-check tests in
`tests/test_k1_algebra.py` (B25, B25b, B28 and the failed-replicate test)
that also need the mask are skipped without it. Two tests run a script's refusal of an
uncommitted tree and are skipped outside a git checkout. Tests of the
integrity guards of the private record are not part of the archive.

## Earlier versions

Versions 1.x of this archive accompanied an earlier analysis of the same
Chinese county panel, which compared Langevin fits of the yearly
standardized agricultural index between two windows. The paper reports a
composition diagnostic of that window comparison, registered before it was
run (Supplementary Note S4; provenance in Supplementary Note S8): the
uncorrected (V0) attractor shift is consistent with composition and
disappears after restandardizing on the stayers (V1), and with
province-clustered intervals neither the V0 shift nor the V0-V1 difference
is resolved. Version 2.0.0 accompanies the paper above; its code,
tables and figures do not depend on versions 1.x.

## Licence and citation

Code: MIT (`LICENSE`). Tables and figures: CC-BY-4.0 (`LICENSE-data`). The
US inputs are public-domain products of BEA and OMB; the Brazilian inputs are
IBGE public statistics, used with attribution; the Chinese county data are
licensed from CSMAR and are not redistributed (`DATA_ACCESS.md`). Please cite
the paper and this archive (`CITATION.cff`).
