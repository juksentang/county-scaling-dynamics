# Numerical verification record

Every numeric claim in the manuscript and its supplementary information was
checked, one at a time, against the output of the analysis pipeline in this
repository. This folder is that record: the inventory of claims, the verdict
for each, and the list of values that changed as a result.

We publish it because the paper's reproducibility statement is only as good as
the evidence behind it. Readers who want to know whether a particular printed
value comes out of the code can look it up here by file and line.

---

## How the inventory was built

1. **Extraction.** Every numeric claim in the eight manuscript source files
   (abstract, introduction, results, discussion, conclusion, methods,
   back matter, supplementary information) and in the table bodies was
   extracted with its file, line, surrounding sentence, and — where the text
   attributes it — its stated source. Table cells were enumerated individually.
   Years used as dates, equation and section numbers, and numbers belonging to
   cited literature were excluded from the claim set (the last are retained but
   marked). Result: **509 claims**.

2. **Reference values.** The pipeline was run from a fixed commit to produce a
   reference value for every quantity the manuscript reports: regression
   coefficients and standard errors, sample and transition-pair counts,
   estimator outputs, bootstrap intervals, test statistics, and descriptive
   tables. These are the CSVs under `results/tables/` in this repository.

3. **Reconciliation.** Each claim was compared with its reference value at the
   precision printed in the manuscript.

## Verdict categories

| Verdict | Meaning | Count |
|---|---|---|
| `clean` | Reproduces exactly at the printed precision | 277 |
| `rounding` | Differs only by rounding at the printed precision | 81 |
| `stale` | A reference value exists and differs beyond rounding | 74 |
| `broken` | No reproducible provenance in the pipeline | 71 |
| `out_of_scope` | Cited from other work, or a qualitative descriptor | 6 |
| | **Total** | **509** |

**70% of the manuscript's numbers (`clean` + `rounding`) reproduced as printed.**

The `stale` entries are dominated by two families whose estimators were revised
during peer review — the escape-time observables and the multifractal surrogate
statistics — so a single change of definition propagates to every downstream
mention. The `broken` entries are concentrated in three tables whose standard
error and R² columns could not be reproduced under any tested specification;
those tables are now generated programmatically by the code in this repository
rather than transcribed.

## Files

| File | Contents |
|---|---|
| `inventory.csv` | The 509 claims: identifier, source file and line, printed value, surrounding context, claim type, attributed source, whether it sits in a table |
| `verdicts.csv` | One row per claim: verdict, the pipeline's value where one exists, and a note on how it was matched or why it differs |
| `table_r1.csv` | The **145** claims verdicted `stale` or `broken`, with submitted value, corrected value, cause, and location. This is the correction table submitted with the point-by-point response |

`table_r1.csv` is exactly the subset of `verdicts.csv` with verdict `stale` or
`broken`; both are generated from the same reconciliation, so the correction
list is complete by construction rather than by inspection.

## How to use this record

- To check a specific printed value: find it in `inventory.csv` by
  `source_file` and `source_line`, then look up its `id` in `verdicts.csv`.
- To see everything that changed between the submitted and revised versions:
  read `table_r1.csv`.
- To recompute any reference value yourself: run `make numbers` at the
  repository root; the CSVs it writes under `results/tables/` are the values
  the reconciliation used.

## Standing practice

Values now enter the manuscript only through programmatic generation from a
fixed pipeline version: table bodies are written by `scripts/make_tex_tables.py`
into LaTeX include files, and no number is transcribed by hand. The failure
mode this record documents — values copied from analysis output at one stage of
a long revision, while the code continued to evolve — is removed by
construction.
