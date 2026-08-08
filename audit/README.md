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

**70% of the manuscript's numbers (`clean` + `rounding`) reproduced as printed**
(359 of 509 after the corrections below were finalised).

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
| `table_r1.csv` | The **150** corrections: submitted value, corrected value, cause, and location. This is the correction table submitted with the point-by-point response |
| `table_r1_applied_check.csv` | One row per correction, recording whether the revised manuscript carries it. Produced by walking `table_r1.csv` backwards against the manuscript source |
| `claim_assertions.csv` | Conditions over generated files, each paired with a phrase the manuscript may not contain while the condition holds. Checks the direction of a claim, not only its numbers |

`table_r1.csv` began as exactly the subset of `verdicts.csv` verdicted `stale`
or `broken`; both come from the same reconciliation, so the correction list is
complete by construction rather than by inspection. Five entries were added
after the first release and one removed, for reasons given below.

### Causes

| Cause | Count |
|---|---|
| Values in tables with no reproducible provenance, now generated programmatically | 69 |
| Values transcribed from earlier drafts during successive rounds of restructuring | 53 |
| Values changed by an error found in the escape-time implementation | 17 |
| Values changed by a corrected estimator or definition | 4 |
| Values where one sentence had mixed two estimators, now reported under one | 3 |
| A quantity computed on the coverage panel rather than the analysis panel | 1 |
| Claims withdrawn as not identified | 3 |

The implementation errors are kept separate from the estimator corrections
deliberately. The first are mistakes — a barrier located at the edge of the
estimation grid instead of on the escape path, and a double integral that mixed
two conventions for the potential — and filing them under a softer label would
make this table less useful, not more.

### Checking that claims match the direction of their evidence

The check above walks a list of printed values. Everything it can see is a
number, and a statement whose *direction* contradicts its evidence passes it
untouched: a null estimate described as confirming an effect, a robustness
check cited as confirmed after it had been withdrawn, an attractor said to move
towards a benchmark it moves away from. `claim_assertions.csv` closes that gap.
Each row names a generated file, a condition over it, and a phrase the
manuscript may not contain while that condition holds --- for example, that the
difference-in-differences estimate is positive with an interval spanning zero,
so the text may not say the reversal is more pronounced in designated counties.
Run against the manuscript as it stood before these were fixed, the assertions
catch three of the four; the fourth was a broken cross-reference, which the
LaTeX build already catches. An assertion whose premise no longer holds is
reported as stale rather than skipped.

### Checking that the corrections were applied

A correction table is only worth submitting if the manuscript actually carries
every entry, so the check runs mechanically rather than by inspection: each
claim is located by the context window recorded in `inventory.csv`, and the
verdict is read from that window alone. Matching on the value would be useless,
since bare tokens such as `2`, `24` or `0.59` occur throughout a manuscript.
Table claims are checked against their own generated table body, cell by cell,
because a bootstrap bound in one table can otherwise collide with a stale cell
reported for another. Results are in `table_r1_applied_check.csv`.

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
