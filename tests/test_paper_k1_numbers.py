"""Fast checks of the K1 paper numbers pipeline (scripts/paper_k1_numbers.py).

Re-runs the script, then checks a handful of key values against the rows of
the committed tables they claim to come from, and that values.tex defines
every macro exactly once with a letters-only name.

PAPER_K1_TEX_DIR (default build/tex) names the directory values.tex is
written to, as for the script itself. Where an expected_outputs.sha256 file
sits at the root (the public archive), every output of the script is also
compared with the sha256 recorded there (build/tex/ entries refer to the
PAPER_K1_TEX_DIR directory).
"""
import hashlib
import os
import re
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALUES = os.path.join(ROOT, "results/tables/paper_k1_values.csv")
TEX = os.path.join(ROOT, os.environ.get("PAPER_K1_TEX_DIR", 'build/tex'), "values.tex")


def _csv(rel):
    return pd.read_csv(os.path.join(ROOT, rel), keep_default_na=False, na_values=[""])


@pytest.fixture(scope="module")
def values():
    subprocess.run([sys.executable, "scripts/paper_k1_numbers.py"], cwd=ROOT, check=True,
                   capture_output=True)
    return pd.read_csv(VALUES, keep_default_na=False).set_index("macro")


def test_outputs_match_recorded_sha256(values):
    rec = os.path.join(ROOT, "expected_outputs.sha256")
    if not os.path.isfile(rec):
        pytest.skip("no expected_outputs.sha256 (not the public archive)")
    tex_dir = os.path.dirname(TEX)
    bad, n = [], 0
    for line in open(rec, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        h, rel = line.rstrip("\n").split("  ", 1)
        if rel.startswith("build/tex/"):
            p = os.path.join(tex_dir, rel[len("build/tex/"):])
        elif rel.startswith("results/tables/"):
            p = os.path.join(ROOT, rel)
        else:
            continue
        n += 1
        if not os.path.isfile(p) or hashlib.sha256(open(p, "rb").read()).hexdigest() != h:
            bad.append(rel)
    assert n > 0
    assert not bad, f"outputs differ from expected_outputs.sha256: {bad}"


def num(values, macro):
    return float(values.loc[macro, "value"])


def test_every_macro_defined_once_letters_only(values):
    with open(TEX, encoding="utf-8") as fh:
        names = re.findall(r"\\newcommand\{\\([^}]*)\}", fh.read())
    assert names, "values.tex is empty"
    assert len(names) == len(set(names)), "duplicate macro in values.tex"
    assert all(re.fullmatch(r"[A-Za-z]+", n) for n in names)
    assert set(names) == set(values.index)


def test_every_value_has_provenance(values):
    for col in ("source_file", "source_row_or_column", "transform"):
        assert (values[col].astype(str).str.len() > 0).all(), col
    for f in values["source_file"]:
        for part in f.split(" | "):
            assert part.startswith("results/tables/"), part
            assert os.path.isfile(os.path.join(ROOT, part)), part


def test_china_primary_ci(values):
    d = _csv("results/tables/expi_rerun_ci_side_by_side.csv")
    r = d[(d.variant == "V0_baseline") & (d.degenerate == "limit")
          & (d.quantity == "xi_trap (during-pre)")].iloc[0]
    assert num(values, "CnVzeroXiShiftLo") == r.rerun_ci_lo
    assert num(values, "CnVzeroXiShiftHi") == r.rerun_ci_hi
    assert num(values, "CnVzeroXiShiftPoint") == r.point_estimate
    lim = d[d.degenerate == "limit"].width_ratio_rerun_over_county
    assert num(values, "CnWidthRatioMin") == lim.min()
    assert num(values, "CnWidthRatioMax") == lim.max()


def test_china_dose(values):
    d = _csv("results/tables/expi_attrition_by_year.csv")
    r = d[(d.saai_spec == "multi") & (d.year_t == 2016)].iloc[0]
    assert num(values, "CnBreakDose") == pytest.approx((r.n_enter + r.n_leave) / r.n_stay, abs=0)


def test_k1c_cells(values):
    g = _csv("results/tables/k1_k1c_f2/k1_gates.csv")
    g = g[(g.gate == "K1c") & (g.rule_set == "limit/tol0") & (g.pipeline == "saai")]
    for cell, L in ((23, "A"), (24, "B"), (25, "C")):
        rb = g[(g.cell_index == cell) & (g.statistic == "V1_relative_bias_xi_shift")].value.iloc[0]
        cv = g[(g.cell_index == cell) & (g.statistic == "V1_ci95_coverage_xi_shift")].value.iloc[0]
        assert num(values, f"KonecRelBias{L}") == rb
        assert num(values, f"KonecCov{L}") == cv
    assert values.loc["KonecCovB", "formatted"] == r"\ensuremath{0.950}"


def test_s0_failure(values):
    s = _csv("results/tables/k1_s0_scope_f2/k1_s0_scope_s0_rows.csv")
    r = s[s.status == "fail"]
    assert len(r) == 1
    assert num(values, "KoneSzeroFailBias") == r.value.iloc[0]
    assert num(values, "KoneSzeroFailTol") == r.threshold.iloc[0]


def test_us_primary_h3b(values):
    h = _csv("results/tables/k2/k2_h3b.csv")
    r = h[(h.scope == "metro") & (h.outcome == "pi") & (h.period == "main") & (h.link == "overlap")].iloc[0]
    assert num(values, "UsPiMainPred") == r.pred_xi_trap
    assert num(values, "UsPiMainObs") == r.observed
    assert num(values, "UsPiMainGate") == pytest.approx(0.5 * r.half_width_V1, rel=1e-15)
    assert values.loc["UsPiMainVerdict", "value"] == "pass"


def test_brazil_ref2002_guard(values):
    p = _csv("results/tables/k2br_predictions.csv")
    r = p[(p.window == "ref2002") & (p.outcome == "pib")].iloc[0]
    assert num(values, "BrTwoPibPred") == r.pred_dxi_star
    assert num(values, "BrTwoPibUnguarded") == r.unguarded_pred_dxi_star
    v = _csv("results/tables/k2br_verdicts.csv")
    assert values.loc["BrTwoPibHfourVerdict", "value"] == \
        v[(v.window == "ref2002") & (v.outcome == "pib")].H4.iloc[0] == "inconsistent"


def test_china_paired_and_ols(values):
    e = _csv("results/tables/expi_paired_estimands.csv")
    def row(est, conv):
        r = e[(e.estimand == est) & (e.convention == conv)]
        assert len(r) == 1, (est, conv)
        return r.iloc[0]
    r = row("xi_trap_V0_minus_V1_during", "limit")
    assert r.role == 'paired_V0_V1' and r.ci_status == "ok"
    assert num(values, "CnVzeroMinusVoneXiDuringLo") == r.ci_lo
    assert num(values, "CnVzeroMinusVoneXiDuringHi") == r.ci_hi
    assert num(values, "CnVzeroMinusVoneXiDuring") == pytest.approx(r.point_estimate, abs=1e-12)
    r = row("xi_shift_V0_minus_V1", "limit")
    assert num(values, "CnVzeroMinusVoneXiShiftLo") == r.ci_lo
    assert num(values, "CnVzeroMinusVoneXiShift") == pytest.approx(r.point_estimate, abs=1e-12)
    assert num(values, "CnVzeroMinusVoneXiShiftNanNonfinite") == row("xi_shift_V0_minus_V1", "nan").n_nonfinite
    a1 = row("a1_V1_during", "none")
    assert num(values, "CnAoneVoneDuring") == a1.point_estimate
    assert num(values, "CnAmpVoneDuring") == pytest.approx(1 / abs(a1.point_estimate), rel=1e-12)
    assert num(values, "CnDoseTimesAmpDuring") == pytest.approx(
        num(values, "CnDoseDuring") / abs(a1.point_estimate), rel=1e-12)
    for m in ("CnAmpDuring", "CnDoseTimesAmp", "CnVzeroAoneDuring", "CnVzeroAonePre"):
        assert m not in values.index


def test_noise_summaries(values):
    s = _csv("results/tables/k2_literal_noise_summary.csv")
    r = s[(s.level == "overall") & (s.quantity == "xi_trap")].iloc[0]
    assert num(values, "UsLiteralXiMax") == r.max_abs_diff
    o = _csv("results/tables/k2br_optimizer_noise_summary.csv")
    r = o[o.window == "ref2002"].iloc[0]
    assert num(values, "BrOptNoiseMaxTwo") == max(r.max_abs_warm_cold, r.max_abs_warm_tight, r.max_abs_cold_tight)


def test_masking_verdicts_and_metrics(values):
    """Controlled masking: macros equal the committed evaluation rows ('null' read as a label)."""
    e = _csv("results/tables/k4_mask/confirmatory/k4_evaluation.csv")
    assert "null" in set(e.scope)
    def met(system, scope, name):
        r = e[(e.system == system) & (e.scope == scope) & (e.name == name) & (e.kind == "metric")]
        assert len(r) == 1, (system, scope, name)
        return r.iloc[0]
    r = met("BR", "active", "calibration_slope")
    assert num(values, "MaskBrSlope") == float(r.value)
    assert num(values, "MaskBrSlopeLo") == r.cluster_ci_lo
    assert num(values, "MaskBrSlopeHi") == r.cluster_ci_hi
    assert num(values, "MaskUsPointAvail") == float(met("US", "active", "point_availability").value)
    assert num(values, "MaskUsNullObs") == float(met("US", "null", "median_abs_obs").value)
    assert num(values, "MaskUsRemoved") == pytest.approx(1 - float(met("US", "active", "recovery_median_ratio").value), abs=0)
    v = e[(e.kind == "verdict") & (e.scope == "active")]
    fz = v[v.criteria_set == "frozen"].set_index(["system", "name"]).value
    pv = v[v.criteria_set == "provisional"].set_index(["system", "name"]).value
    assert (fz == pv).all()
    assert fz[("BR", "T1_point")] == "pass" and values.loc["MaskBrVerdictPoint", "value"] == "pass"
    assert fz[("US", "T2")] == "closer_than_V0_only"
    assert values.loc["MaskUsVerdictRecovery", "value"] == "closer than V0 only"
    assert values.loc["MaskUsVerdictPower", "value"] == "underpowered"
