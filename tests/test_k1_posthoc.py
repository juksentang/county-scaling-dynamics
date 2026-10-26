"""scripts/run_expi_identify.py    China bootstrap identifiability
  scripts/k1_guard_sensitivity.py guard sensitivity of the K1 projection

SYNTHETIC data only: no real county, BEA or IBGE value is read.

Run:  python3 -P -m pytest -q tests/test_k1_posthoc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_expi_identify as I
import run_expi_paired as P
import run_expi_rerun_ci as R
import k1_guard_sensitivity as G
from src.km_mle import derived_observables
from test_expi_paired import synthetic_panel

V0, V1 = I.VARIANTS


@pytest.fixture(scope="module")
def world():
    panel = synthetic_panel()
    windows, members = P.window_setup(panel)
    point, warm = P.evaluate(windows, None)
    return dict(panel=panel, windows=windows, members=members, point=point, warm=warm)


def _params(a0, a1, a2=0.0, a3=0.0, c0=np.log(0.1)):
    return np.array([a0, a1, a2, a3, c0, 0.0, 0.0])


XI = np.random.default_rng(0).normal(0, 1, 3000)
CASES = {
    "stable_far": _params(0.05, -0.1),
    "stable_near_zero": _params(1e-5, -0.1),
    "no_root": _params(0.3, -0.01),
    "unstable_only": _params(0.0, 0.1),
    "two_stable": _params(0.0, 0.3, 0.0, -0.2),
}


@pytest.mark.parametrize("name", list(CASES))
def test_root_status_replicates_derived_observables(name):
    th = CASES[name]
    rs = I.root_status(th, XI)
    lim = derived_observables(th, XI, xi_target=0.0, degenerate="limit")
    nan = derived_observables(th, XI, xi_target=0.0, degenerate="nan")
    assert lim["xi_trap"] == rs["trap"]
    if lim["id_status"] != "subgrid_path":
        assert lim["id_status"] == rs["root_status"]
    if nan["id_status"] != "degenerate_path":
        assert nan["id_status"] == rs["root_status"]
        assert nan["xi_trap"] == rs["trap"]
    assert (lim["id_status"] == "subgrid_path") == rs["subgrid"]
    assert (nan["id_status"] == "degenerate_path") == rs["subgrid"]
    assert rs["support_lo"] == np.percentile(XI, 1) and rs["support_hi"] == np.percentile(XI, 99)


def test_root_status_classes():
    s = {k: I.root_status(v, XI) for k, v in CASES.items()}
    assert s["stable_far"]["root_status"] == "stable_zero"
    assert s["stable_far"]["first_stable"] == pytest.approx(0.5, abs=1e-3)
    assert not s["stable_far"]["subgrid"]
    assert s["stable_near_zero"]["root_status"] == "stable_zero" and s["stable_near_zero"]["subgrid"]
    for k in ("no_root", "unstable_only"):
        assert s[k]["root_status"] == "fallback_argminA" and np.isnan(s[k]["first_stable"])
        assert s[k]["trap"] == s[k]["placeholder"]
    assert s["unstable_only"]["n_unstable"] == 1 and s["unstable_only"]["subgrid"]
    assert s["two_stable"]["n_stable"] == 2 and s["two_stable"]["n_unstable"] == 1
    assert s["two_stable"]["first_stable"] == pytest.approx(-np.sqrt(1.5), abs=1e-2)


def test_capture_leaves_evaluate_unchanged_and_restores(world):
    orig_obs, orig_pairs = R.observables, R.PAIRS
    rec, params, idr = I.evaluate_identified(world["windows"], None)
    assert R.observables is orig_obs and R.PAIRS is orig_pairs
    ref, ref_params = P.evaluate(world["windows"], None)
    assert set(rec) == set(ref)
    for k, v in ref.items():
        assert (np.isnan(v) and np.isnan(rec[k])) if isinstance(v, float) and np.isnan(v) \
            else rec[k] == v, k
    for k in ref_params:
        assert np.array_equal(params[k], ref_params[k])
    for v in I.VARIANTS:
        for p in I.PERIODS:
            assert idr[f"{v}|trap_check_{p}"] is True
            assert idr[f"{v}|root_status_{p}"] in ("stable_zero", "fallback_argminA")


def test_capture_restores_on_exception(world, monkeypatch):
    orig_obs = R.observables
    with pytest.raises(RuntimeError):
        with I.capture_fits():
            raise RuntimeError("boom")
    assert R.observables is orig_obs


def test_failed_fit_is_recorded_as_fit_failed(world, monkeypatch):
    def boom(panel, period):
        raise ValueError("no pairs")
    monkeypatch.setitem(R.PAIRS, V1, boom)
    rec, _, idr = I.evaluate_identified(world["windows"], None)
    assert idr[f"{V1}|root_status_pre"] == "fit_failed"
    assert f"{V1}|fail_pre" in rec and np.isnan(rec[f"{V1}|limit|xi_trap_pre"])
    assert idr[f"{V0}|root_status_pre"] != "fit_failed"


@pytest.fixture(scope="module")
def small(world):
    P._init(world["windows"], world["members"], world["warm"])
    I._init(world["windows"], world["members"], world["warm"])
    mine = pd.DataFrame([I.one_draw(b) for b in range(6)])
    theirs = pd.DataFrame([P.one_draw(b) for b in range(6)])
    point, _, point_id = I.evaluate_identified(world["windows"], None)
    est = P.estimand_table(point, theirs)
    return dict(mine=mine, theirs=theirs, point=point, point_id=point_id, est=est)


def test_rebuilt_draws_equal_the_addendum_draws(small):
    """Draw b of this script carries the addendum's columns bit for bit."""
    for c in small["theirs"].columns:
        a, b = small["theirs"][c].to_numpy(float), small["mine"][c].to_numpy(float)
        assert np.array_equal(a, b, equal_nan=True), c


def _roundtrip(df, tmp_path, name):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return pd.read_csv(p, keep_default_na=False, na_values=[""]) if name.endswith("csv") \
        else pd.read_csv(p)


def test_consistency_kill_rule(small, tmp_path):
    cd = _roundtrip(small["theirs"], tmp_path, "d.csv.gz")
    ce = _roundtrip(small["est"], tmp_path, "e.csv")
    c = I.consistency(small["point"], small["mine"], cd, ce)
    assert (c.tier == "exact").all()
    assert set(c.column[c.table == I.COMMITTED_DRAWS]) == set(cd.columns) - {"replicate"}
    m2 = small["mine"].copy()
    m2[f"{V0}|limit|xi_trap_pre"] += 5e-4
    c = I.consistency(small["point"], m2, cd, ce)
    assert not (c.tier == "fail").any() and (c.tier == "optimiser").any()
    for mut in ("big", "nan", "count", "drop"):
        m3 = small["mine"].copy()
        if mut == "big":
            m3[f"{V1}|limit|xi_trap_during"] += 0.05
        elif mut == "nan":
            m3.loc[0, f"{V0}|nan|xi_trap_pre"] = np.nan
        elif mut == "count":
            m3[f"{V0}|n_pairs_pre"] = m3[f"{V0}|n_pairs_pre"].astype(float)
            m3.loc[0, f"{V0}|n_pairs_pre"] += 1e-6
        else:
            m3 = m3.drop(columns=[f"{V1}|rho1_pre"])
        assert (I.consistency(small["point"], m3, cd, ce).tier == "fail").any(), mut
    pt = dict(small["point"])
    pt[f"{V0}|limit|xi_trap_pre"] += 0.05
    assert (I.consistency(pt, small["mine"], cd, ce).tier == "fail").any()
    c = I.consistency(small["point"], small["mine"].iloc[:3], cd, ce)
    assert (c.tier == "exact").all()
    assert (c.loc[c.table == I.COMMITTED_DRAWS, "n_compared"] == 3).all()


def test_trap_check_rows_fail_on_a_false(small):
    d = small["mine"].copy()
    assert (I.trap_check_rows(d).tier == "exact").all()
    d.loc[2, f"{V1}|trap_check_pre"] = False
    t = I.trap_check_rows(d)
    assert (t.tier == "fail").sum() == 1


def _fake(n=200, seed=1):
    """Draws with known identification: V0 pre not identified on the first
    20 draws (placeholder near 0), V1 during sub-grid on draws 20-29."""
    rng = np.random.default_rng(seed)
    d = {"replicate": np.arange(n)}
    for v in I.VARIANTS:
        for p in I.PERIODS:
            x = rng.normal(0.3 if v == V0 else 0.0, 0.2, n)
            d[f"{v}|root_status_{p}"] = np.array(["stable_zero"] * n, dtype=object)
            d[f"{v}|n_stable_{p}"] = np.ones(n)
            d[f"{v}|subgrid_{p}"] = np.zeros(n, bool)
            for c in I.CONVENTIONS:
                d[f"{v}|{c}|xi_trap_{p}"] = x.copy()
                d[f"{v}|{c}|tau_mfpt_{p}"] = np.exp(rng.normal(0, 0.1, n))
    d[f"{V0}|root_status_pre"][:20] = "fallback_argminA"
    d[f"{V0}|n_stable_pre"][:20] = 0
    d[f"{V0}|limit|xi_trap_pre"][:20] = 0.01
    d[f"{V0}|nan|xi_trap_pre"][:20] = 0.01
    d[f"{V1}|subgrid_during"][20:30] = True
    d[f"{V1}|limit|xi_trap_during"][20:30] = 1e-3
    d[f"{V1}|nan|xi_trap_during"][20:30] = np.nan
    d[f"{V1}|n_stable_pre"][50:55] = 2
    for v in I.VARIANTS:
        for p in I.PERIODS:
            d[f"{v}|rho1_{p}"] = np.full(n, 0.9)
    for q in ("dose", "dose_max", "Mbar"):
        for p in I.PERIODS:
            d[f"{q}_{p}"] = np.full(n, 0.1)
    return pd.DataFrame(d)


def _point_id():
    out = {}
    for v in I.VARIANTS:
        for p in I.PERIODS:
            out |= {f"{v}|root_status_{p}": "stable_zero", f"{v}|n_stable_{p}": 1,
                    f"{v}|subgrid_{p}": False}
    return out


def test_fractions_table():
    d = _fake()
    f = I.fractions_table(_point_id(), d).set_index(["version", "window"])
    assert f.loc[("V0", "pre"), "n_identified"] == 180
    assert f.loc[("V0", "pre"), "frac_identified"] == pytest.approx(0.9)
    assert f.loc[("V0", "pre"), "placeholder_abs_xi_max"] == pytest.approx(0.01)
    assert f.loc[("V1", "during"), "n_subgrid_and_identified"] == 10
    assert f.loc[("V1", "during"), "n_nan_convention_nonfinite"] == 10
    assert f.loc[("V1", "during"), "n_identified"] == 200
    assert f.loc[("V1", "pre"), "n_multiple_stable_zeros"] == 5
    assert (f["n_fit_failed"] == 0).all()


def test_intervals_restricted_to_identified_draws():
    d = _fake()
    point = {k: float(np.nanmedian(d[k])) for k in d.columns if d[k].dtype != object}
    t = I.intervals_table(point, d)
    r = t[(t.estimand == "xi_trap_V0_minus_V1_pre") & (t.convention == "limit")]
    r = r.set_index("subset")
    assert r.loc["all", "n_subset"] == 200 and r.loc["identified", "n_subset"] == 180
    assert r.loc["not_identified", "n_subset"] == 20
    assert r.loc["all", "share_of_draws"] == 1.0
    vals = d[f"{V0}|limit|xi_trap_pre"] - d[f"{V1}|limit|xi_trap_pre"]
    c_all = I.k1_ci(vals.to_numpy())
    c_id = I.k1_ci(vals.to_numpy()[20:])
    assert (r.loc["all", "tol0_lo"], r.loc["all", "tol0_hi"]) == (c_all["ci_lo"], c_all["ci_hi"])
    assert (r.loc["identified", "tol0_lo"], r.loc["identified", "tol0_hi"]) == (
        c_id["ci_lo"], c_id["ci_hi"])
    assert r.loc["identified", "tol0_lo_minus_all"] == pytest.approx(c_id["ci_lo"] - c_all["ci_lo"])
    s = r.loc["not_identified"]
    assert s.share_below_all_supp_interval + s.share_inside_all_supp_interval \
        + s.share_above_all_supp_interval == pytest.approx(1.0)
    est = P.estimand_table(point, d)
    for name in ("xi_trap_V0_minus_V1_pre", "xi_shift_V0_minus_V1"):
        a = t[(t.estimand == name) & (t.convention == "limit") & (t.subset == "all")].iloc[0]
        b = est[(est.estimand == name) & (est.convention == "limit")].iloc[0]
        assert (a.tol0_lo, a.tol0_hi, a.supp_lo, a.supp_hi) == (
            b.ci_lo, b.ci_hi, b.supplement_rule_lo, b.supplement_rule_hi)
    q = t[(t.estimand == "xi_trap_V1_during") & (t.convention == "nan")].set_index("subset")
    assert q.loc["all", "tol0_status"] == "not_evaluable_nonfinite"
    assert q.loc["identified_regular_path", "n_subset"] == 190
    assert q.loc["identified_regular_path", "tol0_status"] == "ok"
    assert set(t.estimand) == set(I.DIAG_ESTIMANDS)
    assert set(t.subset) == set(I.SUBSETS)


def test_conventions_table():
    d = _fake()
    point = {k: float(np.nanmedian(d[k])) for k in d.columns if d[k].dtype != object}
    t = I.intervals_table(point, d)
    c = I.conventions_table(t, d)
    r = c[(c.estimand == "xi_trap_V1_during") & (c.subset == "all")].iloc[0]
    assert r.n_nan_nonfinite_only == 10 and r.n_equal_both_finite == 190
    assert r.tol0_agree is None or pd.isna(r.tol0_agree)
    assert r.supp_agree in (True, False)
    r = c[(c.estimand == "xi_trap_V0_pre") & (c.subset == "all")].iloc[0]
    assert r.tol0_agree is True and r.max_abs_diff_both_finite == 0.0
    assert set(c.subset) == {"all", "identified"}


def test_dev_never_writes_results_tables():
    with pytest.raises(SystemExit):
        I.main(["--dev", "--n-boot", "2", "--out", str(ROOT / "results" / "tables")])


def test_analyse_end_to_end_on_synthetic_panel(world, small, tmp_path):
    """The whole pipeline on the synthetic panel against 'committed' tables
    written from the addendum's own draws: completed, every table written;
    against perturbed committed draws: killed, only the consistency table."""
    com = tmp_path / "com"
    com.mkdir()
    small["theirs"].iloc[:4].to_csv(com / I.COMMITTED_DRAWS, index=False)
    small["est"].to_csv(com / I.COMMITTED_ESTIMANDS, index=False)
    out = tmp_path / "out"
    res = I.analyse(world["panel"], 4, 1, str(com), str(out))
    assert res["status"] == "completed", res
    for f in ("expi_identify_fractions.csv", "expi_identify_intervals.csv",
              "expi_identify_conventions.csv", "expi_identify_draws.csv.gz",
              "expi_identify_consistency.csv"):
        assert (out / f).exists()
    fr = pd.read_csv(out / "expi_identify_fractions.csv")
    assert len(fr) == 4 and (fr.n_draws == 4).all()
    bad = small["theirs"].iloc[:4].copy()
    bad[f"{V0}|limit|xi_trap_pre"] += 0.1
    bad.to_csv(com / I.COMMITTED_DRAWS, index=False)
    out2 = tmp_path / "out2"
    res = I.analyse(world["panel"], 4, 1, str(com), str(out2))
    assert res["killed"] and res["written"] == ["expi_identify_consistency.csv"]
    assert not (out2 / "expi_identify_fractions.csv").exists()


def test_guard_grid():
    g = G.guard_grid()
    assert len(g) == 12 and len(set(g)) == 12
    assert sum(x == G.REGISTERED for x in g) == 1
    assert {s for s, _ in g} == set(G.SUPPORTS) and {f for _, f in g} == set(G.FLOOR_FACTORS)


@pytest.fixture(scope="module")
def br():
    import test_k2br as TB
    m, _ = TB.toy(n0=600, splits={2005: [(150, 1, False)]}, exits={}, entrant_offset=1.5,
                  agri_nonpos=0, seed=16)
    res = TB.K.analyse(m, TB.TOY)
    return G.br_cells_from_res(res, "toy", TB.TOY.outcomes)


@pytest.fixture(scope="module")
def us():
    import run_k2 as K2
    w = K2.make_toy_world(seed=3, entrant_shift=0.0, leaver_shift=-1.5)
    return G.us_cell(w, key=("metro", "pi", "main", "B", "overlap"), n_min=10)


def test_alt_guard_registered_equals_k1_guard(br, us):
    import run_k2 as K2
    import run_k2br as KBR
    for c in br:
        assert G.alt_guard(c["xi_t"], c["rho1"], *G.REGISTERED) == KBR.k1_guard(c["pairs"],
                                                                               c["rho1"])
    assert G.alt_guard(us["xi_t"], us["rho1"], *G.REGISTERED) == K2.k1_guard(us["pairs"],
                                                                            us["rho1"])
    g = G.alt_guard(us["xi_t"], us["rho1"], (5.0, 95.0), 2.0)
    assert g[0] == np.percentile(us["xi_t"], 5) and g[2] == -2.0 * abs(us["rho1"] - 1)
    assert G.alt_guard(us["xi_t"][:10], us["rho1"], (1, 99), 1.0) is None
    assert G.alt_guard(us["xi_t"], 1.0, (1, 99), 1.0) is None


def test_us_cell_is_the_run_pipeline_prediction(us):
    """The cell rebuilt by us_cell gives run_pipeline's prediction (same
    skeleton, fit and k1_prediction call)."""
    import run_k2 as K2
    w = K2.make_toy_world(seed=3, entrant_shift=0.0, leaver_shift=-1.5)
    lin = K2.lineages(w, "metro")
    aggs = {(sc, lab, oc): K2.aggregate(w, sc, lab, oc) for (sc, lab) in w.areas
            for oc in w.outcomes}
    drops, _ = K2.min_n_drops(w, aggs, 10)
    k = ("metro", "pi", "main", "B", "overlap")
    p = K2.run_spec(K2.build_skeleton(w, aggs, k, lin[1], drops[("pi", "main")]))
    e = K2.estimate_spec(p, "B")
    r1 = e["V1"]["limit"]["rho1"]
    ref = K2.k1_prediction(e["V1"]["params"], p["dec"], K2.k1_guard(p["pairs"]["V1"], r1))
    got = G.predict(us, G.registered_guard(us))
    assert got["status"] == ref["status"] == "ok"
    assert got["pred_xi_trap"] == ref["pred_xi_trap"] and got["pred_rho1"] == ref["pred_rho1"]


def _committed_from(cell):
    g = G.registered_guard(cell)
    p = G.predict(cell, g)
    return {"guard_lo": g[0], "guard_hi": g[1], "slope_floor": g[2], **p,
            "observed": cell["observed"]}


def test_reproduction_kill_rule(br, us):
    for c in br + [us]:
        com = _committed_from(c)
        rows = pd.DataFrame(G.reproduction_rows(c, com))
        assert rows["pass"].all(), rows[~rows["pass"]]
        bad = dict(com, pred_xi_trap=com["pred_xi_trap"] + 1e-6)
        assert not pd.DataFrame(G.reproduction_rows(c, bad))["pass"].all()
        bad = dict(com, status="projection_undefined: x")
        assert not pd.DataFrame(G.reproduction_rows(c, bad))["pass"].all()
        bad = dict(com, guard_lo=com["guard_lo"] - 1e-9)
        assert not pd.DataFrame(G.reproduction_rows(c, bad))["pass"].all()


def test_run_cells_kills_before_any_sensitivity(br, monkeypatch):
    com = {c["cell"]: _committed_from(c) for c in br}
    monkeypatch.setattr(G, "committed_row", lambda c, tab=None: com[c["cell"]])
    monkeypatch.setattr(G, "unguarded_committed", lambda c, tab=None: 0.01)
    check, sens = G.run_cells(br)
    assert check["pass"].all() and sens is not None
    assert len(sens) == 12 * len(br)
    reg = sens[sens.registered]
    assert len(reg) == len(br)
    for c in br:
        r = reg[reg.cell == c["cell"]].iloc[0]
        assert r.pred_xi_trap == com[c["cell"]]["pred_xi_trap"]
        assert r.pred_over_registered == pytest.approx(1.0)
    oth = sens[(sens.support_lo_pct == 1.0) & (sens.floor_factor != 1.0)]
    assert oth.status.astype(str).str.len().gt(0).all()
    com2 = dict(com)
    k = br[0]["cell"]
    com2[k] = dict(com[k], pred_rho1=com[k]["pred_rho1"] + 1.0)
    monkeypatch.setattr(G, "committed_row", lambda c, tab=None: com2[c["cell"]])
    check, sens = G.run_cells(br)
    assert not check["pass"].all() and sens is None


def test_sensitivity_statuses_are_reported_not_raised(us):
    rows = pd.DataFrame(G.sensitivity_rows(us, _committed_from(us), np.nan))
    assert len(rows) == 12
    assert rows.status.map(lambda s: s == "ok" or s.startswith("projection_undefined")).all()
    assert rows.registered.sum() == 1
