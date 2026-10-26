"""Run: python3 -P -m pytest -q tests/test_k2.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_k2 as K


@pytest.fixture(autouse=True)
def no_real_outcomes(monkeypatch):
    """Any attempt to load a real outcome or the real world fails the test."""
    def boom(*a, **k):
        raise AssertionError("a unit test tried to read real data")
    monkeypatch.setattr(K, "load_outcome", boom)
    monkeypatch.setattr(K, "load_world", boom)
    monkeypatch.setattr(K.COV, "load_population", boom)
    monkeypatch.setattr(K.COV, "load_vintage", boom)


def _build(world, drops=None):
    lin = {s: K.lineages(world, s) for s in K.SCOPES}
    aggs = {(sc, lab, oc): K.aggregate(world, sc, lab, oc)
            for (sc, lab) in world.areas for oc in world.outcomes}
    keys = K.spec_list(tuple(world.outcomes))
    sks = {k: K.build_skeleton(world, aggs, k, lin[k[0]][1], (drops or {}).get((k[1], k[2])))
           for k in keys}
    return lin, aggs, keys, sks


@pytest.fixture(scope="module")
def toy():
    w = K.make_toy_world()
    lin, aggs, keys, sks = _build(w)
    point = {k: K.run_spec(sk) for k, sk in sks.items()}
    return dict(world=w, lin=lin, aggs=aggs, keys=keys, sks=sks, point=point)


B_OV = ("metro", "pi", "secondary", "B", "overlap")
B_CODE = ("metro", "pi", "main", "B", "code")
A_SEC = ("metro", "pi", "secondary", "A", "none")


def test_in_force_calendar_matches_section14():
    w = K.World(years=list(range(1993, 2025)), pop=pd.DataFrame(), outcomes={},
                vintages=[(lab, date, kind) for lab, date, _, kind in K.COV.VINTAGES],
                areas={})
    want = {**{y: "1993-06" for y in range(1993, 1999)},
            **{y: "1999-06" for y in range(1999, 2003)},
            2003: "2003-06", 2004: "2003-12", 2005: "2004-11", 2006: "2005-12",
            2007: "2006-12", 2008: "2007-11", 2009: "2008-11",
            **{y: "2009-12" for y in range(2010, 2013)},
            **{y: "2013-02" for y in range(2013, 2016)},
            2016: "2015-07", 2017: "2015-07", 2018: "2018-04", 2019: "2018-09",
            **{y: "2020-03" for y in range(2020, 2024)}, 2024: "2023-07"}
    assert {y: w.in_force(y) for y in want} == want
    assert "2017-08" not in {w.in_force(y) for y in range(1993, 2025)}


def test_periods_and_transitions(toy):
    sk_main = toy["sks"][("metro", "pi", "main", "B", "overlap")]
    sk_sec = toy["sks"][B_OV]
    assert sk_main.transitions == list(range(2003, 2024))
    assert sk_sec.transitions == list(range(1993, 2024))
    sk_gdp = toy["sks"][("metro", "gdp", "secondary", "B", "overlap")]
    assert sk_gdp.transitions[0] == 2001
    for k in toy["keys"]:
        if k[4] == "code" or k[0] == "cbsa_all":
            assert k[2] == "main"


def test_sami_weighted_equals_expanded_ols():
    rng = np.random.default_rng(1)
    n = 30
    x = rng.normal(12, 1, n)
    y = 3 + 1.1 * x + rng.normal(0, 0.2, n)
    w = rng.integers(0, 4, n).astype(float)
    m = w > 0
    xi_w, b_w, r2_w, sd_w = K.sami(x[m], y[m], w[m])
    k = w[m].astype(int)
    xe, ye = np.repeat(x[m], k), np.repeat(y[m], k)
    X = np.column_stack([np.ones_like(xe), xe])
    coef, *_ = np.linalg.lstsq(X, ye, rcond=None)
    r = ye - X @ coef
    sd = r.std(ddof=K.SD_DDOF)
    assert abs(b_w - coef[1]) < 1e-10
    assert abs(sd_w - sd) < 1e-12
    np.testing.assert_allclose(np.repeat(xi_w, k), r / sd, atol=1e-10)
    assert abs(K._wmean(xi_w, w[m])) < 1e-12
    assert abs(K._wvar(xi_w, w[m]) - 1.0) < 1e-12


def test_toy_events_overlap_and_code(toy):
    d = toy["point"][B_OV]["dec"].set_index("year_t")
    assert (d.loc[1998, ["n_E", "n_L", "n_changed"]] == [0, 0, 1]).all()
    assert (d.loc[2002, ["n_E", "n_L"]] == [4, 0]).all()
    assert (d.loc[2008, ["n_E", "n_L"]] == [0, 1]).all()
    assert (d.loc[2012, ["n_E", "n_L", "n_changed"]] == [7, 4, 6]).all()
    quiet = d.drop(index=[1998, 2002, 2008, 2012])
    assert (quiet[["n_E", "n_L", "n_changed"]] == 0).all().all()
    dc = toy["point"][B_CODE]["dec"].set_index("year_t")
    assert (dc.loc[2012, ["n_E", "n_L", "n_changed"]] == [8, 5, 6]).all()
    da = toy["point"][("cbsa_all", "pi", "main", "B", "overlap")]["dec"].set_index("year_t")
    assert (da.loc[2012, ["n_E", "n_L"]] == [6, 4]).all()


def test_overlap_link_mutual_majority():
    pop = pd.DataFrame({2000: {"a": 10.0, "b": 1.0, "c": 5.0, "d": 6.0}})
    prev = {"P1": frozenset({"a", "b"}), "P2": frozenset({"c", "d"})}
    nxt = {"N1": frozenset({"a"}),
           "N2": frozenset({"b"}),
           "N3": frozenset({"c", "d"})}
    assert K.COV.link(prev, nxt, pop, 2000, "overlap") == {"N1": "P1", "N3": "P2"}
    assert K.COV.link(prev, nxt, pop, 2000, "code") == {}
    nxt2 = {"M": frozenset({"a", "b", "c", "d"})}
    assert K.COV.link(prev, nxt2, pop, 2000, "overlap") == {}
    pop2 = pop.copy()
    pop2.loc["a", 2000] = 30.0
    assert K.COV.link(prev, nxt2, pop2, 2000, "overlap") == {"M": "P1"}


def test_lineages_and_states(toy):
    w = toy["world"]
    for scope in K.SCOPES:
        unit_lin, area_lin = toy["lin"][scope]
        for (sc, lab), amap in w.areas.items():
            if sc != scope:
                continue
            for c, s in amap.items():
                assert {unit_lin[u] for u in s} == {area_lin[(lab, c)]}
        st = K.lineage_states(w, unit_lin, scope=scope)
        assert len(st) == max(unit_lin.values()) + 1
        assert set(st) <= {u[:2] for u in unit_lin}
    tab = K.lineage_table(w, "metro")
    assert tab["n_panelA_areas"].max() == 2
    assert not K.lineage_summary(tab, w)["exceeds_10pct"]


def test_h1_lemma1_every_transition(toy):
    for k, p in toy["point"].items():
        d = p["dec"]
        assert len(d)
        assert d["identity_residual"].abs().max() < K.H1_TOL, k


def test_h1_under_bootstrap_weights(toy):
    unit_lin, _ = toy["lin"]["metro"]
    n_lin = max(unit_lin.values()) + 1
    W = K.draw_plan(7, 5, 0, 0, n_lin, np.arange(n_lin))
    for b in range(len(W)):
        for k, sk in toy["sks"].items():
            if k[0] != "metro":
                continue
            d = K.run_spec(sk, W[b])["dec"]
            assert d["identity_residual"].abs().max() < K.H1_TOL


def test_h2_panel_a_no_composition(toy):
    for k, p in toy["point"].items():
        if k[3] != "A":
            continue
        d = p["dec"]
        assert (d["n_E"] == 0).all() and (d["n_L"] == 0).all() and (d["M_t"] == 0).all()
        x0, dx0, w0 = p["pairs"]["V0"]
        x1, dx1, w1 = p["pairs"]["V1"]
        assert len(x0) == len(x1)
        assert np.max(np.abs(x0 - x1)) < K.H2_TOL and np.max(np.abs(dx0 - dx1)) < K.H2_TOL
        e = K.estimate_spec(p, "A")
        for c in K.CONVENTIONS:
            for q in K.OBS_KEYS:
                a, b = e["V0"][c][q], e["V1"][c][q]
                assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < K.H2_TOL, (k, c, q)


def test_panel_a_carries_missing_county(toy):
    """The Broomfield-like county (no data before 2002): its area is present
    every year, aggregated over the other counties."""
    ag = toy["aggs"][("metro", "2023-07", "pi")]
    j = ag.years.index(1995)
    assert ag.present[:, j].all()
    assert (ag.n_missing[:, j] > 0).sum() == 1
    assert (ag.n_missing[:, ag.years.index(2005)] == 0).all()


def test_v1_restandardised_on_stayers(toy):
    sk = toy["sks"][B_OV]
    p = toy["point"][B_OV]
    x, dx, w = p["pairs"]["V1"]
    off = 0
    for t in sk.transitions:
        n = len(sk.trans[t]["ia"])
        xt, xt1 = x[off:off + n], x[off:off + n] + dx[off:off + n]
        off += n
        for v in (xt, xt1):
            assert abs(v.mean()) < 1e-12 and abs(v.var(ddof=K.SD_DDOF) - 1) < 1e-10
    assert off == len(x)
    d = p["dec"].set_index("year_t")
    assert abs(d.loc[2012, "xbar_stay_t"]) > 1e-3


def test_v1d_uses_next_delineation(toy):
    sk = toy["sks"][B_OV]
    tr = sk.trans[2012]
    n_v1d = len(tr["v1d"]["lin"])
    ag = toy["aggs"][("metro", "2013-02", "pi")]
    jt, jn = ag.years.index(2012), ag.years.index(2013)
    assert n_v1d == int((ag.present[:, jt] & ag.present[:, jn]).sum())
    d = toy["point"][B_OV]["dec"].set_index("year_t")
    assert abs(d.loc[2012, "mean_inc_V1d"]) < 1e-12
    assert d.loc[2012, "n_V1d_stay"] == d.loc[2012, "n_S"]


def test_h3a_mechanical_term_nonzero(toy):
    d = toy["point"][B_OV]["dec"]
    ev = d[(d.n_E + d.n_L) > 0]
    assert len(ev) == 3
    assert (ev["M_t"].abs() > K.H3A_TOL).all()
    assert (ev["M_t"].abs() <= ev["upper_bound"] + 1e-15).all()


def test_fast_path_equals_literal_county_draw(toy):
    w = toy["world"]
    scope = "metro"
    unit_lin, _ = toy["lin"][scope]
    n_lin = max(unit_lin.values()) + 1
    W = K.draw_plan(11, 2, 0, 0, n_lin, np.arange(n_lin))
    keys = [k for k in toy["keys"] if k[0] == scope]
    for b in range(len(W)):
        lit = K.literal_county_draw(w, scope, unit_lin, W[b], keys, {})
        for k in keys:
            fres = K.run_spec(toy["sks"][k], W[b])
            lres, le = lit[k]
            for v in K.VARIANTS[k[3]]:
                assert K.pair_multiset_diff(fres["pairs"][v], lres["pairs"][v]) < K.LITERAL_PAIR_TOL
            fe = K.estimate_spec(fres, k[3])
            for v in K.VARIANTS[k[3]]:
                for q in ("xi_trap", "rho1", "two_delta_V"):
                    a, c = fe[v]["limit"][q], le[v]["limit"][q]
                    assert (np.isnan(a) and np.isnan(c)) or abs(a - c) <= K.LITERAL_OBS_TOL * max(1, abs(a))
            pd.testing.assert_series_equal(fres["dec"]["n_S"], lres["dec"]["n_S"], check_names=False)


def test_draw_plan_deterministic_and_keyed():
    blk = np.array([0, 0, 1, 2, 2, 3])
    a = K.draw_plan(5, 4, 0, 1, 6, blk)
    b = K.draw_plan(5, 4, 0, 1, 6, blk)
    c = K.draw_plan(5, 4, 1, 1, 6, blk)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert (a[:, 0] == a[:, 1]).all() and (a[:, 3] == a[:, 4]).all()
    assert (a[:, [0, 2, 3, 5]].sum(1) == 4).all()


def test_k1_prediction_zero_without_composition(toy):
    p = toy["point"][A_SEC]
    e = K.estimate(*p["pairs"]["V1"])
    g = K.k1_guard(p["pairs"]["V1"], e["limit"]["rho1"])
    assert g is not None and g[0] < 0 < g[1] and g[2] < 0
    pr = K.k1_prediction(e["params"], p["dec"], g)
    assert pr["status"] == "ok"
    assert abs(pr["pred_xi_trap"]) < 1e-12 and abs(pr["pred_rho1"]) < 1e-12


def test_k1_prediction_follows_mechanical_term():
    """Leavers with low SAMI and no entrants: stayers start above the mean, so
    M_t < 0; the frozen projection and the zeroth-order row both predict
    a lower V0 attractor (sign of M̄)."""
    w = K.make_toy_world(seed=3, entrant_shift=0.0, leaver_shift=-1.5)
    _, _, _, sks = _build(w)
    k = ("metro", "pi", "main", "B", "overlap")
    p = K.run_spec(sks[k])
    d = p["dec"]
    mbar = float((d.n_S * d.M_t).sum() / d.n_S.sum())
    assert mbar < 0
    e = K.estimate_spec(p, "B")
    r1 = e["V1"]["limit"]["rho1"]
    pr = K.k1_prediction(e["V1"]["params"], d, K.k1_guard(p["pairs"]["V1"], r1))
    assert pr["status"] == "ok" and pr["pred_xi_trap"] < 0
    assert K.zeroth_order(d, r1) < 0
    assert np.isfinite(e["V0"]["limit"]["xi_trap"] - e["V1"]["limit"]["xi_trap"])
    bad = K.k1_prediction(np.array([0.0, 0.5, 0.0, 1.0, -3.0, 0.0, 0.0]), d, None)
    assert bad["status"].startswith("projection_undefined") and np.isnan(bad["pred_xi_trap"])
    assert K.k1_prediction(None, d)["status"] == "no_V1_fit"


def test_h3b_rule():
    assert K.h3b_verdict(0.01, 0.0, 0.1, True)[0] == "not_testable"
    assert K.h3b_verdict(0.10, 0.12, 0.1, True)[0] == "pass"
    assert K.h3b_verdict(0.10, 0.131, 0.1, True)[0] == "fail"
    assert K.h3b_verdict(0.10, -0.10, 0.1, True)[0] == "fail"
    assert K.h3b_verdict(-0.10, -0.08, 0.1, True)[0] == "pass"
    assert K.h3b_verdict(np.nan, 0.1, 0.1, True)[0].startswith("not_evaluable")
    assert K.h3b_verdict(0.1, 0.1, np.nan, False)[0].startswith("not_evaluable")
    assert K.h3b_verdict(0.1, 0.1, 0.2, True)[1] == pytest.approx(0.1)


def test_h4_rule():
    assert K.h4_verdict((-1, 1, True), (-0.1, 0.2, True)) == "consistent"
    assert K.h4_verdict((-1, 1, True), (0.01, 0.2, True)) == "inconsistent"
    assert K.h4_verdict((0.1, 1, True), (-0.1, 0.2, True)) == "inconsistent"
    assert K.h4_verdict((-1, 1, False), (-0.1, 0.2, True)).startswith("not_evaluable")


def test_ci_rules_are_k1s():
    v = np.r_[np.arange(1.0, 100.0), np.nan]
    lo, hi, ok = K.ci(v, "tol0")
    assert not ok
    lo, hi, ok = K.ci(v, "tol5")
    assert ok and (lo, hi) == (3.0, 97.0)
    lo, hi, ok = K.ci(v, "literal")
    assert ok and (lo, hi) == (2.0, 98.0)
    lo, hi, ok = K.ci(np.arange(1.0, 201.0), "tol0")
    assert ok and (lo, hi) == (5.0, 196.0)
    lo, hi, ok = K.ci(np.r_[np.zeros(94), [np.nan] * 6], "tol5")
    assert not ok


def test_kill_rule_3_beta_out_of_range(tmp_path):
    w = K.make_toy_world(beta=1.6)
    with pytest.raises(K.KillRule) as e:
        K.run_pipeline(w, tmp_path, n_boot=2, n_jobs=1, n_min=10)
    assert e.value.rule == 3
    assert (tmp_path / "k2_beta_by_year.csv").exists()
    assert not (tmp_path / "k2_verdicts.csv").exists()


def test_kill_rule_1_panel_a_exit(tmp_path):
    """A Panel A area with no outcome in one year = exit and entry in Panel A
    -> H2 fails -> kill rule 1 before any bootstrap or verdict."""
    w = K.make_toy_world()
    s = w.areas[("metro", "2023-07")]
    code = sorted(s)[7]
    for oc in w.outcomes.values():
        oc.loc[list(s[code]), 2010] = np.nan
    with pytest.raises(K.KillRule) as e:
        K.run_pipeline(w, tmp_path, n_boot=2, n_jobs=1, n_min=10)
    assert e.value.rule == 1
    h2 = pd.read_csv(tmp_path / "k2_h2.csv")
    assert not h2["pass"].all()
    assert not (tmp_path / "k2_boot_draws.csv.gz").exists()


def test_kill_rule_4_n_min_drops_adjacent_transitions(toy):
    w, aggs = toy["world"], toy["aggs"]
    n = K.n_areas_by_year(w, aggs, "pi")
    n_min = n[2013] + 1 if n[2013] < n[2012] else n[2012] + 1
    drops, tab = K.min_n_drops(w, aggs, n_min)
    bad = drops[("pi", "main")]
    assert bad and all(n[y] < n_min for y in bad)
    lin = toy["lin"]["metro"][1]
    sk = K.build_skeleton(w, aggs, ("metro", "pi", "main", "B", "overlap"), lin, bad)
    for y in bad:
        assert y not in sk.transitions and y - 1 not in sk.transitions
        assert (y - 1 in sk.dropped) or y - 1 < 2003


def test_end_to_end_synthetic(tmp_path):
    w = K.make_toy_world()
    res = K.run_pipeline(w, tmp_path, n_boot=6, n_jobs=2, n_min=10, literal_checks=1,
                         meta=dict(mode="synthetic-test"))
    for f in ("k2_meta.json", "k2_lineages.csv", "k2_lineage_summary.csv", "k2_beta_by_year.csv",
              "k2_decomposition.csv", "k2_h1.csv", "k2_h2.csv", "k2_h3a.csv", "k2_h3b.csv",
              "k2_observables.csv", "k2_paired_differences.csv", "k2_verdicts.csv",
              "k2_boot_draws.csv.gz", "k2_literal_draw_check.csv", "k2_n_min.csv",
              "k2_aggregation_missing.csv"):
        assert (tmp_path / f).exists(), f
    v = res["verdicts"]
    assert set(v.hypothesis) >= {"H1", "H2", "H3a", "H3b", "H3c", "H4", "H4_state"}
    assert (v[v.hypothesis.isin(["H1", "H2"])].verdict == "pass").all()
    reg = v[v.registered.astype(bool)]
    assert set(reg[reg.hypothesis == "H3b"].outcome) == {"pi", "gdp"}
    assert set(reg[reg.hypothesis == "H3b"].period) == {"main", "secondary"}
    draws = pd.read_csv(tmp_path / "k2_boot_draws.csv.gz")
    s = draws[(draws.scope == "metro") & (draws.panel == "B") & (draws.link == "overlap")
              & (draws.variant == "V1") & (draws.conv == "limit") & (draws.scheme == "lineage")
              & (draws.outcome == "pi") & (draws.period == "main")]
    assert len(s) == 6 and s["xi_trap"].nunique() > 1
    assert set(draws.scheme) == set(K.SCHEMES) and set(draws.conv.astype(str)) == set(K.CONVENTIONS)
    lit = pd.read_csv(tmp_path / "k2_literal_draw_check.csv")
    assert (lit["abs_diff"] <= lit["tol"]).all()
    import json
    meta = json.loads((tmp_path / "k2_meta.json").read_text())
    assert meta["status"] == "complete" and meta["seed"] == K.SEED
