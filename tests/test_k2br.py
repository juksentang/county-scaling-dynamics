"""Run:  python3 -m pytest -q tests/test_k2br.py"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("run_k2br", ROOT / "scripts" / "run_k2br.py")
K = importlib.util.module_from_spec(_spec)
sys.modules["run_k2br"] = K
_spec.loader.exec_module(K)

TOY = K.Config(windows={"toy": (2002, 2011, "toy")}, n_min=100)


def toy(**kw):
    kw.setdefault("seed", 3)
    return K.synthetic_muni(**kw)


@pytest.fixture(scope="module")
def rich():
    """Splits (single parent, two parents, chained same-wave parent),
    selective entry, D3 lag entrants, agri zeros; no extinction (Brazil has
    none after 1991, so Panel A is composition-free)."""
    m, reg = toy(exits={}, entrant_offset=0.8, lag=2)
    return m, reg, K.analyse(m, TOY)


@pytest.fixture(scope="module")
def with_exits():
    """The same plus artificial exits (municipalities losing every record
    from 2007 on): Panel B gets n_L > 0 next to n_E > 0."""
    m, reg = toy(exits={2007: 5}, entrant_offset=0.8, lag=2)
    return m, reg, K.analyse(m, TOY)


def _dec(res, o, pn):
    return pd.DataFrame(res["panels"][(o, pn)]["decomposition"])


@pytest.mark.parametrize("fx", ["rich", "with_exits"])
def test_h1_identity_every_panel_outcome_family(fx, request):
    _, _, res = request.getfixturevalue(fx)
    if fx == "with_exits":
        v0 = _dec(res, "pib", "B").query("family == 'V0'")
        assert (v0.n_L > 0).any() and (v0.n_E > 0).any()
    for o in TOY.outcomes:
        for pn in K.PANELS:
            d = _dec(res, o, pn)
            assert len(d)
            h = K.judge_h1(d)
            assert h["H1"] == "pass", (o, pn, h)
            assert d["identity_residual"].abs().max() < 1e-12


def test_h1_v1p_rewritten_counts(rich):
    """V1p's identity: parents count as leavers AND entrants
    (n_L' = n_L + n_P, n_E' = n_E + n_P) and Lemma 1 still holds exactly."""
    _, _, res = rich
    d = _dec(res, "pib", "B")
    v0 = d[d.family == "V0"].set_index("year_t")
    v0p = d[d.family == "V0p"].set_index("year_t")
    assert len(v0p) == (v0.n_E > 0).sum() >= 2
    for t, r in v0p.iterrows():
        assert r.n_P > 0
        assert r.n_L == v0.at[t, "n_L"] + r.n_P
        assert r.n_E == v0.at[t, "n_E"] + r.n_P
        assert r.n_S == v0.at[t, "n_S"] - r.n_P
        assert abs(r.identity_residual) < 1e-12


def test_h1_detects_a_broken_identity(rich):
    _, _, res = rich
    d = _dec(res, "pib", "B").copy()
    d.loc[d.index[0], "identity_residual"] = 1e-9
    assert K.judge_h1(d)["H1"] == "fail"


def test_h2_panel_A_is_composition_free(rich):
    _, _, res = rich
    for o in TOY.outcomes:
        h = K.judge_h2(res, o)
        assert h["H2"] == "pass", h
        assert h["H2_pairs_max_diff"] == 0.0
        d = _dec(res, o, "A")
        v0 = d[d.family == "V0"]
        assert (v0.n_E == 0).all() and (v0.n_L == 0).all() and (v0.M == 0).all()


def test_h2_fails_with_extinctions(with_exits):
    """A municipality that loses every record removes territory from its AMC
    as well: Panel A is then not composition-free and H2 (a kill rule) fires
    -- the reason H2 is a kill rule only because Brazil has no extinction."""
    _, _, res = with_exits
    assert K.judge_h2(res, "pib")["H2"] == "fail"
    d = _dec(res, "pib", "A")
    assert (d[d.family == "V0"].n_L > 0).any()


def test_kill_rule_1_stops_end_to_end_with_partial_tables(with_exits):
    m, reg, _ = with_exits
    with pytest.raises(K.KillRule) as e:
        K.run_window("toy", m, reg, TOY, n_boot=2, log=lambda *a: None)
    assert "kill rule 1" in str(e.value) and "H2" in str(e.value)
    chk = pd.DataFrame(e.value.partial["checks"])
    assert (chk.dropna(subset=["H1"]).H1 == "pass").all()
    assert len(e.value.partial["decomposition"]) > 0


def test_h2_fails_when_A_loses_an_AMC_year():
    """An AMC with no record in one year makes an exit + entry in A: H2 (a
    kill rule) must fire."""
    m, _ = toy(splits={}, exits={})
    m.Y["pib"][5, 4] = np.nan
    m.Y["vab_agro"][5, 4] = np.nan
    m.N[5, 4] = np.nan
    res = K.analyse(m, TOY)
    assert K.judge_h2(res, "pib")["H2"] == "fail"
    _, msg = K.kill_h1_h2(res, TOY, "toy")
    assert msg and "H2" in msg


def test_panel_A_sums_equal_true_territory_including_D3_lag():
    """A's Y and N are the sums over the AMC's territorial pieces every year,
    also in the lag year where an entrant's population is already listed on
    its own but its output is still inside the parent."""
    m, _, tr = toy(exits={}, agri_nonpos=0, lag=3, return_truth=True)
    assert len(tr["lag_rows"]) == 3
    ids, inv = np.unique(tr["pieces_amc"], return_inverse=True)
    for o in TOY.outcomes:
        A = K.panel_A(m, o)
        assert A.present.all()
        for j in range(len(m.years)):
            want_y = np.bincount(inv, weights=tr["Y"][o][:, j], minlength=len(ids))
            want_n = np.bincount(inv, weights=tr["N"][:, j], minlength=len(ids))
            np.testing.assert_allclose(A.Y[:, j], want_y, rtol=1e-12)
            np.testing.assert_allclose(A.N[:, j], want_n, rtol=1e-12)
        assert A.lag_amc_years.sum() == 3


def test_lag_entrant_enters_B_in_its_first_outcome_year():
    m, _, tr = toy(exits={}, lag=2, return_truth=True)
    B = K.panel_B(m, "pib")
    for c in tr["lag_rows"]:
        first_n = np.flatnonzero(np.isfinite(m.N[c]))[0]
        first_y = np.flatnonzero(np.isfinite(m.Y["pib"][c]))[0]
        assert first_y == first_n + 1
        assert not B.present[c, first_n] and B.present[c, first_y]
        assert c in set(m.edges[:, 0])


def test_d1_nonpositive_rule_drops_unit_in_every_year():
    m, _ = toy(exits={}, agri_nonpos=3)
    Ba = K.panel_B(m, "vab_agro")
    Bp = K.panel_B(m, "pib")
    assert Ba.n_d1_dropped == 3 and Bp.n_d1_dropped == 0
    bad = (np.nan_to_num(m.Y["vab_agro"], nan=1.0) <= 0).any(axis=1)
    assert not Ba.present[bad].any()
    assert Ba.d1_dropped_by_year.max() == 3


def test_panel_A0_untouched_constant_set():
    m, _ = toy(exits={})
    B = K.panel_B(m, "pib")
    A0 = K.panel_A0(m, "pib", B)
    touched = np.zeros(len(m.code), bool)
    touched[np.unique(m.edges.ravel())] = True
    keep = A0.present[:, 0]
    assert (A0.present == keep[:, None]).all()
    assert not (keep & touched).any()
    assert keep.sum() == (B.present.all(axis=1) & ~touched).sum()


def test_build_muni_edges_follow_record_entry_year():
    """build_muni links to its parents every municipality without BOTH
    records in the window's first year (D3 a), also one already in the base
    population list (lag)."""
    years = list(range(2002, 2006))
    pop = {y: pd.Series({"100001": 1000.0, "100002": 2000.0, "100003": 900.0}) for y in years}
    pop[2002] = pd.Series({"100001": 1000.0, "100002": 2000.0, "100004": 50.0})
    pop[2003] = pop[2002]
    for y in (2004, 2005):
        pop[y] = pd.Series({"100001": 1000.0, "100002": 2000.0, "100003": 900.0, "100004": 50.0})
    pop[2003] = pd.Series({"100001": 1000.0, "100002": 2000.0, "100003": 900.0, "100004": 50.0})
    struct = {"pop": pop, "parents": {"100003": ["100001"], "100004": ["100002"]},
              "regions": {"immediate": {}, "uf": {}}}
    rows = []
    for y in years:
        for c in ("100001", "100002"):
            rows.append(("s", c, y))
        if y >= 2003:
            rows.append(("s", "100003", y))
        if y >= 2004:
            rows.append(("s", "100004", y))
    out = pd.DataFrame(rows, columns=["series", "code6", "year"]).assign(pib=1.0, vab_agro=1.0)
    cfg = K.Config(windows={"w": (2002, 2005, "s")})
    m, _ = K.build_muni(struct, "w", out, cfg)
    e = {(m.code[a], m.code[b]) for a, b in m.edges}
    assert e == {("100003", "100001"), ("100004", "100002")}
    assert len(np.unique(m.amc)) == 2


def test_merge_back_single_multi_and_chained_parents():
    Ct = np.array([1, 1, 1, 0, 0, 0, 1], bool)
    Ct1 = np.ones(7, bool)
    edges = np.array([(3, 0), (3, 1), (5, 4), (4, 2)])
    Y = np.array([[10, 6], [20, 15], [30, 20], [0, 9], [0, 7], [0, 3], [5, 5.5]], float)
    N = Y * 2
    ids, mg = K._merge_back(Ct, Ct1, edges, (Y, N))
    yt, _, pt = mg["t"]
    yt1, nt1, pt1 = mg["t1"]
    assert len(ids) == 3 and pt.all() and pt1.all()
    assert sorted(yt) == [5, 30, 30]
    assert sorted(yt1) == [5.5, 30, 30]
    np.testing.assert_allclose(sorted(nt1), [11, 60, 60])


def test_v1m_equals_v1_without_entry_and_drops_parent_jump(rich):
    m, _, res = rich
    d = _dec(res, "pib", "B")
    v1 = d[d.family == "V1"].set_index("year_t")
    v1m = d[d.family == "V1m"].set_index("year_t")
    v0 = d[d.family == "V0"].set_index("year_t")
    for t in v0.index:
        if v0.at[t, "n_E"] == 0:
            assert v1m.at[t, "n_S"] == v1.at[t, "n_S"]
        else:
            assert v1m.at[t, "n_groups_t"] == v1m.at[t, "n_groups_t1"] == v1m.at[t, "n_S"]
            assert v1m.at[t, "n_S"] <= v1.at[t, "n_S"]


def test_v1m_pairs_bitwise_v1_on_a_panel_without_entry():
    m, _ = toy(splits={}, exits={2006: 4})
    res = K.analyse(m, TOY)
    pv1 = res["panels"][("pib", "B")]["pairs"]["V1"]
    pv1m = res["panels"][("pib", "B")]["pairs"]["V1m"]
    pv1p = res["panels"][("pib", "B")]["pairs"]["V1p"]
    for a, b in ((pv1, pv1m), (pv1, pv1p)):
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_kill_rule_5_removes_years_and_their_transitions():
    m, reg = toy(exits={})
    nB = K.panel_B(m, "pib").present.sum(axis=0)
    cfg = replace(TOY, n_min=int(nB[5]))
    res = K.analyse(m, cfg)
    kept = res["keep_years"]["pib"]
    assert set(kept) == set(m.years[nB >= nB[5]]) and len(kept) < len(m.years)
    for pn in K.PANELS:
        yrs = set(_dec(res, "pib", pn)["year_t"])
        assert yrs and all(y in kept and y + 1 in kept for y in yrs)


def test_kill_rule_4_beta_out_of_range_stops_with_partial_tables():
    m, reg = toy(exits={})
    cfg = replace(TOY, beta_range=(1.2, 1.3))
    with pytest.raises(K.KillRule) as e:
        K.run_window("toy", m, reg, cfg, n_boot=2, log=lambda *a: None)
    assert "kill rule 4" in str(e.value)
    assert {"yearly", "checks", "decomposition", "units"} <= set(e.value.partial)
    assert len(e.value.partial["yearly"]) > 0


def test_kill_rule_4_in_window_2_stops_before_any_draw_or_verdict_of_window_1(monkeypatch):
    """Kill rules run on every window before any look at H3 / H4: a beta_t
    miss in the second window stops the run with no bootstrap draw, no
    prediction and no verdict of the first (passing) window."""
    m1, r1 = toy(exits={})
    m2, r2 = toy(exits={}, beta=1.6, seed=4)
    cfg = K.Config(windows={"w1": (2002, 2011, "toy"), "w2": (2002, 2011, "toy")}, n_min=100)
    calls = []
    for name in ("bootstrap", "_draw_worker", "draw_muni", "k1_prediction", "judge_h3a",
                 "judge_h3b", "judge_h4", "finish_window", "flat_contrasts"):
        orig = getattr(K, name)
        monkeypatch.setattr(K, name, lambda *a, _n=name, _o=orig, **k: calls.append(_n) or _o(*a, **k))
    _, msg_w1 = K.kill_beta(K.analyse(m1, cfg), cfg, "w1")
    assert msg_w1 is None
    with pytest.raises(K.KillRule) as e:
        K.run_windows({"w1": (m1, r1), "w2": (m2, r2)}, cfg, n_boot=2, log=lambda *a: None)
    assert "kill rule 4" in str(e.value) and "[w2]" in str(e.value) and "[w1]" not in str(e.value)
    assert calls == []
    assert set(e.value.partial) == set(K.PARTIAL_KEYS)
    for k in ("yearly", "checks", "decomposition", "units"):
        assert set(pd.DataFrame(e.value.partial[k]).window) == {"w1", "w2"}


def test_two_windows_pass_then_both_are_judged():
    m1, r1 = toy(exits={})
    m2, r2 = toy(exits={}, seed=4)
    cfg = K.Config(windows={"w1": (2002, 2011, "toy"), "w2": (2002, 2011, "toy")}, n_min=100)
    out = K.run_windows({"w1": (m1, r1), "w2": (m2, r2)}, cfg, n_boot=2, log=lambda *a: None)
    assert set(pd.DataFrame(out["verdicts"]).window) == {"w1", "w2"}
    assert set(pd.DataFrame(out["draws"]).window) == {"w1", "w2"}


def test_k1_prediction_exactly_zero_without_composition_change():
    m, _ = toy(splits={}, exits={})
    res = K.analyse(m, TOY)
    d = res["panels"][("pib", "B")]
    f = d["fits"]["V1"]
    p = K.k1_prediction(d["k1"], f["params"], f["limit"]["xi_trap"], f["limit"]["rho1"])
    assert p["status"] == "ok", p
    assert p["pred_dxi_star"] == 0.0 and p["pred_drho1"] == 0.0
    assert p["M_bar"] == 0.0 and p["zeroth_order_dxi_star"] == 0.0


def test_k1_prediction_sign_follows_selective_entry():
    """Entrants above the mean (m_E > 0) give M_t < 0: the V0 attractor is
    predicted lower than V1's, as is the zeroth-order row, and the observed
    V0 - V1 difference has the same sign on a strong-dose toy panel."""
    m, _ = toy(n0=600, splits={2005: [(150, 1, False)]}, exits={}, entrant_offset=1.5,
               agri_nonpos=0, seed=16)
    res = K.analyse(m, TOY)
    d = res["panels"][("pib", "B")]
    dec = pd.DataFrame(d["decomposition"])
    assert (dec[(dec.family == "V0") & (dec.n_E > 0)].M < 0).all()
    f = d["fits"]["V1"]
    p = K.k1_prediction(d["k1"], f["params"], f["limit"]["xi_trap"], f["limit"]["rho1"])
    assert p["status"] == "ok", p
    assert p["pred_dxi_star"] < 0 and p["zeroth_order_dxi_star"] < 0
    obs = d["fits"]["V0"]["limit"]["xi_trap"] - f["limit"]["xi_trap"]
    assert obs < 0
    assert abs(obs - p["pred_dxi_star"]) <= 0.30 * abs(p["pred_dxi_star"])


def test_k1_prediction_refuses_an_unconfined_V1_fit():
    """A fitted cubic that escapes (a3 > 0 here) has no stationary law: the
    frozen theory refuses and H3b becomes 'not evaluable', reported."""
    m, _ = toy(n0=300, splits={2005: [(90, 1, False)]}, exits={}, entrant_offset=1.5,
               agri_nonpos=0, seed=11)
    d = K.analyse(m, TOY)["panels"][("pib", "B")]
    f = d["fits"]["V1"]
    assert f["params"][3] > 0
    p = K.k1_prediction(d["k1"], f["params"], f["limit"]["xi_trap"], f["limit"]["rho1"])
    assert p["status"].startswith("prediction_undefined") and "confined" in p["status"]
    assert np.isnan(p["pred_dxi_star"]) and np.isfinite(p["zeroth_order_dxi_star"])
    v = K.judge_h3b(-0.1, p["pred_dxi_star"], _ci(-0.1, 0.1))
    assert v["H3b"] == "not_evaluable_prediction_undefined"


def _unconfined_toy():
    m, _ = toy(n0=300, splits={2005: [(90, 1, False)]}, exits={}, entrant_offset=1.5,
               agri_nonpos=0, seed=11)
    return m, K.analyse(m, TOY)["panels"][("pib", "B")]


def test_k1_guard_is_the_registered_rule():
    _, d = _unconfined_toy()
    xt, _ = d["pairs"]["V1"]
    r1 = d["fits"]["V1"]["limit"]["rho1"]
    g = K.k1_guard(d["pairs"]["V1"], r1)
    assert g == (float(np.percentile(xt, 1)), float(np.percentile(xt, 99)), -abs(r1 - 1.0))
    assert g[0] < 0 < g[1] and g[2] < 0
    assert K.k1_guard((xt[:10], xt[:10]), r1) is None
    assert K.k1_guard(d["pairs"]["V1"], 1.0) is None
    assert K.k1_guard(d["pairs"]["V1"], np.nan) is None


def test_guarded_prediction_defined_where_unguarded_refuses():
    """The escaping fitted cubic (a3 > 0): unguarded the frozen theory
    refuses; with the registered guard the chain is the one K1 simulates and
    the projection is defined, with the sign of the selective entry."""
    _, d = _unconfined_toy()
    f = d["fits"]["V1"]
    args = (d["k1"], f["params"], f["limit"]["xi_trap"], f["limit"]["rho1"])
    g = K.k1_guard(d["pairs"]["V1"], f["limit"]["rho1"])
    pg = K.k1_prediction(*args, guard=g)
    pu = K.k1_prediction(*args, guard=None)
    assert pu["status"].startswith("prediction_undefined") and not pu["guarded"]
    assert pg["status"] == "ok" and pg["guarded"]
    assert (pg["guard_lo"], pg["guard_hi"], pg["guard_slope_floor"]) == g
    assert np.isfinite(pg["pred_dxi_star"]) and pg["pred_dxi_star"] < 0
    m2, _ = toy(n0=600, splits={2005: [(150, 1, False)]}, exits={}, entrant_offset=1.5,
                agri_nonpos=0, seed=16)
    d2 = K.analyse(m2, TOY)["panels"][("pib", "B")]
    f2 = d2["fits"]["V1"]
    a2 = (d2["k1"], f2["params"], f2["limit"]["xi_trap"], f2["limit"]["rho1"])
    q_g = K.k1_prediction(*a2, guard=K.k1_guard(d2["pairs"]["V1"], f2["limit"]["rho1"]))
    q_u = K.k1_prediction(*a2)
    assert q_g["status"] == q_u["status"] == "ok"
    assert q_g["pred_dxi_star"] != q_u["pred_dxi_star"]


def test_window_judges_h3b_on_the_guarded_prediction():
    """finish_window: the H3b prediction is the guarded projection; the
    unguarded one is reported beside it (predictions and verdicts)."""
    m, reg = toy(n0=300, splits={2005: [(90, 1, False)]}, exits={}, entrant_offset=1.5,
                 agri_nonpos=0, seed=11)
    cfg = replace(TOY, beta_range=(0.0, 5.0))
    out = K.run_window("toy", m, reg, cfg, n_boot=3, schemes=("amc",), log=lambda *a: None)
    p = pd.DataFrame(out["predictions"]).set_index("outcome").loc["pib"]
    v = pd.DataFrame(out["verdicts"]).set_index("outcome").loc["pib"]
    assert p["guarded"] and p["status"] == "ok"
    assert str(p["unguarded_status"]).startswith("prediction_undefined")
    assert np.isnan(p["unguarded_pred_dxi_star"])
    assert v["H3b_predicted"] == p["pred_dxi_star"]
    assert v["H3b_prediction_status"] == "ok"
    assert str(v["H3b_unguarded_status"]).startswith("prediction_undefined")
    assert v["H3b"] != "not_evaluable_prediction_undefined"
    d = K.analyse(m, cfg)["panels"][("pib", "B")]
    assert (p["guard_lo"], p["guard_hi"], p["guard_slope_floor"]) == K.k1_guard(
        d["pairs"]["V1"], d["fits"]["V1"]["limit"]["rho1"])


def test_k1_prediction_undefined_is_reported_not_raised():
    p = K.k1_prediction([], np.full(7, np.nan), np.nan, np.nan)
    assert p["status"].startswith("prediction_undefined")
    assert np.isnan(p["pred_dxi_star"])


def test_identity_draw_reproduces_the_data_pipeline(rich):
    """A 'draw' of every AMC once re-runs the whole pipeline from municipal
    values and returns the data's observables (rows permuted only)."""
    m, reg, res = rich
    bb = K.make_block_base(m, reg)
    md = K.muni_from_amc_seq(bb, np.arange(len(bb.amc_ids)))
    r2 = K.analyse(md, TOY, keep_years=res["keep_years"])
    assert r2["panels"][("pib", "B")]["pairs"] is not None
    a, b = K.flat_observables(res), K.flat_observables(r2)
    assert a.keys() == b.keys()
    for key, d in res["panels"].items():
        for v, (x, dx) in d["pairs"].items():
            x2, dx2 = r2["panels"][key]["pairs"][v]
            np.testing.assert_allclose(np.sort(x), np.sort(x2), atol=1e-10)
            np.testing.assert_allclose(np.sort(dx), np.sort(dx2), atol=1e-10)
    for k in a:
        if np.isfinite(a[k]) or np.isfinite(b[k]):
            assert np.isclose(a[k], b[k], rtol=1e-3, atol=1e-6), (k, a[k], b[k])
    da = _dec(res, "pib", "B").sort_values(["family", "year_t"]).reset_index(drop=True)
    db = _dec(r2, "pib", "B").sort_values(["family", "year_t"]).reset_index(drop=True)
    for c in ("n_S", "n_E", "n_L", "n_P"):
        assert (da[c] == db[c]).all()
    np.testing.assert_allclose(da.M, db.M, atol=1e-12)


def test_draw_copies_are_separate_units_with_remapped_genealogy(rich):
    m, reg, _ = rich
    bb = K.make_block_base(m, reg)
    k = int(np.argmax([len(x) for x in bb.members]))
    assert len(bb.members[k]) > 1
    md = K.muni_from_amc_seq(bb, np.array([k, k]))
    n = len(bb.members[k])
    assert len(md.code) == 2 * n
    assert len(np.unique(md.amc)) == 2
    assert md.edges.shape[0] == 2 * bb.local_edges[k].shape[0]
    first = md.edges[: len(md.edges) // 2]
    assert (first < n).all() and (md.edges[len(md.edges) // 2:] >= n).all()


def test_block_assignment_by_most_populous_base_member():
    m, reg = toy(exits={})
    reg = dict(reg)
    reg["immediate"] = np.arange(len(m.code)) % 7
    bb = K.make_block_base(m, reg)
    lab_ids = np.unique(reg["immediate"])
    for a, mem in enumerate(bb.members):
        pb = np.nan_to_num(m.pop_base[mem], nan=-1.0)
        want = reg["immediate"][mem[np.argmax(pb)]]
        assert lab_ids[bb.block_of_amc["immediate"][a]] == want
    for a, mem in enumerate(bb.members):
        assert len(set(reg["uf"][mem])) == 1


def test_bootstrap_deterministic_and_worker_invariant(rich):
    m, reg, res = rich
    bb = K.make_block_base(m, reg)
    warm = {(o, pn, v): f["params"] for (o, pn), d in res["panels"].items()
            for v, f in d["fits"].items()}
    a = K.bootstrap(bb, TOY, res["keep_years"], warm, 0, "amc", 3, workers=1)
    b = K.bootstrap(bb, TOY, res["keep_years"], warm, 0, "amc", 3, workers=2)
    c = K.bootstrap(bb, TOY, res["keep_years"], warm, 0, "immediate", 3, workers=1)
    for x, y in zip(a, b):
        for k in x:
            assert (x[k] == y[k]) or (np.isnan(x[k]) and np.isnan(y[k]))
    assert any(a[0][k] != c[0][k] for k in a[0] if np.isfinite(a[0][k]))


def test_paired_draw_feeds_A_and_B_from_one_sample(rich):
    """One draw, both panels: the AMC multiplicities seen by A equal those
    of B's municipalities (paired bootstrap, H4)."""
    m, reg, _ = rich
    bb = K.make_block_base(m, reg)
    md = K.draw_muni(bb, "amc", np.random.default_rng(5))
    A = K.panel_A(md, "pib")
    assert A.Y.shape[0] == len(np.unique(md.amc)) == len(bb.amc_ids)
    np.testing.assert_allclose(np.nansum(A.N[:, -1]),
                               np.nansum(np.where(np.isfinite(md.N[:, -1]), md.N[:, -1], 0)))


def test_ci_rules():
    v = np.r_[np.linspace(-1, 1, 97), [np.nan] * 3]
    c = K.ci(v)
    assert c["n_draws"] == 100 and c["n_nonfinite"] == 3
    assert c["tol5_ok"] and not c["tol0_ok"]
    assert c["ci_rule"] == "tol0" and c["ci_status"] == "not_evaluable_nonfinite"
    assert np.isnan(c["ci_lo"]) and np.isnan(c["ci_hi"])
    assert np.isfinite(c["tol5_lo"]) and np.isfinite(c["tol5_hi"])
    assert c["literal_hi"] == np.inf and c["literal_lo"] == -np.inf
    c1 = K.ci(np.r_[np.linspace(-1, 1, 99), np.nan])
    assert np.isfinite(c1["literal_lo"]) and np.isfinite(c1["literal_hi"])
    v2 = np.r_[np.linspace(-1, 1, 90), [np.nan] * 10]
    c2 = K.ci(v2)
    assert not c2["tol5_ok"] and c2["ci_status"] == "not_evaluable_nonfinite"
    assert c2["literal_lo"] == -np.inf and c2["literal_hi"] == np.inf
    c3 = K.ci(np.linspace(0, 1, 101))
    assert c3["tol0_ok"] and c3["ci_status"] == "ok"
    assert c3["ci_lo"] == np.linspace(0, 1, 101)[2] and c3["ci_hi"] == np.linspace(0, 1, 101)[98]


def test_ci_tol0_primary_and_symmetric_rank_rule():
    rng = np.random.default_rng(5)
    v = rng.normal(size=1000)
    c = K.ci(v)
    s = np.sort(v)
    assert c["ci_status"] == "ok" and c["ci_rule"] == "tol0"
    assert (c["ci_lo"], c["ci_hi"]) == (s[24], s[975])
    assert (c["tol0_lo"], c["tol0_hi"]) == (c["tol5_lo"], c["tol5_hi"]) == (s[24], s[975])
    assert (c["literal_lo"], c["literal_hi"]) == (s[24], s[975])
    assert K.order_ci(np.arange(40.0)) == (0.0, 39.0)
    v1 = v.copy()
    v1[7] = np.inf
    c1 = K.ci(v1)
    assert not c1["tol0_ok"] and c1["ci_status"] == "not_evaluable_nonfinite"
    assert np.isnan(c1["ci_lo"]) and c1["tol5_ok"] and c1["literal_ok"]
    assert K.judge_h3b(0.1, 0.5, c1)["H3b"] == "not_evaluable_bootstrap"
    assert K.judge_h4(c1, c) == "not_evaluable_bootstrap"


def _ci(lo, hi, ok=True):
    return {"ci_lo": lo, "ci_hi": hi, "ci_status": "ok" if ok else "not_evaluable_nonfinite"}


def test_judge_h3b_gate_and_criteria():
    ci1 = _ci(-0.1, 0.1)
    assert K.judge_h3b(0.2, 0.04, ci1)["H3b"] == "untestable"
    assert K.judge_h3b(0.11, 0.1, ci1)["H3b"] == "pass"
    assert K.judge_h3b(0.131, 0.1, ci1)["H3b"] == "fail"
    assert K.judge_h3b(0.129, 0.1, ci1)["H3b"] == "pass"
    assert K.judge_h3b(-0.1, 0.1, ci1)["H3b"] == "fail"
    assert K.judge_h3b(np.nan, 0.1, ci1)["H3b"] == "fail_nonfinite_observed"
    assert K.judge_h3b(0.1, np.nan, ci1)["H3b"] == "not_evaluable_prediction_undefined"
    assert K.judge_h3b(0.1, 0.1, _ci(np.nan, np.nan, False))["H3b"] == "not_evaluable_bootstrap"


def test_judge_h4():
    assert K.judge_h4(_ci(-1, 1), _ci(-0.1, 0.2)) == "consistent"
    assert K.judge_h4(_ci(0.01, 1), _ci(-0.1, 0.2)) == "inconsistent"
    assert K.judge_h4(_ci(-1, 1), _ci(0.01, 0.2)) == "inconsistent"
    assert K.judge_h4(_ci(-1, 1, False), _ci(-1, 1)) == "not_evaluable_bootstrap"


def test_judge_h3a():
    d = pd.DataFrame({"family": ["V0"] * 3, "n_E": [0, 2, 3], "M": [0.0, 1e-3, -2e-9],
                      "M_over_bound": [np.nan, 1.0, 1.0]})
    assert K.judge_h3a(d)["H3a"] == "zero_on_some_transition"
    d.loc[2, "M"] = -2e-3
    assert K.judge_h3a(d)["H3a"] == "nonzero"
    assert K.judge_h3a(d[d.n_E == 0])["H3a"] == "no_entry_transition"


def test_end_to_end_synthetic_window(tmp_path):
    m, reg = toy(exits={}, entrant_offset=0.5, lag=1)
    out = K.run_window("toy", m, reg, TOY, n_boot=4, log=lambda *a: None)
    v = pd.DataFrame(out["verdicts"])
    assert set(v.outcome) == set(TOY.outcomes)
    assert (v.H1 == "pass").all() and (v.H2 == "pass").all() and (v.H3a == "nonzero").all()
    assert v.H3b.isin(["untestable", "pass", "fail", "fail_nonfinite_observed",
                       "not_evaluable_bootstrap", "not_evaluable_prediction_undefined"]).all()
    for s in K.SCHEMES:
        assert f"H4_{s}" in v
    obs = pd.DataFrame(out["observables"])
    assert set(obs.panel) == set(K.PANELS) and set(obs.variant) == set(K.VARIANTS)
    assert set(obs.convention) == set(K.CONVENTIONS)
    assert {f"{s}_n_draws" for s in K.SCHEMES} <= set(obs.columns)
    assert (obs["amc_n_draws"] == 4).all()
    con = pd.DataFrame(out["contrasts"])
    assert set(con.contrast) == set(K.CONTRASTS)
    files = K.write_tables(out, tmp_path, prefix="t")
    assert all(f.exists() for f in files)


def test_attempt_log_check():
    K.check_attempts([], 0, "abc")
    with pytest.raises(K.KillRule):
        K.check_attempts([{"status": "completed"}], 0, "abc")
    with pytest.raises(K.KillRule):
        K.check_attempts([{"status": "killed", "code_commit": "abc"}], 0, "abc")
    K.check_attempts([{"status": "killed", "code_commit": "old"}], 0, "abc")
    with pytest.raises(K.KillRule):
        K.check_attempts([{"status": "started", "code_commit": "abc"}], 0, "abc")
    K.check_attempts([{"status": "started", "code_commit": "abc"}], 1, "abc")


def test_outcome_frame_checks():
    df = pd.DataFrame({"series": ["ref2010", "ref2010"], "code6": ["110001", "110001"],
                       "year": [2002, 2002], "pib": [1.0, 2.0], "vab_agro": [1.0, 1.0]})
    with pytest.raises(K.KillRule):
        K.check_outcome_frame(df)
    with pytest.raises(K.KillRule):
        K.check_outcome_frame(df.drop(columns="pib"))
    with pytest.raises(K.KillRule):
        K.check_outcome_frame(df.iloc[:1].assign(series="other"))
    assert len(K.check_outcome_frame(df.iloc[:1])) == 1


def test_no_outcome_reader_yet():
    """Order step 2: the PIB reader (step 3) does not exist, so the real run
    cannot read an outcome; the script itself names no PIB file."""
    if (ROOT / "scripts" / "k2br_pib.py").exists():
        pytest.skip("order step 3 reached")
    with pytest.raises(K.KillRule):
        K._pib_module()
    src = (ROOT / "scripts" / "run_k2br.py").read_text()
    assert "Pib_Municipios" not in src and "base_de_dados" not in src


@pytest.mark.skipif(not (K.DATA / "raw" / "dtb" / "DTB_2025.zip").exists(),
                    reason="Brazilian structure data not on this machine")
def test_structure_reproduces_committed_coverage():
    struct = K.load_structure()
    cfg = K.Config(windows={"s": (2000, 2026, "s")})
    m, reg = K.build_muni(struct, "s", None, cfg)
    cov = pd.read_csv(ROOT / "results" / "tables" / "k2br_coverage_panelA_by_year.csv")
    a = cov[cov.panel == "A_AMC2000"].set_index("year")
    a0 = cov[cov.panel == "A0_untouched2000"].set_index("year")
    sizes = pd.Series(m.amc).value_counts()
    assert len(sizes) == a.n_units.iloc[0]
    assert (sizes > 1).sum() == a.n_amc_multi_member.iloc[0]
    assert sizes.max() == a.max_members.iloc[0]
    A = K.panel_A(m, "pib")
    B = K.panel_B(m, "pib")
    A0 = K.panel_A0(m, "pib", B)
    for j, y in enumerate(m.years):
        if y not in a.index:
            continue
        assert A.present[:, j].sum() == a.at[y, "n_units"]
        assert np.isclose(np.nansum(A.N[:, j]), a.at[y, "pop_total"], rtol=1e-12)
        assert A0.present[:, j].sum() == a0.at[y, "n_units"]
    rng = np.random.default_rng(0)
    for o in m.Y:
        m.Y[o] = np.where(np.isfinite(m.Y[o]),
                          m.N ** 1.05 * np.exp(rng.normal(0, 0.5, m.N.shape)), np.nan)
    da = pd.DataFrame(K.build_variants(K.panel_A(m, "pib"), cfg)["decomposition"])
    assert (da[da.family == "V0"][["n_E", "n_L"]] == 0).all().all()
    db = pd.DataFrame(K.build_variants(K.panel_B(m, "pib"), cfg)["decomposition"])
    assert db["identity_residual"].abs().max() < 1e-10
    ent = pd.read_csv(ROOT / "results" / "tables" / "k2br_coverage_panelB_by_transition.csv")
    v0 = db[db.family == "V0"].set_index("year_t")
    n_checked = 0
    for r in ent[ent.gap_years == 1].itertuples():
        if int(r.t) in v0.index and int(r.t) != 2022:
            assert v0.at[int(r.t), "n_E"] == r.n_E and v0.at[int(r.t), "n_S"] == r.n_S
            assert v0.at[int(r.t), "n_P"] == r.n_parents_changed_definition
            n_checked += 1
    assert n_checked >= 20


def test_without_splits_A_and_B_coincide_in_data_and_every_draw():
    """No split: every AMC is one municipality, so A and B are the same
    panel and every H4 contrast is exactly 0 on the data and on each draw
    (the paired bootstrap feeds both the same sample)."""
    m, reg = toy(splits={}, exits={}, agri_nonpos=0)
    out = K.run_window("toy", m, reg, TOY, n_boot=3, log=lambda *a: None)
    con = pd.DataFrame(out["contrasts"])
    h4 = con[con.contrast.isin(["V1B_minus_A", "V0B_minus_A", "V0B_minus_V1B"])]
    assert (h4.estimate == 0).all()
    lim = h4[h4.convention == "limit"]
    for s in K.SCHEMES:
        assert (lim[f"{s}_ci_lo"] == 0).all() and (lim[f"{s}_ci_hi"] == 0).all()
        assert (h4[f"{s}_ci_lo"].fillna(0) == 0).all()
    v = pd.DataFrame(out["verdicts"])
    assert (v.H4 == "consistent").all() and (v.H3a == "no_entry_transition").all()


def test_parent_channel_explains_toy_H4_gap():
    """With large toy splits the V1(B) - A gap is the parent re-definition
    channel: V1m (entrants merged back) is closer to A than V1 is."""
    m, reg = toy(exits={}, agri_nonpos=0)
    res = K.analyse(m, TOY)
    obs = K.flat_observables(res)
    g_v1 = abs(obs[("pib", "B", "V1", "limit", "rho1")] - obs[("pib", "A", "V0", "limit", "rho1")])
    g_v1m = abs(obs[("pib", "B", "V1m", "limit", "rho1")] - obs[("pib", "A", "V0", "limit", "rho1")])
    assert g_v1m < g_v1


def test_execute_classifies_completed_killed_crashed():
    allout = {"yearly": []}
    assert K.execute(lambda: {"existence_records": 3}, allout) == (
        "completed", "completed", {"existence_records": 3})

    def killed():
        raise K.KillRule("kill rule 4: beta", partial={"yearly": [{"y": 1}]})
    d, st, extra = K.execute(killed, allout)
    assert st == "killed" and d.startswith("killed: kill rule 4") and extra == {}
    assert allout["yearly"] == [{"y": 1}]
    for exc in (ValueError("bad"), KeyError("k"), ZeroDivisionError(), MemoryError()):
        def crash(e=exc):
            raise e
        d, st, extra = K.execute(crash, allout)
        assert st == "crashed" and d.startswith(f"crashed: {type(exc).__name__}")
        assert extra["exception_type"] == type(exc).__name__ and "Traceback" in extra["traceback"]


def test_attempt_log_refuses_a_crashed_code_commit():
    with pytest.raises(K.KillRule, match="crashed"):
        K.check_attempts([{"status": "started", "code_commit": "abc"},
                          {"status": "crashed", "code_commit": "abc"}], 1, "abc")
    K.check_attempts([{"status": "started", "code_commit": "old"},
                      {"status": "crashed", "code_commit": "old"}], 1, "abc")


@pytest.mark.parametrize("where,exc", [("_pooled", RuntimeError("boom")),
                                       ("_pooled", FloatingPointError()),
                                       ("process_spec", KeyError("a3")),
                                       ("stationary_law", ZeroDivisionError())])
def test_k1_prediction_any_theory_exception_is_undefined(monkeypatch, where, exc):
    from src import k1_theory as T
    m, _ = toy(splits={}, exits={})
    d = K.analyse(m, TOY)["panels"][("pib", "B")]
    f = d["fits"]["V1"]

    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(T, where, boom)
    p = K.k1_prediction(d["k1"], f["params"], f["limit"]["xi_trap"], f["limit"]["rho1"],
                        guard=K.k1_guard(d["pairs"]["V1"], f["limit"]["rho1"]))
    assert p["status"].startswith(f"prediction_undefined: {type(exc).__name__}")
    assert np.isnan(p["pred_dxi_star"]) and np.isnan(p["pred_drho1"])
    assert np.isfinite(p["zeroth_order_dxi_star"])


def test_unpublished_vab_years_are_removed_by_kill_rule_5_for_vab_only():
    """IBGE publishes no VAB agropecuaria for 2022-2023 (k2br_pib.VAB_NOT_PUBLISHED):
    on a toy panel whose last two years lack every vab_agro record, the data
    pass runs, kill rule 5 removes those years for vab_agro only, and PIB
    keeps every year."""
    m, reg = toy(exits={}, entrant_offset=0.8, lag=2)
    m.Y["vab_agro"][:, -2:] = np.nan
    dp = K.data_pass("toy", m, TOY)
    ky = dp["res"]["keep_years"]
    assert list(ky["pib"]) == list(m.years)
    assert list(ky["vab_agro"]) == list(m.years[:-2])
    assert not dp["kill"], dp["kill"]
    for pn in K.PANELS:
        d = dp["res"]["panels"][("vab_agro", pn)]
        assert len(d["pairs"]["V0"][0]) > 0
    n_a0 = dp["res"]["n_units"]["vab_agro"]["A0"]
    assert (n_a0[:-2] > 0).all()
    B = K.panel_B(m, "pib")
    assert (K.panel_A0(m, "pib", B, m.years).present == K.panel_A0(m, "pib", B).present).all()


def test_unpublished_vab_years_end_to_end_synthetic(tmp_path):
    m, reg = toy(exits={}, entrant_offset=0.8, lag=2)
    m.Y["vab_agro"][:, -2:] = np.nan
    o = K.run_window("toy", m, reg, TOY, n_boot=3, log=lambda *a: None)
    v = pd.DataFrame(o["verdicts"]).set_index("outcome")
    assert v.loc["vab_agro", "H1"] == "pass" and v.loc["vab_agro", "H2"] == "pass"


def test_empty_decomposition_fails_h1_h2_instead_of_crashing():
    assert K.judge_h1(pd.DataFrame([]))["H1"] == "fail"
