"""Run: python3 -P -m pytest -q tests/test_k4_mask.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_k4_mask as M

K2, KBR = M.K2, M.KBR


@pytest.fixture(autouse=True)
def no_real_data(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a unit test tried to read real data")
    for mod, names in ((K2, ("load_outcome", "load_world")),
                       (K2.COV, ("load_population", "load_vintage")),
                       (KBR, ("load_structure", "_pib_module"))):
        for n in names:
            monkeypatch.setattr(mod, n, boom)
    monkeypatch.setattr(M, "load_real_systems", boom)


@pytest.fixture(scope="module")
def us():
    return M.synthetic_us_system(n=240, seed=3, lin_size=2)


@pytest.fixture(scope="module")
def br():
    return M.synthetic_br_system(n=240, seed=4)


def _cell(name, rep=0):
    return [c for c in M.stage_cells("confirmatory", n_rep=rep + 1) if c["cell"] == name
            and c["rep"] == rep][0]


def test_grid_and_pilot_cells():
    g = M.cell_grid()
    assert len(g) == 36 and [c["cell_i"] for c in g] == list(range(36))
    assert len(M.PILOT_CELLS) == 18 and set(M.PILOT_CELLS) <= {c["cell"] for c in g}
    assert len(M.stage_cells("pilot")) == 18
    assert len(M.stage_cells("confirmatory")) == 36 * M.N_REP
    for fl, f in M.FRACTIONS.items():
        assert abs(f / (1 - f) - M.DELTA_EVENT_TARGET[fl]) < 1e-12


@pytest.mark.parametrize("org", M.ORGS)
@pytest.mark.parametrize("fl", list(M.FRACTIONS))
def test_mask_fraction_calibrated(us, org, fl):
    for pos in M.POSITIONS:
        c = _cell(f"{fl}|{pos}|{org}|exit")
        m, att, st = M.construct_mask("US", "dev", 0, c, us.xi_base, us.geo, us.labels)
        assert st == "ok"
        f = M.FRACTIONS[fl]
        if org == "random":
            assert m.sum() == int(round(f * len(m)))
        else:
            assert abs(m.mean() - f) <= M.CAL_TOL * f
            for b in np.unique(us.geo[m]):
                assert m[us.geo == b].all()


@pytest.mark.parametrize("org", M.ORGS)
def test_mask_position_order(us, org):
    means = {}
    for pos in M.POSITIONS:
        c = _cell(f"intermediate|{pos}|{org}|exit")
        m, _, _ = M.construct_mask("US", "dev", 0, c, us.xi_base, us.geo, us.labels)
        means[pos] = us.xi_base[m].mean()
    assert means["low"] < means["mid"] < means["high"]
    assert means["low"] < 0 < means["high"]


def test_random_mask_draws_from_pool(us):
    c = _cell("us_like|low|random|exit")
    m, _, _ = M.construct_mask("US", "dev", 0, c, us.xi_base, us.geo, us.labels)
    q = M.pool_fraction(c["fraction"])
    cut = np.sort(us.xi_base)[int(np.ceil(q * len(m))) - 1]
    assert (us.xi_base[m] <= cut).all()


def test_mask_determinism_and_seed_dependence(us):
    c = _cell("intermediate|high|geo|gap")
    a = M.construct_mask("US", "dev", 0, c, us.xi_base, us.geo, us.labels)[0]
    b = M.construct_mask("US", "dev", 0, c, us.xi_base, us.geo, us.labels)[0]
    assert np.array_equal(a, b)
    c2 = _cell("intermediate|high|random|gap", rep=0)
    r1 = M.construct_mask("US", "dev", 0, c2, us.xi_base, us.geo, us.labels)[0]
    r2 = M.construct_mask("US", "dev", 1, c2, us.xi_base, us.geo, us.labels)[0]
    assert not np.array_equal(r1, r2)


def test_visibility_timing():
    m = np.array([True, False, True])
    T = 22
    v = M.visibility(m, T, "exit")
    assert v[1].all() and not v[0, T // 2:].any() and v[0, :T // 2].all()
    g = M.visibility(m, T, "gap")
    assert not g[0, T // 3:2 * T // 3].any() and g[0, :T // 3].all() and g[0, 2 * T // 3:].all()
    with pytest.raises(ValueError):
        M.visibility(m, T, "other")


def test_infeasible_geo_mask_reported():
    xi = np.linspace(-1, 1, 10)
    geo = np.zeros(10, int)
    c = _cell("us_like|low|geo|exit")
    m, att, st = M.construct_mask("US", "dev", 0, c, xi, geo, np.arange(10).astype(str))
    assert m is None and st == "mask_infeasible" and att == M.MAX_ATTEMPTS


@pytest.mark.parametrize("sysname", ["us", "br"])
def test_dose_matches_fraction(sysname, request):
    s = request.getfixturevalue(sysname)
    for fl in M.FRACTIONS:
        c = _cell(f"{fl}|low|random|exit")
        m, _, _ = M.construct_mask(s.name, "dev", 0, c, s.xi_base, s.geo, s.labels)
        r = s.impl.run(M.visibility(m, s.n_years, "exit"), need_v1=False)
        k = m.sum()
        assert abs(r["dose"]["delta_event"] - k / (len(m) - k)) < 1e-12
        assert r["dose"]["n_event_transitions"] == 1
        g = s.impl.run(M.visibility(m, s.n_years, "gap"), need_v1=False)
        assert g["dose"]["n_event_transitions"] == 2
        assert r["identity_max"] < M.IDENTITY_TOL and g["identity_max"] < M.IDENTITY_TOL


@pytest.mark.parametrize("sysname", ["us", "br"])
def test_forcing_sign(sysname, request):
    s = request.getfixturevalue(sysname)
    mbar = {}
    for pos in ("low", "high"):
        c = _cell(f"china_like|{pos}|random|exit")
        m, _, _ = M.construct_mask(s.name, "dev", 0, c, s.xi_base, s.geo, s.labels)
        r = s.impl.run(M.visibility(m, s.n_years, "exit"), need_v1=False)
        mbar[pos] = r["dose"]["M_bar"]
        ev = json.loads(r["dose"]["M_events"])
        assert len(ev) == 1
    assert mbar["low"] < 0 < mbar["high"]


def test_forcing_sign_on_xi_star():
    """Strong-dose exit of a low (high) block shifts the V0 attractor down
    (up) relative to F on a persistent synthetic panel."""
    s = M.synthetic_us_system(n=400, seed=11, a1=-0.05, cubic=0.02)
    full = s.impl.run(None, need_v1=False)["V0"]["xi_trap"]
    sh = {}
    for pos in ("low", "high"):
        c = _cell(f"china_like|{pos}|random|exit")
        m, _, _ = M.construct_mask("US", "dev", 0, c, s.xi_base, s.geo, s.labels)
        sh[pos] = s.impl.run(M.visibility(m, s.n_years, "exit"), need_v1=False)["V0"]["xi_trap"] - full
    assert sh["low"] < 0 < sh["high"]


def test_project_rejects_non_v1_inputs(us):
    r = us.impl.run(None)
    with pytest.raises(TypeError):
        M.project(r["V0"], r["comp"])
    with pytest.raises(TypeError):
        M.project(r["v1fit"], r["comp"].frame())
    with pytest.raises(TypeError):
        M.project({"theta": r["v1fit"].theta}, r["comp"])


def test_composition_whitelist():
    good = {k: 1.0 for k in M.US_COMP_KEYS}
    M.Composition("US", (good,))
    for bad in ("mean_inc_V0", "identity_residual", "xi_trap_V0", "T1_obs"):
        with pytest.raises(ValueError):
            M.Composition("US", (good | {bad: 0.0},))
    b = {k: 1.0 for k in M.BR_COMP_KEYS}
    with pytest.raises(ValueError):
        M.Composition("BR", (b | {"xi_bar_stay_V0": 0.0},))
    assert "xi_bar_stay_V0" not in M.BR_COMP_KEYS


@pytest.mark.parametrize("sysname", ["us", "br"])
def test_projection_receives_only_v1_fit(sysname, request, monkeypatch):
    """The theta reaching the frozen projection is the masked V1 fit's, and the
    result is unchanged when the V0 fit is replaced by garbage."""
    s = request.getfixturevalue(sysname)
    c = _cell("china_like|low|random|exit")
    m, _, _ = M.construct_mask(s.name, "dev", 0, c, s.xi_base, s.geo, s.labels)
    vis = M.visibility(m, s.n_years, "exit")
    seen = []
    if s.name == "US":
        orig = K2.k1_prediction

        def spy(theta, dec, guard=None):
            seen.append((np.array(theta), set(dec.columns)))
            return orig(theta, dec, guard)
        monkeypatch.setattr(K2, "k1_prediction", spy)
    else:
        orig = KBR.k1_prediction

        def spy(rows, theta, xi_trap_v1, rho1_v1=np.nan, guard=None):
            seen.append((np.array(theta), set(rows[0])))
            return orig(rows, theta, xi_trap_v1, rho1_v1, guard=guard)
        monkeypatch.setattr(KBR, "k1_prediction", spy)
    full = s.impl.run(None, need_v1=False)["V0"]
    row = {"mask_id": "t", "cell": c["cell"]}
    rec, proj, _ = M.point_record(s, row, vis, full, guards=(M.REGISTERED_GUARD,))
    r = s.impl.run(vis)
    assert seen and all(np.array_equal(t, np.asarray(r["V1"]["params"])) for t, _ in seen)
    assert not np.allclose(r["V0"]["params"], r["V1"]["params"])
    assert all(cols == set(M.COMP_KEYS[s.name]) for _, cols in seen)
    real_run = s.impl.run

    def run_bad_v0(visible, *a, **k):
        out = real_run(visible, *a, **k)
        out["V0"] = out["V0"] | {"xi_trap": 99.0, "params": np.full(7, 5.0)}
        return out
    monkeypatch.setattr(s.impl, "run", run_bad_v0)
    rec2, proj2, _ = M.point_record(s, row, vis, full, guards=(M.REGISTERED_GUARD,))
    monkeypatch.undo()
    a = np.array([p["pred"] for p in proj], float)
    b = np.array([p["pred"] for p in proj2], float)
    assert np.array_equal(a, b, equal_nan=True)
    assert [p["status"] for p in proj] == [p["status"] for p in proj2]
    assert rec2["xi_0"] == 99.0
    assert np.array_equal([rec["pred_point"]], [rec2["pred_point"]], equal_nan=True)


def test_project_signature():
    import inspect
    assert list(inspect.signature(M.project).parameters) == ["v1", "comp", "guards"]


def test_guard_set_is_the_posthoc_grid():
    assert len(M.GUARD_SET) == 12
    assert {s for s, _ in M.GUARD_SET} == {(0.5, 99.5), (1.0, 99.0), (2.5, 97.5), (5.0, 95.0)}
    assert {f for _, f in M.GUARD_SET} == {1.0, 0.5, 2.0}
    assert M.REGISTERED_GUARD in M.GUARD_SET


def test_registered_guard_equals_k1_guard(us):
    r = us.impl.run(None)
    v = r["v1fit"]
    g = M.GS.alt_guard(v.xi_t, v.rho1, *M.REGISTERED_GUARD)
    assert g == K2.k1_guard(r["pairs"]["V1"], v.rho1)


def test_us_skeleton_reproduces_registered_panel_a():
    w = K2.make_toy_world()
    s, sk = M.us_system_from_world(w, n_min=1)
    ref = K2.run_spec(sk)
    mine = s.impl.run(None)
    assert M._pair_diff(mine["pairs"]["V0"], ref["pairs"]["V0"]) == 0.0
    e_ref = K2.estimate(*ref["pairs"]["V0"])
    assert mine["V0"]["xi_trap"] == e_ref["limit"]["xi_trap"]
    assert mine["V0"]["rho1"] == e_ref["limit"]["rho1"]


def test_br_panel_reproduces_registered_panel_a_and_draws():
    m, reg = KBR.synthetic_muni(n0=150, seed=5)
    cfg = KBR.Config(n_min=10)
    s = M.br_system_from_muni(m, reg, cfg)
    res = KBR.analyse(m, cfg)
    ref = res["panels"][("pib", "A")]
    mine = s.impl.run(None)
    assert M._pair_diff(mine["pairs"]["V0"], ref["pairs"]["V0"]) == 0.0
    assert mine["V0"]["xi_trap"] == ref["fits"]["V0"]["limit"]["xi_trap"]
    bb = KBR.make_block_base(m, reg)
    seq = np.random.default_rng(0).integers(0, len(bb.amc_ids), len(bb.amc_ids))
    pa = KBR.apply_year_exclusion(KBR.panel_A(KBR.muni_from_amc_seq(bb, seq), "pib"),
                                  s.impl.pA.years)
    pm = s.impl.panel(None, seq)
    assert np.array_equal(pa.present, pm.present)
    assert np.array_equal(np.where(pa.present, pa.Y, 0), np.where(pm.present, pm.Y, 0))
    assert np.array_equal(np.where(pa.present, pa.N, 0), np.where(pm.present, pm.N, 0))


def test_no_mask_equivalence(us, br):
    for s in (us, br):
        r = s.impl.run(None)
        rows = M.equivalence_rows(s, r, None)
        assert rows[0]["pass"] and rows[0]["max_diff"] == 0.0
        assert r["V0"]["xi_trap"] == r["V1"]["xi_trap"]


def test_seed_derivation():
    h = "0123456789abcdef0123456789abcdef01234567"
    assert M.stage_entropy("confirmatory", h) == int(h, 16)
    assert M.stage_entropy("pilot") == M.PILOT_ENTROPY
    for bad in (None, "abc", h.upper(), h[:-1] + "g"):
        with pytest.raises(ValueError):
            M.stage_entropy("confirmatory", bad)
    a = M.mask_seedseq(int(h, 16), "US", "confirmatory", 3, 2, 0).generate_state(4)
    b = M.mask_seedseq(int(h, 16), "US", "confirmatory", 3, 2, 0).generate_state(4)
    c = M.mask_seedseq(int(h, 16) + 1, "US", "confirmatory", 3, 2, 0).generate_state(4)
    d = M.mask_seedseq(int(h, 16), "BR", "confirmatory", 3, 2, 0).generate_state(4)
    p = M.mask_seedseq(int(h, 16), "US", "pilot", 3, 2, 0).generate_state(4)
    assert np.array_equal(a, b) and not np.array_equal(a, c) and not np.array_equal(a, d)
    assert not np.array_equal(a, p)
    bs = M.boot_seedseq(1, "US", "confirmatory").generate_state(2)
    assert not np.array_equal(bs, M.boot_seedseq(1, "US", "pilot").generate_state(2))


def test_pilot_exclusion_redraws(us):
    c = _cell("intermediate|low|random|exit")
    m0, a0, _ = M.construct_mask("US", "dev", 0, c, us.xi_base, us.geo, us.labels)
    h = M.mask_hash("US", "exit", us.labels, m0)
    m1, a1, st = M.construct_mask("US", "dev", 0, c, us.xi_base, us.geo, us.labels, frozenset({h}))
    assert st == "ok" and a1 > a0 and not np.array_equal(m0, m1)


def test_validate_freeze():
    fz = {"guard_set": [[[1.0, 99.0], 1.0], [[5.0, 95.0], 2.0]], "point_guard": [[1.0, 99.0], 1.0],
          "criteria": {"range_coverage_min": 0.85, "slope_band": [0.8, 1.2]}}
    v = M.validate_freeze(fz)
    assert v["point_guard"] == ((1.0, 99.0), 1.0) and v["criteria"]["slope_band"] == (0.8, 1.2)
    with pytest.raises(ValueError):
        M.validate_freeze(fz | {"guard_set": [[[3.0, 97.0], 1.0]]})
    with pytest.raises(ValueError):
        M.validate_freeze(fz | {"point_guard": [[0.5, 99.5], 1.0]})
    with pytest.raises(ValueError):
        M.validate_freeze(fz | {"criteria": {"new_rule": 1}})
    with pytest.raises(ValueError):
        M.validate_freeze(fz | {"n_boot": 100})


def test_freeze_tightening_only():
    assert set(M.CRITERIA_DIRECTION) == set(M.CRITERIA)
    base = {"guard_set": [[[1.0, 99.0], 1.0]], "point_guard": [[1.0, 99.0], 1.0]}
    tighter = {"point_availability_min": 0.9, "slope_band": [0.75, 1.25],
               "intercept_band_rel": 0.2, "sign_agreement_min": 0.95,
               "level_abs_error_rel": 0.25, "range_availability_min": 0.85,
               "range_coverage_min": 0.9, "range_width_rel_max": 0.8,
               "recovery_share_min": 0.85, "recovery_share_ci_lo_min": 0.6,
               "recovery_median_ratio_max": 0.4, "gate_availability_min": 0.9,
               "null_rel_max": 0.2, "comparator_share_ci_lo_min": 0.6}
    for k, val in tighter.items():
        M.validate_freeze(base | {"criteria": {k: val}})
    looser = {"point_availability_min": 0.7, "slope_band": [0.6, 1.3],
              "intercept_band_rel": 0.3, "sign_agreement_min": 0.85,
              "level_abs_error_rel": 0.35, "range_availability_min": 0.7,
              "range_coverage_min": 0.75, "range_width_rel_max": 1.2,
              "recovery_share_min": 0.75, "recovery_share_ci_lo_min": 0.4,
              "recovery_median_ratio_max": 0.6, "gate_availability_min": 0.7,
              "null_rel_max": 0.3, "comparator_share_ci_lo_min": 0.4,
              "gate_factor": 0.25, "power_factor": 1.5, "level_min_masks": 5}
    assert set(looser) == set(M.CRITERIA)
    for k, val in looser.items():
        with pytest.raises(ValueError):
            M.validate_freeze(base | {"criteria": {k: val}})
    for k in ("gate_factor", "power_factor", "level_min_masks"):
        with pytest.raises(ValueError):
            M.validate_freeze(base | {"criteria": {k: M.CRITERIA[k] * 2}})
    with pytest.raises(ValueError):
        M.validate_freeze(base | {"criteria": {"slope_band": [1.1, 0.9]}})
    with pytest.raises(ValueError):
        M.validate_freeze(base | {"criteria": {"sign_agreement_min": float("nan")}})


def test_kill_rule_reproduction(us, tmp_path):
    full = us.impl.run(None, need_v1=False)["V0"]
    ok = {k: full[k] for k in ("xi_trap", "rho1", "n_pairs")} | {"id_status": full["path_status"]}
    assert all(r["pass"] for r in M.reproduction_rows(us, full, ok))
    bad = ok | {"xi_trap": ok["xi_trap"] + 1e-9}
    assert not all(r["pass"] for r in M.reproduction_rows(us, full, bad))
    s2 = M.synthetic_us_system(n=120, seed=8)
    s2.committed = bad | {"xi_trap": 123.0}
    res = M.run_system(s2, "dev", 0, M.stage_cells("pilot")[:1], 2, log=lambda *a: None)
    assert isinstance(res["killed"], M.KillRule) and res["killed"].rule == 2
    assert "masks" not in res


def test_kill_rule_balance():
    w = K2.make_toy_world()
    _, sk = M.us_system_from_world(w, n_min=1)
    y = sk.years[3]
    sk.recs[y] = dict(sk.recs[y], codes=sk.recs[y]["codes"][1:])
    with pytest.raises(M.KillRule) as e:
        M.us_impl_from_skeleton(sk, 10)
    assert e.value.rule == 3


def test_kill_rule_identity(us, monkeypatch):
    c = _cell("us_like|low|random|exit")
    m, _, _ = M.construct_mask("US", "dev", 0, c, us.xi_base, us.geo, us.labels)
    vis = M.visibility(m, us.n_years, "exit")
    full = us.impl.run(None, need_v1=False)["V0"]
    real = us.impl.run

    def broken(*a, **k):
        o = real(*a, **k)
        o["identity_max"] = 1e-6
        return o
    monkeypatch.setattr(us.impl, "run", broken)
    with pytest.raises(M.KillRule) as e:
        M.point_record(us, {"mask_id": "x"}, vis, full)
    assert e.value.rule == 5


def test_single_run_refusal(tmp_path):
    log = tmp_path / "a.jsonl"
    M.check_single_run("pilot", log)
    log.write_text(json.dumps({"stage": "pilot", "status": "started"}) + "\n")
    with pytest.raises(SystemExit):
        M.check_single_run("pilot", log)
    M.check_single_run("confirmatory", log)


def test_dev_refuses_results_tables():
    with pytest.raises(SystemExit):
        M.main(["--dev", "--out", str(M.TAB / "k4_mask" / "x")])


def _fake_points(slope=1.0, noise=0.01, seed=0, n_cells=12, per=5, stratum="active",
                 null_scale=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    pos_t = {"active": (("low", "high"), "exit"), "balanced": (("low", "high"), "gap"),
             "null": (("mid",), "exit")}[stratum]
    for c in range(n_cells):
        fl = list(M.FRACTIONS)[c % 3]
        pos = pos_t[0][(c // 3) % len(pos_t[0])]
        sc = null_scale if stratum == "null" else 1.0
        for r in range(per):
            pred = sc * rng.normal(0, 0.5)
            pred = abs(pred) if pos == "high" else -abs(pred) if pos == "low" else pred
            obs = slope * pred + rng.normal(0, noise)
            rows.append({"system": "US", "mask_id": f"{stratum}{c}-{r}", "cell": f"{stratum}c{c}",
                         "position": pos, "timing": pos_t[1], "stratum": stratum,
                         "T_naive": obs, "id_naive": "stable_zero", "xi_FS": 0.0,
                         "id_FS": "stable_zero", "T3_selection": 0.0, "T3_event": obs,
                         "T2_vs_FS": 0.0,
                         "fraction_level": fl, "pred_point": pred, "T1_obs": obs,
                         "range_lo": pred - 0.05, "range_hi": pred + 0.05,
                         "n_guards": 12, "n_guards_defined": 12,
                         "id_0": "stable_zero", "id_1": "stable_zero", "id_F": "stable_zero",
                         "T1_obs_ci_lo": obs - 0.02, "T1_obs_ci_hi": obs + 0.02,
                         "T2": rng.normal(0, 0.01), "T3": obs, "T3_ci_lo": obs - 0.02,
                         "T3_ci_hi": obs + 0.02, "T2_ci_lo": -0.03, "T2_ci_hi": 0.03,
                         "M_bar": 0.0, "delta_event": 0.3})
    return pd.DataFrame(rows)


def _verdicts(s, crit_set="provisional", scope="active"):
    return s[(s.kind == "verdict") & (s.criteria_set == crit_set)
             & (s.scope == scope)].set_index("name")["value"]


def _metric(s, scope="active"):
    return s[(s.kind == "metric") & (s.scope == scope)].set_index("name")


def test_evaluation_calibrated_passes():
    s, cells = M.evaluate(_fake_points(), 0, "dev", R=200)
    v = _verdicts(s)
    m = _metric(s)
    assert abs(m.loc["calibration_slope", "value"] - 1) < 0.05
    assert v["T1_point"] == "pass" and v["T1_range"] == "pass" and v["T2"] == "recovers"
    assert v["T3_powered"] and v["N_null_control"] == "not_evaluable"
    assert len(cells) == 12 and set(cells.stratum) == {"active"}


def test_evaluation_miscalibrated_fails():
    s, _ = M.evaluate(_fake_points(slope=2.0), 0, "dev", R=200)
    v = _verdicts(s)
    assert v["T1_point"] == "fail" and v["T1_range"] in ("fail", "covers_but_uninformative")


def test_evaluation_availability():
    d = _fake_points()
    d.loc[d.index[:40], "pred_point"] = np.nan
    d.loc[d.index[:40], "n_guards_defined"] = 0
    s, _ = M.evaluate(d, 0, "dev", R=100)
    v = _verdicts(s)
    assert v["T1_point"] == "not_evaluable_availability"
    assert v["T1_range"] == "not_evaluable_availability"


def test_strata_of_grid():
    g = M.cell_grid()
    n = {st: sum(c["stratum"] == st for c in g) for st in M.STRATA}
    assert n == {"active": 12, "balanced": 12, "null": 12}
    for c in g:
        assert c["stratum"] == ("null" if c["position"] == "mid" else
                                "active" if c["timing"] == "exit" else "balanced")


def test_evaluation_primary_is_active_stratum():
    """Verdicts are taken on the active stratum; null (mid) and balanced
    (gap) masks are evaluated separately and do not dilute the calibration."""
    act = _fake_points(seed=1)
    bal = _fake_points(seed=2, stratum="balanced", slope=-1.0)
    nul = _fake_points(seed=3, stratum="null", null_scale=0.02)
    s, cells = M.evaluate(pd.concat([act, bal, nul], ignore_index=True), 0, "dev", R=100)
    v = _verdicts(s)
    assert v["T1_point"] == "pass" and v["N_null_control"] == "pass"
    assert _verdicts(s, scope="balanced")["T1_point"] == "fail"
    assert _verdicts(s, scope="pooled")["T1_point"] == "fail"
    roles = s.drop_duplicates("scope").set_index("scope")["role"]
    assert roles["active"] == "primary" and (roles.drop("active") == "report_only").all()
    assert "T3_powered" not in _verdicts(s, scope="pooled")
    nul_bad = _fake_points(seed=3, stratum="null", null_scale=1.0)
    s2, _ = M.evaluate(pd.concat([act, nul_bad], ignore_index=True), 0, "dev", R=50)
    assert _verdicts(s2)["N_null_control"] == "fail"


def test_positive_control_signed_on_active_exit():
    d = _fake_points()
    s, _ = M.evaluate(d, 0, "dev", R=50)
    m = _metric(s)
    assert m.loc["pc_low_median_T3", "value"] < 0 < m.loc["pc_high_median_T3", "value"]
    assert _verdicts(s)["T3_powered"]
    flip = d.copy()
    lo = flip["position"] == "low"
    flip.loc[lo, ["T3", "T3_ci_lo", "T3_ci_hi"]] = -flip.loc[lo, ["T3", "T3_ci_hi", "T3_ci_lo"]].to_numpy()
    v = _verdicts(M.evaluate(flip, 0, "dev", R=50)[0])
    assert not v["T3_powered_low"] and v["T3_powered_high"] and not v["T3_powered"]
    nul = _fake_points(seed=3, stratum="null", null_scale=0.0)
    s3, _ = M.evaluate(pd.concat([d, nul], ignore_index=True), 0, "dev", R=50)
    assert _metric(s3).loc["pc_low_n", "value"] == _metric(s).loc["pc_low_n", "value"]


def test_gate_unavailable_is_counted():
    d = _fake_points()
    k = d.index[:20]
    d.loc[k, ["T1_obs_ci_lo", "T1_obs_ci_hi", "T3_ci_lo", "T3_ci_hi"]] = np.nan
    e = M.eval_frame(d)
    assert (e.loc[k, "gate_T1_state"] == "gate_unavailable").all()
    assert not e.loc[k, "gated_T1"].any() and not e.loc[k, "gated_T3"].any()
    s, _ = M.evaluate(d, 0, "dev", R=50)
    m, v = _metric(s), _verdicts(s)
    assert m.loc["n_gate_T1_unavailable", "value"] == 20
    assert m.loc["n_gate_T3_unavailable", "value"] == 20
    assert v["T1_point"] == "not_evaluable_gate_availability"
    assert v["T1_range"] == "not_evaluable_gate_availability"
    assert v["T2"] == "not_evaluable_gate_availability"
    assert m.loc["sign_agreement_gate_unavailable_as_fail", "value"] < m.loc["sign_agreement_gated", "value"]


def test_t2_requires_beating_the_naive_comparator():
    d = _fake_points()
    d["T_naive"] = d["T2"] * 0.1
    v = _verdicts(M.evaluate(d, 0, "dev", R=100)[0])
    assert v["T2_V1_share"] and v["T2_V2_ratio"] and not v["T2_V3_beats_naive"]
    assert v["T2"] == "closer_than_V0_only"


def test_frozen_and_provisional_verdicts_both_reported():
    fro = dict(M.CRITERIA) | {"sign_agreement_min": 1.01}
    s, _ = M.evaluate(_fake_points(), 0, "dev", R=50,
                      crit_sets={"frozen": fro, "provisional": M.CRITERIA})
    assert set(s.loc[s.kind == "verdict", "criteria_set"]) == {"frozen", "provisional"}
    assert _verdicts(s, "frozen")["T1_point"] == "fail"
    assert _verdicts(s, "provisional")["T1_point"] == "pass"


def test_cluster_ci_deterministic():
    d = M.eval_frame(_fake_points())
    a = M.cluster_ci(d, M.CRITERIA, M.cluster_seedseq(0, "US", "dev"), 100)
    b = M.cluster_ci(d, M.CRITERIA, M.cluster_seedseq(0, "US", "dev"), 100)
    assert a == b


def test_dev_end_to_end(tmp_path, us, br):
    status, written = M.run_stage([M.synthetic_us_system(n=120, seed=1),
                                   M.synthetic_br_system(n=120, seed=2)], "dev", 0, tmp_path, 4, 1,
                                  cells_fn=lambda s: M.stage_cells("pilot")[:4], R=50,
                                  log=lambda *a: None)
    assert status == "completed"
    for f in ("k4_masks.csv", "k4_points.csv", "k4_projections.csv", "k4_draws.csv.gz",
              "k4_evaluation.csv", "k4_cells.csv", "k4_pilot_diagnostics.csv"):
        assert f in written
    pts = pd.read_csv(tmp_path / "k4_points.csv")
    assert len(pts) == 8 and set(pts.system) == {"US", "BR"}
    pr = pd.read_csv(tmp_path / "k4_projections.csv")
    assert (pr.groupby("mask_id").size() == 12).all()
    dr = pd.read_csv(tmp_path / "k4_draws.csv.gz")
    assert len(dr) == 8 * 4
    assert np.allclose(pts["T1_obs"], pts["xi_0"] - pts["xi_1"], equal_nan=True)
    for c in ("xi_FS", "xi_naive", "T3_selection", "T3_event", "T_naive", "stratum",
              "id_0_unidentified_share"):
        assert c in pts
    assert np.allclose(pts["T3"], pts["T3_selection"] + pts["T3_event"], equal_nan=True)
    ev = pd.read_csv(tmp_path / "k4_evaluation.csv")
    assert set(ev.loc[ev.kind == "verdict", "criteria_set"]) == {"provisional"}


def test_root_status_equals_section11_rule():
    import importlib.util
    spec = importlib.util.spec_from_file_location("expi_id", ROOT / "scripts" / "run_expi_identify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    for _ in range(200):
        p = rng.normal(0, 0.3, 7)
        assert M.root_status(p, x) == mod.root_status(p, x)["root_status"]
    assert M.root_status(None, x) == "failed"


def test_reproduction_uses_path_status(us):
    full = us.impl.run(None, need_v1=False)["V0"]
    com = {"xi_trap": full["xi_trap"], "rho1": full["rho1"], "n_pairs": full["n_pairs"],
           "id_status": full["path_status"]}
    assert all(r["pass"] for r in M.reproduction_rows(us, full, com))


def test_gap_and_mid_forcing_are_balanced(us):
    """gap: exit and re-entry events of opposite sign; mid: M ~ 0."""
    for pos in ("low", "high"):
        c = _cell(f"china_like|{pos}|random|gap")
        m, _, _ = M.construct_mask("US", "dev", 0, c, us.xi_base, us.geo, us.labels)
        r = us.impl.run(M.visibility(m, us.n_years, "gap"), need_v1=False)
        ev = json.loads(r["dose"]["M_events"])
        assert len(ev) == 2 and ev[0] * ev[1] < 0
        ex = us.impl.run(M.visibility(m, us.n_years, "exit"), need_v1=False)["dose"]["M_bar"]
        assert abs(r["dose"]["M_bar"]) < 0.5 * abs(ex)
    c = _cell("china_like|mid|random|exit")
    m, _, _ = M.construct_mask("US", "dev", 0, c, us.xi_base, us.geo, us.labels)
    mid = us.impl.run(M.visibility(m, us.n_years, "exit"), need_v1=False)["dose"]["M_bar"]
    c = _cell("china_like|low|random|exit")
    m, _, _ = M.construct_mask("US", "dev", 0, c, us.xi_base, us.geo, us.labels)
    low = us.impl.run(M.visibility(m, us.n_years, "exit"), need_v1=False)["dose"]["M_bar"]
    assert abs(mid) < 0.2 * abs(low)


@pytest.mark.parametrize("sysname", ["us", "br"])
def test_references_fs_and_naive(sysname, request):
    s = request.getfixturevalue(sysname)
    full = s.impl.run(None, need_v1=False)["V0"]
    nv = s.impl.naive(np.ones((len(s.labels), s.n_years), bool))
    assert nv["xi_trap"] == full["xi_trap"] and nv["n_pairs"] == full["n_pairs"]
    c = _cell("intermediate|low|random|exit")
    m, _, _ = M.construct_mask(s.name, "dev", 0, c, s.xi_base, s.geo, s.labels)
    vis = M.visibility(m, s.n_years, "exit")
    r = s.impl.run(vis, need_v1=False)
    nv = s.impl.naive(vis)
    j = s.n_years // 2 - 1
    n_ev = int((vis[:, j] & vis[:, j + 1]).sum())
    assert M.event_transitions(vis) == {j}
    assert nv["n_pairs"] == r["V0"]["n_pairs"] - n_ev
    vS = M.never_masked(vis)
    assert (vS.all(axis=1) == ~m).all() and (vS == vS[:, :1]).all()
    fs = s.impl.run(vS, need_v1=True, collect=False)
    assert fs["V0"]["xi_trap"] == fs["V1"]["xi_trap"]
    assert fs["V0"]["n_pairs"] == int((~m).sum()) * (s.n_years - 1)
    row = {"mask_id": "t", "cell": c["cell"]}
    rec, _, _ = M.point_record(s, row, vis, full, guards=(M.REGISTERED_GUARD,))
    assert rec["xi_FS"] == fs["V0"]["xi_trap"] and rec["xi_naive"] == nv["xi_trap"]
    assert abs(rec["T3"] - rec["T3_selection"] - rec["T3_event"]) < 1e-12


def _direct_pairs(logN, logY, vis):
    """Independent recomputation of the masked V0 / V1 pairs: per year OLS
    on the visible units (V0), per transition OLS refit on the visible
    stayers of each year (V1); xi = resid / sd(ddof 1)."""
    def std(y, x, sel):
        X = np.column_stack([np.ones(sel.sum()), x[sel]])
        coef, *_ = np.linalg.lstsq(X, y[sel], rcond=None)
        r = y[sel] - X @ coef
        out = np.full(len(y), np.nan)
        out[sel] = r / np.std(r, ddof=1)
        return out
    T = vis.shape[1]
    xi0 = np.column_stack([std(logY[:, j], logN[:, j], vis[:, j]) for j in range(T)])
    p0, p1 = ([], []), ([], [])
    for j in range(T - 1):
        S = vis[:, j] & vis[:, j + 1]
        p0[0].append(xi0[S, j])
        p0[1].append(xi0[S, j + 1] - xi0[S, j])
        a, b = std(logY[:, j], logN[:, j], S), std(logY[:, j + 1], logN[:, j + 1], S)
        p1[0].append(a[S])
        p1[1].append(b[S] - a[S])
    cat = lambda p: (np.concatenate(p[0]), np.concatenate(p[1]))
    return cat(p0), cat(p1)


def _as3(p):
    return (p[0], p[1], np.ones(len(p[0]))) if len(p) == 2 else p


@pytest.mark.parametrize("sysname", ["us", "br"])
@pytest.mark.parametrize("cellname", ["china_like|low|random|exit", "intermediate|high|geo|gap",
                                      "us_like|mid|random|exit"])
def test_masked_pairs_equal_direct_recomputation(sysname, cellname, request):
    s = request.getfixturevalue(sysname)
    c = _cell(cellname)
    m, _, _ = M.construct_mask(s.name, "dev", 0, c, s.xi_base, s.geo, s.labels)
    vis = M.visibility(m, s.n_years, c["timing"])
    if s.name == "US":
        logN, logY = s.impl.logN, s.impl.logY
    else:
        logY, logN = KBR._logs(s.impl.pA)
    d0, d1 = _direct_pairs(logN, logY, vis)
    r = s.impl.run(vis)
    assert K2.pair_multiset_diff(_as3(r["pairs"]["V0"]), _as3(d0)) < 1e-10
    assert K2.pair_multiset_diff(_as3(r["pairs"]["V1"]), _as3(d1)) < 1e-10
    assert K2.pair_multiset_diff(_as3(r["pairs"]["V0"]), _as3(r["pairs"]["V1"])) > 1e-6


def test_masked_us_skeleton_equals_registered_builder_on_open_toy_world():
    """The masked USImpl skeleton equals run_k2.build_skeleton on an
    equivalent OPEN toy world (the masked area-years removed from the
    aggregates' presence), V0 and V1 pairs and the Lemma 1 rows."""
    from dataclasses import replace
    w = K2.make_toy_world()
    s, sk = M.us_system_from_world(w, n_min=1)
    unit_lin, area_lin = K2.lineages(w, "metro")
    aggs = {("metro", lab, oc): K2.aggregate(w, "metro", lab, oc)
            for (sc, lab) in w.areas if sc == "metro" for oc in w.outcomes}
    drops, _ = K2.min_n_drops(w, aggs, 1)
    key = M.US_KEY_A
    for cellname in ("china_like|low|random|exit", "intermediate|high|random|gap"):
        c = _cell(cellname)
        m, _, st = M.construct_mask("US", "dev", 0, c, s.xi_base, s.geo, s.labels)
        assert st == "ok"
        vis = M.visibility(m, s.n_years, c["timing"])
        ag = aggs[("metro", w.panel_a, key[1])]
        pres = ag.present.copy()
        for i, code in enumerate(s.impl.codes):
            for j, y in enumerate(s.impl.years):
                if not vis[i, j]:
                    pres[ag.codes.index(code), ag.years.index(y)] = False
        aggs_m = dict(aggs) | {("metro", w.panel_a, key[1]): replace(ag, present=pres)}
        sk_open = K2.build_skeleton(w, aggs_m, key, area_lin, drops[(key[1], key[2])])
        ref = K2.run_spec(sk_open)
        mine = K2.run_spec(s.impl.skeleton(vis))
        for v in ("V0", "V1"):
            assert K2.pair_multiset_diff(mine["pairs"][v], ref["pairs"][v]) < 1e-12
        cols = ["year_t", "n_S", "n_E", "n_L", "M_t", "xbar_stay_t", "var_stay_t"]
        assert np.allclose(mine["dec"][cols].to_numpy(float), ref["dec"][cols].to_numpy(float),
                           atol=1e-12, equal_nan=True)
        assert (mine["dec"]["n_E"] + mine["dec"]["n_L"]).sum() > 0


def test_preflight_writes_no_masks_no_seeds_no_attempt(tmp_path, monkeypatch):
    sysm = M.synthetic_us_system(n=120, seed=8)
    full = sysm.impl.run(None, need_v1=False)["V0"]
    sysm.committed = {k: full[k] for k in ("xi_trap", "rho1", "n_pairs")} | {
        "id_status": full["path_status"]}
    monkeypatch.setattr(M, "load_real_systems", lambda *a, **k: ([sysm], {}))
    monkeypatch.setattr(M, "integrity_rows", lambda *a, **k: [
        {"system": "US", "file": "f", "committed": "x", "now": "x", "pass": True}])
    monkeypatch.setattr(M, "check_registration", lambda *a, **k: "c" * 40)
    head = "a" * 40
    monkeypatch.setattr(M, "_git", lambda *a: "" if a[0] == "status" else head)
    log = tmp_path / "attempts.jsonl"
    monkeypatch.setattr(M, "ATTEMPT_LOG", log)
    called = []
    monkeypatch.setattr(M, "build_masks", lambda *a, **k: called.append(1))
    monkeypatch.setattr(M, "stage_entropy", lambda *a, **k: called.append(2))
    pdir = tmp_path / "pf"
    rc = M.main(["--preflight", "pilot", "--preflight-dir", str(pdir), "--us-data", str(tmp_path)])
    assert rc == 0 and not called and not log.exists()
    r = json.loads(M.preflight_receipt_path("pilot", pdir).read_text())
    assert r["status"] == "passed" and r["head"] == head
    files = {f.name for f in Path(r["out"]).iterdir()}
    assert "k4_reproduction.csv" in files and "k4_equivalence.csv" in files
    assert not any("mask" in f for f in files)
    assert M.check_preflight("pilot", head, pdir)["status"] == "passed"
    with pytest.raises(SystemExit):
        M.check_preflight("pilot", "b" * 40, pdir)
    with pytest.raises(SystemExit):
        M.check_preflight("confirmatory", head, pdir)
    monkeypatch.setattr(M, "environment_fingerprint", lambda: {"hostname": "elsewhere"})
    with pytest.raises(SystemExit):
        M.check_preflight("pilot", head, pdir)
    sysm.committed = sysm.committed | {"xi_trap": 99.0}
    monkeypatch.undo()
    monkeypatch.setattr(M, "load_real_systems", lambda *a, **k: ([sysm], {}))
    monkeypatch.setattr(M, "integrity_rows", lambda *a, **k: [
        {"system": "US", "file": "f", "committed": "x", "now": "x", "pass": True}])
    monkeypatch.setattr(M, "check_registration", lambda *a, **k: "c" * 40)
    monkeypatch.setattr(M, "_git", lambda *a: "" if a[0] == "status" else head)
    monkeypatch.setattr(M, "ATTEMPT_LOG", log)
    rc = M.main(["--preflight", "pilot", "--preflight-dir", str(pdir), "--us-data", str(tmp_path)])
    assert rc == 3 and not log.exists()
    with pytest.raises(SystemExit):
        M.check_preflight("pilot", head, pdir)


def test_preflight_refuses_repository_output():
    with pytest.raises(SystemExit):
        M.main(["--preflight", "pilot", "--preflight-dir", str(M.ROOT / "results")])


def test_registered_stage_requires_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "check_registration", lambda *a, **k: "c" * 40)
    monkeypatch.setattr(M, "_git", lambda *a: "" if a[0] == "status" else "a" * 40)
    log = tmp_path / "attempts.jsonl"
    monkeypatch.setattr(M, "ATTEMPT_LOG", log)
    monkeypatch.setattr(M, "check_single_run", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        M.main(["--pilot", "--preflight-dir", str(tmp_path / "none")])
    assert not log.exists()
