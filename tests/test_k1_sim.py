def public_guard_skipped(value=None, *args, **kwargs):
    """Integrity guard that checks the authors' private version-controlled record:
    disabled in the public copy. Returns `value` unchanged."""
    return value


import json
import os
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src import k1_sim as S
from src import k1_theory as T
from src.fokker_planck import build_transition_pairs
import k1_gates as G
import k1_run_cells as RC

MASK = S.load_mask()
CELLS = S.registered_cells()


def _test_cell(process, n, B=0):
    return {**CELLS[0], "process": process, "n_override": n, "cell_index": 990, "B": B,
            "gate_use": "test"}


def _v0(res, pipeline="saai"):
    e = res["estimates"]
    return e[(e.pipeline == pipeline) & (e.variant == "V0")].set_index(["period", "quantity"])


def test_truth_recovery_cell_a_ou(monkeypatch):
    monkeypatch.setattr(S, "_w_shock", lambda: 0.0)
    res = S.run_replicate(_test_cell("i_ou", 4000), 0, MASK)
    for pl in ("saai", "direct"):
        e = _v0(res, pl)
        for p in S.PERIODS:
            r = lambda q: e.loc[(p, q)]
            assert abs(r("mean_diffusion").estimate / r("mean_diffusion").truth - 1) < 0.03
            assert abs(r("rho1").estimate - r("rho1").truth) < 0.01
            assert abs(r("xi_trap").estimate - r("xi_trap").truth) < 0.06
            assert abs(r("xi_trap").truth) < 1e-6
            assert r("two_delta_V").estimate < 0.01 and np.isfinite(r("tau_mfpt").truth)


def test_truth_recovery_cell_a_ou_at_registered_w():
    w = S._w_shock()
    assert abs(w - S.calibration()["province_dxi_icc_mean_2000_2018"]) == 0 and w > 0.2
    for p in S.PERIODS:
        proc = S.process("i_ou", p)
        win = S.open_base(MASK, p, 2000)
        rel = []
        for r in range(40):
            X, U = S.simulate_latent(proc, win, np.random.default_rng([11, r]))
            pr = S.pairs_from_matrix(S.xi_matrix_direct(X, np.ones(X.shape, bool)), win)
            tr = S.truth(proc, S.standardisation_table(X, win), p)
            rel.append(S.estimate(pr)["mean_diffusion"] / tr["mean_diffusion"] - 1)
        rel = np.asarray(rel)
        assert abs(rel.mean()) < 3 * rel.std(ddof=1) / np.sqrt(len(rel)), (p, rel.mean())


def test_truth_recovery_state_dependent_B_at_registered_w():
    from src import k1_contract as C
    for p in S.PERIODS:
        proc = S.process("iii_mult", p)
        win = S.open_base(MASK, p, 2000)
        rel, rel0 = [], []
        for r in range(30):
            X, U = S.simulate_latent(proc, win, np.random.default_rng([12, r]))
            pr = S.pairs_from_matrix(S.xi_matrix_direct(X, np.ones(X.shape, bool)), win)
            std = S.standardisation_table(X, win)
            assert np.allclose(std[C.SHOCK_SHARE_COLUMN],
                               S._w_shock() * S.province_herfindahl(win))
            est = S.estimate(pr)["mean_diffusion"]
            rel.append(est / S.truth(proc, std, p)["mean_diffusion"] - 1)
            rel0.append(est / S.truth(proc, std.assign(**{C.SHOCK_SHARE_COLUMN: 0.0}),
                                      p)["mean_diffusion"] - 1)
        rel, rel0 = np.asarray(rel), np.asarray(rel0)
        se = rel.std(ddof=1) / np.sqrt(len(rel))
        assert abs(rel.mean()) < 3 * se, (p, rel.mean(), se)
        if p == "during":
            assert rel0.mean() < -3 * se, (p, rel0.mean(), se)


def test_standardisation_carries_the_absorbed_shock_share():
    from src import k1_contract as C
    win = S.open_base(MASK, "during")
    X, _ = S.simulate_latent(S.process("i_ou", "during"), win, np.random.default_rng(3))
    wH = S._w_shock() * S.province_herfindahl(win)
    assert 0.03 < wH < 0.045
    assert (S.standardisation_table(X, win)[C.SHOCK_SHARE_COLUMN] == wH).all()
    assert (S.standardisation_table(X, win, "persistent")[C.SHOCK_SHARE_COLUMN] == 0).all()
    assert (S.standardisation_table(X, win, w=0.0)[C.SHOCK_SHARE_COLUMN] == 0).all()
    pan = S.simulate_panels(CELLS[26], 1, MASK)
    for p in S.PERIODS:
        assert np.isclose(pan["std"][p][C.SHOCK_SHARE_COLUMN].iloc[0],
                          S._w_shock() * S.province_herfindahl(pan["wins"][p]))


def test_real_k1_theory_predict_row_accepts_the_sim_cell_meta(tmp_path):
    """k1_theory.predict_row(..., cell=contract_cell_meta) accepts the
    simulator's composition tags on a no-exit null cell; its outputs pass
    the contract's pair_predictions, also after the JSON round trip the
    aggregator writes (k1_predict_rows.json.gz), and an edited value breaks
    the digest."""
    from src import k1_contract as C
    res = S.run_replicate(CELLS[34], 0, MASK)
    comp, std = res["composition"], res["standardisation"]
    for pl in S.PIPELINES:
        cell = S.contract_cell_meta(CELLS[34], pl)
        outs = []
        for p in S.PERIODS:
            cp = comp[(comp.pipeline == pl) & (comp.period == p)].reset_index(drop=True)
            sp = std.drop(columns=["cell_index", "replicate", "period"]).reset_index(drop=True)
            o = T.predict_row("projection", cp, res["true_params"][p], p, sp, cell)
            outs.append({"cell_index": 34, "pipeline": pl, "estimate": o["estimate"],
                         "truth": o["truth"], "diag": RC._keep_diag(o["diag"])})
        assert len(C.pair_predictions(cell, outs, "projection")) == 1
        f = str(tmp_path / f"{pl}.json.gz")
        RC.write_predict_rows(f, outs)
        back = RC.read_predict_rows(f)
        assert len(C.pair_predictions(cell, back, "projection")) == 1
        back[0]["estimate"]["rho1"] += 1e-9
        with pytest.raises(ValueError, match="digest"):
            C.pair_predictions(cell, back, "projection")
        bad = {**cell, "exit_dating": "exogenous", "exit_mechanism": "b"}
        with pytest.raises(ValueError):
            T.predict(cp, res["true_params"][p], p, sp, cell=bad)
        one = std[std.period == p].drop(columns=["cell_index", "replicate", "period"])
        with pytest.raises(ValueError, match="complete panel"):
            T.predict(cp, res["true_params"][p], p, one.reset_index(drop=True), cell=cell)


def test_truth_recovery_cell_a_monostable_cubic():
    """Same for a globally stable monostable cubic with state-dependent B (the
    theory line's GOLDEN_CUBIC: in the polynomial family, no guard), w = 0."""
    proc = S.TrueProcess(T.params_vector(T.GOLDEN_CUBIC))
    for p in S.PERIODS:
        win = S.open_base(MASK, p, 4000)
        X, U = S.simulate_latent(proc, win, np.random.default_rng(1), w=0.0)
        assert not U.any()
        pr = S.pairs_from_matrix(S.xi_matrix_direct(X, np.ones(X.shape, bool)), win)
        est = S.estimate(pr)
        tr = S.truth(proc, S.standardisation_table(X, win, w=0.0), p)
        assert abs(est["mean_diffusion"] / tr["mean_diffusion"] - 1) < 0.03
        assert abs(est["rho1"] - tr["rho1"]) < 0.01
        assert abs(est["xi_trap"] - tr["xi_trap"]) < 0.06


def test_truth_is_k1_theory_truth():
    proc = S.process("i_ou", "pre")
    win = S.open_base(MASK, "pre", 500)
    X, _ = S.simulate_latent(proc, win, np.random.default_rng(2))
    std = S.standardisation_table(X, win)
    a = S.truth(proc, std, "pre")
    b = T.truth(proc.as_dict(), std.drop(columns="period"), "pre")
    assert {q: a[q] for q in S.QUANTITIES} == {q: b[q] for q in S.QUANTITIES}
    assert set(std.columns) >= {"year", "n", "mu_x", "sigma_x", "x_units"}
    assert (std["x_units"] == "latent_x").all() and (std["n"] == win.n).all()


def test_true_process_declares_its_guard_and_matches_the_theory_chain():
    from src import k1_contract as C
    x = np.linspace(-15, 15, 3001)
    for name, p in [("i_ou", "pre"), ("ii_cubic", "pre"), ("ii_cubic", "during"),
                    ("iii_mult", "pre"), ("iii_mult", "during"), ("v_posctrl", "pre"),
                    ("iv_bistable_2", "during")]:
        proc = S.process(name, p, S.design_constants(name) if name.startswith("iv_") else None)
        d = proc.as_dict()
        assert set(d) == set(C.TRUE_PROCESS_KEYS)
        theta, guard = C.check_true_params(d)
        if proc.guard is None:
            assert guard is None and all(d[k] is None for k in C.GUARD_KEYS)
        else:
            assert guard == (proc.guard[0], proc.guard[1], proc.slope_floor)
            assert proc.slope_floor == -abs(S._ou_a1(p if name != "v_posctrl" else "pre"))
        np.testing.assert_allclose(T.drift(theta, guard, x), proc.A(x), rtol=0, atol=1e-12)
        np.testing.assert_allclose(T.log_diffusion(theta, guard, x), proc.logB(x),
                                   rtol=0, atol=1e-12)


def test_theory_compat_compares_the_guarded_law():
    from src import k1_contract as C
    assert "guard_active" not in S.COMPAT_TOL
    for name in ("i_ou", "ii_cubic", "iii_mult"):
        for p in S.PERIODS:
            r = S.theory_compat(name, p)
            assert r["compatible"] and r["theory_law"] == "guarded", (name, p, r)
            assert set(r["checks"]) == {"edge_mass", "mean_sd", "sd_rel"}
    r = S.theory_compat("ii_cubic", "during")
    assert r["guard_active_frac"] > 1e-3
    proc = S.process("ii_cubic", "during")
    with pytest.raises(ValueError, match="not confined"):
        T.stationary_law(proc.params, None)
    d = proc.as_dict()
    for k in C.GUARD_KEYS:
        d.pop(k)
    with pytest.raises(ValueError, match="lacks"):
        C.check_true_params(d)


def test_pairs_from_matrix_equals_build_transition_pairs():
    rng = np.random.default_rng(1)
    win = S.Window("during", np.arange(100, 130), np.repeat([1, 2, 3], 10), None)
    xi = rng.standard_normal((30, len(win.years)))
    xi[rng.random(xi.shape) < 0.3] = np.nan
    i, t = np.nonzero(~np.isnan(xi))
    long = pd.DataFrame({"CountyCode": win.codes[i], "Year": win.years[t], "SAAI": xi[i, t]})
    a = build_transition_pairs(long, "during").reset_index(drop=True)
    b = S.pairs_from_matrix(xi, win)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


@pytest.mark.parametrize("pipeline", ["direct", "saai"])
def test_composition_identity_random_exits_toy(pipeline):
    """mech_term == stayers' mean Δξ to 1e-10 under random exits AND entries
    (identity only: nothing is estimated); contract columns present."""
    from src.k1_contract import COMPOSITION_COLUMNS
    rng = np.random.default_rng(7)
    n = 80
    win = S.Window("during", 800_000 + np.arange(n), np.repeat(np.arange(8), 10), None)
    T_ = len(win.years)
    X = rng.standard_normal((n, T_)).cumsum(1) * 0.3 + rng.standard_normal((n, 1))
    P = rng.random((n, T_)) > 0.3
    lp, ll = S.output_inputs(win, rng)
    xi = S.xi_matrix_saai(X, P, lp, ll, win) if pipeline == "saai" else S.xi_matrix_direct(X, P)
    c = S.composition_table(xi, X, P, win, "shock", "exogenous")
    assert (c["n_leave"] > 0).all() and (c["n_enter"] > 0).all()
    assert c["identity_residual"].abs().max() < 1e-10
    assert np.allclose(c["mech_term"], c["mean_dxi_stay"], atol=1e-10, rtol=0)
    need = set(COMPOSITION_COLUMNS) - {"replicate", "pipeline"}
    assert need <= set(c.columns)
    j = 2
    S_ = P[:, j] & P[:, j + 1]
    assert np.isclose(c.loc[j, "mu_x_t"], X[P[:, j], j].mean())
    assert np.isclose(c.loc[j, "var_stay"], xi[S_, j].var())


def test_seed_reproducibility_and_rule():
    cell = CELLS[5]
    r1 = S.run_replicate(cell, 3, MASK, B=2)
    r2 = S.run_replicate(cell, 3, MASK, B=2)
    assert r1["seed"] == KC.PRECHECK_SEED_BASE + 5 * 10_000 + 3
    assert S.replicate_seed(CELLS[1], 3) == 20261021 + 1 * 10_000 + 3
    assert S.replicate_seed(cell, 3, stream="implcheck") == KC.IMPLCHECK_SEED_BASE + 5 * 10_000 + 3
    assert len({KC.S0_SEED_BASE, KC.PRECHECK_SEED_BASE, KC.IMPLCHECK_SEED_BASE,
                KC.S0_SEED_BASE_FREEZE1, S.SEED_BASE}) == 5
    with pytest.raises(ValueError, match="implcheck"):
        S.replicate_seed(CELLS[1], 3, stream="implcheck")
    for k in ("estimates", "composition", "truth", "standardisation"):
        pd.testing.assert_frame_equal(r1[k], r2[k])
    r3 = S.run_replicate(cell, 4, MASK, B=2)
    assert not r1["estimates"]["estimate"].equals(r3["estimates"]["estimate"])


def test_cell_a_V1_is_V0_and_failed_draws_are_counted():
    res = S.run_replicate(CELLS[0], 0, MASK, B=2)
    e = res["estimates"]
    for pl in S.PIPELINES:
        a = e[(e.pipeline == pl) & (e.variant == "V0")].drop(columns="variant").reset_index(drop=True)
        b = e[(e.pipeline == pl) & (e.variant == "V1")].drop(columns="variant").reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
    assert {"n_nan_boot", "ci_lo_lit", "ci_hi_lit"} <= set(e.columns) and (e["B"] == 2).all()
    assert set(e["obs_rule"]) == {"limit"}


def test_registered_grid_seeds_and_appended_cells():
    assert [c["cell_index"] for c in CELLS] == list(range(52))
    assert [(c["process"], c["mechanism"]) for c in CELLS[:5]] == [("i_ou", m) for m in "abcde"]
    assert sum(c["gate_use"] == "S0" for c in CELLS) == len(S.PROCESS_NAMES)
    assert all(c["gate_use"] in ("K1a_null", "control", "control_null") for c in CELLS[34:46])
    assert all(c["gate_use"] == "K1a_shocked_null" and c["dated_shocks"] for c in CELLS[46:50])
    assert [(c["process"], c["mechanism"], c["gate_use"], c.get("drop_breaks"))
            for c in CELLS[50:]] == [(p, "e", S.SENS_NO2006_GATE_USE, (2006,))
                                     for p in ("i_ou", "ii_cubic")]
    assert not any(c["dated_shocks"] for c in CELLS[:46])
    seeds = {S.SEED_BASE + c["cell_index"] * 10_000 + r for c in CELLS for r in range(500)}
    assert len(seeds) == len(CELLS) * 500
    for c in CELLS:
        S.check_cell(c)
        assert c["gated"] == (c["re_placement"] == "shock")


@pytest.mark.parametrize("ci", [3, 4, 8, 9])
def test_null_baseline_matches_its_k1a_cell(ci):
    cell = CELLS[ci]
    null = S.null_cell_for(cell)
    assert null["mechanism"] == "a" and null["structure"] == cell["mechanism"]
    assert null["B"] == 0 and null["gated"]
    keys = ("process", "N", "windows", "pipeline", "re_placement", "gated")
    for pl in S.PIPELINES:
        a, b = S.contract_cell_meta(cell, pl), S.contract_cell_meta(null, pl)
        assert {k: a[k] for k in keys} == {k: b[k] for k in keys}
        assert b["exit_mechanism"] == "a" and b["exit_dating"] == "none"
    wc, wn = S.cell_windows(cell, MASK), S.cell_windows(null, MASK)
    for p in S.PERIODS:
        assert (wc[p].codes == wn[p].codes).all() and wn[p].presence.all() and not wn[p].targets
    ctrl = [c for c in CELLS if c["gate_use"] == "control"
            and (c["process"], c["mechanism"]) == (cell["process"], cell["mechanism"])][0]
    assert not ctrl["gated"] and ctrl["re_placement"] == "persistent"
    assert S.null_cell_for(ctrl)["gate_use"] == "control_null"


def test_re_placement_is_enforced_before_simulating(monkeypatch):
    monkeypatch.setattr(S, "cell_windows", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("simulated a refused cell")))
    for bad in ({**CELLS[0], "re_placement": "persistent"},
                {**CELLS[0], "gated": False},
                {**CELLS[10], "re_placement": "persistent", "gated": False},
                {**CELLS[0], "re_placement": "initial_condition"},
                {**CELLS[0], "gated": "yes"}):
        with pytest.raises(ValueError):
            S.run_replicate(bad, 0, MASK, B=0)


def test_province_shock_keeps_each_county_law_and_is_stationary():
    """'shock': every county's innovation has variance B (w only splits it),
    and the between-province share at the window end equals the start (the
    old initial-condition assignment decayed as (1+a1)^t)."""
    proc = S.process("i_ou", "pre")
    win = S.open_base(MASK, "pre", 20_000)
    w = S._w_shock()
    assert 0.0 < w < 1.0
    X, U = S.simulate_latent(proc, win, np.random.default_rng(3))
    assert not U.any()
    eta = X[:, 1:] - X[:, :-1] - proc.A(X[:, :-1])
    assert abs(eta.var() / proc.B(0.0) - 1) < 0.05
    s0, s1 = S.province_share(X[:, 0], win.prov_idx), S.province_share(X[:, -1], win.prov_idx)
    assert s0 > 0.05 and abs(s1 - s0) < 0.5 * s0
    X0, _ = S.simulate_latent(proc, win, np.random.default_rng(3), w=0.0)
    assert S.province_share(X0[:, -1], win.prov_idx) < 0.01


def test_persistent_placement_is_an_observation_offset():
    proc = S.process("i_ou", "during")
    win = S.open_base(MASK, "during", 5000)
    X, U = S.simulate_latent(proc, win, np.random.default_rng(4), placement="persistent")
    Y = S.observed_latent(X, U)
    assert S.province_share(X[:, -1], win.prov_idx) < 0.01
    assert S.province_share(Y[:, 0], win.prov_idx) > 0.1
    assert S.province_share(Y[:, -1], win.prov_idx) > 0.1
    assert np.allclose(Y - X, U[:, None])


class _Reached(Exception):
    pass


@pytest.mark.parametrize('ci', [47])
def test_estimation_on_exit_cells_needs_a_committed_s0_pass(monkeypatch, ci):
    from src import k1_contract as C
    monkeypatch.setattr(S, "_check_guard", lambda m: None)
    monkeypatch.setattr(C, "_require_frozen", lambda: {"frozen_commit": "x"})
    for c in (CELLS[ci], S.exit_source_cell(CELLS[ci])):
        S.contract_cell_meta(c, "saai")
    monkeypatch.setattr(S, "cell_windows", lambda *a, **k: (_ for _ in ()).throw(_Reached()))
    monkeypatch.setattr(S, "estimate_full", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("an estimator ran")))
    cell = CELLS[ci]
    with pytest.raises(S.GuardRefused, match="S0"):
        S.run_replicate(cell, 0, MASK, B=0, estimate=True)
    with pytest.raises(_Reached):
        S.run_replicate(cell, 0, MASK, B=0, estimate=False)
    assert S.exit_guards(CELLS[0], True) == "no exit design"
    monkeypatch.setattr(RC, "guard_cells", lambda cells: "mocked freeze")
    with pytest.raises(S.GuardRefused):
        RC.main(["run", "--cells", str(ci), "--reps", "0:1", "--out", "/nonexistent", "--dev"])


def test_exit_dating_labels():
    """(d), (e): the leavers get the dated shock δ_b before exit, so exit is
    decided on information at or before t ('info_t'), not MCAR."""
    from src import k1_contract as C
    want = {"a": "none", "b": "exogenous", "c": "info_t", "d": "info_t", "e": "info_t"}
    assert S.EXIT_DATING == want
    for ci in (1, 3, 4, 46):
        m = C.check_cell_config(S.contract_cell_meta(CELLS[ci], "saai"))
        assert m["exit_dating"] == want[CELLS[ci]["mechanism"]] and m["valid_for_law"]


@pytest.mark.parametrize("ci", [46, 47, 48, 49])
def test_shocked_null_has_the_k1a_shocks_and_no_exit(ci):
    cell = CELLS[ci]
    src = S.exit_source_cell(cell)
    assert src["gate_use"] == "K1a/K1b" and src["mechanism"] == cell["structure"]
    assert S.null_cell_for(src)["gate_use"] == "K1a_null"
    assert S.shocked_null_for(src) == cell
    keys = ("process", "N", "windows", "pipeline", "re_placement", "gated")
    a, b = S.contract_cell_meta(src, "saai"), S.contract_cell_meta(cell, "saai")
    assert {k: a[k] for k in keys} == {k: b[k] for k in keys} and b["exit_mechanism"] == "a"
    ws, wn = S.cell_windows(src, MASK), S.cell_windows(cell, MASK)
    for p in S.PERIODS:
        assert (ws[p].codes == wn[p].codes).all() and wn[p].presence.all() and not wn[p].targets
        assert [(s_, list(i), y) for s_, i, y in ws[p].offsets] == \
               [(s_, list(i), y) for s_, i, y in wn[p].offsets]
        drops = getattr(wn[p], "drop_pairs", [])
        assert len(drops) == sum(s_ > 0 for s_, _, _ in ws[p].offsets)
        xi = np.zeros((wn[p].n, len(wn[p].years)))
        pr = S.pairs_from_matrix(xi, wn[p])
        assert len(pr) == wn[p].n * (len(wn[p].years) - 1) - sum(len(i) for _, i in drops)
    bad = {**cell, "structure": "b"}
    with pytest.raises(ValueError):
        S.check_cell(bad)


def _no_exit_run(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("guard bypassed: a replicate was about to run")
    monkeypatch.setattr(S, "run_replicate", boom)


def _git(cwd, *a):
    subprocess.run(["git", "-C", str(cwd), *a], check=True, capture_output=True)


def _toy_repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    (r / "README").write_text("x\n")
    _git(r, "add", ".")
    _git(r, "commit", "-qm", "base")
    return r


def _head(r):
    return subprocess.check_output(["git", "-C", str(r), "rev-parse", "HEAD"], text=True).strip()


def _fake_theory(monkeypatch):
    fake = types.ModuleType("src.k1_theory")
    tru = lambda tp, std, p, cell=None: {q: 1.0 for q in S.QUANTITIES} | {"id_status": "ok"}
    fake.truth = tru
    from src import k1_contract as C

    def predict(comp, tp, p, std, design="observed", cell=None):
        return {"estimate": {q: 1.0 for q in S.QUANTITIES},
                "diag": {"cell_checked": cell is not None, "design": design, "period": p,
                         "replicate": int(comp["replicate"].iloc[0]),
                         "cell_key": C.cell_key(C.check_cell_config(cell))}}
    fake.predict = predict
    fake.closed_form_observables = lambda comp, tp, p, std, cell=None: {q: 1.0 for q in S.QUANTITIES}
    fake.zeroth_order_observables = (lambda comp, tp, p, std, cell=None, form="first_order":
                                     {q: 1.0 for q in S.QUANTITIES})

    def predict_row(row, comp, tp, p, std, cell):
        base = predict(comp, tp, p, std, design=C.ROW_DESIGN[row], cell=cell)
        est, tr = base["estimate"], tru(tp, std, p)
        reg = C.check_registered_true_params(str(cell.get("process")), p, tp)
        diag = {**base["diag"], "row": row, "design": C.ROW_DESIGN[row],
                "theory_version": "fake", "contract_version": C.CONTRACT_VERSION,
                "true_params": reg, "true_params_check": C.REGISTERED_CHECK}
        diag["digest"] = C.prediction_digest(diag, est, tr)
        return {"estimate": est, "truth": tr, "bias": {q: 0.0 for q in S.QUANTITIES},
                "diag": diag}
    fake.predict_row = predict_row
    monkeypatch.setitem(sys.modules, "src.k1_theory", fake)
    import src
    monkeypatch.setattr(src, "k1_theory", fake, raising=False)


def test_composition_only_chunks_aggregate_without_estimates(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "estimate_full", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("an estimator ran in composition-only mode")))
    RC.main(["run", "--cells", "34", "--reps", "0:2", "--out", str(tmp_path / "c"),
             "--format", "csv.gz", "--dev", "--composition-only"])
    tabs, metas = RC.load_chunks([str(tmp_path / "c")])
    assert len(tabs["estimates"]) == 0 and len(tabs["composition"]) > 0
    assert {m["mode"] for m in metas} == {"composition_only"} and metas[0]["B"] == 0
    _, rep, _, _ = RC.check_chunks(metas, allow_dev=True, registered=False)
    assert rep[34]["mode"] == "composition_only"
    mixed = metas + [{**metas[0], "mode": "full", "replicates": [2, 3]}]
    with pytest.raises(SystemExit, match="mix modes"):
        RC.check_chunks(mixed, allow_dev=True, registered=False)
    _fake_theory(monkeypatch)
    pred, _, praw = RC.predictions_table(tabs["composition"], tabs["standardisation"],
                                         tabs["estimates"], metas)
    assert set(pred["replicate"]) == {0, 1} and pred["sim_estimate"].isna().all()
    assert len(praw) == 2 * 2 * (5 + 2)
    for f in (RC.bias_table, RC.coverage_table, RC.power_table, RC.refit_table):
        f(tabs["estimates"])


def test_ci_rules_count_failed_and_nan_draws():
    est = pd.DataFrame({
        "cell_index": 0, "replicate": [0, 1, 2, 3], "pipeline": "saai", "saai_spec": "multi",
        "variant": "V1", "obs_rule": "limit", "period": "contrast", "quantity": "xi_trap",
        "B": 200, "truth": 0.0, "estimate": 0.0,
        "ci_lo": [-1.0, -1.0, np.nan, -1.0], "ci_hi": [1.0, 1.0, np.nan, 1.0],
        "ci_lo_lit": [-1.0, -np.inf, 0.5, 0.2], "ci_hi_lit": [1.0, np.inf, 0.9, 0.9],
        "n_nan_boot": [0, 11, 0, 10]})
    c = RC.coverage_table(est).set_index("ci_nan_rule")
    assert c.loc["tol5", "n_ci_failed"] == 2 and c.loc["tol5", "coverage"] == 0.5
    assert c.loc["tol0", "n_ci_failed"] == 3 and c.loc["tol0", "coverage"] == 0.25
    assert c.loc["literal", "n_ci_failed"] == 0 and c.loc["literal", "coverage"] == 0.5


def test_bootstrap_ci_uses_symmetric_order_statistics():
    """k = ceil(0.025 B): B = 200 → 5th smallest and 5th largest (the old
    inverted_cdf upper bound was the 6th largest)."""
    v = np.arange(200, dtype=float)
    rng = np.random.default_rng(0)
    lo, hi, nn, lol, hil = S._ci(rng.permutation(v), 200)
    assert (lo, hi, nn) == (4.0, 195.0, 0) and (lol, hil) == (4.0, 195.0)
    assert S.order_ci(np.arange(500.0)) == (12.0, 487.0)
    assert S.order_ci(np.arange(40.0)) == (0.0, 39.0)
    lo, hi = S.order_ci(np.r_[-np.inf, np.arange(199.0)])
    assert lo == 3.0 and hi == 194.0


def test_literal_ci_keeps_nan_draws_in_the_denominator():
    v = np.r_[np.linspace(-1, 1, 190), [np.nan] * 10]
    lo, hi, nn, lol, hil = S._ci(v, 200)
    assert nn == 10 and np.isfinite(lo) and np.isfinite(hi)
    assert lol == -np.inf and hil == np.inf
    v2 = np.r_[np.linspace(-1, 1, 199), [np.nan]]
    lo2, hi2, _, lol2, hil2 = S._ci(v2, 200)
    assert np.isfinite(lol2) and np.isfinite(hil2) and lol2 <= lo2 and hil2 >= hi2


def test_contrast_values_use_the_contract_zero_convention():
    from src.k1_contract import _ratio
    cases = [(0.0, 0.0), (0.3, 0.0), (0.0, 0.2), (0.4, 0.2), (np.nan, 1.0)]
    for d, p in cases:
        got = S.contrast_values("tau_mfpt", d, p, 0.5, 0.5)
        want = _ratio({"tau_mfpt": d, "xi_trap": 0.5}, {"tau_mfpt": p, "xi_trap": 0.5}, "tau_mfpt")
        assert (np.isnan(got) and np.isnan(want)) or got == want
    assert S.contrast_values("tau_mfpt", 0.3, 0.2, 0.5, 0.0) == np.inf


def test_results_tables_path_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(RC, "tree_dirty", lambda: False)
    sub = os.path.join(RC.RESULTS_TABLES, "k1_precheck_a")
    assert RC.is_under(RC.RESULTS_TABLES) and RC.is_under(sub)
    assert not RC.is_under(str(tmp_path)) and not RC.is_under(RC.RESULTS_TABLES + "_x")
    with pytest.raises(SystemExit, match="results/tables"):
        RC.main(["aggregate", "--in", str(tmp_path), "--out", sub, "--non-registered"])
    with pytest.raises(SystemExit, match="results/tables"):
        RC.main(["aggregate", "--in", str(tmp_path), "--out", sub, "--dev"])


def test_failed_replicate_is_a_nan_row_and_grid_must_be_complete(monkeypatch, tmp_path):
    monkeypatch.delenv("K1_THEORY_FROZEN", raising=False)
    real = S.run_replicate

    def flaky(cell, rep, mask, **kw):
        if rep == 1:
            raise FloatingPointError("injected")
        return real(cell, rep, mask, **kw)
    monkeypatch.setattr(S, "run_replicate", flaky)
    RC.main(["run", "--cells", "5", "--reps", "0:2", "--B", "2", "--out", str(tmp_path),
             "--format", "csv.gz", "--dev"])
    tabs, metas = RC.load_chunks([str(tmp_path)])
    est = tabs["estimates"]
    f = est[est.replicate == 1]
    assert len(f) == len(est[est.replicate == 0])
    assert f["estimate"].isna().all() and (f["replicate_status"] == "failed:FloatingPointError").all()
    assert metas[0]["failed_replicates"][0]["replicate"] == 1
    b = RC.bias_table(est)
    assert (b["R"] == 2).all() and (b["n_replicate_failed"] == 1).all()
    assert (b[b.variant != "V0_neural"]["n_est_nan"] >= 1).all()
    out = tmp_path / "agg"
    RC.main(["aggregate", "--in", str(tmp_path), "--out", str(out), "--dev", "--non-registered"])
    meta = json.load(open(out / "k1_sim_meta.json"))
    assert meta["registered"] is False and meta["run_label"] == "NON-REGISTERED"
    assert meta["mask_content_sha256"] == S.mask_content_sha256(MASK)
    assert pd.read_csv(out / "k1_sim_bias.csv")["run_label"].eq("NON-REGISTERED").all()


def _meta(ci, **kw):
    return {"cell": CELLS[ci], "replicates": [0, 1], "code_commit": "c", "dev": False,
            "B": CELLS[ci]["B"], "_stem": "s", "versions": {},
            "mask": {"content_sha256": "m"}, **kw}


def test_chunk_theory_freeze_must_match_aggregator():
    with pytest.raises(SystemExit, match="theory freeze"):
        RC.check_chunks([_meta(1, theory_frozen="X")], registered=False, frozen="Y")


def test_chunk_without_mask_hash_is_refused():
    m = _meta(0)
    m["mask"] = {}
    with pytest.raises(SystemExit, match="mask"):
        RC.check_chunks([m], registered=False)


def test_unconverged_offsets_are_refused():
    m = _meta(3, theory_frozen="Y", offsets_diagnostics={"during": {"2016": {
        "converged": False, "target": -0.45, "achieved": -0.2, "bracketed": False}}})
    with pytest.raises(SystemExit, match="not converged"):
        RC.check_chunks([m], registered=False, frozen="Y")


def test_verify_ship_detects_modified_and_extra_files(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    man = {"commit": "abc", "files": {"src/a.py": RC._sha(str(tmp_path / "src" / "a.py"))}}
    (tmp_path / "K1_SHIP.json").write_text(json.dumps(man))
    assert RC.verify_ship(str(tmp_path)) == "abc"
    (tmp_path / "src" / "b.py").write_text("y = 2\n")
    with pytest.raises(SystemExit, match="unlisted"):
        RC.verify_ship(str(tmp_path))
    (tmp_path / "src" / "b.py").unlink()
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    with pytest.raises(SystemExit, match="modified"):
        RC.verify_ship(str(tmp_path))
    (tmp_path / "K1_SHIP.json").write_text(json.dumps({"commit": "abc"}))
    with pytest.raises(SystemExit, match="per-file"):
        RC.verify_ship(str(tmp_path))


def test_mask_content_hash_is_format_independent(tmp_path):
    p = tmp_path / "m.csv.gz"
    MASK.to_csv(p, index=False)
    m2 = S.load_mask(str(p))
    assert S.mask_provenance(m2)["content_sha256"] == S.mask_provenance(MASK)["content_sha256"]
    m3 = MASK.copy()
    m3.loc[0, "available"] = not m3.loc[0, "available"]
    assert S.mask_content_sha256(m3) != S.mask_content_sha256(MASK)
    assert S.mask_provenance(MASK)["content_sha256"] == S.MASK_CONTENT_SHA256
    p3 = tmp_path / "m3.csv.gz"
    m3.to_csv(p3, index=False)
    with pytest.raises(ValueError, match="pinned"):
        S.load_mask(str(p3))
    assert S.mask_content_sha256(S.load_mask(str(p3), allow_unpinned=True)) != S.MASK_CONTENT_SHA256


def test_calibrate_offsets_reports_convergence():
    """Latent chain only (no estimator)."""
    proc = S.process("i_ou", "during")
    n = 300
    pres = np.ones((n, 7), dtype=bool)
    pres[:40, 3:] = False
    pres[40:45, 5:] = False
    win = S.Window("during", 600_000 + np.arange(n), np.repeat(np.arange(10), 30), pres,
                   offsets=[(0, np.arange(40), 2015), (0, np.arange(40, 45), 2017)],
                   targets={2015: -0.5, 2017: -0.3})
    for placement in S.RE_PLACEMENTS:
        d, diag = S.calibrate_offsets(proc, win, seed=1, placement=placement, n_rep=8)
        for y in (2015, 2017):
            assert diag[y]["converged"] and abs(diag[y]["residual"]) < S.OFFSET_TOL
        assert diag[2017]["small_n"] and diag[2015]["placement"] == placement
    win.targets = {2015: -80.0}
    d, diag = S.calibrate_offsets(proc, win, seed=1, n_rep=4, sweeps=1)
    assert not diag[2015]["bracketed"] and not diag[2015]["converged"]


def _frame(rows, **common):
    return pd.DataFrame([{**common, **r} for r in rows])


def _cellcols(process, mechanism, n_override=np.nan, ci=0, gate_use="registered", **kw):
    return dict(cell_index=ci, process=process, mechanism=mechanism, n_override=n_override,
                gate_use=gate_use, theory_compatible=True, structure=np.nan, **kw)


def _verdict(rows, rule_set=None, unblocked=True, gate=None):
    v = G.verdicts(pd.DataFrame(rows))
    v = v[v["gate"] == gate] if gate else v[v["gate"] != "S0_neural"]
    v = v[v["primary"]] if rule_set is None else v[v["rule_set"] == rule_set]
    assert len(v) == 1
    return v.iloc[0]["verdict_unblocked" if unblocked else "verdict"]


from src import k1_contract as KC

R5 = KC.R_PER_CELL
FROZEN = {"frozen_commit": "f" * 40}
K1A_SPECS = [(3, "i_ou", "d", "K1a/K1b", None), (4, "i_ou", "e", "K1a/K1b", None),
             (8, "ii_cubic", "d", "K1a/K1b", None), (9, "ii_cubic", "e", "K1a/K1b", None),
             (34, "i_ou", "a", "K1a_null", "d"), (35, "i_ou", "a", "K1a_null", "e"),
             (36, "ii_cubic", "a", "K1a_null", "d"), (37, "ii_cubic", "a", "K1a_null", "e"),
             (46, "i_ou", "a", "K1a_shocked_null", "d"), (47, "i_ou", "a", "K1a_shocked_null", "e"),
             (48, "ii_cubic", "a", "K1a_shocked_null", "d"),
             (49, "ii_cubic", "a", "K1a_shocked_null", "e")]


def _frozen(monkeypatch, s0_pass=True):
    monkeypatch.setattr(KC, "_require_frozen", lambda: FROZEN)
    if s0_pass:
        monkeypatch.setattr(KC, "_require_s0_pass", lambda p, pl: {"pass": True})


def _fab_rows(meta, ci, pl, row, delta, R=R5, truth=0.5):
    """predict_row-shaped outputs: estimate = truth in 'pre', truth + delta in
    'during', so every predicted contrast is delta (difference and ratio)."""
    ck = [list(x) for x in KC.cell_key(KC.check_cell_config(meta))]
    out = []
    for r in range(R):
        for per in KC.PERIODS:
            tru = {q: truth for q in S.QUANTITIES}
            est = {q: truth + (delta if per == "during" else 0.0) for q in S.QUANTITIES}
            diag = {"cell_checked": True, "cell_key": ck, "row": row,
                    "design": KC.ROW_DESIGN[row], "replicate": r, "period": per,
                    "theory_version": "fabricated", "contract_version": KC.CONTRACT_VERSION,
                    "true_params": KC.registered_true_params(meta["process"], per),
                    "true_params_check": KC.REGISTERED_CHECK}
            diag["digest"] = KC.prediction_digest(diag, est, tru)
            out.append({"cell_index": ci, "pipeline": pl, "estimate": est, "truth": tru,
                        "diag": diag})
    return out


def _k1a_inputs(R=R5, pred_delta=0.2, sim_delta=0.2, null_sd=0.01, own_offset=0.05,
                pipelines=("saai",)):
    """(pred CSV with the simulated contrast rows, cell_meta, raw predict rows)."""
    rng = np.random.default_rng(7)
    rows, raw = [], []
    cm = {}
    for ci, proc, mech, gu, st in K1A_SPECS:
        cm[ci] = {pl: S.contract_cell_meta(CELLS[ci], pl) for pl in pipelines}
        for pl in pipelines:
            is_null = mech == "a"
            for rep in range(R):
                sb = (rng.normal(0, null_sd) if is_null else sim_delta + rng.normal(0, null_sd))
                for q in S.QUANTITIES:
                    rows.append({**_cellcols(proc, mech, ci=ci, gate_use=gu), "structure": st,
                                 "replicate": rep, "pipeline": pl, "saai_spec": "multi",
                                 "obs_rule": "limit", "row": "projection",
                                 "period": "contrast", "quantity": q, "pred_bias": np.nan,
                                 "sim_bias": sb, "truth": np.nan})
            for row in (G.REPORT_ROWS if pl == "saai" else ("projection", "fallback")):
                d = own_offset if (is_null and row == "zeroth_order") else (
                    0.0 if is_null else pred_delta)
                raw += _fab_rows(cm[ci][pl], ci, pl, row, d, R)
    return pd.DataFrame(rows), cm, raw


def _s0_fake_record(status=None, missing=(), bias=0.001, tol=0.01):
    """A stand-in for k1_gates._s0_record: every (process, pipeline) has a
    committed record with the given verdict; `missing` lack one."""
    def rec(proc, pl):
        if (proc, pl) in missing:
            return None, "not_evaluable_no_s0_record", "RuntimeError: no S0 record"
        st = status or "pass"
        res = {"R": R5, "cell": {"N": KC.S0_N}, "pass": st == "pass"}
        for per in KC.PERIODS:
            res[per] = {q: {"status": st, "median_bias": bias, "tol": tol} for q in S.QUANTITIES}
        return {"result": res, "pass": st == "pass", "attempt": 1}, st, "fake record"
    return rec


ALL_S0 = [(p, pl) for p in G.S0_PROCESSES for pl in S.PIPELINES]


def _s0ok(**kw):
    return G.gate_s0(0.0, _s0_fake_record(**kw), _s0_fake_record(**kw))


def test_s0_mean_diffusion_blocked_while_truth_lacks_the_shock_share(monkeypatch):
    assert getattr(T, G.SHOCK_SHARE_FLAG) is True
    monkeypatch.setattr(T, G.SHOCK_SHARE_FLAG, False)
    out = pd.DataFrame(G.gate_s0(S._w_shock(), _s0_fake_record(), _s0_fake_record()))
    md = out[out.quantity == "mean_diffusion"]
    assert (md["status"] == G.BLOCK_SHOCK_SHARE).all() and (md["contract_status"] == "pass").all()
    assert (out[out.quantity != "mean_diffusion"]["status"] == "pass").all()
    assert _verdict(out) == "not_evaluable"
    monkeypatch.setattr(T, G.SHOCK_SHARE_FLAG, True, raising=False)
    assert _verdict(G.gate_s0(S._w_shock(), _s0_fake_record())) == "pass"


def test_s0_backend_refuses_blocked_processes_before_simulating(monkeypatch):
    from src import k1_s0_backend as SB
    ok = lambda p, per: {"compatible": True}
    monkeypatch.setattr(S, "simulate_panels", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("simulated a blocked S0")))
    assert SB.theory_has_shock_share_term()
    monkeypatch.setattr(T, SB.SHOCK_SHARE_FLAG, False)
    with pytest.raises(SB.S0Blocked, match="shock-share"):
        SB.s0_precheck("ou", "saai", compat=ok)
    with pytest.raises(SB.S0Blocked, match="shock-share"):
        SB.cell_meta("ou", "a", "saai", s0=True)
    seed = KC.s0_seeds("ou", "saai")[0]
    meta = S.contract_cell_meta(CELLS[26], "saai")
    with pytest.raises(SB.S0Blocked):
        SB.simulate_no_exit(meta, 0, seed)
    monkeypatch.setattr(T, SB.SHOCK_SHARE_FLAG, True)
    bad = lambda p, per: {"compatible": per != "during", "reason": "x"}
    with pytest.raises(SB.S0Blocked, match="not the one the theory computes"):
        SB.s0_precheck("cubic", "saai", w=0.0, compat=bad)
    with pytest.raises(SB.S0Blocked, match="FAMILY"):
        SB.s0_precheck("cubic_bistable", "saai", w=0.0, compat=ok)
    with monkeypatch.context() as mp:
        mp.setitem(KC._CAL, "design", {})
        with pytest.raises(SB.S0Blocked, match="registered constants"):
            SB.s0_precheck("cubic_shift", "saai", w=0.0, compat=ok)
    assert SB.s0_precheck("ou", "direct", w=0.0, compat=ok)["cell_index"] == 26
    monkeypatch.setattr(SB, "s0_precheck", lambda *a, **k: CELLS[26])
    meta = SB.cell_meta("ou", "a", "saai", s0=True)
    assert S.N_S0 == KC.S0_N
    assert meta["N"] == 2 * KC.S0_N and meta["N_windows"] == KC.s0_window_sizes()
    KC.check_s0_N(meta)
    with pytest.raises(ValueError, match="per window"):
        KC.check_s0_N({**meta, "N": KC.S0_N, "N_windows": {"pre": KC.S0_N // 2,
                                                           "during": KC.S0_N // 2}})
    with pytest.raises(ValueError, match="sum of its windows"):
        KC.check_s0_N({**meta, "N": KC.S0_N})


def test_s0_backend_simulates_what_the_grid_simulates(monkeypatch):
    """simulate_no_exit = run_replicate's V0 pairs (same seed, same panel);
    a seed other than the registered one, a meta that is not the cell's, and
    composition requests for no-exit cells are refused."""
    from src import k1_s0_backend as SB
    monkeypatch.setattr(SB, "s0_precheck", lambda *a, **k: None)
    small = {**CELLS[26], "n_override": 300}
    monkeypatch.setattr(SB, "_sim_cell", lambda m: (small, m["pipeline"]))
    seed = KC.s0_seeds("ou", "direct")[1]
    out = SB.simulate_no_exit({"pipeline": "direct"}, 1, seed)
    assert out["true_params"] == {p: KC.registered_true_params("ou", p) for p in S.PERIODS}
    est = {p: SB.estimate(out["pairs"][p]) for p in S.PERIODS}
    res = S.run_replicate(small, 1, MASK, pipelines=("direct",))
    e = res["estimates"]
    for p in S.PERIODS:
        g = e[(e.variant == "V0") & (e.period == p)].set_index("quantity")["estimate"]
        assert all(np.isclose(est[p][q], g[q], equal_nan=True) for q in S.QUANTITIES)
    assert set(out["standardisation"]["period"]) == set(S.PERIODS)
    assert not out["standardisation"]["year"].duplicated().any()
    with pytest.raises(ValueError, match="registered S0 seed"):
        SB.simulate_no_exit({"pipeline": "direct"}, 1, seed + 1)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="not registered cell"):
        SB._sim_cell({**S.contract_cell_meta(CELLS[26], "saai"), "N": 1})
    with pytest.raises(ValueError, match="K1a exit cells"):
        SB.simulate_composition(S.contract_cell_meta(CELLS[34], "saai"), 0)


def test_s0_end_to_end_writes_one_record_per_process_pipeline(monkeypatch, tmp_path):
    import importlib.util
    import shutil
    from src import k1_s0_backend as SB
    spec = importlib.util.spec_from_file_location("k1_s0", os.path.join(S.ROOT, "scripts/k1_s0.py"))
    K0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K0)
    repo = tmp_path / "repo"
    (repo / 'results/records').mkdir(parents=True)
    (repo / "src").mkdir()
    shutil.copy(os.path.join(S.ROOT, ".gitattributes"), repo / ".gitattributes")
    shutil.copy(os.path.join(S.ROOT, KC.ARCHIVE_COMMIT_FILE), repo / KC.ARCHIVE_COMMIT_FILE)
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True,
                                    capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    git("config", "user.name", "t")
    git("config", "user.email", "t@t")
    git("add", "-A")
    git("commit", "-qm", "code")
    monkeypatch.setattr(KC, "REPO", repo)
    monkeypatch.setattr(KC, "_FROZEN", {}, raising=False)
    monkeypatch.setattr(KC, "_require_frozen", lambda: {"frozen_commit": "F1"})
    monkeypatch.setattr(KC, "R_PER_CELL", 2)
    monkeypatch.setattr(T, SB.SHOCK_SHARE_FLAG, True, raising=False)
    monkeypatch.setattr(S, "theory_compat", lambda p, per, *a: {"compatible": True})
    recs = []
    for pl in S.PIPELINES:
        KC.start_s0_attempt("cubic_mult", pl)
        K0.run("cubic_mult", pl, range(0, 2), tmp_path / f"raw_{pl}")
        out = K0.record("cubic_mult", pl, tmp_path / f"raw_{pl}")
        recs.append(json.loads(out["path"].read_text()))
        assert not git("status", "--porcelain")
    assert [r["result"]["pipeline"] for r in recs] == list(S.PIPELINES)
    assert all(r["seeds"] == KC.s0_seeds("cubic_mult", pl) for r, pl in zip(recs, S.PIPELINES))
    st = [(e["pipeline"], e["stage"]) for e in KC._read_attempts()]
    assert st == [("saai", "started"), ("saai", "result"), ("direct", "started"),
                  ("direct", "result")]
    assert set(pd.DataFrame(G.gate_s0(0.0)).query("process == 'cubic_mult'")["status"]) \
        == {"fail_R"}
    monkeypatch.setattr(G, "R_REGISTERED", 2)
    rows = pd.DataFrame(G.gate_s0(0.0))
    for r, pl in zip(recs, S.PIPELINES):
        g = rows[(rows.process == "cubic_mult") & (rows.pipeline == pl)]
        assert len(g) == 2 * len(S.QUANTITIES)
        assert set(g["status"]) <= {"pass", "fail", "fail_(non-finite)", "fail (non-finite)"}
        assert (g["status"] == "pass").all() == r["pass"]
    assert (rows[rows.process == "ou"]["status"] == "not_evaluable_no_s0_record").all()
    with pytest.raises(RuntimeError, match="already logged under this freeze"):
        KC.start_s0_attempt("cubic_mult", "saai")


def test_s0_cells_use_the_registered_seeds_per_pipeline():
    c = CELLS[26]
    assert c["gate_use"] == "S0"
    for pl in S.PIPELINES:
        assert [S.replicate_seed(c, r, pl) for r in range(5)] == KC.s0_seeds("ou", pl)[:5]
    with pytest.raises(ValueError):
        S.replicate_seed(c, 3)
    with pytest.raises(ValueError):
        S.replicate_seed(c, R5, "saai")
    assert S.replicate_seed(CELLS[6], 3) == S.SEED_BASE + 6 * 10_000 + 3
    assert S.replicate_seed(CELLS[5], 3) == KC.PRECHECK_SEED_BASE + 5 * 10_000 + 3
    small = {**c, "n_override": 300, "cell_index": 26}
    res = S.run_replicate(small, 1, MASK)
    assert res["seed"] == {pl: KC.s0_seeds("ou", pl)[1] for pl in S.PIPELINES}
    assert set(res["truth"]["pipeline"]) == set(S.PIPELINES)
    e = res["estimates"]
    a = e[(e.pipeline == "saai") & (e.variant == "V0") & (e.period == "pre")]
    b = e[(e.pipeline == "direct") & (e.variant == "V0") & (e.period == "pre")]
    assert not np.allclose(a["truth"].to_numpy(), b["truth"].to_numpy())


def test_gate_k1a_on_the_real_contract(monkeypatch):
    _frozen(monkeypatch)
    pred, cm, raw = _k1a_inputs()
    rows, claims = G.gate_k1a(pred, cm, G.index_predict_rows(raw))
    out = pd.DataFrame(rows)
    gating = out[out.gating]
    assert set(gating["statistic"].str.split(":").str[0]) == {"projection"}
    assert set(gating["cell_index"].astype(int)) == {3, 4, 8, 9} and _verdict(out) == "pass"
    assert (gating["status"] == "pass").all() and set(gating["null_variant"]) == {"registered"}
    assert claims["projection"] == "prediction confirmed" and claims["prl_condition_1"]
    assert claims["closed_form"] == "law under linear drift"
    sh = out[out.null_variant == "shocked"]
    assert len(sh) and not sh["gating"].any() and sh["report_only"].all()
    pred2, cm2, raw2 = _k1a_inputs(pred_delta=-0.2)
    assert _verdict(G.gate_k1a(pred2, cm2, G.index_predict_rows(raw2))[0]) == "fail"
    pred3, cm3, raw3 = _k1a_inputs(sim_delta=0.005, pred_delta=0.2)
    o3 = pd.DataFrame(G.gate_k1a(pred3, cm3, G.index_predict_rows(raw3))[0])
    assert "indistinguishable_from_null" in set(o3[o3.gating]["status"])


def test_gate_k1a_without_s0_record_or_with_short_runs_is_never_a_pass(monkeypatch):
    _frozen(monkeypatch, s0_pass=False)
    pred, cm, raw = _k1a_inputs()
    rows, claims = G.gate_k1a(pred, cm, G.index_predict_rows(raw))
    out = pd.DataFrame(rows)
    assert set(out[out.gating]["status"]) == {"not_evaluable_refused"}
    assert set(claims) == {"not_evaluable"} and "refused" in claims["not_evaluable"][0]
    assert set(out["status"]) <= {"not_evaluable_refused", "missing"}
    ref = out[out.status == "not_evaluable_refused"]
    assert ref["note"].str.contains("S0").all()
    _frozen(monkeypatch)
    small, cm2, raw2 = _k1a_inputs(R=20)
    assert _verdict(G.gate_k1a(small, cm2, G.index_predict_rows(raw2))[0]) == "not_evaluable"
    assert _verdict(G.gate_k1a(None, {}, None)[0]) == "not_run"
    no_null = {k: v for k, v in cm.items() if k not in (34, 35, 36, 37)}
    assert _verdict(G.gate_k1a(pred, no_null, G.index_predict_rows(raw))[0]) == "not_run"


def test_zeroth_order_own_offset_rule(monkeypatch):
    _frozen(monkeypatch)
    pred, cm, raw = _k1a_inputs(own_offset=0.05)
    out = pd.DataFrame(G.gate_k1a(pred, cm, G.index_predict_rows(raw))[0])
    z = out[(out.row == "zeroth_order") & out.statistic.str.endswith("_sign")
            & (out.null_variant == "registered") & out.report_only.eq(False)]
    assert (z["status"] == "pass").all()
    assert ((0 < z["own_offset_null_pred_median"]) & (z["own_offset_null_pred_median"]
                                                       < z["median_pred"])).all()
    assert G.k1a_uses_contract_own_offset_rule()
    pred, cm, raw = _k1a_inputs(own_offset=-0.3)
    rows, claims = G.gate_k1a(pred, cm, G.index_predict_rows(raw))
    out = pd.DataFrame(rows)
    z = out[(out.row == "zeroth_order") & out.statistic.str.endswith("_sign")
            & (out.null_variant == "registered") & out.report_only.eq(False)]
    assert (z["status"] == KC.OWN_OFFSET_STATUS.replace(" ", "_")).all()
    assert claims["zeroth_order"] == "not a law" and not claims["prl_condition_1"]
    assert claims["projection"] == "prediction confirmed"
    raw_bad = [r for r in raw if not (r["cell_index"] in (34, 35, 36, 37)
                                      and r["diag"]["row"] == "zeroth_order")]
    rows, claims = G.gate_k1a(pred, cm, G.index_predict_rows(raw_bad))
    out = pd.DataFrame(rows)
    z = out[(out.row == "zeroth_order") & (out.null_variant == "registered")
            & out.report_only.eq(False)]
    assert set(z["status"]) == {"not_evaluable_refused"} and len(z)
    assert set(claims) == {"not_evaluable"}


def test_report_only_cells_never_reach_law_claims(monkeypatch):
    for ci in (46, 47, 48, 49, 38, 42):
        m = S.contract_cell_meta(CELLS[ci], "saai")
        assert m["report_only"] is True
    m46 = KC.check_cell_config(S.contract_cell_meta(CELLS[46], "saai"))
    assert m46["gated"] and m46["valid_for_law"]
    for ci in (3, 4, 8, 9, 34, 26):
        assert S.contract_cell_meta(CELLS[ci], "saai")["report_only"] is False
    _frozen(monkeypatch)
    pred, cm, raw = _k1a_inputs()
    for ci in (34, 35, 36, 37):
        cm[ci] = {"saai": {**cm[ci]["saai"], "report_only": True}}
    rows, claims = G.gate_k1a(pred, cm, G.index_predict_rows(raw))
    out = pd.DataFrame(rows)
    assert claims is None and not out["gating"].any()
    with pytest.raises(ValueError, match="report-only"):
        G.law_claims_gated([{"report_only": True, "label": "x"}])


def test_s0_precedence_blocks_every_other_gate(monkeypatch):
    df = pd.DataFrame(G.gate_s0(0.0, _s0_fake_record("fail"))
                      + G.gate_k1b(_cov_rows(0.95)) + G.gate_k2(_k2_tab(0.85, 0.05)))
    v = G.verdicts(df, signed=True)
    assert v[(v.gate == "S0") & v.primary]["verdict"].iloc[0] == "fail"
    assert (v[~v.gate.isin(G.S0_GATES) & (v.gate != "K2")]["verdict"] == "blocked_by_S0").all()
    assert (v[v.gate == "K2"]["verdict"] == G.K2_VERDICT).all()
    assert v[(v.gate == "K1b") & v.primary]["verdict_unblocked"].iloc[0] == "pass"
    df = pd.DataFrame(G.gate_s0(0.0, _s0_fake_record()) + G.gate_k1b(_cov_rows(0.95)))
    v = G.verdicts(df, signed=True)
    assert v[(v.gate == "K1b") & v.primary]["verdict"].iloc[0] == "pass"


def test_law_claims_skip_theory_incompatible_cells(monkeypatch):
    _frozen(monkeypatch)
    pred, cm, raw = _k1a_inputs()
    rows, claims = G.gate_k1a(pred, cm, G.index_predict_rows(raw))
    assert claims["zeroth_order"] == "law" and claims["prl_condition_1"]
    for bad_ci in (8, 36):
        p2 = pred.copy()
        p2.loc[p2.cell_index == bad_ci, "theory_compatible"] = False
        rows, claims = G.gate_k1a(p2, cm, G.index_predict_rows(raw))
        out = pd.DataFrame(rows)
        g = out[out.gating & (out.cell_index == 8)]
        assert (g["status"] == "blocked_theory_incompatible").all() and len(g)
        assert set(claims) == {"not_evaluable"}
        assert any("blocked_theory_incompatible" in x for x in claims["not_evaluable"])
        assert "prl_condition_1" not in claims and "zeroth_order" not in claims


def test_k3_is_not_an_unsigned_sim_rule_gate():
    assert "K3" not in G.UNSIGNED_RULE_GATES and G.K3_DIAG_GATE not in G.UNSIGNED_RULE_GATES
    s0ok = _s0ok()
    k3 = [{"gate": "K3", "rule_set": G.PRIMARY_RULE_SET["K3"], "gating": True,
           "status": "pass"}]
    for signed in (False, True):
        v = G.verdicts(pd.DataFrame(s0ok + k3), signed=signed)
        assert v[v.gate == "K3"]["verdict"].iloc[0] == "pass"


def test_aggregator_ties_the_chunks_code_commit_to_its_own(tmp_path, monkeypatch):
    repo = tmp_path / "r"
    repo.mkdir()
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True,
                                    capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    git("config", "user.name", "t")
    git("config", "user.email", "t@t")
    (repo / "src").mkdir()
    (repo / "src/k1_sim.py").write_text("x = 1\n")
    git("add", "-A")
    git("commit", "-qm", "code")
    c0 = git("rev-parse", "HEAD")
    monkeypatch.setattr(RC, "ROOT", str(repo))
    assert RC.same_code(c0) is None
    (repo / "results/tables").mkdir(parents=True)
    (repo / "results/tables/k1_sim_bias.csv").write_text("a\n")
    (repo / 'results/records').mkdir(parents=True)
    (repo / "results/records/k1_s0_attempts.jsonl").write_text("{}\n")
    git("add", "-A")
    git("commit", "-qm", "tables")
    assert RC.same_code(c0) is None
    (repo / "src/k1_sim.py").write_text("x = 2\n")
    git("commit", "-qam", "fix")
    assert "code changed" in RC.same_code(c0)
    assert "not a full commit" in RC.same_code("abc")
    git("checkout", "-q", "--orphan", "other")
    git("commit", "-qm", "unrelated")
    assert "not an ancestor" in RC.same_code(c0)
    monkeypatch.setattr(RC, "load_chunks", lambda d: ({k: pd.DataFrame() for k in RC.TABLES},
                                                      [{"cell": CELLS[5]}]))
    monkeypatch.setattr(RC, "guard_cells", lambda c: None)
    monkeypatch.setattr(RC, "check_chunks", lambda *a, **k: (c0, {}, [], "m"))
    monkeypatch.setattr(RC, "tree_dirty", lambda: False)
    with pytest.raises(SystemExit, match="not at the aggregator's code state"):
        RC.main(["aggregate", "--in", str(tmp_path), "--out", str(tmp_path / "o"),
                 "--non-registered"])
    with pytest.raises(SystemExit, match="another code state"):
        G.same_code_as_aggregate(c0)


def test_k1c_not_evaluable_while_the_posctrl_design_fails():
    meta = {"23": {"error": "ValueError: f(a) and f(b) must have different signs"},
            "24": {"error": "x"}, "25": {"error": "x"}, "0": {}}
    vc = [c["cell_index"] for c in CELLS if c["process"] == "v_posctrl"
          and c["gate_use"] == "K1c"]
    assert vc == [23, 24, 25]
    err = G.posctrl_design_error(meta)
    assert err and "cell 23" in err
    assert G.posctrl_design_error({str(i): {"lam": 0.01, "beta": 0.01} for i in vc}) is None
    assert G.posctrl_design_error({"23": None}) is not None
    nanb = pd.DataFrame([{**_cellcols("v_posctrl", m, ci=23), "pipeline": "saai",
                          "saai_spec": "multi", "variant": "V1", "obs_rule": "limit",
                          "period": "contrast", "quantity": "xi_trap", "median_bias": np.nan,
                          "median_truth": -0.31, "n_est_nan": 500, "R": 500}
                         for m in ("a", "d", "e")])
    cov = pd.DataFrame([{**_cellcols("v_posctrl", m, ci=23), "pipeline": "saai",
                         "variant": "V1", "obs_rule": "limit", "ci_nan_rule": "tol0",
                         "period": "contrast", "quantity": "xi_trap", "coverage": 0.0,
                         "n_truth_nan": 0, "n_ci_failed": 500, "R": 500}
                        for m in ("a", "d", "e")])
    pw = cov.assign(power_excl_null_failed_as_neg=0.0)
    s0ok = _s0ok()
    raw = G.verdicts(pd.DataFrame(s0ok + G.gate_k1c(nanb, cov, pw)), signed=True)
    assert raw[(raw.gate == "K1c") & raw.primary]["verdict"].iloc[0] == "fail"
    v = G.verdicts(pd.DataFrame(s0ok + G.gate_k1c(nanb, cov, pw, err)), signed=True)
    assert v[(v.gate == "K1c") & v.primary]["verdict"].iloc[0] == "not_evaluable"


def test_run_cells_refuses_s0_cells():
    s0 = [c["cell_index"] for c in CELLS if c["gate_use"] == "S0"]
    assert s0 and not set(s0) & {c["cell_index"] for c in RC.parse_cells("all")}
    with pytest.raises(SystemExit, match="k1_s0.py"):
        RC.refuse_s0_cells([CELLS[s0[0]]])


def test_literal_ci_rule_never_gates():
    for rows in (G.gate_k1b(_cov_rows(0.95)), G.gate_k2(_k2_tab(0.85, 0.05))):
        out = pd.DataFrame(rows)
        assert not out[out.ci_nan_rule == "literal"]["gating"].any()
    out = pd.DataFrame(G.gate_k1b(_cov_rows(0.95)))
    assert out[out.ci_nan_rule == "tol0"]["gating"].any()
    assert not pd.DataFrame(G.gate_k2(_k2_tab(0.85, 0.05)))["gating"].any()


def test_shocked_null_report_is_never_gating():
    rows = []
    for ci, proc, _, gu, st in K1A_SPECS[8:]:
        for rep in range(5):
            for p in S.PERIODS:
                for q in S.QUANTITIES:
                    rows.append({**_cellcols(proc, "a", ci=ci, gate_use=gu), "structure": st,
                                 "replicate": rep, "pipeline": "saai", "saai_spec": "multi",
                                 "obs_rule": "limit", "period": p, "quantity": q,
                                 "estimate": 1.1, "truth": 1.0, "replicate_status": "ok"})
    out = pd.DataFrame(G.gate_shocked_null(pd.DataFrame(rows)))
    assert not out["gating"].any() and np.allclose(out["value"], 0.1)
    assert (out["status"] == "report").all()


def _cov_rows(cov, truth_nan=0, R=500):
    rows = []
    for obs in G.OBS_RULES:
        for cir in G.CI_RULES:
            for proc in ("i_ou", "ii_cubic", "iii_mult"):
                for mech in "bcde":
                    for q in ("xi_trap", "tau_mfpt"):
                        rows.append({**_cellcols(proc, mech), "pipeline": "saai",
                                     "saai_spec": "multi", "variant": "V1", "obs_rule": obs,
                                     "ci_nan_rule": cir, "period": "contrast", "quantity": q,
                                     "R": R, "coverage": cov, "n_ci_failed": 3,
                                     "n_truth_nan": truth_nan})
    return pd.DataFrame(rows)


def test_gate_k1b():
    assert _verdict(G.gate_k1b(_cov_rows(0.93))) == "pass"
    assert _verdict(G.gate_k1b(_cov_rows(0.89))) == "fail"
    assert _verdict(G.gate_k1b(_cov_rows(0.95, truth_nan=1))) == "not_evaluable"
    assert _verdict(G.gate_k1b(_cov_rows(0.95, R=200))) == "fail"
    part = _cov_rows(0.95)
    assert _verdict(G.gate_k1b(part[part["mechanism"] != "e"])) == "not_evaluable"
    ou_tau = _cov_rows(0.95)
    ou_tau.loc[(ou_tau.process == "i_ou") & (ou_tau.quantity == "tau_mfpt"), "coverage"] = 0.10
    rows = pd.DataFrame(G.gate_k1b(ou_tau))
    assert _verdict(G.gate_k1b(ou_tau)) == "pass"
    t = rows[(rows.process == "i_ou") & (rows.quantity == "tau_mfpt")]
    assert len(t) and not t["gating"].any() and \
        (t["status"] == G.K1B_NOT_EVALUABLE_STATUS).all()
    inc = _cov_rows(0.95)
    inc.loc[inc.process == "ii_cubic", "theory_compatible"] = False
    assert _verdict(G.gate_k1b(inc)) == "not_evaluable"
    ctrl = pd.concat([_cov_rows(0.95), _cov_rows(0.10).assign(gate_use="control")])
    assert _verdict(G.gate_k1b(ctrl)) == "pass"


def test_gate_k1c():
    def tabs(med_bias, cov, n_est_nan=0):
        common = dict(pipeline="saai", saai_spec="multi", variant="V1", period="contrast",
                      quantity="xi_trap", R=500)
        b = pd.DataFrame([{**_cellcols("v_posctrl", m), **common, "obs_rule": o,
                           "median_bias": med_bias, "median_truth": -0.31,
                           "n_est_nan": n_est_nan, "n_truth_nan": 0}
                          for m in "ade" for o in G.OBS_RULES])
        c = pd.DataFrame([{**_cellcols("v_posctrl", m), **common, "obs_rule": o,
                           "ci_nan_rule": r, "coverage": cov, "n_ci_failed": 0,
                           "n_truth_nan": 0} for m in "ade" for o in G.OBS_RULES
                          for r in G.CI_RULES])
        p = pd.DataFrame([{**_cellcols("v_posctrl", m), **common, "obs_rule": o,
                           "ci_nan_rule": r, "n_ci_failed": 0,
                           "power_excl_null_failed_as_neg": 0.6} for m in "ade"
                          for o in G.OBS_RULES for r in G.CI_RULES])
        return b, c, p
    assert _verdict(G.gate_k1c(*tabs(0.05, 0.92))) == "pass"
    assert _verdict(G.gate_k1c(*tabs(0.10, 0.92))) == "fail"
    assert _verdict(G.gate_k1c(*tabs(0.05, 0.80))) == "fail"
    assert _verdict(G.gate_k1c(*tabs(0.05, 0.92, n_est_nan=1))) == "fail"
    out = pd.DataFrame(G.gate_k1c(*tabs(0.05, 0.92)))
    assert (out[out.statistic == "V1_power_ci_excludes_0"]["gating"] == False).all()


def _k2_tab(power2, fpr0, v0_power2=None):
    rows = []
    for k in (0, 1, 2, 4):
        for m in "ae":
            for period in ("pre", "during"):
                for var in ("V0", "V1"):
                    for o in G.OBS_RULES:
                        for r in G.CI_RULES:
                            p2 = v0_power2 if (var == "V0" and v0_power2 is not None) else power2
                            rows.append({**_cellcols(f"iv_bistable_{k}", m), "pipeline": "saai",
                                         "saai_spec": "multi", "variant": var, "obs_rule": o,
                                         "ci_nan_rule": r, "period": period, "R": 500,
                                         "quantity": "two_delta_V", "n_ci_failed": 0,
                                         "median_truth": float(k),
                                         "detect_lo_gt_0p5_failed_as_neg": p2 if k == 2 else 0.5,
                                         "detect_lo_gt_0p5_failed_as_pos": fpr0 if k == 0 else 0.5})
    return pd.DataFrame(rows)


def test_gate_k2():
    for t in (_k2_tab(0.85, 0.05), _k2_tab(0.75, 0.05), _k2_tab(0.85, 0.12),
              _k2_tab(0.85, 0.05, v0_power2=0.5)):
        v = G.verdicts(pd.DataFrame(G.gate_k2(t)))
        assert len(v) and (v["verdict"] == G.K2_VERDICT).all()
        assert (v["verdict_unblocked"] == G.K2_VERDICT).all() and not v["gated"].any()
        assert (v["n_gating_rows"] == 0).all()
    assert G.K2_VERDICT == "report-only (amendment 10)"


def _k2_pvt(det):
    rows = []
    for k in (0, 1, 2, 4):
        for m in "ae":
            for b in ("(1.5, 2.0]", "(2.0, 2.5]", "truth_nan"):
                for var in ("V0", "V1"):
                    for r in G.CI_RULES:
                        for period in ("pre", "during"):
                            rows.append({**_cellcols(f"iv_bistable_{k}", m), "pipeline": "saai",
                                         "saai_spec": "multi", "variant": var,
                                         "obs_rule": "limit", "ci_nan_rule": r,
                                         "period": period, "quantity": "two_delta_V",
                                         "truth_bin": b, "n": 50, "n_ci_failed": 1,
                                         "detect_failed_as_neg": det,
                                         "detect_failed_as_pos": det + 0.02,
                                         "median_truth": float(k)})
    return pd.DataFrame(rows)


def test_R10a_k2_not_gated_iv_a_and_e_reported_nominal_and_by_truth_bin():
    t = _k2_tab(0.85, 0.05)
    rows = pd.DataFrame(G.gate_k2(t, _k2_pvt(0.3)))
    k2 = rows[rows.gate == "K2"]
    assert len(k2) == len(rows) and not k2["gating"].any()
    assert set(k2["status"]) == {G.K2_REPORT_STATUS} == {'report_only_amendment10'}
    assert k2["threshold"].isna().all()
    assert set(k2["truth_flag"]) == {"truth biased for w > 0 (re-derivation 8)"}
    assert set(k2["mechanism"]) == {"a", "e"}
    nom = k2[k2.statistic != "detection_rate_lo_gt_0.5_by_truth_bin"]
    assert len(nom) == 4 * 2 * 2 * 2 * len(G.CI_RULES)
    assert set(nom["nominal_two_delta_V"]) == {0.0, 1.0, 2.0, 4.0}
    pw = nom[nom.statistic == "power_lo_gt_0.5"]
    fp = nom[nom.statistic == "false_positive_rate_lo_gt_0.5"]
    assert set(pw["process"]) == {"iv_bistable_2"} and np.allclose(pw["value"], 0.85)
    assert set(fp["process"]) == {"iv_bistable_0"} and np.allclose(fp["value"], 0.05)
    assert set(pw["mechanism"]) == set(fp["mechanism"]) == {"a", "e"}
    assert np.allclose(pw["registered_threshold"], G.K2_POWER)
    assert np.allclose(fp["registered_threshold"], G.K2_FPR)
    tb = k2[k2.statistic == "detection_rate_lo_gt_0.5_by_truth_bin"]
    assert len(tb) == 4 * 2 * 3 * 2 * len(G.CI_RULES) * 2
    assert set(tb["mechanism"]) == {"a", "e"} and not tb["gating"].any()
    assert "truth_nan" in set(tb["truth_bin"]) and np.allclose(tb["value"], 0.3)
    for tab, pvt in ((_k2_tab(0.10, 0.90), _k2_pvt(0.0)), (_k2_tab(0.99, 0.0), _k2_pvt(1.0)),
                     (t, None), (t[t.mechanism == "e"], None)):
        v = G.verdicts(pd.DataFrame(G.gate_k2(tab, pvt)))
        assert (v["verdict"] == G.K2_VERDICT).all() and not v["gated"].any()
    rows0 = pd.DataFrame(G.gate_k2(t))
    mi = rows0[rows0.statistic == "cell not in power_vs_truth"]
    assert len(mi) == 8 and not mi["gating"].any()
    import inspect
    for f in (KC.law_claims, G.law_claims_gated, G.gate_k1a):
        assert "K2" not in inspect.getsource(f) and "gate_k2" not in inspect.getsource(f)
    ev = inspect.getsource(G.evaluate)
    assert "k1a_rows, claims = gate_k1a(" in ev and ev.count("claims =") == 2
    assert G.REPORT_ONLY_GATES == ("K2",)


def test_gate_k3_diagnostic_is_never_gating():
    def tab(bn, nan_n=0):
        rows = []
        for proc in ("i_ou", "ii_cubic"):
            for m in "de":
                for period in ("pre", "during", "contrast"):
                    for var, mb in (("V0", -0.4), ("V0_neural", bn)):
                        for o in G.OBS_RULES:
                            rows.append({**_cellcols(proc, m), "pipeline": "saai",
                                         "saai_spec": "multi", "variant": var, "obs_rule": o,
                                         "period": period, "quantity": "xi_trap", "R": 500,
                                         "median_bias": mb, "n_est_nan": 0, "n_truth_nan": 0,
                                         "n_bias_nan": nan_n if var == "V0_neural" else 0})
        return pd.DataFrame(rows)

    def st(t):
        out = pd.DataFrame(G.gate_k3_diagnostic(t))
        assert (out["gate"] == G.K3_DIAG_GATE).all() and not out["gating"].any()
        return set(out["status"])
    assert st(tab(-0.3)) == {"pass"}
    assert st(tab(-0.1)) == {"fail"} and st(tab(0.4)) == {"fail"}
    assert st(tab(-0.3, nan_n=2)) == {"fail_nan"}
    no_neural = tab(-0.3)
    assert st(no_neural[no_neural.variant == "V0"]) == {"missing"}
    assert _verdict(G.gate_k3_diagnostic(tab(-0.1)), gate=G.K3_DIAG_GATE) == "not_run"


def _k3_est(cm, nn_shift, R=R5, cell_shift=0.3):
    rng = np.random.default_rng(11)
    rows = []
    for ci, proc, mech, gu, st in K1A_SPECS:
        if gu not in ("K1a/K1b", "K1a_null"):
            continue
        for rep in range(R):
            for per in S.PERIODS:
                for var in ("V0", "V0_neural"):
                    for q in S.QUANTITIES:
                        sh = 0.0
                        if gu == "K1a/K1b" and per == "during":
                            sh = cell_shift + (nn_shift if var == "V0_neural" else 0.0)
                        rows.append({"cell_index": ci, "replicate": rep, "pipeline": "saai",
                                     "saai_spec": "multi", "variant": var, "obs_rule": "limit",
                                     "period": per, "quantity": q, "truth": 0.5,
                                     "estimate": 0.5 + sh + rng.normal(0, 0.01)})
    return pd.DataFrame(rows)


def test_gates_refuse_non_registered_tables_under_results(tmp_path, monkeypatch):
    sub = os.path.join(RC.RESULTS_TABLES, "k1_precheck_a")
    monkeypatch.setattr(G.subprocess, "check_output", lambda *a, **k: "")
    with pytest.raises(SystemExit, match="results/tables"):
        G.main(["--tables", sub, "--dev"])


def test_design_registry_is_what_the_simulator_simulates(monkeypatch):
    from src import k1_s0_backend as SB
    d = json.load(open(os.path.join(S.ROOT, KC.DESIGN_PROCESS_FILE)))
    for name in ("iv_bistable_0", "iv_bistable_1", "iv_bistable_2", "iv_bistable_4"):
        key = S.PROCESS_KEY[name]
        if key not in d["members"]:
            assert key in d["unregistered"]
            with pytest.raises(ValueError, match="no registered constants"):
                S.process(name, "pre")
            continue
        for p in S.PERIODS:
            proc = S.process(name, p)
            assert proc.as_dict() == KC.registered_true_params(key, p)
            KC.check_registered_true_params(key, p, proc.as_dict())
        if not d["design_rule"]["signed"]:
            with pytest.raises(SB.S0Blocked, match="not signed"):
                SB.s0_precheck(key, "saai", w=0.0, compat=lambda *a: {"compatible": True})
        monkeypatch.setattr(SB, "design_rule_signed", lambda: False)
        with pytest.raises(SB.S0Blocked, match="not signed"):
            SB.s0_precheck(key, "saai", w=0.0, compat=lambda *a: {"compatible": True})
        monkeypatch.setattr(SB, "design_rule_signed", lambda: True)
        assert SB.s0_precheck(key, "saai", w=0.0,
                              compat=lambda *a: {"compatible": True})["process"] == name
        monkeypatch.undo()
    assert d["design_rule"]["signed"] is False or public_guard_skipped(True)
    if "cubic_shift" in d["members"]:
        proc = S.process("v_posctrl", "during")
        assert proc.as_dict() == KC.registered_true_params("cubic_shift", "during")
    else:
        with pytest.raises(ValueError, match="no registered constants"):
            S.process("v_posctrl", "during")


def _neural_k3():
    k3 = pytest.importorskip("src.k1_neural_k3")
    from src import k1_neural_sde as nsde
    return k3, nsde


@pytest.mark.parametrize("ci", [3, 4, 8, 9])
def test_exit_cell_with_neural_and_no_neural_s0_refuses_not_nan(monkeypatch, ci):
    from src import k1_contract as C
    k3, nsde = _neural_k3()
    monkeypatch.setattr(nsde, "TUNING_FROZEN", True)
    monkeypatch.setattr(S, "_check_guard", lambda m: None)
    monkeypatch.setattr(S, "_check_coverage_precheck", lambda *a, **k: None)
    monkeypatch.setattr(C, "_require_frozen", lambda: {"frozen_commit": "x"})
    for c in (CELLS[ci], S.exit_source_cell(CELLS[ci])):
        S.contract_cell_meta(c, "saai")
    monkeypatch.setattr(C, "assert_exit_simulation_allowed", lambda cell: cell)
    monkeypatch.setattr(C, "assert_exit_estimation_allowed",
                        lambda cell: C.check_cell_config(cell))
    monkeypatch.setattr(S, "cell_windows", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("simulated before the neural guard")))
    monkeypatch.setattr(nsde, "estimate", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("a neural fit ran")))
    cell = CELLS[ci]
    assert S.is_neural_cell(cell) and cell["mechanism"] != "a"
    assert S.exit_guards(cell, True) == "estimation allowed"
    with pytest.raises(S.GuardRefused, match="neural refused") as ei:
        S.exit_guards(cell, True, neural=k3.sim_estimate)
    assert isinstance(ei.value.__cause__, k3.NeuralRefused)
    assert "S0" in str(ei.value)
    with pytest.raises(S.GuardRefused, match="neural refused"):
        S.run_replicate(cell, 0, MASK, B=0, neural=k3.sim_estimate)
    monkeypatch.setattr(RC, "guard_cells", lambda cells: "mocked freeze")
    monkeypatch.setattr(RC, "load_neural", lambda: (k3.sim_estimate, "k3"))
    with pytest.raises(S.GuardRefused, match="neural refused"):
        RC.main(["run", "--cells", str(ci), "--reps", "0:1", "--out", "/nonexistent", "--dev"])


def test_neural_refusal_inside_the_fit_loop_propagates(monkeypatch):
    k3, nsde = _neural_k3()
    cell = CELLS[34]
    assert S.is_neural_cell(cell) and cell["mechanism"] == "a"

    def refusing(pairs, **kw):
        raise k3.NeuralRefused("stub refusal")
    refusing.preflight = lambda meta: meta
    with pytest.raises(S.GuardRefused, match="stub refusal"):
        S.run_replicate(cell, 0, MASK, B=0, neural=refusing, pipelines=("saai",))

    def no_preflight(pairs, **kw):
        raise AssertionError("must not be called")
    with pytest.raises(S.GuardRefused, match="preflight"):
        S.run_replicate(cell, 0, MASK, B=0, neural=no_preflight, pipelines=("saai",))

    monkeypatch.setattr(nsde, "TUNING_FROZEN", False)
    with pytest.raises(S.GuardRefused, match="not frozen"):
        S.exit_guards(cell, True, neural=k3.sim_estimate)
    with pytest.raises(S.GuardRefused, match="not frozen"):
        S.run_replicate(cell, 0, MASK, B=0, neural=k3.sim_estimate, pipelines=("saai",))

    def crashing(pairs, **kw):
        raise FloatingPointError("fit blew up")
    crashing.preflight = lambda meta: meta
    out = S.run_replicate(cell, 0, MASK, B=0, neural=crashing, pipelines=("saai",))
    nr = out["estimates"][out["estimates"].variant == "V0_neural"]
    assert len(nr) and nr["estimate"].isna().all()
    assert all(set(str(v).split("|")) == {"neural_failed:FloatingPointError"}
               for v in nr["id_status"])


def test_v_search_takes_the_smallest_root():
    assert S.V_SCAN_NODES == 25 and S.V_BETA_BRACKET == (1e-4, 0.05)
    assert S.V_LAM_BRACKET == (0.003, 0.08) and S.V_SEARCH_NOTE.startswith("amendment 5 (b)")
    f = lambda x: (x - 0.2) * (x - 0.7)
    assert abs(S._first_crossing(f, 0.0, 1.0) - 0.2) < 1e-6
    assert abs(S._first_crossing(lambda x: x - 0.55, 0.0, 1.0) - 0.55) < 1e-6
    with pytest.raises(RuntimeError, match="no sign change"):
        S._first_crossing(lambda x: x + 1.0, 0.0, 1.0)
    with pytest.raises(RuntimeError, match="non-finite"):
        S._first_crossing(lambda x: np.nan, 0.0, 1.0)


def test_design_registry_after_amendment_5_b():
    d = json.load(open(os.path.join(S.ROOT, KC.DESIGN_PROCESS_FILE)))
    assert set(d["members"]) == set(KC.DESIGN_MEMBERS) and d["unregistered"] == {}
    assert d["design_rule"]["signed"] is True and public_guard_skipped(True)
    v = d["design_constants"]["cubic_shift"]
    assert v["search"] == S.V_SEARCH_NOTE and "smallest root" in v["search"]
    assert S.V_BETA_BRACKET[0] <= v["beta"] <= S.V_BETA_BRACKET[1]
    assert S.V_LAM_BRACKET[0] <= v["lam"] <= S.V_LAM_BRACKET[1]
    assert abs(v["residual_shift"]) <= 0.01 and abs(v["residual_logtau"]) <= 0.02
    for key, periods in KC.DESIGN_MEMBERS.items():
        name = {v_: k for k, v_ in S.PROCESS_KEY.items()}[key]
        for p in periods:
            assert S.theory_compat(name, p)["compatible"], (name, p)


def test_s0_block_is_scoped_per_gate():
    assert G.GATE_S0_SCOPE["K1a"] == ("ou", "cubic")
    assert set(G.GATE_S0_SCOPE["K2"]) == {m for m in KC.PROCESSES if m.startswith("cubic_bistable_")}
    gates = ["K1a", "K1b", "K1c", "K2", "K3", "shocked_null"]

    def blocked(mle_missing=(), mle_fail=False, neural_missing=()):
        rows = pd.DataFrame(G.gate_s0(0.0, _s0_fake_record("fail" if mle_fail else None,
                                                          missing=mle_missing),
                                      _s0_fake_record(missing=neural_missing)))
        return G.s0_blocked_gates(rows, gates)
    assert blocked() == set()
    assert blocked([("cubic_bistable_1", "direct")]) == {"K2", "shocked_null"}
    assert blocked([("cubic_shift", "saai")]) == {"K1c", "shocked_null"}
    assert blocked([("cubic_mult", "saai")]) == {"K1b", "shocked_null"}
    assert blocked([("ou", "direct")]) == {"K1a", "K1b", "K3", "shocked_null"}
    assert blocked(neural_missing=[("cubic", "saai")]) == {"K3"}
    assert blocked(mle_fail=True) == set(gates)
    rows = pd.DataFrame(G.gate_s0(0.0, _s0_fake_record(), _s0_fake_record()))
    rows = rows[~((rows.gate == "S0") & (rows.process == "cubic") & (rows.pipeline == "direct"))]
    assert G.s0_blocked_gates(rows, gates) == {"K1a", "K1b", "K3", "shocked_null"}
    df = pd.DataFrame(G.gate_s0(0.0, _s0_fake_record(missing=[("cubic_bistable_4", "saai")]),
                                _s0_fake_record())
                      + G.gate_k1b(_cov_rows(0.95)) + G.gate_k2(_k2_tab(0.85, 0.05)))
    v = G.verdicts(df, signed=True)
    assert v[(v.gate == "K1b") & v.primary]["verdict"].iloc[0] == "pass"
    assert v[(v.gate == "K2") & v.primary]["verdict"].iloc[0] == G.K2_VERDICT
    assert v[(v.gate == "K2") & v.primary]["verdict_unblocked"].iloc[0] == G.K2_VERDICT


def test_item9_exit_base_reaches_the_registered_pair_counts():
    for p, target in (("pre", 5860), ("during", 1734)):
        assert S.observed_pairs(MASK, p) == target
        n = S.exit_base_n(MASK, p)
        T = S.WINDOWS[p][1] - S.WINDOWS[p][0] + 1
        expect = n * sum(0.9 ** k for k in range(1, T))
        assert abs(expect - target) <= 0.5 * sum(0.9 ** k for k in range(1, T))
        for ci in (1, 2):
            assert S.cell_windows(CELLS[ci], MASK)[p].n == n
        assert S.cell_windows(CELLS[0], MASK)[p].n == S.open_base(MASK, p).n
    win = S.cell_windows(CELLS[1], MASK)["during"]
    pairs = []
    for r in range(30):
        P = S.presence_for("b", np.zeros((win.n, len(win.years))), win,
                           np.random.default_rng([5, r]))
        pairs.append(int((P[:, :-1] & P[:, 1:]).sum()))
    assert abs(np.mean(pairs) / 1734 - 1) < 0.02


def test_item4_only_tol0_gates():
    rows = pd.DataFrame(G.gate_k1b(_cov_rows(0.95)) + G.gate_k2(_k2_tab(0.85, 0.05)))
    assert G.PRIMARY_CI == "tol0"
    assert not rows[rows.ci_nan_rule != "tol0"]["gating"].any()
    ne = (rows.gate == "K1b") & (rows.process == "i_ou") & (rows.quantity == "tau_mfpt")
    ne |= rows.gate == "K2"
    assert rows[(rows.ci_nan_rule == "tol0") & (rows.pipeline == "saai") & ~ne
                & rows.statistic.isin(["V1_ci95_coverage", "power_lo_gt_0.5",
                                       "false_positive_rate_lo_gt_0.5"])]["gating"].all()
    assert not rows[ne]["gating"].any()


def test_item5_failed_cis_count_against_power_and_for_false_positives():
    t = _k2_tab(0.85, 0.05)
    t["detect_lo_gt_0p5_failed_as_pos"] = np.where(t["median_truth"] == 0, 0.5, 0.95)
    rows = pd.DataFrame(G.gate_k2(t))
    pw = rows[rows.statistic == "power_lo_gt_0.5"]
    fp = rows[rows.statistic == "false_positive_rate_lo_gt_0.5"]
    assert np.allclose(pw["value"], 0.85)
    assert np.allclose(fp["value"], 0.5)
    assert (fp["status"] == G.K2_REPORT_STATUS).all() and not fp["gating"].any()


def test_decision_G_sign_checks_need_both_null_bands(monkeypatch):
    v_reg = {"checks": {"xi_trap_sign": "pass", "xi_trap_rel": "pass"}, "law_status": "pass"}
    v_ok = {"checks": {"xi_trap_sign": "pass", "xi_trap_rel": "pass"}, "law_status": "pass"}
    v_in = {"checks": {"xi_trap_sign": "indistinguishable from null", "xi_trap_rel": "pass"},
            "law_status": "fail"}
    assert G.combine_with_shocked_null(v_reg, v_ok)["checks"]["xi_trap_sign"] == "pass"
    c = G.combine_with_shocked_null(v_reg, v_in)
    assert c["checks"]["xi_trap_sign"].endswith("(dated-shock null)")
    assert c["checks"]["xi_trap_rel"] == "pass" and c["law_status"] != "pass"
    m = G.combine_with_shocked_null(v_reg, None)
    assert m["checks"]["xi_trap_sign"] == G.SHOCKED_NULL_MISSING and m["law_status"] != "pass"
    _frozen(monkeypatch)
    pred, cm, raw = _k1a_inputs()
    rows, claims = G.gate_k1a(pred, cm, G.index_predict_rows(raw))
    g = pd.DataFrame(rows)
    g = g[g.gating & g.statistic.str.endswith("_sign")]
    assert len(g) and (g["status"] == "pass").all()
    p2 = pred.copy()
    shk = p2.cell_index.isin([46, 47, 48, 49])
    p2.loc[shk, "sim_bias"] = 0.2 + np.random.default_rng(3).normal(0, 0.05, shk.sum())
    rows2, claims2 = G.gate_k1a(p2, cm, G.index_predict_rows(raw))
    g2 = pd.DataFrame(rows2)
    g2 = g2[g2.gating & g2.statistic.str.endswith("_sign")]
    assert not (g2["status"] == "pass").any()
    assert claims2["zeroth_order"] == "not a law" and not claims2["prl_condition_1"]


def test_itemE_2006_break_sensitivity_is_report_only():
    for c in CELLS[50:]:
        meta = S.contract_cell_meta(c, "saai")
        assert meta["report_only"] and c["B"] == 0
        w = S.cell_windows(c, MASK)["pre"]
        w0 = S.cell_windows(CELLS[4 if c["process"] == "i_ou" else 9], MASK)["pre"]
        assert 2006 not in {o[2] for o in w.offsets} and 2006 in {o[2] for o in w0.offsets}
        j = 2006 - S.WINDOWS["pre"][0]
        lv = np.where(w0.presence[:, j] & ~w0.presence[:, j + 1])[0]
        assert len(lv) == 3 and w.presence[lv, j + 1:].all()
        assert ({o[2] for o in w0.offsets} - {2006}) == {o[2] for o in w.offsets}
    assert S.SENS_NO2006_GATE_USE in S.REPORT_ONLY_GATE_USES
    assert S.SENS_NO2006_GATE_USE in RC.PRED_GATE_USES


def test_S2_window_stratified_province_block_rerun_bootstrap():
    cell = CELLS[0]
    pan = S.simulate_panels(cell, S.replicate_seed(cell, 0), MASK)
    wins, sim = pan["wins"], pan["sim"]
    rng = np.random.default_rng([1, S.BOOT_STREAM])
    draws = S.province_block_draws(wins, 6, rng)
    for p in S.PERIODS:
        w = wins[p]
        assert len(draws[p]) == 6
        for idx in draws[p]:
            provs = w.prov_idx[idx]
            sizes = np.bincount(w.prov_idx)
            k = np.bincount(provs, minlength=w.n_prov) / sizes
            assert np.allclose(k, np.round(k)) and int(np.round(k).sum()) == w.n_prov
    assert draws["pre"][0].max() < wins["pre"].n and draws["during"][0].max() < wins["during"].n
    for pl in S.PIPELINES:
        idx = draws["pre"][0]
        pr = S.bootstrap_draw_pairs(pl, "V0", "a", sim["pre"], wins["pre"], idx)
        X, Y, P, lp, ll = sim["pre"]
        wd = S.Window("pre", np.arange(len(idx)), wins["pre"].prov[idx], P[idx])
        xi = S.xi_matrix(pl, (X[idx], Y[idx], P[idx], lp[idx], ll[idx]), wd)
        assert np.allclose(np.nanmean(xi, axis=0), 0, atol=1e-9)
        if pl == "direct":
            assert np.allclose(np.nanstd(xi, axis=0), 1, atol=1e-9)
        full = S.pairs_from_matrix(S.xi_matrix(pl, sim["pre"], wins["pre"]), wins["pre"])
        assert len(pr) == len(idx) * (len(wins["pre"].years) - 1)
        assert not np.allclose(np.sort(pr["xi_t"].to_numpy())[:len(full)],
                               np.sort(full["xi_t"].to_numpy())[:len(pr)])
    Pm = sim["pre"][2].copy()
    Pm[:40, 5:] = False
    simx = (sim["pre"][0], sim["pre"][1], Pm, sim["pre"][3], sim["pre"][4])
    idx = draws["pre"][1]
    v1 = S.bootstrap_draw_pairs("direct", "V1", "b", simx, wins["pre"], idx)
    wd = S.Window("pre", np.arange(len(idx)), wins["pre"].prov[idx], Pm[idx])
    ref = S.v1_pairs("direct", simx[1][idx], Pm[idx], simx[3][idx], simx[4][idx], wd)
    pd.testing.assert_frame_equal(v1.reset_index(drop=True), ref.reset_index(drop=True))
    r1 = S.run_replicate(cell, 2, MASK, B=3, pipelines=("saai",))
    r2 = S.run_replicate(cell, 2, MASK, B=3, pipelines=("saai",))
    pd.testing.assert_frame_equal(r1["estimates"], r2["estimates"])
    assert (r1["estimates"]["n_nan_boot"] >= 0).all()


def test_expi_rerun_ci_province_block_resample():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_expi_rerun", os.path.join(S.ROOT, "scripts", "run_expi_rerun_ci.py"))
    X = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(X)
    assert X.SEED == 20261023 and X.N_BOOT == 1000 and X.SEED != 106
    codes = np.array([110101, 110102, 120101, 130101, 130102, 130103])
    wp = pd.DataFrame([{"CountyCode": c, "Year": y, "v": float(c % 7 + y)}
                       for c in codes for y in (2013, 2014, 2015)])
    provs = np.unique(X.province_of(codes))
    members = {int(p): codes[X.province_of(codes) == p] for p in provs}
    rng = np.random.default_rng([X.SEED, 0])
    d = X.resampled_window(wp, members, provs, rng)
    k = d["CountyCode"] // 1_000_000
    orig = d["CountyCode"] % 1_000_000
    assert sorted(k.unique()) == [1, 2, 3]
    for slot, g in d.groupby(k):
        pv = np.unique(X.province_of(g["CountyCode"] % 1_000_000))
        assert len(pv) == 1 and set(g["CountyCode"] % 1_000_000) == set(members[int(pv[0])])
        assert (g.groupby("CountyCode")["Year"].nunique() == 3).all()
    assert d["CountyCode"].nunique() == len(d) // 3
    assert set(orig) <= set(codes)
