"""Run:  python3 -m pytest -q tests/test_expi_paired.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_expi_paired as P
import run_expi_rerun_ci as R
import run_k2br as KBR


def synthetic_panel(seed: int = 0, n_prov: int = 6, n_per: int = 14) -> pd.DataFrame:
    """County panel 2000-2019 with the multi-SAAI columns.  Latent AR(1)
    anomaly; province 14 leaves 2016 -> 2017 and comes back in 2018, a few
    counties of province 12 enter in 2005 -- an open panel in both windows."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_prov):
        prov = 11 + k
        shock = rng.normal(0, 0.1, 20)
        for j in range(n_per):
            code = prov * 10000 + 21 + j
            pop = np.exp(rng.normal(3.5, 0.5))
            area = np.exp(rng.normal(4.0, 0.4))
            x = rng.normal(0, 0.3)
            for t, year in enumerate(range(2000, 2020)):
                x = 0.8 * x + shock[t] + rng.normal(0, 0.15)
                if prov == 14 and year == 2017:
                    continue
                if prov == 12 and j < 4 and year < 2005:
                    continue
                gov = np.exp(0.3 + 0.9 * np.log(pop) + 0.3 * np.log(area) + x - 0.4 * (prov == 14))
                rows.append({"CountyCode": code, "Year": year, "GrossOutputValue": gov,
                             "PopNum": pop * np.exp(rng.normal(0, 0.01)), "CultivateArea": area})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def world():
    panel = synthetic_panel()
    windows, members = P.window_setup(panel)
    point, warm = P.evaluate(windows, None)
    return dict(panel=panel, windows=windows, members=members, point=point, warm=warm)


def test_draw_panels_are_the_supplements_resamples(world, monkeypatch):
    """For given seed and b, draw_panels builds exactly the panels that the
    supplement's own one_draw builds (spy on its resampled_window)."""
    orig = R.resampled_window
    seen = []

    def spy(*a, **k):
        out = orig(*a, **k)
        seen.append(out)
        return out

    def no_fit(*a, **k):
        raise RuntimeError("skip the fit in this test")

    R._init(None, world["windows"], world["members"], world["warm"])
    for b in (0, 1, 7):
        mine = P.draw_panels(b, world["windows"], world["members"])
        seen.clear()
        monkeypatch.setattr(R, "resampled_window", spy)
        monkeypatch.setattr(R, "PAIRS", {v: no_fit for v in R.VARIANTS})
        R.one_draw(b)
        monkeypatch.undo()
        assert len(seen) == len(R.PERIODS)
        for p, theirs in zip(R.PERIODS, seen):
            pd.testing.assert_frame_equal(mine[p], theirs)
    a0 = P.draw_panels(0, world["windows"], world["members"])
    a1 = P.draw_panels(1, world["windows"], world["members"])
    assert not a0["pre"]["CountyCode"].equals(a1["pre"]["CountyCode"])


def test_draw_values_equal_supplement_one_draw(world):
    """Every column the supplement records for draw b is reproduced bit for
    bit by this script's draw (same pairs, same warm start, same fit)."""
    R._init(None, world["windows"], world["members"], world["warm"])
    P._init(world["windows"], world["members"], world["warm"])
    for b in (0, 3):
        theirs = R.one_draw(b)
        mine = P.one_draw(b)
        for k, v in theirs.items():
            if isinstance(v, float) and np.isnan(v):
                assert np.isnan(mine[k]), k
            else:
                assert mine[k] == v, k
        assert {f"{v}|rho1_{p}" for v in R.VARIANTS for p in R.PERIODS} <= set(mine)


def test_window_setup_matches_supplement_rule(world):
    for p in R.PERIODS:
        provs = set(world["members"][p])
        assert provs == set(np.unique(R.province_of(world["windows"][p]["CountyCode"])))
        for pv, codes in world["members"][p].items():
            assert np.all(R.province_of(codes) == pv)


def test_rho1_is_the_us_brazil_pooled_ols_definition(world):
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 400)
    dx = -0.12 * x + rng.normal(0, 0.3, 400)
    r = KBR.rho1_from_pairs(x, dx)
    b = np.polyfit(x, dx, 1)[0]
    assert r == pytest.approx(1 + b, abs=1e-12)
    xc = x - x.mean()
    assert r == pytest.approx(1 + (xc * (dx - dx.mean())).sum() / (xc * xc).sum(), abs=1e-14)
    pr = R.PAIRS["V1_stayer_restandardised"](world["windows"]["during"], "during")
    assert world["point"]["V1_stayer_restandardised|rho1_during"] == KBR.rho1_from_pairs(
        pr["xi_t"].to_numpy(), pr["delta_xi"].to_numpy())


def test_slope_recovers_ar1_on_synthetic_pairs():
    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, 20000)
    dx = -0.2 * x + rng.normal(0, 0.4, 20000)
    a1 = KBR.rho1_from_pairs(x, dx) - 1
    assert a1 == pytest.approx(-0.2, abs=0.01)
    ev = P.estimand_values({"V0_baseline|rho1_pre": 1 + a1, "V1_stayer_restandardised|rho1_pre": 1 + a1})
    assert ev[("amp_V1_pre", "none")] == pytest.approx(1 / abs(a1))


def _fake_draws(n=1000, seed=2, nan_at=None, inf_at=None):
    rng = np.random.default_rng(seed)
    d = {}
    common = {p: rng.normal(0, 0.5, n) for p in R.PERIODS}
    for c in R.CONVENTIONS:
        for p in R.PERIODS:
            d[f"V0_baseline|{c}|xi_trap_{p}"] = common[p] + 0.3 + rng.normal(0, 0.01, n)
            d[f"V1_stayer_restandardised|{c}|xi_trap_{p}"] = common[p] + rng.normal(0, 0.01, n)
            d[f"V0_baseline|{c}|tau_mfpt_{p}"] = np.exp(rng.normal(1, 0.2, n))
            d[f"V1_stayer_restandardised|{c}|tau_mfpt_{p}"] = np.exp(rng.normal(1, 0.2, n))
    for p in R.PERIODS:
        d[f"V0_baseline|rho1_{p}"] = 0.8 + rng.normal(0, 0.01, n)
        d[f"V1_stayer_restandardised|rho1_{p}"] = 0.85 + rng.normal(0, 0.01, n)
        d[f"dose_{p}"] = 0.5 + rng.normal(0, 0.02, n)
        d[f"dose_max_{p}"] = 1.0 + rng.normal(0, 0.02, n)
        d[f"Mbar_{p}"] = -0.05 + rng.normal(0, 0.01, n)
    df = pd.DataFrame(d)
    if nan_at is not None:
        df.loc[nan_at, "V1_stayer_restandardised|limit|xi_trap_during"] = np.nan
    if inf_at is not None:
        df.loc[inf_at, "V0_baseline|limit|tau_mfpt_pre"] = 0.0
    return df


def test_paired_difference_is_drawwise_and_narrower_than_marginal():
    d = _fake_draws()
    ev = P.estimand_values(d)
    diff = ev[("xi_trap_V0_minus_V1_during", "limit")]
    assert np.allclose(diff, d["V0_baseline|limit|xi_trap_during"]
                       - d["V1_stayer_restandardised|limit|xi_trap_during"])
    c = KBR.ci(diff)
    assert c["ci_status"] == "ok" and c["ci_lo"] > 0.25 and c["ci_hi"] < 0.35
    m0 = KBR.ci(d["V0_baseline|limit|xi_trap_during"])
    m1 = KBR.ci(d["V1_stayer_restandardised|limit|xi_trap_during"])
    assert (c["ci_hi"] - c["ci_lo"]) < 0.2 * ((m0["ci_hi"] - m0["ci_lo"]) + (m1["ci_hi"] - m1["ci_lo"]))
    shift = ev[("xi_shift_V0_minus_V1", "limit")]
    exp = ((d["V0_baseline|limit|xi_trap_during"] - d["V0_baseline|limit|xi_trap_pre"])
           - (d["V1_stayer_restandardised|limit|xi_trap_during"]
              - d["V1_stayer_restandardised|limit|xi_trap_pre"]))
    assert np.allclose(shift, exp)
    tr = ev[("tau_ratio_V0_over_V1", "nan")]
    exp = ((d["V0_baseline|nan|tau_mfpt_during"] / d["V0_baseline|nan|tau_mfpt_pre"])
           / (d["V1_stayer_restandardised|nan|tau_mfpt_during"]
              / d["V1_stayer_restandardised|nan|tau_mfpt_pre"]))
    assert np.allclose(tr, exp)
    a1 = ev[("a1_V1_pre", "none")]
    assert np.allclose(a1, d["V1_stayer_restandardised|rho1_pre"] - 1)
    assert np.allclose(ev[("dose_x_amp_V1_pre", "none")], d["dose_pre"] / np.abs(a1))
    assert np.allclose(ev[("zeroth_order_Mbar_over_abs_a1_V1_during", "none")],
                       d["Mbar_during"] / np.abs(d["V1_stayer_restandardised|rho1_during"] - 1))


def test_k_rank_rule_and_tol0_failures():
    v = np.arange(1, 1001, dtype=float)
    c = KBR.ci(v)
    assert (c["ci_lo"], c["ci_hi"]) == (25.0, 976.0)
    d = _fake_draws(nan_at=[3], inf_at=[5])
    t = P.estimand_table({k: float(np.nanmedian(np.where(np.isfinite(d[k]), d[k], np.nan)))
                          for k in d.columns}, d)
    r = t[(t.estimand == "xi_trap_V0_minus_V1_during") & (t.convention == "limit")].iloc[0]
    assert r.ci_status == "not_evaluable_nonfinite" and np.isnan(r.ci_lo) and r.n_nonfinite == 1
    assert r.supplement_rule_n == 999 and np.isfinite(r.supplement_rule_lo)
    r = t[(t.estimand == "tau_ratio_V0_over_V1") & (t.convention == "limit")].iloc[0]
    assert r.n_nonfinite == 1 and r.ci_status == "not_evaluable_nonfinite"
    r = t[(t.estimand == "xi_trap_V0_minus_V1_pre") & (t.convention == "limit")].iloc[0]
    assert r.ci_status == "ok"
    assert set(t.estimand) == set(P.ESTIMAND_ROLE)
    assert t[t.convention == "none"].primary.all()


def test_supplement_quantile_rule_matches_summarise():
    d = _fake_draws()
    rows = R.summarise(d.assign(**{f"{v}|limit|{q}_{p}": 1.0 for v in R.VARIANTS for p in R.PERIODS
                                   for q in ("delta_V", "gamma", "mean_diffusion", "B_ratio_w")}),
                       "V0_baseline", "limit")
    r = [x for x in rows if x["quantity"] == "xi_trap (during-pre)"][0]
    v = d["V0_baseline|limit|xi_trap_during"] - d["V0_baseline|limit|xi_trap_pre"]
    m, lo, hi, n = P.supplement_quantile_ci(v)
    assert (m, lo, hi, n) == (r["median"], r["ci_lo"], r["ci_hi"], r["n_draws"])


def test_composition_dose_and_mbar_definitions(world):
    """dose = sum(n_E + n_L)/sum(n_S) = n_S-weighted mean of (n_E+n_L)/n_S
    (k2br_coverage_summary rule); M_bar = n_S-weighted M_t = mean stayer
    increment of the V0 pairs (Lemma 1)."""
    from run_attrition_diagnostic import decompose_year
    from src.saai import compute_saai
    for p in R.PERIODS:
        wp = world["windows"][p]
        c = P.composition(wp, p)
        sa = compute_saai(wp, **R.SPEC)
        a, b = P.FP_PERIODS[p]
        d = pd.DataFrame([decompose_year(sa, t) for t in range(a, b)])
        ratio = (d.n_enter.fillna(0) + d.n_leave) / d.n_stay
        assert c["dose"] == pytest.approx(float((ratio * d.n_stay).sum() / d.n_stay.sum()), abs=1e-14)
        assert c["dose_max"] == pytest.approx(float(ratio.max()))
        pr = R.v0_pairs(wp, p)
        assert c["Mbar"] == pytest.approx(float(pr["delta_xi"].mean()), abs=1e-12)
    assert world["point"]["dose_during"] > 0
    sa = compute_saai(world["windows"]["during"], **R.SPEC)
    assert decompose_year(sa, 2016)["n_leave"] == 14


def test_v1_mbar_is_zero_by_construction(world):
    pr = R.v1_pairs(world["windows"]["during"], "during")
    for t, g in pr.groupby("Year_t"):
        assert abs(g["delta_xi"].mean()) < 1e-10


def _committed_like(world, tmp_path, draws):
    var, lev = P.marginal_tables(world["point"], draws)
    var.to_csv(tmp_path / "expi_rerun_ci_variants.csv", index=False)
    lev.to_csv(tmp_path / "expi_rerun_ci_levels.csv", index=False)
    rows = []
    for p in R.PERIODS:
        rows.append({"dataset": "China counties multi-SAAI (expi_attrition_by_year.csv)",
                     "window": p, "nS_weighted_comp_shock": world["point"][f"dose_{p}"],
                     "max_comp_shock": world["point"][f"dose_max_{p}"]})
    pd.DataFrame(rows).to_csv(tmp_path / "k2br_coverage_summary.csv", index=False)
    pd.DataFrame([{"period": p, "mechanical_term_weighted": world["point"][f"Mbar_{p}"]}
                  for p in R.PERIODS]).to_csv(tmp_path / "expi_attrition_by_period.csv", index=False)


@pytest.fixture(scope="module")
def small_draws(world):
    P._init(world["windows"], world["members"], world["warm"])
    return pd.DataFrame([P.one_draw(b) for b in range(6)])


def test_consistency_tiers(world, small_draws, tmp_path):
    _committed_like(world, tmp_path, small_draws)
    c = P.consistency(world["point"], small_draws, tab=str(tmp_path))
    assert (c.tier == "exact").all() and len(c) > 100
    d2 = small_draws.copy()
    d2["V0_baseline|limit|xi_trap_during"] += 5e-4
    c = P.consistency(world["point"], d2, tab=str(tmp_path))
    assert not (c.tier == "fail").any() and (c.tier == "optimiser").any()
    assert set(c.loc[c.tier == "outside_not_gated", "key"]) <= {
        f"V0_baseline|limit|{q} (during/pre)" for q in ("xi_trap", "delta_V", "gamma")}
    d3 = small_draws.copy()
    d3["V0_baseline|limit|xi_trap_during"] += 0.05
    assert (P.consistency(world["point"], d3, tab=str(tmp_path)).tier == "fail").any()
    d4 = small_draws.copy()
    d4.loc[0, "V0_baseline|nan|xi_trap_pre"] = np.nan
    assert (P.consistency(world["point"], d4, tab=str(tmp_path)).tier == "fail").any()
    pt = dict(world["point"])
    pt["dose_during"] += 1e-9
    assert (P.consistency(pt, small_draws, tab=str(tmp_path)).tier == "fail").any()


def test_dev_run_never_writes_results_tables():
    with pytest.raises(SystemExit):
        P.main(["--dev", "--n-boot", "2", "--out", str(ROOT / "results" / "tables")])


def test_dev_main_end_to_end_on_synthetic_panel(tmp_path, monkeypatch):
    """--dev smoke run with the real-data loader replaced by the synthetic
    panel: the comparison with the committed (real) tables must fail, so the
    status is 'killed'; a --dev run still writes its estimand table."""
    import src.data_merger as DM
    monkeypatch.setattr(DM, "load_panel", lambda name: synthetic_panel())
    monkeypatch.setattr(P, "_sha256", lambda path: "synthetic")
    P.main(["--dev", "--n-boot", "3", "--workers", "1", "--out", str(tmp_path)])
    import json
    meta = json.loads((tmp_path / "expi_paired.meta.json").read_text())
    assert meta["status"].startswith("killed") and meta["dev"] and meta["n_boot"] == 3
    est = pd.read_csv(tmp_path / "expi_paired_estimands.csv")
    assert set(est.estimand) == set(P.ESTIMAND_ROLE)
    dr = pd.read_csv(tmp_path / "expi_paired_draws.csv.gz")
    assert list(dr.replicate) == [0, 1, 2]


def test_g07_literal_noise_summary_on_synthetic_table():
    import k2_literal_noise_summary as L
    rows = []
    for scope in ("metro", "cbsa_all"):
        for scheme in ("lineage", "state"):
            for conv in ("limit", "nanpath"):
                for draw in range(3):
                    rows.append(dict(scope=scope, scheme=scheme, draw=draw, outcome="pi",
                                     period="main", panel="B", link="overlap", variant="V0",
                                     conv=conv, quantity="xi_trap", fast=0.1, literal=0.1,
                                     abs_diff=1e-4 * (draw + 1) * (2 if scope == "cbsa_all" else 1),
                                     tol=1e-4, over_tol=draw > 0))
                rows.append(dict(scope=scope, scheme=scheme, draw=0, outcome="pi", period="main",
                                 panel="B", link="overlap", variant="V0", conv="pairs",
                                 quantity="pairs", fast=np.nan, literal=np.nan, abs_diff=1e-13,
                                 tol=1e-10, over_tol=False))
    s = L.summarise(pd.DataFrame(rows))
    ov = s[(s.level == "overall") & (s.quantity == "xi_trap")].iloc[0]
    assert ov.max_abs_diff == pytest.approx(6e-4) and ov.argmax_row.startswith("cbsa_all")
    pr = s[(s.level == "primary") & (s.quantity == "xi_trap")].iloc[0]
    assert pr.max_abs_diff == pytest.approx(3e-4) and pr.n_rows == 3 and pr.n_over_tol == 2
    assert "pairs" not in set(s.quantity)


def test_g04_term_share_on_synthetic_tables():
    import k1_rho1_term_share as S
    term = pd.DataFrame([dict(process=pr, variant=v, period=pe, n=20000, P=21,
                              term_expected=-0.001 if pe == "during" else -0.0008)
                         for pr in ("ou", "cubic", "cubic_bistable_0") for v in ("registered", "w0")
                         for pe in ("pre", "during")])
    eff = pd.DataFrame([dict(process=pr, pipeline=pl, quantity=q, effect=0.01 if pl == "saai" else 0.008)
                        for pr in ("ou", "cubic", "cubic_bistable_0") for pl in ("saai", "direct")
                        for q in ("rho1", "xi_trap")])
    t = S.table(term, eff)
    assert len(t) == 2 * 2 * 2 and set(t.process) == {"ou", "cubic"}
    r = t[(t.process == "ou") & (t.period == "during") & (t.pipeline == "direct")].iloc[0]
    assert r.share_abs_term == pytest.approx(0.125) and r.share_estimator_half == pytest.approx(0.0625)
    assert t.label.str.contains("NOT a bound").all()


def test_g06_refits_reproduce_the_registered_draw_path():
    import k2br_optimizer_noise as N
    cfg = KBR.Config(windows={"toy": (2002, 2011, "toy")}, n_min=100)
    m, regions = KBR.synthetic_muni(seed=3)
    rows = pd.DataFrame(N.draw_rows("toy", m, regions, cfg, draws=(0, 1), outcomes=cfg.outcomes))
    assert set(rows.status) == {"ok"}
    assert len(rows) == 2 * len(cfg.outcomes) * len(N.PANELS) * len(N.FIT_VARIANTS)
    res = KBR.analyse(m, cfg)
    warm = {(o, pn, v): f["params"] for (o, pn), d in res["panels"].items()
            for v, f in d["fits"].items() if np.all(np.isfinite(f["params"]))}
    KBR._W.update(bb=KBR.make_block_base(m, regions), cfg=cfg, keep=res["keep_years"], warm=warm)
    for b in (0, 1):
        _, flat = KBR._draw_worker((0, KBR.SCHEMES.index(N.SCHEME), b))
        for r in rows[rows.draw == b].itertuples():
            reg = flat[(r.outcome, r.panel, r.variant, "limit", "xi_trap")]
            assert r.xi_warm == reg and r.xi_registered_path == reg
    assert (rows.nll_warm_minus_tight >= -1e-9).all() and (rows.nll_cold_minus_tight >= -1e-9).all()
    s = N.summary(rows)
    assert set(s.window) == {"all", "toy"} and np.isfinite(s.max_abs_warm_cold).all()


def test_report_only_dev_guards():
    import k2br_optimizer_noise as N
    with pytest.raises(SystemExit):
        N.main(["--dev", "--out", str(ROOT / "results" / "tables")])
