"""results/tables/expi_rerun_ci_variants.csv        (contrasts, both conventions)
    results/tables/expi_rerun_ci_levels.csv          (per-period levels)
    results/tables/expi_rerun_ci_variants.meta.json

The county data are licensed (CSMAR) and not shipped: this runs locally, with --workers processes.  Rule 1: refuses a dirty tree (src/,
scripts/) unless --dev, and a --dev run never writes into results/tables.

    python3 scripts/run_expi_rerun_ci.py [--n-boot 1000] [--workers 8]
    python3 scripts/run_expi_rerun_ci.py --n-boot 10 --dev --out /tmp/x
    python3 scripts/run_expi_rerun_ci.py --compare   (side-by-side table from the
        committed tables: expi_rerun_ci_side_by_side.csv)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import pandas as pd

from src.config import FP_PERIODS
from src.fokker_planck import build_transition_pairs
from src.km_mle import derived_observables, fit_mle
from src.saai import compute_saai

TAB = os.path.join(ROOT, "results", "tables")
SEED = 20261023
N_BOOT = 1000
PERIODS = ("pre", "during")
SPEC = dict(y_col="GrossOutputValue", x_cols=["PopNum", "CultivateArea"])
VARIANTS = ("V0_baseline", "V1_stayer_restandardised")
CONVENTIONS = ("limit", "nan")
OBS = ("xi_trap", "delta_V", "mean_diffusion", "gamma", "tau_mfpt", "B_ratio_w")
SCHEME = ("window-stratified province block (provinces drawn with replacement within "
          "each window), full pipeline re-run per draw (compute_saai multi: scaling "
          "regression + yearly standardisation; V1 per-transition restandardisation); "
          "amendment 7 (S2) scheme on real data")
LABEL = "China V0/V1 CIs, per-draw pipeline re-run (amendment 7 supplement)"


def _git(*a) -> str:
    return subprocess.check_output(["git", "-C", ROOT, *a], text=True).strip()


def province_of(codes) -> np.ndarray:
    """ProvinceCode = first two digits of the 6-digit county code
    (src.data_cleaner)."""
    return (np.asarray(codes, dtype=np.int64) // 10000).astype(np.int64)


def v0_pairs(window_panel: pd.DataFrame, period: str) -> pd.DataFrame:
    sa = compute_saai(window_panel, **SPEC)
    return build_transition_pairs(sa, period)


def v1_pairs(window_panel: pd.DataFrame, period: str) -> pd.DataFrame:
    """Per transition, refit SAAI on that transition's stayers only
    (run_attrition_diagnostic.pairs_v1_restandardised on this panel)."""
    sa = compute_saai(window_panel, **SPEC)
    a, b = FP_PERIODS[period]
    out = []
    for t in range(a, b):
        stay = (set(sa.loc[sa["Year"] == t, "CountyCode"])
                & set(sa.loc[sa["Year"] == t + 1, "CountyCode"]))
        if len(stay) < 10:
            continue
        sub = window_panel[window_panel["Year"].isin([t, t + 1])
                           & window_panel["CountyCode"].isin(stay)]
        out.append(build_transition_pairs(compute_saai(sub, **SPEC), period, window=(t, t + 1)))
    return pd.concat(out, ignore_index=True)


PAIRS = {"V0_baseline": v0_pairs, "V1_stayer_restandardised": v1_pairs}


def observables(pairs: pd.DataFrame, warm=None) -> tuple[dict, np.ndarray]:
    xi, dxi = pairs["xi_t"].to_numpy(), pairs["delta_xi"].to_numpy()
    fit = fit_mle(xi, dxi, x0=warm)
    out = {}
    for conv in CONVENTIONS:
        o = derived_observables(fit["params"], xi, xi_target=0.0, degenerate=conv)
        out[conv] = {k: float(o[k]) for k in OBS} | {"id_status": o["id_status"]}
    return out, fit["params"]


_STATE = {}


def _init(panel_path_or_df, windows, members, warm):
    _STATE.update(windows=windows, members=members, warm=warm)


def resampled_window(win_panel: pd.DataFrame, members: dict, provs: np.ndarray,
                     rng) -> pd.DataFrame:
    """The window panel of one draw: G provinces with replacement, every
    county of each with multiplicity, relabelled k·10⁶ + code so that
    duplicates are distinct counties."""
    G = len(provs)
    drawn = provs[rng.integers(0, G, G)]
    parts = []
    for k, pv in enumerate(drawn):
        sub = win_panel[win_panel["CountyCode"].isin(members[pv])].copy()
        sub["CountyCode"] = sub["CountyCode"].astype(np.int64) + (k + 1) * 1_000_000
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def one_draw(b: int) -> dict:
    """Both variants and periods of draw b (rng [SEED, b]: per window, then
    the same resampled panels for V0 and V1)."""
    rng = np.random.default_rng([SEED, int(b)])
    rec = {"replicate": int(b)}
    panels = {}
    for p in PERIODS:
        provs = np.array(sorted(_STATE["members"][p]))
        panels[p] = resampled_window(_STATE["windows"][p], _STATE["members"][p], provs, rng)
    for var in VARIANTS:
        for p in PERIODS:
            try:
                pr = PAIRS[var](panels[p], p)
                o, _ = observables(pr, warm=_STATE["warm"][(var, p)])
                for conv in CONVENTIONS:
                    for k in OBS:
                        rec[f"{var}|{conv}|{k}_{p}"] = o[conv][k]
                rec[f"{var}|n_pairs_{p}"] = len(pr)
            except Exception as exc:
                for conv in CONVENTIONS:
                    for k in OBS:
                        rec[f"{var}|{conv}|{k}_{p}"] = np.nan
                rec[f"{var}|fail_{p}"] = f"{type(exc).__name__}: {exc}"[:200]
    return rec


def summarise(draws: pd.DataFrame, var: str, conv: str) -> list[dict]:
    """bootstrap_mle_quantities' summary rules on the draws of one variant
    and convention (plus n_failed: draws with a NaN in either period)."""
    rows = []
    p1, p2 = PERIODS
    for q in ("delta_V", "gamma", "mean_diffusion", "tau_mfpt", "B_ratio_w", "xi_trap"):
        v1, v2 = draws[f"{var}|{conv}|{q}_{p1}"], draws[f"{var}|{conv}|{q}_{p2}"]
        nfail = int((v1.isna() | v2.isna()).sum())
        diff = (v2 - v1).dropna()
        rows.append({"quantity": f"{q} ({p2}-{p1})", "kind": "difference",
                     "median": diff.median(), "ci_lo": diff.quantile(0.025),
                     "ci_hi": diff.quantile(0.975), "P_gt0": float((diff > 0).mean()),
                     "n_draws": len(diff), "n_failed": nfail})
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (v2 / v1).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({"quantity": f"{q} ({p2}/{p1})", "kind": "ratio",
                     "median": ratio.median(), "ci_lo": ratio.quantile(0.025),
                     "ci_hi": ratio.quantile(0.975), "P_gt1": float((ratio > 1).mean()),
                     "n_draws": len(ratio), "n_failed": nfail})
    return rows


SIDE_QUANTITIES = ("xi_trap (during-pre)", "tau_mfpt (during/pre)", "gamma (during-pre)",
                   "mean_diffusion (during/pre)", "B_ratio_w (during/pre)")
OLD_TABLES = {"limit": "expi_attrition_variants_limit.csv", "nan": "expi_attrition_variants.csv"}


def side_by_side(tab: str = TAB) -> pd.DataFrame:
    """--compare: the published county-block intervals (seed 106, B = 1000,
    full-sample ξ reused) beside the per-draw re-run province-block ones of
    this script, per variant, convention and headline quantity.  Reads the
    committed tables only (no recomputation)."""
    new = pd.read_csv(os.path.join(tab, "expi_rerun_ci_variants.csv"), keep_default_na=False,
                      na_values=[""])
    rows = []
    for conv, fn in OLD_TABLES.items():
        old = pd.read_csv(os.path.join(tab, fn), keep_default_na=False, na_values=[""])
        for var in VARIANTS:
            for q in SIDE_QUANTITIES:
                o = old[(old["variant"] == var) & (old["quantity"] == q)]
                n = new[(new["variant"] == var) & (new["quantity"] == q)
                        & (new["degenerate"].astype(str) == conv)]
                if len(o) != 1 or len(n) != 1:
                    raise SystemExit(f"missing row {var} {q} {conv}")
                o, n = o.iloc[0], n.iloc[0]
                rows.append({"variant": var, "degenerate": conv, "quantity": q,
                             "point_estimate": n["point_estimate"],
                             "county_block_median": o["median"], "county_block_ci_lo": o["ci_lo"],
                             "county_block_ci_hi": o["ci_hi"], "county_block_n_draws": o["n_draws"],
                             "rerun_median": n["median"], "rerun_ci_lo": n["ci_lo"],
                             "rerun_ci_hi": n["ci_hi"], "rerun_n_draws": n["n_draws"],
                             "rerun_n_failed": n["n_failed"],
                             "width_ratio_rerun_over_county": (n["ci_hi"] - n["ci_lo"])
                             / (o["ci_hi"] - o["ci_lo"])})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    if argv is None and "--compare" in sys.argv[1:]:
        out = side_by_side()
        out.to_csv(os.path.join(TAB, "expi_rerun_ci_side_by_side.csv"), index=False)
        print(out.to_string(index=False))
        return
    ap = argparse.ArgumentParser(description=LABEL)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--dev", action="store_true", help="dirty tree / small B; never results/tables")
    ap.add_argument("--out", default=TAB)
    a = ap.parse_args(argv)
    dirty = _git("status", "--porcelain", "--", "src", "scripts")
    out_real = os.path.realpath(a.out)
    if a.dev and os.path.commonpath([out_real, os.path.realpath(TAB)]) == os.path.realpath(TAB):
        sys.exit("--dev output never goes to results/tables")
    if not a.dev and (dirty or a.n_boot != N_BOOT):
        sys.exit("refusing: the registered run needs committed code (src/, scripts/ clean) "
                 f"and n_boot = {N_BOOT}; use --dev for a throw-away run")
    from src.data_merger import load_panel
    panel = load_panel("panel_scaling")
    sa = compute_saai(panel, **SPEC)
    windows, members, point, warm = {}, {}, {}, {}
    for p in PERIODS:
        lo, hi = FP_PERIODS[p]
        wp = panel[(panel["Year"] >= lo) & (panel["Year"] <= hi)].copy()
        sw = sa[(sa["Year"] >= lo) & (sa["Year"] <= hi)]
        provs = np.unique(province_of(sw["CountyCode"]))
        wp = wp[np.isin(province_of(wp["CountyCode"]), provs)]
        windows[p] = wp
        members[p] = {int(pv): np.unique(wp.loc[province_of(wp["CountyCode"]) == pv,
                                               "CountyCode"]).astype(np.int64)
                      for pv in provs}
        for var in VARIANTS:
            pr = PAIRS[var](wp, p)
            o, th = observables(pr)
            point[(var, p)] = {"n_pairs": len(pr), **o}
            warm[(var, p)] = th
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(None, windows, members, warm)) as ex:
        recs = list(ex.map(one_draw, range(a.n_boot), chunksize=4))
    secs = time.time() - t0
    draws = pd.DataFrame(recs).sort_values("replicate")
    rows, levels = [], []
    for var in VARIANTS:
        for conv in CONVENTIONS:
            for r in summarise(draws, var, conv):
                q, kind = r["quantity"].split(" (")[0], r["kind"]
                pe = {p: point[(var, p)][conv][q] for p in PERIODS}
                est = (pe["during"] - pe["pre"]) if kind == "difference" else (
                    pe["during"] / pe["pre"] if pe["pre"] != 0 else np.nan)
                rows.append({"variant": var, "degenerate": conv, "point_estimate": est,
                             "n_pairs_pre": point[(var, "pre")]["n_pairs"],
                             "n_pairs_during": point[(var, "during")]["n_pairs"], **r})
            for p in PERIODS:
                for k in OBS:
                    col = draws[f"{var}|{conv}|{k}_{p}"].dropna()
                    levels.append({"variant": var, "degenerate": conv, "period": p,
                                   "quantity": k, "point_estimate": point[(var, p)][conv][k],
                                   "median": col.median(), "ci_lo": col.quantile(0.025),
                                   "ci_hi": col.quantile(0.975), "n_draws": len(col)})
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "expi_rerun_ci_variants.csv"), index=False)
    pd.DataFrame(levels).to_csv(os.path.join(a.out, "expi_rerun_ci_levels.csv"), index=False)
    fails = {c: int(draws[c].notna().sum()) for c in draws.columns if "|fail_" in c}
    meta = {"label": LABEL, "script": "scripts/run_expi_rerun_ci.py",
            "git": _git("rev-parse", "HEAD"), "dirty": bool(dirty), "dev": bool(a.dev),
            "seed": SEED, "seed_rule": "rng [SEED, b] per draw b", "n_boot": a.n_boot,
            "scheme": SCHEME, "saai_spec": "multi", "variants": list(VARIANTS),
            "conventions": list(CONVENTIONS),
            "provinces": {p: len(members[p]) for p in PERIODS},
            "counties": {p: int(sum(len(v) for v in members[p].values())) for p in PERIODS},
            "failed_draws": fails, "seconds": secs, "workers": a.workers,
            "companions": "expi_attrition_variants.csv ('nan'), expi_attrition_variants_limit.csv "
                          "('limit'): county block, seed 106, B = 1000 — unchanged",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(a.out, "expi_rerun_ci_variants.meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    show = pd.DataFrame(rows)
    show = show[show["quantity"].isin(["xi_trap (during-pre)", "tau_mfpt (during/pre)",
                                       "gamma (during-pre)", "mean_diffusion (during/pre)",
                                       "B_ratio_w (during/pre)"])]
    print(show[["variant", "degenerate", "quantity", "point_estimate", "median", "ci_lo",
                "ci_hi", "n_draws", "n_failed"]].to_string(index=False))
    print(f"{a.n_boot} draws in {secs:.0f} s ({a.workers} workers)")


if __name__ == "__main__":
    main()
