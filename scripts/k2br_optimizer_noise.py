"""draws    : both windows (ref2010, ref2002), primary scheme 'amc',
             b = 0..4 (DRAWS), each rebuilt exactly as run_k2br._draw_worker
             (rng default_rng([SEED, window index, scheme index, b]),
             draw_muni, analyse with the data's kept years and the data-fit
             warm starts, cache=True);
  cells    : outcomes pib and vab_agro x panels A and B x variants V0, V1;
  fits     : warm  = the registered fit (fit_mle, x0 = the data-fit params);
             cold  = fit_mle with x0 = None (moment start);
             tight = L-BFGS-B from the lower-NLL of warm / cold with
                     ftol = 1e-15, gtol = 1e-12, maxiter = maxfun = 1e5
                     (TIGHT_OPTIONS);
  reported : xi_trap ('limit', derived_observables as run_k2br.estimate) and
             NLL of each fit, |warm - cold|, |warm - tight|, |cold - tight|,
             and |warm - committed| against k2br_bootstrap_draws.csv.gz
             (the registered value of that draw; expected 0).

Reads the Brazilian data (public IBGE statistics) through run_k2br's own
loaders; run locally.  Refuses a dirty tree unless --dev (--dev never
writes into results/tables).

    python3 -P scripts/k2br_optimizer_noise.py [--workers 4]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
from scipy import optimize

import run_k2br as K
from src.km_mle import derived_observables, fit_mle, negloglik

TAB = ROOT / "results" / "tables"
DRAWS = (0, 1, 2, 3, 4)
SCHEME = "amc"
OUTCOMES = ("pib", "vab_agro")
PANELS = ("A", "B")
FIT_VARIANTS = ("V0", "V1")
TIGHT_OPTIONS = {"ftol": 1e-15, "gtol": 1e-12, "maxiter": 100000, "maxfun": 100000}
LABEL = "report only: cold / warm / tight refits of fixed Brazilian AMC draws"


def _git(*a) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *a], text=True).strip()


def xi_limit(params, xt) -> float:
    return float(derived_observables(params, xt, xi_target=0.0, degenerate="limit")["xi_trap"])


def refits(xt: np.ndarray, dx: np.ndarray, warm_params: np.ndarray, warm_x0) -> dict:
    """Warm (the registered fit's params, passed in), cold and tight fits of
    one pair set."""
    cold = fit_mle(xt, dx, x0=None)
    nll_w = float(negloglik(warm_params, xt, dx))
    start = warm_params if nll_w <= cold["nll"] else cold["params"]
    t = optimize.minimize(negloglik, start, args=(xt, dx), method="L-BFGS-B",
                          options=TIGHT_OPTIONS)
    out = {"xi_warm": xi_limit(warm_params, xt), "xi_cold": xi_limit(cold["params"], xt),
           "xi_tight": xi_limit(t.x, xt), "nll_warm": nll_w, "nll_cold": cold["nll"],
           "nll_tight": float(t.fun), "tight_success": bool(t.success),
           "tight_start": "warm" if start is warm_params else "cold",
           "warm_x0_given": warm_x0 is not None}
    out["abs_warm_cold"] = abs(out["xi_warm"] - out["xi_cold"])
    out["abs_warm_tight"] = abs(out["xi_warm"] - out["xi_tight"])
    out["abs_cold_tight"] = abs(out["xi_cold"] - out["xi_tight"])
    out["nll_warm_minus_tight"] = nll_w - out["nll_tight"]
    out["nll_cold_minus_tight"] = cold["nll"] - out["nll_tight"]
    return out


def draw_rows(window: str, m, regions: dict, cfg, draws=DRAWS, committed: pd.DataFrame | None = None,
              outcomes=OUTCOMES) -> list[dict]:
    """All report rows of one window (data pass for the warm starts, then
    each fixed draw rebuilt as run_k2br._draw_worker)."""
    res = K.analyse(m, cfg)
    warm = {(o, pn, v): f["params"] for (o, pn), d in res["panels"].items()
            for v, f in d["fits"].items() if np.all(np.isfinite(f["params"]))}
    bb = K.make_block_base(m, regions)
    w_idx = list(cfg.windows).index(window) if window in cfg.windows else 0
    s_idx = K.SCHEMES.index(SCHEME)
    rows = []
    for b in draws:
        rng = np.random.default_rng([K.SEED, w_idx, s_idx, b])
        md = K.draw_muni(bb, SCHEME, rng)
        rd = K.analyse(md, cfg, keep_years=res["keep_years"], warm=warm, collect=True, cache=True)
        for o in outcomes:
            for pn in PANELS:
                d = rd["panels"][(o, pn)]
                for v in FIT_VARIANTS:
                    xt, dx = d["pairs"][v]
                    f = d["fits"][v]
                    base = {"window": window, "scheme": SCHEME, "draw": b, "outcome": o,
                            "panel": pn, "variant": v, "n_pairs": int(len(xt)),
                            "xi_registered_path": float(f["limit"]["xi_trap"])}
                    if len(xt) < 20 or not np.all(np.isfinite(f["params"])):
                        rows.append(base | {"status": "no_fit"})
                        continue
                    r = refits(xt, dx, f["params"], warm.get((o, pn, v)))
                    base["status"] = "ok"
                    if committed is not None:
                        c = committed[(committed["window"] == window)
                                      & (committed["scheme"] == SCHEME) & (committed["draw"] == b)]
                        col = f"{o}|{pn}|{v}|limit|xi_trap"
                        base["xi_committed"] = float(c[col].iloc[0]) if len(c) and col in c else np.nan
                        base["abs_warm_committed"] = abs(r["xi_warm"] - base["xi_committed"])
                    rows.append(base | r)
    return rows


def summary(rows: pd.DataFrame) -> pd.DataFrame:
    ok = rows[rows["status"] == "ok"]
    out = []
    for key, g in [("all", ok)] + [(w, g) for w, g in ok.groupby("window")]:
        rec = {"window": key, "n_cells": len(g), "label": LABEL}
        for c in ("abs_warm_cold", "abs_warm_tight", "abs_cold_tight", "nll_warm_minus_tight",
                  "nll_cold_minus_tight", "abs_warm_committed"):
            if c in g:
                rec[f"max_{c}"] = float(g[c].max())
                rec[f"median_{c}"] = float(g[c].median())
        out.append(rec)
    return pd.DataFrame(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=LABEL)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--out", default=str(TAB))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.dev and out.resolve() == TAB.resolve():
        sys.exit("--dev output never goes to results/tables")
    dirty = _git("status", "--porcelain")
    if dirty and not a.dev:
        sys.exit("refusing: dirty tree (use --dev for a throw-away run)")
    pib = K._pib_module()
    cfg = K.Config()
    outcomes = K.check_outcome_frame(pib.outcome_values(K.DATA))
    struct = K.load_structure()
    committed = pd.read_csv(TAB / "k2br_bootstrap_draws.csv.gz")
    t0 = time.time()
    rows = []
    for w in K.WINDOWS:
        m, regions = K.build_muni(struct, w, outcomes, cfg)
        rows += draw_rows(w, m, regions, cfg, committed=committed)
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "k2br_optimizer_noise.csv", index=False)
    s = summary(df)
    s.to_csv(out / "k2br_optimizer_noise_summary.csv", index=False)
    import scipy
    meta = {"label": LABEL, "script": "scripts/k2br_optimizer_noise.py",
            "git": _git("rev-parse", "HEAD"), "dirty": bool(dirty), "dev": bool(a.dev),
            "draws": list(DRAWS), "scheme": SCHEME, "seed": K.SEED,
            "seed_rule": "default_rng([SEED, window index, scheme index, b]) (run_k2br._draw_worker)",
            "tight_options": TIGHT_OPTIONS, "cells": {"outcomes": OUTCOMES, "panels": PANELS,
                                                      "variants": FIT_VARIANTS},
            "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
            "seconds": time.time() - t0, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "k2br_optimizer_noise.meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
