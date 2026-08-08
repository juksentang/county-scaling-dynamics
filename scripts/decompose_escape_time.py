"""
Decompose the change in mean first-passage time between policy periods.

The escape time rises by a factor of 4.2 (multi-factor index) to 11.0
(single-factor), and the manuscript attributed that to a deepening barrier and
compressed noise. Multiplying the barrier factor by the noise factor accounts
for only 1.8x of the multi-factor change and 2.1x of the single-factor change,
so most of the effect comes from somewhere else. This measures where.

Ratios cannot be multiplied together to answer the question. In the Gardiner
double integral the escape distance and the restoring drift enter the exponent
entangled with each other, so a ratio of attractor positions is not a
"contribution factor" in any well-defined sense -- it is not even
dimensionally the same kind of object. The decomposition instead substitutes
one estimated component at a time, from the pre-policy values towards the
during-policy values, and recomputes the whole first-passage integral at each
step.

Which components can be substituted is set by what is estimated rather than
derived. The barrier is not a free knob: V_FP = -int A/B and xi* is a zero of
A, so both follow from the drift and diffusion. The three independent knobs
are therefore

    B      the diffusion profile
    shift  the attractor location, applied as a rigid translation of A so that
           its stable zero sits at the during-policy attractor
    shape  whatever remains of the drift change after that translation

and the barrier is reported as a derived diagnostic of each hybrid rather than
as an input. With three knobs there are six substitution orders; a component's
marginal effect depends on the order, and that dependence is the interaction.
All six are run: the two extremes bound it and the Shapley value -- the average
marginal contribution over every order -- summarises it.

Writes results/tables/escape_decomposition.csv.
"""
import itertools
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data_merger import load_panel
from src.saai import compute_saai, compute_saai_single_factor
from src.fokker_planck import (KMResult, build_transition_pairs,
                               estimate_kramers_moyal,
                               compute_escape_observables)

OUT = "results/tables/escape_decomposition.csv"
KNOBS = ("B", "shift", "shape")


def attractor(km) -> float:
    """The stable drift zero the escape calculation starts from."""
    return compute_escape_observables(km)["xi_trap"]


def common_grid(pre, during, n: int = 400):
    """A grid spanning both periods' reliable support.

    Each period is binned over its own range, and when the two attractors sit
    far apart -- as they do under the single-factor index -- interpolating one
    period onto the other's bin centres silently truncates it, so the
    fully-substituted hybrid fails to reproduce the period it substitutes to.
    Both periods are put on a shared grid instead.
    """
    lo = min(pre.bin_centers[pre.reliable].min(),
             during.bin_centers[during.reliable].min())
    hi = max(pre.bin_centers[pre.reliable].max(),
             during.bin_centers[during.reliable].max())
    return np.linspace(lo, hi, n)


def regrid(km, grid):
    """A KM result carrying the same A and B, evaluated on ``grid``."""
    return replace(
        km,
        bin_centers=grid,
        A=np.interp(grid, km.bin_centers, km.A),
        B=np.interp(grid, km.bin_centers, km.B),
        A_se=np.interp(grid, km.bin_centers, km.A_se),
        B_se=np.interp(grid, km.bin_centers, km.B_se),
        n_per_bin=np.interp(grid, km.bin_centers, km.n_per_bin),
        reliable=np.interp(grid, km.bin_centers,
                           km.reliable.astype(float)) > 0.5,
    )


def hybrid(pre, during, active: frozenset):
    """A KM result with the named components taken from the during period.

    ``shift`` translates the pre-policy drift so its zero sits at the
    during-policy attractor; ``shape`` replaces the translated drift with the
    during-policy drift outright. Applying both is simply the during drift.
    """
    grid = pre.bin_centers
    A, B = pre.A.copy(), pre.B.copy()

    if "B" in active:
        B = np.interp(grid, during.bin_centers, during.B)

    if "shape" in active and "shift" in active:
        A = np.interp(grid, during.bin_centers, during.A)
    elif "shift" in active:
        # Rigid translation: same restoring force, new attractor position.
        delta = attractor(during) - attractor(pre)
        A = np.interp(grid - delta, pre.bin_centers, pre.A)
    elif "shape" in active:
        # During-policy drift translated back onto the pre-policy attractor,
        # so the shape changes while the location does not.
        delta = attractor(during) - attractor(pre)
        A = np.interp(grid + delta, during.bin_centers, during.A)

    return replace(pre, A=A, B=B)


def tau_of(pre, during, active: frozenset) -> dict:
    esc = compute_escape_observables(hybrid(pre, during, active))
    return {"tau": esc["tau_mfpt"], "delta_V": esc["delta_V"],
            "xi_trap": esc["xi_trap"], "D_bar": esc["mean_diffusion"]}


def decompose(pre, during) -> pd.DataFrame:
    """Marginal contribution of each knob under every substitution order."""
    cache = {frozenset(s): tau_of(pre, during, frozenset(s))
             for r in range(len(KNOBS) + 1)
             for s in itertools.combinations(KNOBS, r)}
    tau0 = cache[frozenset()]["tau"]

    rows = []
    for order in itertools.permutations(KNOBS):
        active, running = set(), tau0
        for knob in order:
            active.add(knob)
            after = cache[frozenset(active)]["tau"]
            rows.append({"order": "+".join(order), "knob": knob,
                         "tau_before": running, "tau_after": after,
                         "factor": after / running if running else np.nan})
            running = after
    return pd.DataFrame(rows), cache


def summarise(spec: str, pre, during, table: pd.DataFrame, cache: dict) -> list:
    tau_pre = cache[frozenset()]["tau"]
    tau_dur = cache[frozenset(KNOBS)]["tau"]
    out = []
    for knob in KNOBS:
        sel = table[table.knob == knob]
        out.append({
            "spec": spec, "component": knob,
            "shapley_factor": float(np.exp(np.log(sel.factor).mean())),
            "min_factor": float(sel.factor.min()),
            "max_factor": float(sel.factor.max()),
            "order_spread": float(sel.factor.max() / sel.factor.min()),
        })
    # A free-diffusion benchmark: if escape were purely diffusive over the
    # same distance, how long would it take? Divergence from it measures how
    # much the restoring drift still opposes escape.
    for label, km in (("pre", pre), ("during", during)):
        esc = compute_escape_observables(km)
        L = abs(esc["xi_trap"])
        out.append({
            "spec": spec, "component": f"free_diffusion_benchmark_{label}",
            "shapley_factor": np.nan, "min_factor": np.nan, "max_factor": np.nan,
            "order_spread": np.nan,
            "tau_observed": esc["tau_mfpt"],
            "tau_free_diffusion": L ** 2 / (2 * esc["mean_diffusion"])
            if esc["mean_diffusion"] > 0 else np.nan,
            "delta_V": esc["delta_V"], "xi_trap": esc["xi_trap"],
            "D_bar": esc["mean_diffusion"],
        })
    out.append({"spec": spec, "component": "TOTAL",
                "shapley_factor": tau_dur / tau_pre,
                "tau_observed": tau_dur, "tau_pre": tau_pre})
    return out


def mle_km(period: str, grid: np.ndarray):
    """A KM-shaped object carrying the fitted parametric drift and diffusion.

    The parametric fit is the primary estimator: its cubic drift admits at most
    two stable fixed points and in fact yields exactly one per period, whereas
    the binned drift hovers near zero over a wide interval and crosses it
    several times, so which crossing the escape calculation starts from is not
    stable across resamples.
    """
    f = pd.read_csv("results/tables/mle_fits.csv")
    r = f[f.period == period].iloc[0]
    A = r.a0 + r.a1 * grid + r.a2 * grid ** 2 + r.a3 * grid ** 3
    B = np.exp(np.clip(r.c0 + r.c1 * grid + r.c2 * grid ** 2, -20, 20))
    zeros = np.ones_like(grid)
    return KMResult(period=period, bin_centers=grid, A=A, B=B,
                    A_se=zeros * np.nan, B_se=zeros * np.nan,
                    n_per_bin=zeros, n_pairs_total=int(r.n_pairs),
                    reliable=np.ones_like(grid, dtype=bool))


def main():
    panel = load_panel("panel_scaling")
    specs = {
        "multi": compute_saai(panel, y_col="GrossOutputValue",
                              x_cols=["PopNum", "CultivateArea"]),
        "single": compute_saai_single_factor(panel),
    }

    summary, detail = [], []

    # Primary estimator first.
    grid = np.linspace(-2.5, 2.5, 400)
    kms = {p: mle_km(p, grid) for p in ("pre", "during")}
    table, cache = decompose(kms["pre"], kms["during"])
    table.insert(0, "spec", "mle")
    detail.append(table)
    summary += summarise("mle", kms["pre"], kms["during"], table, cache)

    for spec, saai in specs.items():
        kms = {}
        for period in ("pre", "during"):
            pairs = build_transition_pairs(saai, period)
            kms[period] = estimate_kramers_moyal(pairs, period=period)
        grid = common_grid(kms["pre"], kms["during"])
        kms = {k: regrid(v, grid) for k, v in kms.items()}
        table, cache = decompose(kms["pre"], kms["during"])
        table.insert(0, "spec", "binned_" + spec)
        detail.append(table)
        summary += summarise("binned_" + spec, kms["pre"], kms["during"], table, cache)

    summary = pd.DataFrame(summary)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    summary.to_csv(OUT, index=False)
    pd.concat(detail).to_csv(OUT.replace(".csv", "_orders.csv"), index=False)

    print(f"wrote {OUT}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
