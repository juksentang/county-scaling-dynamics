"""Figures of the K1 methods paper, from committed tables only.

Reads ONLY committed CSV/JSON under results/tables/ (including
results/tables/paper_k1_values.csv, the provenance table written by
scripts/paper_k1_numbers.py) and writes, under paper_k1/latex/figures/,

Output is vector PDF only (no PNG).
Environment: PAPER_K1_FIG_OUTDIR redirects the output (used by the test);
PAPER_K1_FIG_DEBUGDIR saves a PDF of a figure that fails the lint, for inspection.

Rules:
  * no analysis is run; every plotted value is read from one committed cell
    or is a deterministic re-expression (difference, ratio, product) done
    here and named in the code; the only simulation is the labelled
    illustration of Fig. 1 (fixed seed);
  * every figure passes src/paper_k1_style.lint (font >= 5 pt at final size,
    Arial, width 89/183 mm, height <= 170 mm, no clipped or overlapping text,
    greyscale-separable series, contrast); a violation raises and the script
    exits non-zero.

Usage: python3 scripts/paper_k1_figures.py [fig1 fig2 ...]
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from src import paper_k1_style as S

S.apply()
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter, NullLocator

TABLES = os.path.join(ROOT, "results", "tables")
OUTDIR = os.environ.get("PAPER_K1_FIG_OUTDIR") or os.path.join(ROOT, 'results', 'figures')
VALUES_CSV = os.path.join(TABLES, "paper_k1_values.csv")


class SourceError(RuntimeError):
    pass


def tab(rel: str) -> pd.DataFrame:
    p = os.path.join(TABLES, rel)
    if not os.path.isfile(p):
        raise SourceError(f"missing committed table {rel}")
    return pd.read_csv(p)


def one(df: pd.DataFrame, **kw) -> pd.Series:
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        if k not in df.columns:
            raise SourceError(f"column {k} missing")
        m &= (df[k] == v).to_numpy()
    if m.sum() != 1:
        raise SourceError(f"filter {kw} matched {int(m.sum())} rows (need 1)")
    return df[m].iloc[0]


_VALS = None


def _vals():
    global _VALS
    if _VALS is None:
        _VALS = pd.read_csv(VALUES_CSV).set_index("macro")
    return _VALS


def v(macro: str) -> float:
    """Value of a paper macro (paper_k1_values.csv)."""
    if macro not in _vals().index:
        raise SourceError(f"macro {macro} not in paper_k1_values.csv")
    return float(_vals().loc[macro, "value"])


def ftxt(macro: str) -> str:
    """The macro exactly as the tex prints it, as figure text."""
    s = str(_vals().loc[macro, "formatted"])
    s = s.replace("\\ensuremath{", "").rstrip("}")
    s = s.replace("{,}", ",").replace("\\,", "")
    return S.minus(s)


def fnum(x: float, nd: int = 2, signed: bool = False) -> str:
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return S.minus(s)


def sym_ci_err(point, lo, hi):
    return np.array([[point - lo], [hi - point]])


CC = S.COUNTRY_COLOR
CM = S.COUNTRY_MARKER


ILLUSTRATION_SEED = 20261025


def _illustration():
    """Small fixed-seed open panel: OU latent, province effects, one block exit.

    Not a result.  Returns the presence mask, roles, z (yearly standardised),
    and the V0/V1 transition pairs of both windows.
    """
    rng = np.random.default_rng(ILLUSTRATION_SEED)
    P, n_per, T = 10, 36, 18
    N = P * n_per
    a1, sig = -0.12, 0.40
    u = np.sort(rng.normal(0.2, 0.2, P))
    u[:2] = -1.0
    prov = np.repeat(np.arange(P), n_per)
    x = np.empty((N, T))
    x[:, 0] = rng.normal(0, sig / np.sqrt(1 - (1 + a1) ** 2), N)
    for t in range(T - 1):
        x[:, t + 1] = x[:, t] + a1 * x[:, t] + sig * rng.normal(size=N)
    y = u[prov][:, None] + x
    present = np.ones((N, T), bool)
    t_break = 14
    leave_block = np.isin(prov, [0, 1])
    present[leave_block, t_break + 1:] = False
    enter_block = prov == 6
    t_enter = 4
    present[enter_block, :t_enter] = False
    churn = rng.random((N, T)) < 0.015
    present &= ~churn
    z = np.full((N, T), np.nan)
    for t in range(T):
        m = present[:, t]
        z[m, t] = (y[m, t] - y[m, t].mean()) / y[m, t].std(ddof=1)
    role = np.zeros((N, T), int)
    role[present] = 1
    role[leave_block[:, None] & present] = 2
    role[enter_block[:, None] & present] = 3

    def pairs(t0, t1, v1):
        X, D = [], []
        for t in range(t0, t1):
            s = present[:, t] & present[:, t + 1]
            if v1:
                a = (y[s, t] - y[s, t].mean()) / y[s, t].std(ddof=1)
                b = (y[s, t + 1] - y[s, t + 1].mean()) / y[s, t + 1].std(ddof=1)
            else:
                a, b = z[s, t], z[s, t + 1]
            X.append(a)
            D.append(b - a)
        return np.concatenate(X), np.concatenate(D)

    win = {"pre": (0, 11), "during": (11, T - 1)}
    out = {}
    for w, (t0, t1) in win.items():
        for var, flag in (("V0", False), ("V1", True)):
            out[(w, var)] = pairs(t0, t1, flag)
    return dict(present=present, role=role, z=z, prov=prov, P=P, n_per=n_per,
                T=T, t_break=t_break, t_enter=t_enter, pairs=out, a1=a1,
                leave=leave_block, enter=enter_block)


def _ols(xv, dv):
    A = np.vstack([np.ones_like(xv), xv]).T
    (b0, b1), *_ = np.linalg.lstsq(A, dv, rcond=None)
    return b0, b1


def fig1():
    I = _illustration()
    fig = S.figure(183, 54)
    h = 36.0
    axa = S.ax_mm(fig, 10, 9.5, 36, h)
    axb = S.ax_mm(fig, 57.5, 9.5, 36, h)
    axc = S.ax_mm(fig, 104.5, 9.5, 36, h)
    axd = S.ax_mm(fig, 152, 9.5, 29, h)

    role = I["role"]
    order = np.lexsort((np.arange(len(role)), I["prov"]))
    cmap = ListedColormap(["#FFFFFF", S.STAYER_COLOR, S.LEAVER_COLOR, S.ENTRANT_COLOR])
    axa.pcolormesh(np.arange(I["T"] + 1) - 0.5, np.arange(len(role) + 1) - 0.5, role[order], cmap=cmap,
                   vmin=-0.5, vmax=3.5, shading="flat", antialiased=False, rasterized=False)
    axa.set_ylim(len(role) - 0.5, -0.5)
    for p in range(1, I["P"]):
        axa.axhline(p * I["n_per"] - 0.5, color="white", lw=0.6)
    tb = I["t_break"]
    axa.axvline(tb + 0.5, color="black", lw=0.6, ls=(0, (2, 1.2)))
    axa.set_xlim(-0.5, I["T"] - 0.5)
    axa.set_xticks([0, 5, 10, 15, 17])
    axa.set_xticklabels(["1", "6", "11", "16", "18"])
    axa.set_yticks([I["n_per"] * (p + 0.5) - 0.5 for p in range(I["P"])])
    axa.set_yticklabels([f"P{p + 1}" for p in range(I["P"])])
    axa.tick_params(axis="y", length=0, pad=1.0, labelsize=5)
    axa.set_xlabel("Year")
    axa.set_ylabel("Units, grouped by province", labelpad=1)
    for (x0, x1, lab) in ((-0.4, 10.9, "first window"), (11.1, 17.4, "second window")):
        axa.annotate("", xy=(x0, -14), xytext=(x1, -14), xycoords="data",
                     arrowprops=dict(arrowstyle="-", lw=0.5, color="black",
                                     shrinkA=0, shrinkB=0), annotation_clip=False)
        axa.text((x0 + x1) / 2, -17, lab, ha="center", va="bottom", fontsize=5.5)
    axa.text(tb + 0.9, 6, "block\nexit", fontsize=5, color="black", va="top", ha="left")
    hs = [mpatches.Patch(fc=S.STAYER_COLOR, ec="none", label="stayer"),
          mpatches.Patch(fc=S.LEAVER_COLOR, ec="none", label="leaving block"),
          mpatches.Patch(fc=S.ENTRANT_COLOR, ec="none", label="entering block"),
          mpatches.Patch(fc="white", ec="#9A9A9A", lw=0.4, label="absent")]
    axa.legend(handles=hs, loc="upper left", bbox_to_anchor=(-0.02, -0.17), ncol=4,
               fontsize=5, handlelength=0.9, columnspacing=0.6, handletextpad=0.3)
    axa.spines["left"].set_visible(False)

    z, pres = I["z"], I["present"]
    st = pres[:, tb] & pres[:, tb + 1]
    lv = pres[:, tb] & ~pres[:, tb + 1]
    bins = np.linspace(-3.0, 3.0, 31)
    w = bins[1] - bins[0]
    hS, _ = np.histogram(z[st, tb], bins)
    hL, _ = np.histogram(z[lv, tb], bins)
    hS1, _ = np.histogram(z[st, tb + 1], bins)
    c = bins[:-1] + w / 2
    sc = 0.95 / max((hS + hL).max(), hS1.max())
    up = 1.25
    axb.bar(c, hS * sc, width=w * 0.9, bottom=up, color=S.STAYER_COLOR, lw=0)
    axb.bar(c, hL * sc, width=w * 0.9, bottom=up + hS * sc, color=S.LEAVER_COLOR, lw=0)
    axb.bar(c, hS1 * sc, width=w * 0.9, bottom=0.0, color=S.STAYER_COLOR, lw=0)
    mS0, mS1 = z[st, tb].mean(), z[st, tb + 1].mean()
    mL = z[lv, tb].mean()
    axb.plot([mS0, mS0], [up - 0.04, up + 1.0], color="black", lw=0.6)
    axb.plot([mS1, mS1], [-0.04, 1.0], color="black", lw=0.6)
    axb.plot([mS0, mS0], [1.0, up - 0.04], color="black", lw=0.4, ls=":")
    axb.plot([0, 0], [-0.04, up + 1.0], color=S.REF_COLOR, lw=0.4, zorder=0)
    axb.annotate("", xy=(mS1, 1.10), xytext=(mS0, 1.10),
                 arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black",
                                 mutation_scale=5, shrinkA=0, shrinkB=0))
    axb.text(mS0 + 0.12, 1.10, "$M_t$", fontsize=6, ha="left", va="center")
    axb.plot([mL], [up - 0.02], marker="^", ms=3, color=S.LEAVER_COLOR, mew=0)
    axb.text(mL, up - 0.12, r"$m_L$", fontsize=5.5, ha="center", va="top")
    axb.text(-3.0, up + 0.93, "year $t$", fontsize=5.5, ha="left", va="top")
    axb.text(-3.0, 0.93, "year $t$+1", fontsize=5.5, ha="left", va="top")
    axb.text(3.0, up + 0.93, "stayers\n+ leaving block", fontsize=5, ha="right", va="top")
    axb.text(3.0, 0.93, "stayers", fontsize=5, ha="right", va="top")
    axb.text(-3.0, up + 1.30, r"$M_t=(n_Lm_L-n_Em_E)/n_S$", fontsize=5.5,
             ha="left", va="top")
    axb.set_ylim(-0.04, up + 1.32)
    axb.set_xlim(-3.0, 3.0)
    S.despine(axb, left=False)
    axb.set_xlabel(r"Standardized index $\xi$")

    edges = np.linspace(-2.4, 2.4, 13)
    for var, sty, off in (("V0", S.V0_STYLE, -0.04), ("V1", S.V1_STYLE, 0.04)):
        xv, dv = I["pairs"][("during", var)]
        idx = np.digitize(xv, edges)
        xm, dm, de = [], [], []
        for k in range(1, len(edges)):
            s = idx == k
            if s.sum() >= 15:
                xm.append(xv[s].mean() + off)
                dm.append(dv[s].mean())
                de.append(1.96 * dv[s].std(ddof=1) / np.sqrt(s.sum()))
        eb = axc.errorbar(xm, dm, yerr=de, fmt=sty["marker"], ms=2.6, mfc=sty["mfc"], mec=sty["mec"],
                          ecolor=sty["color"], elinewidth=0.5, mew=0.6, zorder=3)
        S.tag(eb, var)
        b0, b1 = _ols(xv, dv)
        xx = np.linspace(-2.4, 2.4, 50)
        ln, = axc.plot(xx, b0 + b1 * xx, color=sty["color"], lw=0.8, ls=sty["ls"], zorder=2)
        S.tag(ln, var)
        xi = -b0 / b1
        axc.plot([xi], [0], marker="|", ms=6, mew=0.9, color=sty["color"], zorder=4)
        if var == "V0":
            xi0 = xi
        else:
            xi1 = xi
    xv, dv = I["pairs"][("pre", "V1")]
    b0, b1 = _ols(xv, dv)
    axc.axhline(0, color=S.REF_COLOR, lw=0.4, zorder=0)
    axc.annotate("", xy=(xi0, 0.30), xytext=(xi1, 0.30),
                 arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black", mutation_scale=5,
                                 shrinkA=0, shrinkB=0))
    axc.plot([xi0, xi0], [0.02, 0.27], color=S.V0_COLOR, lw=0.4, ls=":")
    axc.plot([xi1, xi1], [0.02, 0.27], color=S.V1_COLOR, lw=0.4, ls=":")
    axc.text((xi0 + xi1) / 2, 0.34, r"spurious shift of $\xi^*$", ha="center", va="bottom", fontsize=5.5)
    axc.text(-2.3, -0.30, "V0: standardized on all units", color=S.V0_COLOR, fontsize=5.5, va="center")
    axc.text(-2.3, -0.38, "V1: restandardized on stayers", color=S.V1_COLOR, fontsize=5.5, va="center")
    axc.set_xlim(-2.5, 2.5)
    axc.set_ylim(-0.45, 0.5)
    axc.set_xlabel(r"$\xi_t$ (second window, simulated), binned")
    axc.set_ylabel(r"Mean increment E[$\Delta\xi\,|\,\xi_t$]")

    rho = np.linspace(0.70, 0.995, 300)
    axd.plot(rho, 1.0 / (1.0 - rho), color="black", lw=0.8)
    pts = [("CN", v("CnAoneVoneDuring"), "China " + ftxt("CnAmpVoneDuring")),
           ("BR", v("BrTenPibAone"), "Brazil " + ftxt("BrTenPibAmp")),
           ("US", v("UsPiMainAone"), "US " + ftxt("UsPiMainAmp"))]
    for k, a1, lab in pts:
        r = 1.0 + a1
        amp = 1.0 / abs(a1)
        h = axd.plot([r], [amp], marker=CM[k], ms=3.6, color=CC[k], mec="white", mew=0.4, ls="none", zorder=4)
        S.tag(h, k)
        axd.plot([r, r], [1, amp], color=CC[k], lw=0.4, ls=":", zorder=1)
        dx, ha = (-0.006, "right")
        axd.text(r + dx, amp * 1.05, lab, color=CC[k], fontsize=5.5, ha=ha, va="bottom")
    axd.set_yscale("log")
    axd.set_ylim(2, 300)
    axd.set_xlim(0.70, 1.0)
    axd.set_xticks([0.7, 0.8, 0.9, 1.0])
    axd.set_xlabel(r"Persistence $\rho_1=1+a_1$")
    axd.set_ylabel(r"Amplification $1/|a_1|$")
    for ax, l in ((axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")):
        S.panel_label(fig, ax, l, dx_mm=-9.5 if ax is not axb else -3.0, dy_mm=5.0 if ax is axa else 1.5)
    S.save(fig, OUTDIR, "fig1_mechanism")


K1C_DIR = "k1_k1c_f2"
K1C_CELLS = [(23, "no exit", "no exit"), (24, "province block", "province-\nblock exit"),
             (25, "mask replay", "observed-\nmask replay")]
QLAB = {"xi_trap": r"$\xi^*$", "two_delta_V": r"$2\Delta V$", "tau_mfpt": r"$\tau$",
        "mean_diffusion": r"$\bar D$", "B_ratio_w": r"$B_w$", "rho1": r"$\rho_1$"}
PROC_LAB = {"ou": "Ornstein–Uhlenbeck", "cubic": "monostable cubic", "cubic_mult": "cubic, mult. noise",
            "cubic_bistable_0": r"bistable, 2$\Delta V$ = 0", "cubic_bistable_1": r"bistable, 2$\Delta V$ = 1",
            "cubic_bistable_2": r"bistable, 2$\Delta V$ = 2", "cubic_bistable_4": r"bistable, 2$\Delta V$ = 4",
            "cubic_shift": "positive control"}
PROC_GATE = {"ou": "formula, null cov.", "cubic": "formula, null cov.", "cubic_mult": "null coverage",
             "cubic_shift": "recovery test", "cubic_bistable_0": "barrier (report)",
             "cubic_bistable_1": "barrier (report)", "cubic_bistable_2": "barrier (report)",
             "cubic_bistable_4": "barrier (report)"}


def _k1c():
    b = tab(f"{K1C_DIR}/k1_sim_bias.csv")
    c = tab(f"{K1C_DIR}/k1_sim_coverage.csv")
    p = tab(f"{K1C_DIR}/k1_power.csv")
    b = b[b.quantity == "xi_trap"]
    c = c[(c.quantity == "xi_trap") & (c.period == "contrast") & (c.ci_nan_rule == "tol0")]
    p = p[(p.quantity == "xi_trap") & (p.period == "contrast") & (p.ci_nan_rule == "tol0")]
    return b, c, p


def _cell_axis(ax, labels=True):
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks([0, 1, 2])
    if labels:
        ax.set_xticklabels([t for _, _, t in K1C_CELLS], fontsize=5)
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0, pad=2)


def _vv(ax, x, y, var, pipe, yerr=None, ms=3.2):
    sty = S.V0_STYLE if var == "V0" else S.V1_STYLE
    gated = pipe == "saai"
    mk = "o" if gated else "D"
    size = ms if gated else ms * 0.62
    if yerr is not None:
        h = ax.errorbar([x], [y], yerr=yerr, fmt=mk, ms=size, mfc=sty["mfc"], mec=sty["mec"], mew=0.6,
                        ecolor=sty["color"], elinewidth=0.6 if gated else 0.4, zorder=3)
    else:
        h = ax.plot([x], [y], mk, ms=size, mfc=sty["mfc"], mec=sty["mec"], mew=0.6, zorder=3)
    S.tag(h, f"{var}-{pipe}")
    return h


OFF = {("V0", "saai"): -0.17, ("V1", "saai"): 0.09, ("V0", "direct"): -0.29, ("V1", "direct"): 0.21}


def fig2():
    b, c, p = _k1c()
    fig = S.figure(183, 104)
    yt, ht = 61.0, 33.0
    axa = S.ax_mm(fig, 11, yt, 45, ht)
    axb = S.ax_mm(fig, 68, yt, 32, ht)
    axc = S.ax_mm(fig, 111, yt, 32, ht)
    axd = S.ax_mm(fig, 153, yt, 28, ht)

    for i, (cell, mech, _) in enumerate(K1C_CELLS):
        for pipe in ("saai", "direct"):
            for var in ("V0", "V1"):
                r = one(b, cell_index=cell, pipeline=pipe, variant=var, period="contrast")
                _vv(axa, i + OFF[(var, pipe)], r.median_estimate, var, pipe, yerr=r.sd_bias)
        t = one(b, cell_index=cell, pipeline="saai", variant="V1", period="contrast").median_truth
        axa.plot([i - 0.38, i + 0.34], [t, t], color=S.TRUTH_COLOR, lw=0.8, zorder=2)
        axa.text(i + 0.36, t, fnum(t), fontsize=5, va="center", ha="left")
    axa.axhline(0, color=S.REF_COLOR, lw=0.4, zorder=0)
    axa.set_ylim(-2.05, 0.35)
    axa.set_ylabel(r"$\xi^*$ shift, second $-$ first window")
    _cell_axis(axa)
    axa.text(0.0, 0.22, r"V0 $\equiv$ V1", fontsize=5, ha="center")
    hs = [Line2D([], [], marker="o", ls="none", ms=3.2, mfc="white", mec=S.V0_COLOR, mew=0.6, label="V0"),
          Line2D([], [], marker="o", ls="none", ms=3.2, mfc=S.V1_COLOR, mec=S.V1_COLOR, label="V1"),
          Line2D([], [], marker="D", ls="none", ms=2.0, mfc="#666666", mec="#666666", label="direct-$x$ (report only)"),
          Line2D([], [], color="black", lw=0.8, label="truth (median)")]
    axa.legend(handles=hs, loc="lower left", fontsize=5, ncol=2, columnspacing=0.8,
               bbox_to_anchor=(0.0, 0.0), handletextpad=0.3)

    for i, (cell, _, _) in enumerate(K1C_CELLS):
        for pipe in ("saai", "direct"):
            for var in ("V0", "V1"):
                r = one(b, cell_index=cell, pipeline=pipe, variant=var, period="contrast")
                _vv(axb, i + OFF[(var, pipe)], abs(r.rel_median_bias), var, pipe)
    thr = v("KonecBiasThreshold")
    axb.axhline(thr, color="black", lw=0.5, ls=(0, (3, 1.5)))
    axb.text(2.5, thr * 1.25, "gate " + ftxt("KonecBiasThreshold"), fontsize=5, ha="right", va="bottom")
    axb.set_yscale("log")
    axb.set_ylim(1e-3, 10)
    axb.set_ylabel("|Relative median bias|")
    _cell_axis(axb)

    R = v("KoneR")
    nom = 0.95
    half = 1.96 * np.sqrt(nom * (1 - nom) / R)
    axc.axhspan(nom - half, nom + half, color=S.WINDOW_SHADE, lw=0, zorder=0)
    axc.axhline(nom, color=S.REF_COLOR, lw=0.5)
    cthr = v("KonecCovThreshold")
    axc.axhline(cthr, color="black", lw=0.5, ls=(0, (3, 1.5)))
    axc.text(2.5, cthr - 0.006, "gate " + ftxt("KonecCovThreshold"), fontsize=5, ha="right", va="top")
    for i, (cell, _, _) in enumerate(K1C_CELLS):
        for pipe in ("saai", "direct"):
            for var in ("V0", "V1"):
                r = one(c, cell_index=cell, pipeline=pipe, variant=var)
                _vv(axc, i + OFF[(var, pipe)], r.coverage, var, pipe)
    axc.set_ylim(0.66, 1.005)
    axc.set_ylabel("Coverage of true shift, 95% CI")
    _cell_axis(axc)

    for i, (cell, _, _) in enumerate(K1C_CELLS):
        for pipe in ("saai", "direct"):
            for var in ("V0", "V1"):
                r = one(p, cell_index=cell, pipeline=pipe, variant=var)
                _vv(axd, i + OFF[(var, pipe)], r.power_excl_null_failed_as_neg, var, pipe)
    axd.set_ylim(0, 1)
    axd.set_ylabel("Power, CI excludes 0 (report only)")
    _cell_axis(axd)

    axe = S.ax_mm(fig, 11, 10, 52, 33)
    xs = []
    for i, (cell, _, _) in enumerate(K1C_CELLS):
        for j, per in enumerate(("pre", "during")):
            x0 = i * 2.6 + j * 1.0
            xs.append((x0, per))
            for var in ("V0", "V1"):
                r = one(b, cell_index=cell, pipeline="saai", variant=var, period=per)
                _vv(axe, x0 + (-0.16 if var == "V0" else 0.16), r.median_estimate, var, "saai",
                    yerr=r.sd_bias, ms=2.8)
            t = one(b, cell_index=cell, pipeline="saai", variant="V1", period=per).median_truth
            axe.plot([x0 - 0.36, x0 + 0.36], [t, t], color="black", lw=0.8, zorder=2)
        axe.text(i * 2.6 + 0.5, 0.82, K1C_CELLS[i][1], fontsize=5.5, ha="center")
    axe.axhline(0, color=S.REF_COLOR, lw=0.4, zorder=0)
    axe.set_xticks([x for x, _ in xs])
    axe.set_xticklabels(["first" if per == "pre" else "second" for _, per in xs], fontsize=5)
    axe.tick_params(axis="x", length=0)
    axe.set_xlim(-0.6, 2 * 2.6 + 1.6)
    axe.set_ylim(-1.85, 0.95)
    axe.set_ylabel(r"$\xi^*$ (window level)")

    g = tab(f"{K1C_DIR}/k1_gates.csv")
    g = g[(g.gate == "S0") & (g.rule_set == "limit")]
    procs = ["ou", "cubic", "cubic_mult", "cubic_bistable_0", "cubic_bistable_1",
             "cubic_bistable_2", "cubic_bistable_4", "cubic_shift"]
    quants = ["xi_trap", "two_delta_V", "tau_mfpt", "mean_diffusion", "B_ratio_w", "rho1"]
    rows = [(pr, pi) for pr in procs for pi in ("saai", "direct")]
    cols = [(pe, q) for pe in ("pre", "during") for q in quants]
    axf = S.ax_mm(fig, 100, 10, 50, 37)
    axf._k1_box_ok = True
    M = np.full((len(rows), len(cols)), np.nan)
    marks = []
    for ri, (pr, pi) in enumerate(rows):
        for ci, (pe, q) in enumerate(cols):
            r = one(g, process=pr, pipeline=pi, period=pe, quantity=q)
            ratio = abs(r.value) / r.threshold
            st = str(r.status)
            if st == "pass":
                M[ri, ci] = ratio
            else:
                marks.append((ri, ci, st, ratio))
    norm = LogNorm(vmin=1e-3, vmax=1.0)
    cm = plt.get_cmap("Greys").copy()
    im = axf.pcolormesh(np.arange(len(cols) + 1) - 0.5, np.arange(len(rows) + 1) - 0.5,
                        np.ma.masked_invalid(np.clip(M, 1e-3, 1.0)),
                        cmap=ListedColormap(cm(np.linspace(0.06, 0.62, 256))), norm=norm, shading="flat",
                        antialiased=False, rasterized=False)
    axf.set_xlim(-0.5, len(cols) - 0.5)
    axf.set_ylim(len(rows) - 0.5, -0.5)
    for ri, ci, st, ratio in marks:
        if st == "fail":
            axf.add_patch(mpatches.Rectangle((ci - 0.5, ri - 0.5), 1, 1, fc="black", ec="black", lw=0.5))
            axf.text(ci, ri + 0.05, "×", color="white", fontsize=6, ha="center", va="center",
                     fontweight="bold", gid="on-dark")
        elif st.startswith("not_evaluable"):
            axf.add_patch(mpatches.Rectangle((ci - 0.5, ri - 0.5), 1, 1, fc="white", ec="black", lw=0.4))
            axf.plot([ci], [ri], marker="o", ms=1.6, color="black")
        else:
            axf.add_patch(mpatches.Rectangle((ci - 0.5, ri - 0.5), 1, 1, fc="white", ec="black", lw=0.4))
            axf.plot([ci - 0.3, ci + 0.3], [ri, ri], color="black", lw=0.5)
    for k in range(1, len(procs)):
        axf.axhline(2 * k - 0.5, color="white", lw=0.8)
    axf.axvline(len(quants) - 0.5, color="white", lw=1.6)
    axf.set_xticks(range(len(cols)))
    axf.set_xticklabels([QLAB[q] for _, q in cols], fontsize=5)
    axf.set_yticks([2 * k + 0.5 for k in range(len(procs))])
    axf.set_yticklabels([PROC_LAB[pr] for pr in procs], fontsize=5)
    axf.tick_params(length=0, pad=1.5)
    for sp in axf.spines.values():
        sp.set_visible(False)
    axf.text(2.5, -1.0, "first window", fontsize=5.5, ha="center", va="bottom")
    axf.text(8.5, -1.0, "second window", fontsize=5.5, ha="center", va="bottom")
    for k, pr in enumerate(procs):
        axf.text(len(cols) - 0.2, 2 * k + 0.5, PROC_GATE[pr], fontsize=5, va="center", ha="left")
    axf.text(len(cols) - 0.2, -1.0, "needed by", fontsize=5, ha="left", va="bottom")
    axf.text(-0.75, len(rows) + 1.9, "rows per process: regression pipeline (top), direct-$x$ (bottom)",
             fontsize=5, ha="left", va="top")
    cax = S.ax_mm(fig, 166, 30, 2.2, 17)
    cb = fig.colorbar(im, cax=cax)
    cb.solids.set_rasterized(False)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5, width=0.4, length=1.5)
    cb.set_label("|median bias| / tolerance\n(pass rows)", fontsize=5, labelpad=2)
    kx, ky = 165.5, 22.5
    W, H = fig.get_size_inches() / S.MM
    key = [("fail", "fail"), ("not_evaluable", "not evaluable"), ("report", "report only")]
    for n, (st, lab) in enumerate(key):
        yy = ky - n * 4.2
        ax_ = S.ax_mm(fig, kx, yy, 2.6, 2.6)
        ax_.set_xlim(0, 1); ax_.set_ylim(0, 1); ax_.axis("off")
        fc = "black" if st == "fail" else "white"
        ax_.add_patch(mpatches.Rectangle((0, 0), 1, 1, fc=fc, ec="black", lw=0.5, transform=ax_.transAxes,
                                         clip_on=False))
        if st == "fail":
            ax_.text(0.5, 0.52, "×", color="white", fontsize=6, ha="center", va="center",
                     fontweight="bold", gid="on-dark")
        elif st == "not_evaluable":
            ax_.plot([0.5], [0.5], marker="o", ms=1.6, color="black")
        else:
            ax_.plot([0.2, 0.8], [0.5, 0.5], color="black", lw=0.5)
        fig.text((kx + 3.6) / W, (yy + 1.3) / H, lab, fontsize=5, va="center")
    for ax, l, dx in ((axa, "a", -10), (axb, "b", -10), (axc, "c", -10), (axd, "d", -10),
                      (axe, "e", -10), (axf, "f", -28)):
        S.panel_label(fig, ax, l, dx_mm=dx, dy_mm=2.5)
    S.save(fig, OUTDIR, "fig2_simulation")


def _ladder_points():
    """One row per (country, outcome, window): dose, 1/|a1|, observed V0-V1 xi*, CI, guarded projection, gate."""
    pts = []
    for per, Per, lab in (("pre", "Pre", "2000–12"), ("during", "During", "2013–19")):
        pts.append(dict(c="CN", out="agri", lab=lab, dose=v(f"CnDose{Per}"), amp=v(f"CnAmpVone{Per}"),
                        y=v(f"CnVzeroMinusVoneXi{Per}"), lo=v(f"CnVzeroMinusVoneXi{Per}Lo"),
                        hi=v(f"CnVzeroMinusVoneXi{Per}Hi"), pred=np.nan, gate=np.nan,
                        primary=(per == "during"), filled=True))
        if abs(pts[-1]["dose"] * pts[-1]["amp"] - v(f"CnDoseTimesAmp{Per}")) > 1e-9:
            raise SourceError(f"China {per}: dose x amp disagrees with CnDoseTimesAmp{Per}")
    pdiff = tab("k2/k2_paired_differences.csv")
    h3b = tab("k2/k2_h3b.csv")
    cov = tab("k2_coverage_panelB_summary.csv")
    us = [("pi", "main", "2003-2024", "UsPiMainAmp", "PI 2003–24"),
          ("pi", "secondary", "1993-2024", "UsPiSecAmp", "PI 1993–2024"),
          ("gdp", "main", "2003-2024", "UsGdpMainAmp", "GDP 2003–24"),
          ("gdp", "secondary", "2001-2024", "UsGdpSecAmp", "GDP 2001–24")]
    for out, per, win, ampm, lab in us:
        r = one(pdiff, scope="metro", outcome=out, period=per, link="overlap", contrast="V0-V1",
                conv="limit", quantity="xi_trap")
        d = one(cov, scope="metro", link_rule="overlap", period=win).nS_weighted_mech_bound_per_unit_m
        hb = one(h3b, scope="metro", outcome=out, period=per, link="overlap")
        pred = hb.pred_xi_trap if str(hb.status) == "ok" else np.nan
        gate = hb.gate if np.isfinite(hb.gate) else v("HthreebGateFactor") * hb.half_width_V1
        pts.append(dict(c="US", out=out, lab=lab, dose=d, amp=v(ampm), y=r.estimate,
                        lo=r.lineage_tol0_lo, hi=r.lineage_tol0_hi, pred=pred, gate=gate,
                        primary=(out == "pi" and per == "main"), filled=(out == "pi")))
    con = tab("k2br_contrasts.csv")
    prd = tab("k2br_predictions.csv")
    bv = tab("k2br_verdicts.csv")
    for win, W, dm, wl in (("ref2010", "Ten", "BrDoseTen", "2010 ref."), ("ref2002", "Two", "BrDoseTwo", "2002 ref.")):
        for out, O, ol in (("pib", "Pib", "GDP"), ("vab_agro", "Vab", "agri. VA")):
            r = one(con, window=win, outcome=out, contrast="V0B_minus_V1B", convention="limit",
                    quantity="xi_trap")
            pr = one(prd, window=win, outcome=out)
            g = one(bv, window=win, outcome=out).H3b_gate_threshold
            pts.append(dict(c="BR", out=out, lab=f"{ol}, {wl}", dose=v(dm), amp=v(f"Br{W}{O}Amp"),
                            y=r.estimate, lo=r.amc_ci_lo, hi=r.amc_ci_hi,
                            pred=pr.pred_dxi_star if pr.status == "ok" else np.nan, gate=g,
                            primary=(win == "ref2010" and out == "pib"), filled=(out == "pib")))
    return pts


SECOND_TINT = 0.55
UNTESTABLE_COLOR = "#8A8A8A"
GATE_COLOR = "#E4E4E4"
LABEL_COLOR = {"CN": CC["CN"], "US": CC["US"], "BR": "#3E7A30"}


def _marker_kw(P, ms):
    col = CC[P["c"]]
    return dict(marker=CM[P["c"]], ms=ms, mfc=col if P["filled"] else S.tint(col, SECOND_TINT), mec=col, mew=0.7)


PROJ_DODGE = 1.25


def _plot_ladder(ax, pts, xkey, labels):
    for P in pts:
        x = P["dose"] * P["amp"] if xkey == "dxa" else P["dose"]
        col = CC[P["c"]]
        ms = 4.2 if P["primary"] else 3.4
        if np.isfinite(P["gate"]):
            hg = ax.plot([x, x], [-P["gate"], P["gate"]], color=GATE_COLOR, lw=3.2, solid_capstyle="butt",
                         zorder=1)
            S.tag(hg, "gate")
        kw = _marker_kw(P, ms)
        if np.isfinite(P["lo"]):
            top = ax.get_ylim()[1]
            hi = min(P["hi"], top)
            h = ax.errorbar([x], [P["y"]], yerr=sym_ci_err(P["y"], P["lo"], hi), fmt=kw.pop("marker"),
                            ecolor=col, elinewidth=0.7, zorder=4, **kw)
            if P["hi"] > top:
                ha_ = ax.plot([x], [top], marker="^", ms=3.0, color=col, ls="none", clip_on=False, zorder=5)
                S.tag(ha_, "trunc")
                ax.annotate("to " + fnum(P["hi"], 2, signed=True), (x, top), xytext=(4, -3),
                            textcoords="offset points", fontsize=5, ha="left", va="top", color=LABEL_COLOR[P["c"]])
        else:
            h = ax.plot([x], [P["y"]], ls="none", zorder=4, **kw)
        S.tag(h, P["c"] + ("" if P["filled"] else "-2nd"))
        if np.isfinite(P["pred"]):
            testable = abs(P["pred"]) > P["gate"]
            xp = x * PROJ_DODGE
            hp = ax.plot([xp], [P["pred"]], marker="D", ms=3.4 if testable else 3.0, mfc="white",
                         mec=S.PROJ_COLOR if testable else UNTESTABLE_COLOR, mew=0.7 if testable else 0.5,
                         ls="none", zorder=6)
            S.tag(hp, "proj" if testable else "proj-untestable")
            if testable and labels:
                ax.annotate("projection", (xp, P["pred"]), xytext=(0, -5), textcoords="offset points",
                            fontsize=5, ha="center", va="top", color=S.PROJ_COLOR,
                            bbox=dict(fc="white", ec="none", pad=0.4))
        if labels and P.get("txt"):
            dx, dy, ha = P["txt"]
            ax.annotate(P["lab"], (x, P["y"]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=5, ha=ha, va="center", color=LABEL_COLOR[P["c"]])
    ax.axhline(0, color=S.REF_COLOR, lw=0.4, zorder=0)
    ax.set_xscale("log")


def _dose_series(ax, years, dose, col, mk, ylim=(1e-5, 3)):
    base = ylim[0]
    nz = dose > 0
    ax.vlines(years[nz], base, dose[nz], color=col, lw=0.6, zorder=2)
    h = ax.plot(years[nz], dose[nz], mk, ms=2.4, color=col, mec="white", mew=0.3, ls="none", zorder=3)
    S.tag(h, "dose")
    hz = ax.plot(years[~nz], np.full((~nz).sum(), base * 1.8), "|", ms=2.2, color="#7A7A7A", mew=0.5,
                 ls="none", zorder=3)
    S.tag(hz, "zero")
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.yaxis.set_minor_locator(NullLocator())


def _window_box(ax, a, b_, y0, y1, hatch=None, fill=None):
    """A window as a filled band (fill) or an outlined, hatched band (overlapping second window)."""
    if fill:
        ax.axvspan(a, b_, color=fill, lw=0, zorder=0)
    else:
        ax.add_patch(mpatches.Rectangle((a, y0), b_ - a, y1 - y0, fc="none", ec="#9A9A9A", lw=0.4,
                                        hatch=hatch, zorder=0))


def fig3():
    pts = _ladder_points()
    fig = S.figure(183, 122)
    YL = (-1.18, 0.42)
    axa = S.ax_mm(fig, 13, 64, 92, 50)
    axb = S.ax_mm(fig, 13, 11, 40, 38)
    axf = S.ax_mm(fig, 66, 11, 39, 38)
    lab_pos = {("CN", "2013–19"): (-6, -3, "right"), ("CN", "2000–12"): (6, 4, "left"),
               ("US", "PI 2003–24"): (-7, -2, "right"), ("US", "PI 1993–2024"): (5, -6, "left"),
               ("US", "GDP 2003–24"): (-6, 6, "right"), ("US", "GDP 2001–24"): (5, 8, "left"),
               ("BR", "GDP, 2010 ref."): (-6, -6, "right"), ("BR", "agri. VA, 2010 ref."): (6, 6, "left"),
               ("BR", "GDP, 2002 ref."): (6, -3, "left"), ("BR", "agri. VA, 2002 ref."): (6, 6, "left")}
    for P in pts:
        P["txt"] = lab_pos[(P["c"], P["lab"])]
    axa.set_xlim(1.5e-3, 12)
    axa.set_ylim(*YL)
    _plot_ladder(axa, pts, "dxa", True)
    axa.set_xlabel(r"Screening index $\bar\delta\,/\,|a_1|$ (dose $\times$ amplification; not a bound)")
    axa.set_ylabel(r"Observed $\xi^*_{V0}-\xi^*_{V1}$ (within window)")
    hs = [Line2D([], [], marker=CM[k], ls="none", ms=3.6, color=CC[k], label=S.COUNTRY_NAME[k]) for k in ("CN", "US", "BR")]
    hs += [Line2D([], [], marker="s", ls="none", ms=3.4, mfc=S.tint(CC["US"], SECOND_TINT), mec=CC["US"], mew=0.7,
                  label="light fill: second outcome"),
           Line2D([], [], marker="D", ls="none", ms=3.0, mfc="none", mec="black", mew=0.5,
                  label="guarded projection, testable"),
           Line2D([], [], marker="D", ls="none", ms=3.0, mfc="none", mec=UNTESTABLE_COLOR, mew=0.5,
                  label="guarded projection, below gate"),
           Line2D([], [], color=GATE_COLOR, lw=3.2, label="± resolution gate")]
    axa.legend(handles=hs, loc="lower left", fontsize=5, ncol=1, bbox_to_anchor=(0.0, 0.0))

    axb.set_xlim(3e-5, 1)
    axb.set_ylim(*YL)
    _plot_ladder(axb, pts, "dose", False)
    axb.set_xlabel(r"Composition dose $\bar\delta=\overline{(n_E+n_L)/n_S}$")
    axb.set_ylabel(r"$\xi^*_{V0}-\xi^*_{V1}$")
    axb.text(4e-5, YL[0] + 0.04, "same points,\nno amplification", fontsize=5, va="bottom")

    ex = tab("expi_attrition_by_year.csv")
    ex = ex[(ex.saai_spec == "multi") & ex.period.isin(["pre", "during"])]
    d_cn = (ex.n_enter + ex.n_leave) / ex.n_stay
    dec = tab("k2/k2_decomposition.csv")
    dec = dec[(dec.scope == "metro") & (dec.panel == "B") & (dec.link == "overlap") & (dec.period == "secondary")]
    dbr = tab("k2br_decomposition.csv")
    dbr = dbr[(dbr.panel == "B") & (dbr.family == "V0")]
    series = [("CN", "agri", d_cn, ex.mechanical_term, True)]
    for out, filled in (("pi", True), ("gdp", False)):
        q = dec[(dec.outcome == out) & ((dec.n_E + dec.n_L) > 0)]
        series.append(("US", out, (q.n_E + q.n_L) / q.n_S, q.M_t, filled))
    for out, filled in (("pib", True), ("vab_agro", False)):
        q = dbr[(dbr.outcome == out) & ((dbr.n_E + dbr.n_L) > 0)]
        series.append(("BR", out, (q.n_E + q.n_L) / q.n_S, q.M, filled))
    for k, out, d, m, filled in series:
        d = np.asarray(d, float)
        m = np.asarray(m, float)
        ok = (d > 0) & (np.abs(m) > 0)
        h = axf.plot(d[ok], np.abs(m[ok]), CM[k], ms=2.6, mfc=CC[k] if filled else S.tint(CC[k], SECOND_TINT),
                     mec=CC[k], mew=0.6, ls="none", alpha=0.9)
        S.tag(h, k + ("" if filled else "-2nd"))
    xx = np.array([1e-4, 3])
    axf.plot(xx, xx, color=S.REF_COLOR, lw=0.5, ls=(0, (3, 1.5)), zorder=0)
    axf.text(1.5e-1, 0.5, r"$|M_t|=\delta_t$ (slope 1)", fontsize=5, rotation=0, ha="right", color="#6E6E6E")
    axf.set_xscale("log")
    axf.set_yscale("log")
    axf.set_xlim(1e-4, 3)
    axf.set_ylim(3e-5, 1)
    axf.set_xlabel(r"Per-transition dose $\delta_t=(n_E+n_L)/n_S$")
    axf.set_ylabel(r"Composition term $|M_t|$ (Lemma 1)")

    x0, w = 124, 57
    axc = S.ax_mm(fig, x0, 87, w, 26)
    axd = S.ax_mm(fig, x0, 49, w, 26)
    axe = S.ax_mm(fig, x0, 11, w, 26)
    allcn = tab("expi_attrition_by_year.csv")
    allcn = allcn[allcn.saai_spec == "multi"]
    _window_box(axc, 2000.5, 2012.5, 0, 0, fill=S.WINDOW_SHADE)
    _window_box(axc, 2013.5, 2019.5, 0, 0, fill=S.WINDOW_SHADE)
    inw = allcn.period.isin(["pre", "during"]).to_numpy()
    yr_cn = allcn.year_t.to_numpy() + 1
    dose_cn = ((allcn.n_enter + allcn.n_leave) / allcn.n_stay).to_numpy()
    _dose_series(axc, yr_cn[inw], dose_cn[inw], CC["CN"], CM["CN"])
    _dose_series(axc, yr_cn[~inw], dose_cn[~inw], "#B5B5B5", CM["CN"])
    axc.set_xlim(1999, 2025)
    axc.text(2006.5, 2.2e-5, "2000–12", fontsize=5, ha="center", va="bottom")
    axc.text(2016.5, 2.2e-5, "2013–19", fontsize=5, ha="center", va="bottom")
    axc.plot([2017], [v("CnBreakDose")], "o", ms=4.0, mfc="none", mec="black", mew=0.5, zorder=4)
    axc.text(2017.9, v("CnBreakDose"), ftxt("CnBreakDose"), fontsize=5, va="center", ha="left")
    axc.text(1999.4, 1.9, "China counties", fontsize=5.5, color=CC["CN"], va="top", fontweight="bold")
    u = tab("k2_coverage_panelB_by_year.csv")
    u = u[(u.scope == "metro") & (u.link_rule == "overlap")]
    uy = u.transition.str.slice(6, 10).astype(int).to_numpy()
    _window_box(axd, 2003.5, 2024.5, 0, 0, fill="#E2E2E2")
    _window_box(axd, 1993.5, 2003.5, 0, 0, fill=S.WINDOW_SHADE)
    _dose_series(axd, uy, u.mech_bound_per_unit_m.to_numpy(), CC["US"], CM["US"])
    axd.set_xlim(1991.3, 2026)
    axd.text(1991.7, 1.9, "US metropolitan areas", fontsize=5.5, color=CC["US"], va="top", fontweight="bold")
    axd.text(2014, 1.9, "main 2003–24", fontsize=5, ha="center", va="top")
    axd.text(1998.5, 0.35, "+ 1993–2003", fontsize=5, ha="center", va="top")
    b = tab("k2br_coverage_panelB_by_transition.csv")
    _window_box(axe, 2002.5, 2023.5, 0, 0, fill=S.WINDOW_SHADE)
    _window_box(axe, 1999.5, 2012.5, 1.15e-5, 2.6, hatch="////")
    _dose_series(axe, b.t1.to_numpy(), b["comp_shock_(nE+nL)/nS"].to_numpy(), "#4E8F3E", CM["BR"])
    axe.set_xlim(1999.3, 2024.2)
    axe.text(1999.6, 1.9, "Brazilian municipalities", fontsize=5.5, color=LABEL_COLOR["BR"], va="top",
             fontweight="bold")
    axe.text(2006, 0.35, "2002 ref. (hatched)", fontsize=5, ha="center", va="top")
    axe.text(2018, 0.35, "2010 ref.", fontsize=5, ha="center", va="top")
    axe.set_xlabel("Transition end year")
    for ax in (axc, axd, axe):
        ax.set_ylabel(r"$\delta_t$", labelpad=1)
        ax.set_yticklabels(["10$^{-4}$", "10$^{-3}$", "10$^{-2}$", "10$^{-1}$", "1"])
        lo, hi = ax.get_xlim()
        ax.text(hi - 0.2, 1e-5 * 1.8, "0", fontsize=5, va="center", ha="right", color="#555555")
    for ax, l, dx in ((axa, "a", -11), (axb, "b", -11), (axf, "c", -11), (axc, "d", -11), (axd, "e", -11),
                      (axe, "f", -11)):
        S.panel_label(fig, ax, l, dx_mm=dx, dy_mm=2.0)
    S.save(fig, OUTDIR, "fig3_dose_ladder")


PROV_EN = {11: "Beijing", 12: "Tianjin", 13: "Hebei", 14: "Shanxi", 15: "Inner Mongolia", 21: "Liaoning",
           22: "Jilin", 23: "Heilongjiang", 31: "Shanghai", 32: "Jiangsu", 33: "Zhejiang", 34: "Anhui",
           35: "Fujian", 36: "Jiangxi", 37: "Shandong", 41: "Henan", 42: "Hubei", 43: "Hunan",
           44: "Guangdong", 45: "Guangxi", 46: "Hainan", 50: "Chongqing", 51: "Sichuan", 52: "Guizhou",
           53: "Yunnan", 54: "Tibet", 61: "Shaanxi", 62: "Gansu", 63: "Qinghai", 64: "Ningxia", 65: "Xinjiang"}
TRAP_Q = [("xi_trap (during-pre)", r"$\xi^*$ shift (second $-$ first)", "lin", 0.0),
          ("tau_mfpt (during/pre)", r"$\tau$ ratio (second / first)", "log", 1.0),
          ("mean_diffusion (during/pre)", r"$\bar D$ ratio", "lin", 1.0),
          ("B_ratio_w (during/pre)", r"$B_w$ ratio (poverty side)", "lin", 1.0)]


def _trap_panel(ax, sbs, q, lab, scale, ref, show_legend=False, paired=None):
    rows = [("V0_baseline", "V0"), ("V1_stayer_restandardised", "V1")]
    for i, (var, vl) in enumerate(rows):
        r = one(sbs, variant=var, degenerate="limit", quantity=q)
        y = 1 - i
        sty = S.V0_STYLE if vl == "V0" else S.V1_STYLE
        hp = ax.plot([r.rerun_ci_lo, r.rerun_ci_hi], [y, y], color=sty["color"], lw=1.6,
                     solid_capstyle="butt", ls="-")
        S.tag(hp, f"prov-{vl}")
        hm = ax.plot([r.point_estimate], [y], "o", ms=3.0, mfc=sty["mfc"], mec=sty["mec"], mew=0.7)
        S.tag(hm, f"pt-{vl}")
        ax.text(1.02, y, "×" + fnum(r.width_ratio_rerun_over_county, 1), transform=ax.get_yaxis_transform(),
                fontsize=5, va="center", ha="left")
    ax.axvline(ref, color="black", lw=0.4, ls=(0, (2, 1.5)), zorder=0)
    if scale == "log":
        ax.set_xscale("log")
    yt, yl, ylo = [1, 0], ["V0", "V1"], -0.55
    if paired is not None:
        hpd = ax.plot([paired.ci_lo, paired.ci_hi], [-1, -1], color="black", lw=1.6, solid_capstyle="butt")
        S.tag(hpd, "paired")
        hpp = ax.plot([paired.point_estimate], [-1], "D", ms=2.8, mfc="white", mec="black", mew=0.7)
        S.tag(hpp, "pt-paired")
        yt, yl, ylo = [1, 0, -1], ["V0", "V1", "V0$-$V1"], -1.55
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=5.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(ylo, 1.55)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(lab, labelpad=1)


def fig4():
    fig = S.figure(183, 128)
    cov = tab("k1_coverage_province_year.csv")
    cnt = cov.pivot(index="ProvinceCode", columns="Year", values="n_multi_saai")
    tot = cov.pivot(index="ProvinceCode", columns="Year", values="n_counties")
    keep = cnt.fillna(0).max(axis=1) > 0
    cnt, tot = cnt[keep], tot[keep]
    years = list(cnt.columns)
    share = (cnt / tot).to_numpy()
    first = np.array([np.argmax(np.nan_to_num(r) > 0) for r in cnt.to_numpy()])
    order = np.lexsort((cnt.index.to_numpy(), first))
    cnt, share, tot = cnt.iloc[order], share[order], tot.iloc[order]
    nP = len(cnt)
    x0, wr = 24, 96
    axr = S.ax_mm(fig, x0, 10, wr, 66)
    axr._k1_box_ok = True
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("cn", ["#F3B98C", CC["CN"]])
    im = axr.pcolormesh(np.arange(years[0], years[-1] + 2) - 0.5, np.arange(nP + 1) - 0.5,
                        np.ma.masked_invalid(np.where(np.nan_to_num(share) > 0, share, np.nan)), cmap=cmap,
                        vmin=0, vmax=1, shading="flat", antialiased=False, rasterized=False)
    axr.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    for i in range(nP):
        for j, yr in enumerate(years):
            val = cnt.iloc[i, j]
            if np.isnan(val):
                axr.add_patch(mpatches.Rectangle((yr - 0.5, i - 0.5), 1, 1, fc="#EDEDED", ec="none", lw=0))
            if j + 1 < len(years):
                nxt = cnt.iloc[i, j + 1]
                if val > 0 and (np.isnan(nxt) or nxt == 0):
                    axr.plot([yr + 0.5, yr + 0.5], [i - 0.45, i + 0.45], color="black", lw=1.0,
                             solid_capstyle="butt")
    for i in range(1, nP):
        axr.axhline(i - 0.5, color="white", lw=0.4)
    axr.set_yticks(range(nP))
    axr.set_yticklabels([PROV_EN[c] for c in cnt.index], fontsize=5)
    axr.set_xticks([2000, 2004, 2008, 2012, 2016, 2020, 2023])
    axr.tick_params(axis="y", length=0, pad=1.5)
    for sp in axr.spines.values():
        sp.set_visible(False)
    axr.set_xlabel("Year")
    for (a, b_, lab) in ((2000, 2012, "first window"), (2013, 2019, "second")):
        axr.plot([a - 0.4, b_ + 0.4], [-0.85, -0.85], color="black", lw=0.5, clip_on=False)
        axr.text((a + b_) / 2, -1.0, lab, fontsize=5, ha="center", va="bottom")
    axr.set_ylim(nP - 0.5, -0.5)
    for code, mac in ((51, "CnSichuanLeave"), (62, "CnGansuLeave")):
        i = list(cnt.index).index(code)
        axr.text(2017.0, i, S.minus("-") + ftxt(mac), fontsize=5, va="center", ha="left")
        axr.add_patch(mpatches.Rectangle((years[0] - 0.5, i - 0.5), len(years), 1.0, fc="none", ec="black",
                                         lw=0.6, zorder=5))
    axm = S.ax_mm(fig, x0 + wr + 2, 10, 12, 66)
    nmax = tot.max(axis=1).to_numpy()
    axm.barh(range(nP), nmax, height=0.7, color="#BDBDBD", lw=0)
    axm.set_ylim(nP - 0.5, -0.5)
    axm.set_yticks([])
    axm.spines["left"].set_visible(False)
    axm.set_xticks([0, 100])
    axm.set_xlabel("Counties", labelpad=1)
    cax = S.ax_mm(fig, 150, 10, 2.2, 22)
    cb = fig.colorbar(im, cax=cax)
    cb.solids.set_rasterized(False)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5, width=0.4, length=1.5)
    cb.set_label("Share of counties\nwith the agricultural index", fontsize=5, labelpad=2)
    kx = 166
    ka = S.ax_mm(fig, kx, 26, 3, 2.6)
    ka.set_xlim(0, 1); ka.set_ylim(0, 1); ka.axis("off")
    ka.add_patch(mpatches.Rectangle((0, 0), 1, 1, fc="#EDEDED", ec="none", transform=ka.transAxes))
    W, H = fig.get_size_inches() / S.MM
    fig.text((kx + 4) / W, 27.3 / H, "not in panel", fontsize=5, va="center")
    kb = S.ax_mm(fig, kx, 21, 3, 2.6)
    kb.set_xlim(0, 1); kb.set_ylim(0, 1); kb.axis("off")
    kb.add_patch(mpatches.Rectangle((0, 0), 1, 1, fc="white", ec="#9A9A9A", lw=0.4, transform=kb.transAxes))
    fig.text((kx + 4) / W, 22.3 / H, "not reported", fontsize=5, va="center")
    kc = S.ax_mm(fig, kx, 16, 3, 2.6)
    kc.set_xlim(0, 1); kc.set_ylim(0, 1); kc.axis("off")
    kc.plot([0.5, 0.5], [0.05, 0.95], color="black", lw=1.0)
    fig.text((kx + 4) / W, 17.3 / H, "province stops", fontsize=5, va="center")

    ex = tab("expi_attrition_by_year.csv")
    ex = ex[ex.saai_spec == "multi"]
    xt = ex.year_t.to_numpy() + 0.5
    dose = ((ex.n_enter + ex.n_leave) / ex.n_stay).to_numpy()
    axd = S.ax_mm(fig, x0, 97, wr, 20)
    axM = S.ax_mm(fig, x0, 80, wr, 15)
    colw = {"pre": CC["CN"], "during": CC["CN"], "cross": "#B5B5B5", "post": "#B5B5B5"}
    for per in ("pre", "during", "cross", "post"):
        m = (ex.period == per).to_numpy()
        hb = axd.bar(xt[m], dose[m], width=0.7, color=colw[per], lw=0)
        S.tag(list(hb), "dose-in" if per in ("pre", "during") else "dose-out")
        st = axM.vlines(xt[m], 0, ex.mechanical_term.to_numpy()[m], color=colw[per], lw=0.7)
        hm = axM.plot(xt[m], ex.mechanical_term.to_numpy()[m], "o", ms=2.0, color=colw[per], mew=0)
        S.tag(hm, "M-in" if per in ("pre", "during") else "M-out")
    i16 = int(np.where(ex.year_t.to_numpy() == 2016)[0][0])
    axd.annotate(ftxt("CnBreakDose"), (xt[i16], dose[i16]), xytext=(0, 1.5), textcoords="offset points",
                 fontsize=5, ha="center", va="bottom")
    axM.annotate(S.minus(fnum(ex.mechanical_term.to_numpy()[i16], 2)), (xt[i16], ex.mechanical_term.to_numpy()[i16]),
                 xytext=(4, 0), textcoords="offset points", fontsize=5, ha="left", va="center")
    for ax in (axd, axM):
        ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
        ax.set_xticks([2000, 2004, 2008, 2012, 2016, 2020, 2023])
        ax.tick_params(labelbottom=False)
    axd.set_ylim(0, 1.08 * np.nanmax(dose))
    axd.set_ylabel("$(n_E+n_L)/n_S$", labelpad=1)
    axM.axhline(0, color=S.REF_COLOR, lw=0.4)
    axM.set_ylim(-0.6, 0.45)
    axM.set_ylabel("$M_t$", labelpad=1)
    axd.text(years[0] - 0.3, 1.04 * np.nanmax(dose), "grey: transitions outside the two windows", fontsize=5, va="top",
             color="#6E6E6E")

    sbs = tab("expi_rerun_ci_side_by_side.csv")
    pe = pd.read_csv(os.path.join(TABLES, "expi_paired_estimands.csv"), keep_default_na=False, na_values=[""])
    pr = one(pe, estimand="xi_shift_V0_minus_V1", convention="limit", role='paired_V0_V1')
    if pr.ci_status != "ok":
        raise SourceError("paired shift difference: tol0 interval not usable under the limit convention")
    pr = pr.copy()
    for c_ in ("ci_lo", "ci_hi", "point_estimate"):
        pr[c_] = float(pr[c_])
    if abs(pr.point_estimate - v("CnVzeroMinusVoneXiShift")) > 5e-3:
        raise SourceError("paired shift difference disagrees with CnVzeroMinusVoneXiShift")
    axes = []
    xr = 152
    for k, (q, lab, sc, ref) in enumerate(TRAP_Q[:2] + TRAP_Q[3:]):
        hgt = 18 if k == 0 else 14
        ax = S.ax_mm(fig, xr, 97 if k == 0 else 99 - k * 25, 24, hgt)
        _trap_panel(ax, sbs, q, lab, sc, ref, paired=pr if k == 0 else None)
        axes.append(ax)
    axes[0].set_xlim(-2.0, 0.8)
    axes[1].set_xlim(0.02, 200)
    axes[2].set_xlim(0.3, 1.5)
    hs = [Line2D([], [], color="#555555", lw=1.6, label="95% CI, province block,\npipeline re-run per draw")]
    axes[0].legend(handles=hs, loc="lower left", bbox_to_anchor=(-0.25, 1.02), fontsize=5, ncol=1,
                   handlelength=1.2)
    axes[0].text(1.02, 1.22, "width vs\ncounty\nblock", transform=axes[0].transAxes, fontsize=5, ha="left",
                 va="bottom")
    S.panel_label(fig, axd, "a", dx_mm=-17, dy_mm=1.0)
    S.panel_label(fig, axr, "b", dx_mm=-17, dy_mm=3.0)
    S.panel_label(fig, axes[0], "c", dx_mm=-9, dy_mm=8.0)
    S.save(fig, OUTDIR, "fig4_china_coverage")


ROWSTY = {"projection": dict(marker="o", mfc="black", mec="black", label="pooled projection"),
          "closed_form": dict(marker="s", mfc="white", mec="black", label="closed form (OU only)"),
          "zeroth_order": dict(marker="^", mfc="#8A8A8A", mec="#8A8A8A", label=r"zeroth order"),
          "zeroth_order_mbar": dict(marker="v", mfc="white", mec="#8A8A8A", label=r"$\bar M/|a_1|$ (report only)")}


def figS1():
    a = tab("k1_exploratory/k1_exploratory_k1a.csv")
    a = a[(a.pipeline == "saai") & (a.null_variant == "registered")]
    fig = S.figure(183, 62)
    specs = [("xi_trap", r"$\xi^*$ shift", "rel"), ("two_delta_V", r"$2\Delta V$ shift", "sign"),
             ("tau_mfpt", r"$\tau$ ratio", "sign"), ("rho1", r"$\rho_1$ shift", "sign")]
    axes = [S.ax_mm(fig, 12 + k * 44.0, 12, 31, 36) for k in range(4)]
    for ax, (q, lab, kind) in zip(axes, specs):
        sub = a[(a.quantity == q) & a.statistic.str.endswith("_rel" if q in ("xi_trap",) else ("_sign" if q != "rho1" else "_sign"))]
        lo = min(np.nanmin(sub.median_sim), np.nanmin(sub.median_pred[np.isfinite(sub.median_pred)]))
        hi = max(np.nanmax(sub.median_sim), np.nanmax(sub.median_pred[np.isfinite(sub.median_pred)]))
        pad = 0.12 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        if q == "xi_trap":
            xx = np.linspace(lo, hi, 10)
            ax.fill_between(xx, xx * 0.7, xx * 1.3, color=S.WINDOW_SHADE, lw=0, zorder=0)
        if q == "rho1":
            ax.fill_between([0, hi], lo, 0, color="#EDEDED", lw=0, zorder=0)
            ax.fill_between([lo, 0], 0, hi, color="#EDEDED", lw=0, zorder=0)
            ax.text(0.03, 0.97, "shaded: wrong sign", transform=ax.transAxes, fontsize=5, ha="left", va="top",
                    color="#555555")
        ax.plot([lo, hi], [lo, hi], color=S.REF_COLOR, lw=0.5, zorder=1)
        ax.axhline(0, color=S.REF_COLOR, lw=0.3)
        ax.axvline(0, color=S.REF_COLOR, lw=0.3)
        for row, st in ROWSTY.items():
            r = sub[sub.row == row].drop_duplicates("cell_index")
            ok = np.isfinite(r.median_pred.to_numpy())
            h = ax.plot(r.median_sim.to_numpy()[ok], r.median_pred.to_numpy()[ok], ls="none", ms=3.0, mew=0.6,
                        marker=st["marker"], mfc=st["mfc"], mec=st["mec"])
            S.tag(h, row)
            if row == "projection":
                offs = {3: (3, -6), 4: (3, 2), 8: (-7, 2), 9: (3, -3)}
                if q == "rho1":
                    offs = {3: (3, -6), 4: (-9, 3), 8: (-7, 2), 9: (3, -3)}
                for _, rr in r[ok].iterrows():
                    ax.annotate(str(int(rr.cell_index)), (rr.median_sim, rr.median_pred),
                                xytext=offs[int(rr.cell_index)], textcoords="offset points", fontsize=5)
        if q == "tau_mfpt":
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(0.5, 200); ax.set_ylim(0.5, 200)
            ax.plot([0.5, 200], [0.5, 200], color=S.REF_COLOR, lw=0.5)
        else:
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Simulated median V0 bias, " + lab, fontsize=5.5)
        ax.set_ylabel("Predicted", labelpad=1)
    axes[0].text(0.03, 0.97, "shaded: ±30%", transform=axes[0].transAxes, fontsize=5, va="top")
    axes[2].text(0.03, 0.97, "cell 3: prediction infinite\n(not shown)", transform=axes[2].transAxes,
                 fontsize=5, va="top")
    hs = [Line2D([], [], ls="none", ms=3.0, mew=0.6, marker=st["marker"], mfc=st["mfc"], mec=st["mec"],
                 label=st["label"]) for st in ROWSTY.values()]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(12 / 183, 1 - 2 / 62), ncol=4, fontsize=5.5,
               columnspacing=1.2)
    fig.text(1 - 2 / 183, 1 - 3 / 62, "Exploratory",
             fontsize=5.5, ha="right", va="top", fontweight="bold")
    for ax, l in zip(axes, "abcd"):
        S.panel_label(fig, ax, l, dx_mm=-10, dy_mm=1.5)
    S.save(fig, OUTDIR, "figS1_k1a_exploratory")


def figS2():
    b = tab("k1_exploratory/k1_exploratory_k1b.csv")
    b = b[b.ci_nan_rule == "tol0"]
    fig = S.figure(183, 58)
    ax = S.ax_mm(fig, 12, 13, 100, 36)
    cells = sorted(b.cell_index.unique())
    procs = {"i_ou": "(i) OU", "ii_cubic": "(ii) cubic", "iii_mult": "(iii) cubic, mult."}
    R = v("KoneR")
    half = 1.96 * np.sqrt(0.95 * 0.05 / R)
    ax.axhspan(0.95 - half, 0.95 + half, color=S.WINDOW_SHADE, lw=0, zorder=0)
    ax.axhline(0.95, color=S.REF_COLOR, lw=0.5)
    ax.axhline(0.90, color="black", lw=0.5, ls=(0, (3, 1.5)))
    for i, c in enumerate(cells):
        for q, dx, qmk in (("xi_trap", -0.15, "o"), ("tau_mfpt", 0.15, "s")):
            for pipe, dd in (("saai", 0.0), ("direct", 0.07)):
                r = one(b, cell_index=c, quantity=q, pipeline=pipe)
                ne = "not_evaluable" in str(r.exploratory_outcome)
                col = "#9A9A9A" if ne else S.V1_COLOR
                mk = qmk if pipe == "saai" else "D"
                h = ax.plot([i + dx + dd], [r.value], mk, ms=3.0 if pipe == "saai" else 2.0, color=col,
                            mfc=col, mec=col, mew=0.6)
                S.tag(h, f"{q}-{pipe}-{ne}")
                if "fail" in str(r.exploratory_outcome) and pipe == "saai":
                    ax.annotate(fnum(r.value, 3), (i + dx, r.value), xytext=(4, 0), textcoords="offset points",
                                fontsize=5, va="center")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"{c}\n({one(b, cell_index=c, quantity='xi_trap', pipeline='saai').mechanism})" for c in cells],
                       fontsize=5)
    ax.tick_params(axis="x", length=0)
    for k, pr in enumerate(procs):
        idx = [i for i, c in enumerate(cells) if one(b, cell_index=c, quantity="xi_trap", pipeline="saai").process == pr]
        ax.text(np.mean(idx), 1.006, procs[pr], fontsize=5.5, ha="center", va="bottom")
        if k:
            ax.axvline(idx[0] - 0.5, color=S.REF_COLOR, lw=0.4)
    ax.set_ylim(0.83, 1.0)
    ax.set_xlim(-0.6, len(cells) - 0.4)
    ax.set_ylabel("V1 coverage of 95% CI")
    ax.set_xlabel("Cell (exit mechanism)")
    hs = [Line2D([], [], marker="o", ls="none", ms=3, color=S.V1_COLOR, label=r"$\xi^*$ shift"),
          Line2D([], [], marker="s", ls="none", ms=3, color=S.V1_COLOR, label=r"$\tau$ ratio"),
          Line2D([], [], marker="s", ls="none", ms=3, color="#9A9A9A", label="not evaluable (amendment 7)"),
          Line2D([], [], marker="D", ls="none", ms=2, mfc="#555555", mec="#555555", label="direct-$x$")]
    ax.legend(handles=hs, loc="lower left", ncol=4, fontsize=5)
    c = tab("k1_exploratory/k1_exploratory_controls_k1a.csv")
    axb = S.ax_mm(fig, 132, 13, 48, 36)
    outs = [("exploratory:pass", "pass", "#D9D9D9"), ("exploratory:fail", "fail", "#4D4D4D"),
            ("exploratory:fail_(non-finite)", "fail (non-finite)", "#8A8A8A"),
            ("exploratory:indistinguishable_from_null", "indistinguishable", "#FFFFFF")]
    cc = sorted(c.cell_index.unique())
    left = np.zeros(len(cc))
    for key, lab, col in outs:
        n = np.array([(c[(c.cell_index == x)].exploratory_outcome == key).sum() for x in cc])
        hb = axb.barh(range(len(cc)), n, left=left, color=col, ec="black", lw=0.3, height=0.7, label=lab)
        S.tag(list(hb), lab)
        left += n
    axb.set_yticks(range(len(cc)))
    axb.set_yticklabels([str(x) for x in cc], fontsize=5)
    axb.invert_yaxis()
    axb.set_xlabel("Rows (report-only controls)")
    axb.set_ylabel("Cell", labelpad=1)
    axb.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=5, handlelength=0.9)
    fig.text(16 / 183, 1 - 1.5 / 58, "Exploratory",
             fontsize=5.5, ha="left", va="top", fontweight="bold")
    S.panel_label(fig, ax, "a", dx_mm=-10, dy_mm=4)
    S.panel_label(fig, axb, "b", dx_mm=-9, dy_mm=4.5)
    S.save(fig, OUTDIR, "figS2_k1b_controls")


US_SPECS = [("metro", "pi", "main", "overlap"), ("metro", "pi", "main", "code"), ("metro", "pi", "secondary", "overlap"),
            ("metro", "gdp", "main", "overlap"), ("metro", "gdp", "main", "code"), ("metro", "gdp", "secondary", "overlap"),
            ("cbsa_all", "pi", "main", "overlap"), ("cbsa_all", "pi", "main", "code"),
            ("cbsa_all", "gdp", "main", "overlap"), ("cbsa_all", "gdp", "main", "code")]


def _spec_label(sc, out, per, link):
    return f"{'metro' if sc == 'metro' else 'all CBSA'}, {out.upper()}, {per}, {link}"


def _forest(ax, ys, est, lo1, hi1, lo2, hi2, col, filled):
    for y, e, a1, b1, a2, b2, f in zip(ys, est, lo1, hi1, lo2, hi2, filled):
        h2 = ax.plot([a2, b2], [y - 0.18, y - 0.18], color=S.SUPERSEDED_COLOR, lw=0.8, solid_capstyle="butt")
        S.tag(h2, "alt")
        h1 = ax.plot([a1, b1], [y, y], color=col, lw=1.3, solid_capstyle="butt")
        S.tag(h1, "main")
        hm = ax.plot([e], [y], "o", ms=2.6, mfc=col if f else S.tint(col, SECOND_TINT), mec=col, mew=0.6)
        S.tag(hm, "pt" + ("" if f else "o"))
    ax.axvline(0, color="black", lw=0.4, ls=(0, (2, 1.5)))
    ax.set_ylim(len(ys) - 0.4, -0.6)


def figS3():
    d = tab("k2/k2_paired_differences.csv")
    fig = S.figure(183, 74)
    cols = [("V0-V1", "xi_trap", r"V0 $-$ V1, $\xi^*$"), ("V0-V1", "rho1", r"V0 $-$ V1, $\rho_1$"),
            ("V1(B)-A", "xi_trap", r"V1(B) $-$ A, $\xi^*$ (H4)"), ("V1(B)-A", "rho1", r"V1(B) $-$ A, $\rho_1$ (H4)"),
            ("V1-V1d", "rho1", r"V1 $-$ V1d (redefinition), $\rho_1$")]
    W0 = 52
    w = (183 - W0 - 4 * 5 - 3) / 5
    axes = [S.ax_mm(fig, W0 + k * (w + 5), 12, w, 52) for k in range(5)]
    ys = np.arange(len(US_SPECS))
    for ax, (con, q, lab) in zip(axes, cols):
        rows = [one(d, scope=sc, outcome=o, period=p_, link=l, contrast=con, conv="limit", quantity=q)
                for sc, o, p_, l in US_SPECS]
        _forest(ax, ys, [r.estimate for r in rows], [r.lineage_tol0_lo for r in rows], [r.lineage_tol0_hi for r in rows],
                [r.state_tol0_lo for r in rows], [r.state_tol0_hi for r in rows], CC["US"],
                [o == "pi" for _, o, _, _ in US_SPECS])
        ax.set_xlabel(lab, fontsize=5.5)
        ax.set_yticks(ys)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    axes[0].set_yticklabels([_spec_label(*sp) + ("" if (sp[0] == "metro" and sp[3] == "overlap") else " †")
                             for sp in US_SPECS], fontsize=5)
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    hs = [Line2D([], [], color=CC["US"], lw=1.3, label="lineage block (primary)"),
          Line2D([], [], color=S.SUPERSEDED_COLOR, lw=0.8, label="state block")]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(W0 / 183, 1 - 1 / 74), ncol=2, fontsize=5.5)
    fig.text(1 - 2 / 183, 1 - 2.5 / 74, "95% percentile CIs, B = " + ftxt("UsB") + "; failed draws count",
             fontsize=5.5, ha="right", va="top")
    for ax, l in zip(axes, "abcde"):
        S.panel_label(fig, ax, l, dx_mm=-48 if l == "a" else -3, dy_mm=1.0)
    S.save(fig, OUTDIR, "figS4_us_detail")


BR_CELLS = [("ref2010", "pib"), ("ref2010", "vab_agro"), ("ref2002", "pib"), ("ref2002", "vab_agro")]


def figS4():
    c = tab("k2br_contrasts.csv")
    c = c[c.convention == "limit"]
    cons = [("V0B_minus_V1B", r"V0 $-$ V1"), ("V1B_minus_A", r"V1 $-$ fixed geogr."), ("V0B_minus_A", r"V0 $-$ fixed geogr."),
            ("V1pB_minus_A", r"V1 parents out $-$ fixed"), ("V1mB_minus_A", r"V1 merged $-$ fixed"),
            ("V1B_minus_V1mB", r"V1 $-$ V1 merged")]
    fig = S.figure(183, 92)
    W0 = 26
    w = (183 - W0 - 5 * 4.5 - 3) / 6
    ys = np.arange(len(BR_CELLS))
    bc = "#4E8F3E"
    for row, (q, qlab) in enumerate((("xi_trap", r"$\xi^*$"), ("rho1", r"$\rho_1$"))):
        for k, (con, lab) in enumerate(cons):
            ax = S.ax_mm(fig, W0 + k * (w + 4.5), 55 - row * 41, w, 25)
            mx = max(max(abs(one(c, window=wn, outcome=o, contrast=con, quantity=q)[f"{sch}_ci_{e_}"])
                         for sch in ("amc", "immediate", "uf") for e_ in ("lo", "hi")) for wn, o in BR_CELLS)
            ex_ = int(np.floor(np.log10(mx))) if mx < 0.01 else 0
            sc_ = 10.0 ** (-ex_)
            for i, (win, out) in enumerate(BR_CELLS):
                r = one(c, window=win, outcome=out, contrast=con, quantity=q).copy()
                for col_ in [x for x in r.index if x.endswith("_ci_lo") or x.endswith("_ci_hi")] + ["estimate"]:
                    r[col_] = r[col_] * sc_
                for dy, sch, lw_, colr in ((-0.22, "amc", 1.3, bc), (0.0, "immediate", 0.8, "#9A9A9A"),
                                           (0.22, "uf", 0.8, "#C9C9C9")):
                    lo_, hi_ = r[f"{sch}_ci_lo"], r[f"{sch}_ci_hi"]
                    h = ax.plot([lo_, hi_], [i + dy, i + dy], color=colr, lw=lw_, solid_capstyle="butt")
                    S.tag(h, sch)
                hm = ax.plot([r.estimate], [i - 0.22], CM["BR"], ms=2.8, mfc=bc if out == "pib" else S.tint(bc, SECOND_TINT),
                             mec=bc, mew=0.6)
                S.tag(hm, "pt" + out)
            ax.axvline(0, color="black", lw=0.4, ls=(0, (2, 1.5)))
            ax.set_ylim(len(BR_CELLS) - 0.4, -0.6)
            ax.set_yticks(ys)
            ax.set_yticklabels([f"{'GDP' if o == 'pib' else 'agri. VA'}, {wn}" for wn, o in BR_CELLS] if k == 0 else [],
                               fontsize=5)
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            unit = f" ($\\times10^{{{ex_}}}$)" if ex_ else ""
            ax.set_xlabel(f"{lab}\n{qlab}{unit}", fontsize=5.5)
            if k == 0:
                S.panel_label(fig, ax, "ab"[row], dx_mm=-24, dy_mm=3.0)
    hs = [Line2D([], [], color=bc, lw=1.3, label="AMC block (primary)"),
          Line2D([], [], color="#9A9A9A", lw=0.8, label="immediate region block"),
          Line2D([], [], color="#C9C9C9", lw=0.8, label="state (UF) block")]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(W0 / 183, 1 - 1 / 92), ncol=3, fontsize=5.5)
    fig.text(1 - 2 / 183, 1 - 2.5 / 92, "95% percentile CIs; failed draws count", fontsize=5.5, ha="right", va="top")
    S.save(fig, OUTDIR, "figS6_brazil_detail")


def _mass(status):
    import re
    m = re.search(r"stationary mass ([0-9.eE+-]+)", str(status))
    if not m:
        return ""
    x = float(m.group(1))
    if x >= 0.01:
        return f"{x:.2f}"
    e = int(np.floor(np.log10(x)))
    mant = x / 10.0 ** e
    return f"${mant:.1f}\\times10^{{{e}}}$"


def figS5():
    h = tab("k2/k2_h3b.csv")
    pdiff = tab("k2/k2_paired_differences.csv")
    bv = tab("k2br_verdicts.csv")
    con = tab("k2br_contrasts.csv")
    rows = []
    for sc, o, p_, l in US_SPECS:
        r = one(h, scope=sc, outcome=o, period=p_, link=l)
        dd = one(pdiff, scope=sc, outcome=o, period=p_, link=l, contrast="V0-V1", conv="limit", quantity="xi_trap")
        ro = (sc != "metro" or l != "overlap")
        rows.append(dict(c="US", lab=_spec_label(sc, o, p_, l) + (" †" if ro else ""), obs=r.observed,
                         lo=dd.lineage_tol0_lo,
                         hi=dd.lineage_tol0_hi, g=r.pred_xi_trap if str(r.status) == "ok" else np.nan,
                         ug=r.unguarded_pred_xi_trap, z=r.zeroth_order_xi_trap,
                         gate=r.gate if np.isfinite(r.gate) else v("HthreebGateFactor") * r.half_width_V1,
                         note=(("guarded: defined" if str(r.status) == "ok"
                                else "guarded: undefined (" + _mass(r.status) + ")")
                               + ";  unguarded: " + ("defined" if str(r.unguarded_status) == "ok"
                                                     else "undefined (" + _mass(r.unguarded_status) + ")")),
                         filled=(o == "pi")))
    for win, out in BR_CELLS:
        r = one(bv, window=win, outcome=out)
        cc = one(con, window=win, outcome=out, contrast="V0B_minus_V1B", convention="limit", quantity="xi_trap")
        rows.append(dict(c="BR", lab=f"Brazil {'GDP' if out == 'pib' else 'agri. VA'}, {win}", obs=r.H3b_observed,
                         lo=cc.amc_ci_lo, hi=cc.amc_ci_hi, g=r.H3b_predicted, ug=r.H3b_unguarded_predicted,
                         z=r.H3b_zeroth_order_predicted, gate=r.H3b_gate_threshold,
                         note=("guarded: " + ("defined" if r.H3b_prediction_status == "ok" else str(r.H3b_prediction_status))
                               + ";  unguarded: " + ("defined" if r.H3b_unguarded_status == "ok"
                                                     else str(r.H3b_unguarded_status))),
                         filled=(out == "pib")))
    fig = S.figure(183, 90)
    ax = S.ax_mm(fig, 44, 11, 76, 66)
    for i, R_ in enumerate(rows):
        col = CC[R_["c"]] if R_["c"] == "US" else "#4E8F3E"
        ax.add_patch(mpatches.Rectangle((-R_["gate"], i - 0.42), 2 * R_["gate"], 0.84, fc="#EFEFEF", ec="none", zorder=0))
        he = ax.errorbar([R_["obs"]], [i], xerr=sym_ci_err(R_["obs"], R_["lo"], R_["hi"]), fmt=CM[R_["c"]], ms=3.0,
                         mfc=col if R_["filled"] else S.tint(col, SECOND_TINT), mec=col, ecolor=col, elinewidth=0.8,
                         mew=0.6, zorder=3)
        S.tag(he, R_["c"] + ("" if R_["filled"] else "o"))
        if np.isfinite(R_["g"]):
            hg = ax.plot([R_["g"]], [i - 0.25], "D", ms=3.0, mfc="none", mec="black", mew=0.6, zorder=4)
            S.tag(hg, "guarded")
        if np.isfinite(R_["ug"]):
            hu = ax.plot([R_["ug"]], [i + 0.25], "D", ms=3.0, mfc="#9A9A9A", mec="#9A9A9A", mew=0.6, zorder=4)
            S.tag(hu, "unguarded")
        hz = ax.plot([R_["z"]], [i], "x", ms=2.8, color="#5A5A5A", mew=0.6, zorder=4)
        S.tag(hz, "zeroth")
        ax.text(1.02, i, R_["note"], transform=ax.get_yaxis_transform(), fontsize=5, va="center", ha="left")
    ax.axvline(0, color="black", lw=0.4, ls=(0, (2, 1.5)))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([R_["lab"] for R_ in rows], fontsize=5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlabel(r"$\xi^*_{V0}-\xi^*_{V1}$")
    hs = [Line2D([], [], marker="s", ls="none", ms=3, color=CC["US"], label="observed (95% CI)"),
          Line2D([], [], marker="D", ls="none", ms=3, mfc="none", mec="black", label="guarded projection"),
          Line2D([], [], marker="D", ls="none", ms=3, mfc="#9A9A9A", mec="#9A9A9A", label="unguarded projection"),
          Line2D([], [], marker="x", ls="none", ms=3, color="#5A5A5A", label="zeroth order (report only)"),
          mpatches.Patch(fc="#EFEFEF", ec="none", label="± resolution gate"),
          Line2D([], [], ls="none", label="† report-only specification")]
    ax.text(1.02, -0.9, "projection status (stationary mass\nbeyond the stable fixed points)",
            transform=ax.get_yaxis_transform(), fontsize=5, va="bottom")
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(44 / 183, 1 - 1 / 90), ncol=5, fontsize=5,
               columnspacing=0.8)
    S.save(fig, OUTDIR, "figS5_guarded")


def figS6():
    sbs = tab("expi_rerun_ci_side_by_side.csv")
    sbs["degenerate"] = sbs["degenerate"].fillna("nan")
    pe = pd.read_csv(os.path.join(TABLES, "expi_paired_estimands.csv"), keep_default_na=False, na_values=[""])
    qs = [("xi_trap (during-pre)", r"$\xi^*$ shift", "lin", 0.0), ("tau_mfpt (during/pre)", r"$\tau$ ratio", "log", 1.0),
          ("gamma (during-pre)", r"$\Gamma$ difference (descriptive)", "lin", 0.0),
          ("mean_diffusion (during/pre)", r"$\bar D$ ratio", "lin", 1.0), ("B_ratio_w (during/pre)", r"$B_w$ ratio", "lin", 1.0)]
    fig = S.figure(183, 68)
    w = 26
    grid = {}
    for row, conv in enumerate(("limit", "nan")):
        for k, (q, lab, sc, ref) in enumerate(qs):
            ax = S.ax_mm(fig, 14 + k * (w + 9), 42 - row * 31, w, 15)
            grid.setdefault(k, []).append(ax)
            for i, (var, vl) in enumerate((("V0_baseline", "V0"), ("V1_stayer_restandardised", "V1"))):
                r = one(sbs, variant=var, degenerate=conv, quantity=q)
                sty = S.V0_STYLE if vl == "V0" else S.V1_STYLE
                y = 1 - i
                h1 = ax.plot([r.county_block_ci_lo, r.county_block_ci_hi], [y + 0.2, y + 0.2], color=S.SUPERSEDED_COLOR,
                             lw=1.0, solid_capstyle="butt")
                S.tag(h1, "county")
                h2 = ax.plot([r.rerun_ci_lo, r.rerun_ci_hi], [y - 0.12, y - 0.12], color=sty["color"], lw=1.5,
                             solid_capstyle="butt")
                S.tag(h2, "prov" + vl)
                h3 = ax.plot([r.point_estimate], [y - 0.12], "o", ms=2.6, mfc=sty["mfc"], mec=sty["mec"], mew=0.6)
                S.tag(h3, "pt" + vl)
            ax.axvline(ref, color="black", lw=0.4, ls=(0, (2, 1.5)))
            if sc == "log":
                ax.set_xscale("log")
            yt, yl = [1, 0], ["V0", "V1"]
            if k == 0:
                pr = one(pe, estimand="xi_shift_V0_minus_V1", convention=conv)
                if pr.ci_status == "ok":
                    hpd = ax.plot([pr.ci_lo, pr.ci_hi], [-1.12, -1.12], color="black", lw=1.5, solid_capstyle="butt")
                    S.tag(hpd, "paired")
                else:
                    ax.annotate("no tol0 interval", (pr.point_estimate, -1.12), xytext=(-4, 0), textcoords="offset points",
                                fontsize=5, ha="right", va="center", bbox=dict(fc="white", ec="none", pad=0.3))
                hpp = ax.plot([pr.point_estimate], [-1.12], "D", ms=2.4, mfc="white", mec="black", mew=0.6)
                S.tag(hpp, "pt-paired")
                yt, yl = [1, 0, -1], ["V0", "V1", "V0$-$V1"]
            ax.set_yticks(yt)
            ax.set_yticklabels(yl, fontsize=5.5)
            if k > 0:
                ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
            ax.set_ylim(-1.55, 1.55)
            ax.set_xlabel(lab, fontsize=5.5, labelpad=1)
            if k == 0:
                S.panel_label(fig, ax, "ab"[row], dx_mm=-12, dy_mm=2.0)
                ax.text(-0.25, 1.25, f"degenerate-draw convention: {conv}", transform=ax.transAxes, fontsize=5.5, va="bottom")
    for k, axs in grid.items():
        lo_ = min(ax_.get_xlim()[0] for ax_ in axs)
        hi_ = max(ax_.get_xlim()[1] for ax_ in axs)
        for ax_ in axs:
            ax_.set_xlim(lo_, hi_)
    hs = [Line2D([], [], color=S.SUPERSEDED_COLOR, lw=1.0, label="county block (superseded; trap 2)"),
          Line2D([], [], color=S.V0_COLOR, lw=1.5, label="province block, re-run per draw (primary), V0"),
          Line2D([], [], color=S.V1_COLOR, lw=1.5, label="province block (primary), V1"),
          Line2D([], [], color="black", lw=1.5, marker="D", ms=2.4, mfc="white", mec="black",
                 label="paired V0$-$V1 (same draws; non-finite draws = failures)")]
    fig.legend(handles=hs, loc="upper right", bbox_to_anchor=(1 - 2 / 183, 1 - 1 / 68), ncol=2, fontsize=5.5)
    S.save(fig, OUTDIR, "figS3_china_ci")


MASK_DIR = "k4_mask/confirmatory"
MASK_DOSES = (("us_like", "US-like"), ("intermediate", "intermediate"), ("china_like", "China-like"))
MASK_SYS = (("US", "US metropolitan areas"), ("BR", "Brazilian municipalities"))


def _shade(color, f):
    """Darker shade of a colour (mix with black by fraction f)."""
    from matplotlib.colors import to_rgb, to_hex
    r, g, b = to_rgb(color)
    return to_hex((r * (1 - f), g * (1 - f), b * (1 - f)))


MASK_DOSE_COLOR = {"US": {"us_like": S.tint(CC["US"], 0.62), "intermediate": S.tint(CC["US"], 0.3),
                          "china_like": CC["US"]},
                   "BR": {"us_like": S.tint(CC["BR"], 0.45), "intermediate": CC["BR"],
                          "china_like": _shade(CC["BR"], 0.42)}}
ZEROTH_COLOR = "#7A7A7A"
NAIVE_COLOR = "#9A9A9A"


def mask_tab(name):
    """A committed controlled-masking table; 'null' (a stratum label) must stay a string."""
    p = os.path.join(TABLES, MASK_DIR, name)
    if not os.path.isfile(p):
        raise SourceError(f"missing committed table {MASK_DIR}/{name}")
    return pd.read_csv(p, low_memory=False, keep_default_na=False, na_values=[""])


def mask_points():
    p = mask_tab("k4_points.csv")
    gf = json.load(open(os.path.join(TABLES, MASK_DIR, "k4_meta.json")))["criteria"]["gate_factor"]
    for q in ("T1_obs", "T3"):
        p[f"gate_{q}"] = p[q].abs() > gf * (p[f"{q}_ci_hi"] - p[f"{q}_ci_lo"]) / 2.0
    p["point_ok"] = (p.pred_point_status == "ok") & (p.id_0 == "stable_zero") & (p.id_1 == "stable_zero")
    p["covered"] = (p.T1_obs >= p.range_lo) & (p.T1_obs <= p.range_hi)
    return p


def _schematic(ax):
    """Panel a: visibility rule (illustration, no data). Units ranked by the base-year index."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    vis = "#E6E6E6"
    col = {"active": "#3A3A3A", "balanced": "#7E7E7E", "null": "#B4B4B4"}
    for x0, lab, (a, b), st in ((2, "exit", (0.5, 1.0), "active"), (52, "gap", (1 / 3, 2 / 3), "balanced")):
        w, h, y0 = 40, 56, 34
        ax.add_patch(mpatches.Rectangle((x0, y0), w, h, fc=vis, ec="none"))
        for yb in (0.0, 0.75):
            ax.add_patch(mpatches.Rectangle((x0 + a * w, y0 + yb * h), (b - a) * w, h * 0.25, fc=col[st], ec="none"))
        ax.add_patch(mpatches.Rectangle((x0 + a * w, y0 + 0.4 * h), (b - a) * w, 0.2 * h, fc=col["null"], ec="none"))
        ax.text(x0 + w / 2, y0 + h + 3, lab, fontsize=6, ha="center", va="bottom")
        ax.text(x0 + w / 2, y0 - 2, "years", fontsize=5, ha="center", va="top")
    ax.text(0, 34 + 28, "units ranked by\nbase-year index", fontsize=5, rotation=90, ha="right", va="center")
    for yy, t in ((34 + 56 * 0.125, "low"), (34 + 56 * 0.5, "mid"), (34 + 56 * 0.875, "high")):
        ax.text(93.5, yy, t, fontsize=5, ha="left", va="center")
    for yy, st, t in ((19, "active", "masked, exit (active: net forcing)"),
                      (10, "balanced", "masked, gap (balanced: report-only)"),
                      (1, "null", "masked, middle (null control)")):
        ax.add_patch(mpatches.Rectangle((2, yy), 5, 5.5, fc=col[st], ec="none"))
        ax.text(9, yy + 2.75, t, fontsize=5, va="center")


def fig5():
    p = mask_points()
    act = p[p.stratum == "active"]
    fig = S.figure(183, 162)
    axa = S.ax_mm(fig, 10, 114, 50, 42)
    _schematic(axa)
    axb = S.ax_mm(fig, 82, 120, 44, 32)
    ex = tab("expi_attrition_by_year.csv")
    ex = ex[(ex.saai_spec == "multi") & ex.period.isin(["pre", "during"])]
    real = [("CN", ((ex.n_enter + ex.n_leave) / ex.n_stay).to_numpy(), "China counties")]
    u = tab("k2_coverage_panelB_by_year.csv")
    u = u[(u.scope == "metro") & (u.link_rule == "overlap")]
    real.append(("US", u.mech_bound_per_unit_m.to_numpy(), "US metro areas"))
    b = tab("k2br_coverage_panelB_by_transition.csv")
    real.append(("BR", b["comp_shock_(nE+nL)/nS"].to_numpy(), "Brazil municipalities"))
    ylab = []
    for k, (c, d, lab) in enumerate(real):
        d = d[np.asarray(d, float) > 0]
        y = 2.0 - 0.7 * k
        h = axb.plot(d, np.full(len(d), y), "|", ms=4.0, mew=0.6, color=CC[c], ls="none")
        S.tag(h, "real-" + c)
        ylab.append((y, lab))
    for (sy, _n), yy in zip(MASK_SYS, (4.1, 3.1)):
        q = act[act.system == sy]
        for lv, _l in MASK_DOSES:
            r = q[q.fraction_level == lv].sort_values(["cell", "rep"])
            jit = ((np.arange(len(r)) % 9) - 4) * 0.07
            h = axb.plot(r.delta_event, yy + jit, CM[sy], ms=2.0, mew=0.3, mfc=MASK_DOSE_COLOR[sy][lv],
                         mec=CC[sy], ls="none")
            S.tag(h, f"{sy}-{lv}")
        ylab.append((yy, "US masks" if sy == "US" else "Brazil masks"))
    for lv, L in (("us_like", "MaskDoseUsLike"), ("intermediate", "MaskDoseInter"), ("china_like", "MaskDoseChinaLike")):
        axb.axvline(v(L), color=S.REF_COLOR, lw=0.4, ls=(0, (2, 1.5)), zorder=0)
    axb.text(v("MaskDoseUsLike") * 0.9, 4.75, "US-like", fontsize=5, ha="right", va="bottom")
    axb.text(v("MaskDoseInter"), 4.75, "interm.", fontsize=5, ha="center", va="bottom")
    axb.text(v("MaskDoseChinaLike") * 1.12, 4.75, "China-like", fontsize=5, ha="left", va="bottom")
    axb.set_xscale("log")
    axb.set_xlim(1e-4, 3)
    axb.set_ylim(0.3, 5.2)
    ylab.sort()
    axb.set_yticks([y for y, _ in ylab])
    axb.set_yticklabels([l for _, l in ylab], fontsize=5)
    axb.tick_params(axis="y", length=0)
    axb.spines["left"].set_visible(False)
    axb.set_xlabel(r"Event dose $(n_E+n_L)/n_S$")
    axc = S.ax_mm(fig, 141, 120, 38, 32)
    for i, (sy, _n) in enumerate(MASK_SYS):
        for j, sc in enumerate(("active", "null")):
            q = p[(p.system == sy) & (p.stratum == sc)].sort_values("T1_obs", key=np.abs)
            x0 = i * 2.6 + j
            off = np.linspace(-0.28, 0.28, len(q))
            col = CC[sy] if sc == "active" else S.tint(CC[sy], 0.5)
            h = axc.plot(x0 + off, q.T1_obs.abs(), CM[sy], ms=1.4, mew=0.2, mfc=col, mec=CC[sy], ls="none", alpha=0.9)
            S.tag(h, f"null-{sy}-{sc}")
            med = q[q.point_ok].T1_obs.abs().median()
            hm = axc.plot([x0 - 0.36, x0 + 0.36], [med, med], color="black", lw=0.8)
            S.tag(hm, "median")
        P_ = "Us" if sy == "US" else "Br"
        hb_ = axc.plot([i * 2.6 + 0.6, i * 2.6 + 1.4], [v(f"Mask{P_}NullBar")] * 2, color="black", lw=0.6,
                       ls=(0, (2, 1)))
        S.tag(hb_, "bar")
    axc.set_yscale("log")
    axc.set_ylim(1e-4, 10 ** 0.5)
    axc.set_xticks([0, 1, 2.6, 3.6])
    axc.set_xticklabels(["active", "null", "active", "null"], fontsize=5)
    axc.text(0.5, -0.11, "US", fontsize=5.5, ha="center", va="top", transform=axc.get_xaxis_transform(),
             color=CC["US"], fontweight="bold")
    axc.text(3.1, -0.11, "Brazil", fontsize=5.5, ha="center", va="top", transform=axc.get_xaxis_transform(),
             color="#3E7A30", fontweight="bold")
    axc.set_xlim(-0.55, 4.15)
    axc.set_ylabel(r"$|O|$ per mask")
    lim3 = {"US": (-1.0, 1.25), "BR": (-1.6, 1.6)}
    axd = {}
    for i, (sy, name) in enumerate(MASK_SYS):
        ax = S.ax_mm(fig, 12 + i * 47, 66, 36, 36)
        axd[sy] = ax
        lo, hi = lim3[sy]
        q = act[act.system == sy]
        ax.plot([lo, hi], [lo, hi], color=S.V0_COLOR, lw=0.5, ls=(0, (3, 1.5)), zorder=0)
        ax.axhline(0, color=S.REF_COLOR, lw=0.4, zorder=0)
        h = ax.plot(q.T3, q.T2, CM[sy], ms=2.4, mew=0.3, mfc=CC[sy], mec=CC[sy], ls="none", zorder=2)
        S.tag(h, f"v1-{sy}")
        hn = ax.plot(q.T3, q.T_naive, "D", ms=1.6, mew=0.4, mfc="none", mec="#5A5A5A", ls="none", zorder=3)
        S.tag(hn, "naive")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(r"Uncorrected shift $T_3=\xi^*_{V0}-\xi^*_F$")
        ax.set_ylabel(r"Residual after correction" if i == 0 else "")
        P_ = "Us" if sy == "US" else "Br"
        ax.text(0.97, 0.03, f"med.|$T_2$|/med.|$T_3$| = {ftxt(f'Mask{P_}Ratio')}\n"
                            f"V1 beats naive: {ftxt(f'Mask{P_}Naive')}", transform=ax.transAxes, fontsize=5,
                ha="right", va="bottom", linespacing=1.15, bbox=dict(fc="white", ec="none", pad=0.3))
        ax.text(0.03, 0.97, "US (pilot-informed)" if sy == "US" else "Brazil", transform=ax.transAxes,
                fontsize=5.5, ha="left", va="top", color=CC[sy] if sy == "US" else "#3E7A30", fontweight="bold")
    axs = S.ax_mm(fig, 108, 66, 17, 36)
    for i, (sy, _n) in enumerate(MASK_SYS):
        q = act[(act.system == sy) & act.gate_T3].sort_values("T3")
        dlt = (q.T2.abs() - q.T_naive.abs()).to_numpy()
        jit = ((np.arange(len(q)) % 11) - 5) * 0.045
        h = axs.plot(i + jit, dlt, CM[sy], ms=1.4, mew=0.2, mfc=CC[sy], mec=CC[sy], ls="none", alpha=0.85)
        S.tag(h, f"strip-{sy}")
    axs.axhline(0, color="black", lw=0.5)
    axs.set_xlim(-0.5, 1.5)
    axs.set_ylim(-0.075, 0.075)
    axs.set_xticks([0, 1])
    axs.set_xticklabels(["US", "Brazil"], fontsize=5)
    axs.set_ylabel(r"$|T_2|-|T_{\mathrm{naive}}|$", labelpad=1)
    axs.text(0.5, 0.02, "V1 closer", transform=axs.transAxes, fontsize=5, ha="center", va="bottom")
    axe = S.ax_mm(fig, 143, 66, 36, 36)
    lo, hi = -1.0, 0.25
    axe.plot([lo, hi], [lo, hi], color="black", lw=0.5, ls=(0, (1, 1)), zorder=0)
    axe.axhline(0, color=S.REF_COLOR, lw=0.5, zorder=0)
    for sy, _n in MASK_SYS:
        q = act[act.system == sy]
        h = axe.plot(q.T3_selection, q.T2, CM[sy], ms=2.0, mew=0.3, mfc=CC[sy], mec=CC[sy], ls="none", alpha=0.85)
        S.tag(h, f"sel-{sy}")
    axe.set_xlim(lo, hi)
    axe.set_ylim(lo, hi)
    axe.set_xlabel(r"Selection $\xi^*_{F_S}-\xi^*_F$")
    axe.set_ylabel(r"$T_2=\xi^*_{V1}-\xi^*_F$", labelpad=1)
    axe.text(-0.97, -0.62, r"$V1=F_S$", fontsize=5, rotation=45, ha="left", va="bottom")
    axe.text(-0.97, 0.03, r"$V1=F$", fontsize=5, ha="left", va="bottom")
    hs = [Line2D([], [], marker="s", ls="none", ms=2.4, mfc=CC["US"], mec=CC["US"], label=r"V1: $T_2=\xi^*_{V1}-\xi^*_F$"),
          Line2D([], [], marker="D", ls="none", ms=1.6, mew=0.4, mfc="none", mec="#5A5A5A",
                 label="naive: event pairs dropped"),
          Line2D([], [], color=S.V0_COLOR, lw=0.5, ls=(0, (3, 1.5)), label="V0 (no correction)"),
          Line2D([], [], marker="s", ls="none", ms=2.4, mfc=CC["US"], mec=CC["US"], label="US"),
          Line2D([], [], marker="^", ls="none", ms=2.4, mfc=CC["BR"], mec=CC["BR"], label="Brazil")]
    fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(20 / 183, 104.5 / 162), ncol=5, fontsize=5,
               columnspacing=1.2, handletextpad=0.4, borderaxespad=0, frameon=False)
    axf = S.ax_mm(fig, 12, 12, 46, 36)
    xs = []
    for i, (sy, _n) in enumerate(MASK_SYS):
        q = act[act.system == sy]
        for j, (lv, ll) in enumerate(MASK_DOSES):
            r = q[q.fraction_level == lv]
            x = i * 3.8 + j
            xs.append((x, ftxt({"us_like": "MaskDoseUsLike", "intermediate": "MaskDoseInter",
                                "china_like": "MaskDoseChinaLike"}[lv])))
            av = r.point_ok.mean()
            hb = axf.bar([x], [av], width=0.72, color=MASK_DOSE_COLOR[sy][lv], ec=CC[sy], lw=0.5)
            S.tag(hb.patches, "bars")
            hc = axf.plot([x], [r.covered.mean()], marker="_", ms=6, mew=1.1, color="black", ls="none", zorder=4)
            S.tag(hc, "coverage")
        axf.text(i * 3.8 + 1, -0.12, "US" if sy == "US" else "Brazil", fontsize=5.5, ha="center", va="top",
                 color=CC[sy] if sy == "US" else "#3E7A30", fontweight="bold",
                 transform=axf.get_xaxis_transform())
    axf.axhline(v("MaskCritAvail"), color="black", lw=0.5, ls=(0, (3, 1.5)), zorder=3)
    axf.text(2.5, v("MaskCritAvail") + 0.02, "availability\ncriterion", fontsize=5, ha="center", va="bottom")
    axf.set_xticks([x for x, _ in xs])
    axf.set_xticklabels([l for _, l in xs], fontsize=5)
    axf.set_xlabel("Event dose of the mask", labelpad=11)
    axf.set_ylim(0, 1.1)
    axf.set_xlim(-0.6, 7.4)
    axf.set_yticks([0, 0.5, 1.0])
    axf.set_ylabel("Share of active masks")
    lims = {"US": (-0.95, 1.25), "BR": (-1.65, 2.05)}
    axg = {}
    for i, (sy, name) in enumerate(MASK_SYS):
        ax = S.ax_mm(fig, 75 + i * 54, 12, 36, 36)
        axg[sy] = ax
        lo, hi = lims[sy]
        q = act[act.system == sy]
        ax.plot([lo, hi], [lo, hi], color=S.REF_COLOR, lw=0.5, zorder=0)
        ax.axhline(0, color=S.REF_COLOR, lw=0.3, zorder=0)
        ax.axvline(0, color=S.REF_COLOR, lw=0.3, zorder=0)
        ok = q[q.point_ok]
        hz = ax.plot(ok.zeroth_order, ok.T1_obs, "x", ms=2.2, mew=0.5, color=ZEROTH_COLOR, ls="none", zorder=2)
        S.tag(hz, "zeroth")
        for lv, _l in MASK_DOSES:
            r = ok[ok.fraction_level == lv]
            h = ax.plot(r.pred_point, r.T1_obs, CM[sy], ms=2.6, mew=0.4, mfc=MASK_DOSE_COLOR[sy][lv], mec=CC[sy],
                        ls="none", zorder=3)
            S.tag(h, f"{sy}-{lv}")
            r0 = q[~q.point_ok & (q.fraction_level == lv)]
            if len(r0):
                hr = ax.plot(np.full(len(r0), lo + 0.02 * (hi - lo)), r0.T1_obs, "_", ms=4.0, mew=0.7,
                             color=MASK_DOSE_COLOR[sy][lv] if lv != "us_like" else S.tint(CC[sy], 0.45),
                             ls="none", zorder=3, clip_on=False)
                S.tag(hr, f"rug-{sy}-{lv}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(r"Projected $P$ (frozen guarded projection)")
        ax.set_ylabel(r"Observed $O=\xi^*_{V0}-\xi^*_{V1}$")
        P_ = "Us" if sy == "US" else "Br"
        txt = (f"computable {ftxt(f'Mask{P_}PointN')}/{ftxt(f'Mask{P_}Active')}\n"
               f"slope {ftxt(f'Mask{P_}Slope')} [{ftxt(f'Mask{P_}SlopeLo')}, {ftxt(f'Mask{P_}SlopeHi')}]\n"
               f"zeroth order: slope {ftxt(f'Mask{P_}SlopeZeroth')}")
        ax.text(0.97, 0.03, txt, transform=ax.transAxes, fontsize=5, ha="right", va="bottom", linespacing=1.15)
        ax.text(0.06, 0.97, "US (pilot-informed)" if sy == "US" else "Brazil (not used in the pilot)",
                transform=ax.transAxes, fontsize=5.5, ha="left", va="top",
                color=CC[sy] if sy == "US" else "#3E7A30", fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=0.4))
    hs = [mpatches.Patch(fc="#BDBDBD", ec="#333333", lw=0.5, label="computable (f)"),
          Line2D([], [], marker="_", ms=6, mew=1.1, color="black", ls="none", label="$O$ in envelope (f)")]
    hs += [Line2D([], [], marker="s", ls="none", ms=2.6, mfc=MASK_DOSE_COLOR["US"][lv], mec=CC["US"], mew=0.4,
                  label=ll) for lv, ll in MASK_DOSES]
    hs += [Line2D([], [], marker="x", ls="none", ms=2.2, mew=0.5, color=ZEROTH_COLOR, label=r"zeroth order $\bar M/|a_1|$"),
           Line2D([], [], marker="_", ls="none", ms=4.0, mew=0.7, color="#555555", label="refused (tick at $O$)")]
    fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(20 / 183, 52.0 / 162), ncol=7, fontsize=5,
               columnspacing=1.0, handletextpad=0.4, borderaxespad=0, frameon=False)
    S.panel_label(fig, axa, "a", dx_mm=-9, dy_mm=-3.0)
    S.panel_label(fig, axb, "b", dx_mm=-26, dy_mm=3.0)
    S.panel_label(fig, axc, "c", dx_mm=-10, dy_mm=3.0)
    S.panel_label(fig, axd["US"], "d", dx_mm=-11, dy_mm=2.0)
    S.panel_label(fig, axe, "e", dx_mm=-11, dy_mm=2.0)
    S.panel_label(fig, axf, "f", dx_mm=-11, dy_mm=2.0)
    S.panel_label(fig, axg["US"], "g", dx_mm=-11, dy_mm=2.0)
    S.save(fig, OUTDIR, "fig5_masking")


_MASK_DOSE_SHORT = {"us_like": "US-like", "intermediate": "interm.", "china_like": "China-like"}


def figS7():
    """Controlled masking, every cell: O and P per mask; T3, T2 and the naive comparator per mask."""
    p = mask_points()
    order = {"active": 0, "balanced": 1, "null": 2}
    cells = (p[p.system == "US"][["cell", "stratum"]].drop_duplicates()
             .assign(o=lambda d: d.stratum.map(order)).sort_values(["o", "cell"]))
    if len(cells) != 36:
        raise SourceError("figS7: need 36 cells")
    ypos = {c: 35 - i for i, c in enumerate(cells.cell)}
    fig = S.figure(183, 168)
    x0s = [37, 74, 111, 148]
    axes = [S.ax_mm(fig, x, 12, 32, 146) for x in x0s]
    for k, (sy, _n) in enumerate(MASK_SYS):
        q = p[p.system == sy]
        axo, axr = axes[2 * k], axes[2 * k + 1]
        y = q.cell.map(ypos).to_numpy() + ((q.rep.to_numpy() - 4.5) * 0.03)
        ok = q.point_ok.to_numpy()
        h = axo.plot(q.T1_obs, y, CM[sy], ms=1.6, mew=0.2, mfc=CC[sy], mec=CC[sy], ls="none")
        S.tag(h, "obs")
        h = axo.plot(q.pred_point[ok], y[ok] - 0.28, "|", ms=2.2, mew=0.5, color="black", ls="none")
        S.tag(h, "pred")
        h = axr.plot(q.T3, y, "o", ms=1.6, mew=0.3, mfc="white", mec=S.V0_COLOR, ls="none")
        S.tag(h, "v0")
        h = axr.plot(q.T_naive, y - 0.28, "D", ms=1.3, mew=0.3, mfc="white", mec="#555555", ls="none")
        S.tag(h, "naive")
        h = axr.plot(q.T2, y + 0.28, CM[sy], ms=1.6, mew=0.2, mfc=CC[sy], mec=CC[sy], ls="none")
        S.tag(h, "v1")
        for ax, lab in ((axo, r"$O$ (marker), $P$ (bar)"), (axr, r"$T_3$ (open), $T_2$ (filled, above), naive (below)")):
            ax.axvline(0, color=S.REF_COLOR, lw=0.4, zorder=0)
            for yb in (23.5, 11.5):
                ax.axhline(yb, color="#9A9A9A", lw=0.4)
            ax.set_ylim(-0.7, 35.7)
            ax.set_yticks(range(36))
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
            ax.set_xlabel(lab, fontsize=5)
        fig.text((x0s[2 * k] + 32 + 2.5) / 183, (12 + 146 + 4.5) / 168,
                 "US (pilot-informed)" if sy == "US" else "Brazil (not used in the pilot)",
                 fontsize=5.5, ha="center", va="bottom", fontweight="bold", color=CC[sy] if sy == "US" else "#3E7A30")
    labs = []
    for c in reversed(list(cells.cell)):
        lv, pos, org, tim = c.split("|")
        labs.append(f"{_MASK_DOSE_SHORT[lv]}, {pos}, {org}, {tim}")
    axes[0].set_yticklabels(labs, fontsize=5)
    for yb, t in ((29.5, "active"), (17.5, "balanced"), (5.5, "null")):
        fig.text(1.5 / 183, (12 + 146 * (yb + 0.7) / 36.4) / 168, t, rotation=90, fontsize=5.5, va="center",
                 ha="left", fontweight="bold")
    for ax, l in zip(axes, "abcd"):
        S.panel_label(fig, ax, l, dx_mm=-3, dy_mm=1.0)
    S.save(fig, OUTDIR, "figS7_mask_cells")


def figS8():
    """Controlled masking: every guard of the envelope (active masks)."""
    p = mask_points()
    act = p[p.stratum == "active"]
    pj = mask_tab("k4_projections.csv").merge(act[["mask_id", "system", "T1_obs", "fraction_level"]],
                                              on="mask_id", how="inner")
    guards = pj[["support_lo_pct", "support_hi_pct", "floor_factor", "registered_guard"]].drop_duplicates()
    guards = guards.sort_values(["support_lo_pct", "floor_factor"]).reset_index(drop=True)
    if len(guards) != 12:
        raise SourceError("figS8: need 12 guards")
    fig = S.figure(183, 72)
    axa = S.ax_mm(fig, 34, 12, 40, 52)
    axb = S.ax_mm(fig, 84, 12, 40, 52)
    axc = S.ax_mm(fig, 138, 12, 41, 52)
    labs = []
    for i, g in guards.iterrows():
        yy = 11 - i
        reg = str(g.registered_guard) == "True"
        labs.append(f"{fnum(g.support_lo_pct, 1)}–{fnum(g.support_hi_pct, 1)}, floor ×{fnum(g.floor_factor, 1)}"
                    + (" (point)" if reg else ""))
        for k, (sy, _n) in enumerate(MASK_SYS):
            q = pj[(pj.system == sy) & (pj.support_lo_pct == g.support_lo_pct) & (pj.floor_factor == g.floor_factor)]
            for lv, _l in MASK_DOSES:
                r = q[q.fraction_level == lv]
                dy = (k - 0.5) * 0.36 + (["us_like", "intermediate", "china_like"].index(lv) - 1) * 0.1
                ok = r[r.status == "ok"]
                h = axa.plot([len(ok) / len(r)], [yy + dy], CM[sy], ms=2.2, mew=0.3, mfc=MASK_DOSE_COLOR[sy][lv],
                             mec=CC[sy], ls="none")
                S.tag(h, f"{sy}-{lv}")
                if len(ok):
                    e = float((ok.T1_obs - ok.pred).abs().median() / ok.T1_obs.abs().median())
                    h = axb.plot([e], [yy + dy], CM[sy], ms=2.2, mew=0.3, mfc=MASK_DOSE_COLOR[sy][lv], mec=CC[sy],
                                 ls="none")
                    S.tag(h, f"{sy}-{lv}")
    for ax in (axa, axb):
        ax.set_ylim(-0.7, 11.7)
        ax.set_yticks(range(12))
        ax.set_yticklabels(list(reversed(labs)) if ax is axa else [], fontsize=5)
        ax.tick_params(axis="y", length=0)
        for yb in (2.5, 5.5, 8.5):
            ax.axhline(yb, color="#D0D0D0", lw=0.4, zorder=0)
    axa.set_xlim(-0.03, 1.03)
    axa.set_xlabel("Share of active masks with the guard defined")
    axb.set_xscale("log")
    axb.set_xticks([0.03, 0.1, 0.3, 1.0])
    axb.set_xticklabels(["0.03", "0.1", "0.3", "1"])
    axb.xaxis.set_minor_formatter(NullFormatter())
    axb.axvline(v("MaskCritLevel"), color="black", lw=0.5, ls=(0, (3, 1.5)))
    axb.set_xlabel(r"median $|O-\mathrm{pred}|$ / median $|O|$ where defined")
    axb.text(v("MaskCritLevel") * 1.08, 11.4, "per-dose criterion", fontsize=5, va="center", ha="left")
    u = act[act.system == "US"]
    for n_, r in u.groupby("n_guards_defined"):
        h = axc.plot([n_], [r.covered.mean()], "s", ms=1.5 + 0.12 * len(r), mew=0.3, mfc=S.tint(CC["US"], 0.3),
                     mec=CC["US"], ls="none")
        S.tag(h, "us-cov")
        axc.text(n_, r.covered.mean() + 0.06, fnum(len(r), 0), fontsize=5, ha="center", va="bottom")
    b_ = act[act.system == "BR"]
    h = axc.plot([b_.n_guards_defined.iloc[0]], [b_.covered.mean()], "^", ms=4.0, mew=0.3, mfc=CC["BR"],
                 mec=CC["BR"], ls="none")
    S.tag(h, "br-cov")
    axc.text(b_.n_guards_defined.iloc[0] - 0.2, b_.covered.mean() - 0.04, "Brazil, all " + fnum(len(b_), 0),
             fontsize=5, ha="right", va="top", color="#3E7A30")
    axc.axhline(v("MaskCritCoverage"), color="black", lw=0.5, ls=(0, (3, 1.5)))
    axc.set_ylim(-0.05, 1.18)
    axc.set_xlim(2, 12.8)
    axc.set_xlabel("Guards defined for the mask (of 12)")
    axc.set_ylabel("Envelope coverage of $O$")
    axc.text(2.2, 1.15, "US: number of masks printed", fontsize=5, va="top")
    hs = [Line2D([], [], marker=CM[sy], ls="none", ms=2.2, mew=0.3, mfc=MASK_DOSE_COLOR[sy][lv], mec=CC[sy],
                 label=f"{'US' if sy == 'US' else 'Brazil'}, {_l}") for sy, _n in MASK_SYS for lv, _l in MASK_DOSES]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(34 / 183, 1 - 1 / 72), ncol=6, fontsize=5,
               columnspacing=0.9, handletextpad=0.3)
    for ax, l in ((axa, "a"), (axb, "b"), (axc, "c")):
        S.panel_label(fig, ax, l, dx_mm=-8 if ax is not axa else -33, dy_mm=1.0)
    S.save(fig, OUTDIR, "figS8_mask_guards")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "figS1": figS1, "figS2": figS2,
        "figS3": figS6, "figS4": figS3, "figS5": figS5, "figS6": figS4, "figS7": figS7, "figS8": figS8}


def main(argv):
    names = argv or list(FIGS)
    failed = []
    for n in names:
        try:
            FIGS[n]()
            print(f"{n}: ok")
        except S.FigureLintError as e:
            print(str(e), file=sys.stderr)
            failed.append(n)
    if failed:
        print(f"FAILED lint: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
