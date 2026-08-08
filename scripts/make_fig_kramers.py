"""
Generate fig_kramers.pdf: the dynamical potential V_FP(xi) from the binned
Kramers-Moyal estimates (compute_escape_observables).

Shows V_FP for pre and during only — the post window (2020-2023) is not
identifiable due to selective attrition under yearly cross-sectional
standardisation and is deliberately absent.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_merger import load_panel
from src.saai import compute_saai
from src.fokker_planck import (build_transition_pairs, estimate_kramers_moyal,
                               compute_escape_observables)

COLORS = {"pre": "#4575b4", "during": "#d73027"}
LABELS = {"pre": "Pre (2000--2012)", "during": "During (2013--2019)"}

panel = load_panel("panel_scaling")
sa = compute_saai(panel, y_col="GrossOutputValue",
                  x_cols=["PopNum", "CultivateArea"])

fig, ax = plt.subplots(figsize=(7.5, 5), dpi=300)
for period in ("pre", "during"):
    pairs = build_transition_pairs(sa, period)
    km = estimate_kramers_moyal(pairs, period=period)
    esc = compute_escape_observables(km)
    xi, V = esc["V_fp_xi"], esc["V_fp"]
    ax.plot(xi, V, "-", lw=2, color=COLORS[period],
            label=(f"{LABELS[period].replace('--','–')}:  "
                   f"$\\xi^*={esc['xi_trap']:.2f}$, "
                   f"$\\Delta V={esc['delta_V']:.3f}$, "
                   f"$\\tau={esc['tau_mfpt']:.1f}$ yr"))
    ax.plot(esc["xi_trap"],
            float(__import__('numpy').interp(esc["xi_trap"], xi, V)),
            "o", ms=7, color=COLORS[period])

ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.7)
ax.text(0.03, ax.get_ylim()[1]*0.95, "escape target $\\xi=0$",
        fontsize=9, color="gray", va="top")
ax.set_xlabel(r"SAAI ($\xi$)")
ax.set_ylabel(r"$V_{\mathrm{FP}}(\xi) = -\int A/B\,\mathrm{d}\xi'$")
ax.set_title("Dynamical potential across policy regimes (binned KM)")
ax.legend(fontsize=9, loc="upper center")
ax.grid(alpha=0.25)
fig.tight_layout()
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/fig_kramers.pdf")
print("wrote results/figures/fig_kramers.pdf")
