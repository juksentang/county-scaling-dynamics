"""A synthetic open panel on the real county / province / window structure of
data_clean/k1_mask.parquet.  Per window (pre 2000–2012, during 2013–2019) the
latent county state x follows the period's true Euler chain in km_mle's form

    x_{t+1} = x_t + A(x_t) + sqrt(B(x_t)) · η_{i,t}

with A cubic and log B quadratic, started from a 60-year burn-in that runs
the same chain (so the window starts at the chain's law, province structure
included).

Outputs are generated as log Y = c + β·log X + λ·y (y = x for 'shock',
y = x + u for 'persistent'; β, c, X moments and λ = the yearly SAAI residual
s.d. from src/k1_sim_calibration.json) and fed to the real compute_saai
('saai' pipeline); the 'direct' pipeline standardises y itself.

Exit mechanisms (presence masks): (a) none; (b) MCAR 10 %/yr; (c) logit in the
current ξ, 10 %/yr, slope set by the 2010→11 leaver/stayer gap; (d) the
2016→17 block exit (192 of the 385 counties of the 2016 cross-section, which
is Sichuan + Gansu + 2 of Shaanxi); (e) replay of the mask.  δ_b is set so
that the expected leaver mean of ξ at the break equals the observed (multi)
leaver mean; target, achieved value, residual, convergence and n_leave of
every break are recorded, and the aggregator refuses unconverged breaks.
"""
from __future__ import annotations

def public_guard_skipped(value=None, *args, **kwargs):
    """Integrity guard that checks the authors' private version-controlled record:
    disabled in the public copy. Returns `value` unchanged."""
    return value


import functools
import hashlib
import inspect
import json
import os

import numpy as np
import pandas as pd
from scipy import optimize, special

from src import k1_contract as C
from src.config import FP_PERIODS
from src.k1_contract import PERIOD_CONTRASTS, QUANTITIES, TRUE_PARAM_KEYS, X_UNITS
from src.km_mle import fit_mle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL_PATH = os.path.join(ROOT, "src", "k1_sim_calibration.json")

PERIODS = ("pre", "during")
WINDOWS = {p: tuple(FP_PERIODS[p]) for p in PERIODS}
WINDOWS_LABEL = ";".join(f"{p}:{a}-{b}" for p, (a, b) in WINDOWS.items())
SEED_BASE = 20261021
R_REGISTERED = 500
B_REGISTERED = 200
N_S0 = 20_000
EXIT_RATE = 0.10
BURN_IN = 60
CI_NAN_RULES = {"tol0": 0.0, "tol5": 0.05, "literal": None}
PRIMARY_CI_RULE = "tol0"
CI_TAIL = 0.025
BOOT_STREAM = 4
BOOTSTRAP_SCHEME = ("window-stratified province block (provinces drawn with replacement "
                    "within each window), full pipeline re-run per draw (scaling "
                    "regression, yearly standardisation, V1 restandardisation); amendment 7 (S2)")
QUANTILE_METHOD = "symmetric order statistics, k = ceil(0.025 n)"
OBS_RULES = ("limit",)
N_DESIGN_SEEDS = 16
SMALL_N_LEAVE = 20
OFFSET_TOL = 0.01
REFIT_INFORMATIVE = True
PIPELINES = ("saai", "direct")
SAAI_SPEC = dict(y_col="GrossOutputValue", x_cols=["PopNum", "CultivateArea"])
SAAI_SPEC_LABEL = {"saai": "multi", "direct": "direct_x"}
MECHANISMS = ("a", "b", "c", "d", "e")
RE_PLACEMENTS = ("shock", "persistent")
GATED_RE_PLACEMENT, CONTROL_RE_PLACEMENT = "shock", "persistent"
EXIT_DATING = {"a": "none", "b": "exogenous", "c": "info_t", "d": "info_t",
               "e": "info_t"}
PROCESS_KEY = {"i_ou": "ou", "ii_cubic": "cubic", "iii_mult": "cubic_mult",
               "iv_bistable_0": "cubic_bistable_0", "iv_bistable_1": "cubic_bistable_1",
               "iv_bistable_2": "cubic_bistable_2", "iv_bistable_4": "cubic_bistable_4",
               "v_posctrl": "cubic_shift"}
K1A_CELLS = (("i_ou", "d"), ("i_ou", "e"), ("ii_cubic", "d"), ("ii_cubic", "e"))
COMPAT_TOL = {"edge_mass": C.EDGE_MASS_TOL, "mean_sd": 0.02, "sd_rel": 0.02}


@functools.lru_cache(maxsize=1)
def calibration() -> dict:
    with open(CAL_PATH) as f:
        return json.load(f)


def _w_shock() -> float:
    return float(calibration()["province_dxi_icc_mean_2000_2018"])


def _rho_level() -> float:
    return float(calibration()["province_share_mean_2000_2019"])


def _poly_A(p, x):
    return p[0] + p[1] * x + p[2] * x**2 + p[3] * x**3


def _poly_dA(p, x):
    return p[1] + 2 * p[2] * x + 3 * p[3] * x**2


def _poly_logB(p, x):
    return np.clip(p[4] + p[5] * x + p[6] * x**2, -20.0, 20.0)


class TrueProcess:
    """A(x), B(x) of one period in latent units.

    params : 7-vector (a0..a3, c0..c2) as in km_mle.
    guard  : None, or (lo, hi): inside [lo, hi] A and log B are the
        polynomials; outside, log B is frozen at the edge value and A is
        continued linearly with slope min(A'(edge), slope_floor) (< 0), so the
        chain cannot escape through an unstable outer root of a fitted cubic or
        an exploding exp(c2·x²).  The guard edges are the 1–99 % support of the
        real V1 fit, i.e. the polynomials are used exactly where they were
        estimated.
    """

    def __init__(self, params, guard=None, slope_floor=None, label=""):
        self.params = np.asarray(params, dtype=float)
        self.guard = None if guard is None else (float(guard[0]), float(guard[1]))
        self.label = label
        self.slope_floor = None
        if self.guard is not None:
            lo, hi = self.guard
            fl = -abs(slope_floor) if slope_floor is not None else -0.05
            self.slope_floor = float(fl)
            self._s_lo = min(_poly_dA(self.params, lo), fl)
            self._s_hi = min(_poly_dA(self.params, hi), fl)

    def as_dict(self) -> dict:
        d = dict(zip(TRUE_PARAM_KEYS, map(float, self.params)))
        if self.guard is None:
            d.update(dict.fromkeys(C.GUARD_KEYS))
        else:
            d.update(guard_lo=self.guard[0], guard_hi=self.guard[1],
                     slope_floor=self.slope_floor)
        return d

    def A(self, x):
        x = np.asarray(x, dtype=float)
        if self.guard is None:
            return _poly_A(self.params, x)
        lo, hi = self.guard
        a = _poly_A(self.params, np.clip(x, lo, hi))
        return a + np.where(x > hi, self._s_hi * (x - hi), 0.0) \
                 + np.where(x < lo, self._s_lo * (x - lo), 0.0)

    def logB(self, x):
        x = np.asarray(x, dtype=float)
        if self.guard is not None:
            x = np.clip(x, *self.guard)
        return _poly_logB(self.params, x)

    def B(self, x):
        return np.exp(self.logB(x))

    def describe(self) -> dict:
        return {"label": self.label, "params": [float(v) for v in self.params],
                "guard": self.guard}


def _fit(period, form):
    return np.array(calibration()["v1_mle_fits"][period][form], dtype=float)


def _support(period):
    return tuple(calibration()["v1_mle_fits"][period]["support_p01_p99"])


def _ou_a1(period):
    return float(_fit(period, "ou")[1])


def bistable_params(period: str, two_dv: float, alpha: float | None = None) -> np.ndarray:
    """(iv) symmetric family A = r x − α x³, constant B (anchors: B and |a1|
    of the period's V1 OU fit).  r = |a1|/2 (well curvature −2r = a1) and
    analytically 2ΔV = r²/(2αB) for two_dv > 0; two_dv = 0 is the monostable
    member r = −|a1| with the 2ΔV = 1 value of α."""
    a1 = abs(_ou_a1(period))
    B = float(np.exp(_fit(period, "ou")[4]))
    r = a1 / 2.0
    if two_dv > 0:
        al = r**2 / (2.0 * B * two_dv) if alpha is None else alpha
        return np.array([0.0, r, 0.0, -al, np.log(B), 0.0, 0.0])
    al = r**2 / (2.0 * B * 1.0) if alpha is None else alpha
    return np.array([0.0, -a1, 0.0, -al, np.log(B), 0.0, 0.0])


def posctrl_during_params(lam: float, beta: float) -> np.ndarray:
    """(v) during shape: A = −λx + βx² − γx³ with γ = β²/(2λ) (so that
    β² = 2λγ < 4λγ: a single root, globally monostable, no guard needed) and
    constant B = 2λ (stationary s.d. ≈ 1).  λ sets the relaxation time and β
    the asymmetry that puts the trap below the stationary mean."""
    gam = beta**2 / (2 * lam) if beta != 0 else 0.0
    return np.array([0.0, -lam, beta, -gam, np.log(2 * lam), 0.0, 0.0])


PROCESS_NAMES = ("i_ou", "ii_cubic", "iii_mult", "iv_bistable_0", "iv_bistable_1",
                 "iv_bistable_2", "iv_bistable_4", "v_posctrl")


def registered_process(name: str, period: str) -> TrueProcess:
    """The (iv) member / (v) 'during' chain exactly as REGISTERED in
    src.k1_contract.DESIGN_PROCESS_FILE (contract 8): the simulator simulates
    the committed constants, never a re-calibration at run time, so the
    simulated chain and k1_contract.registered_true_params cannot drift
    apart.  Refuses (ValueError) while the member is not registered."""
    tp = C.registered_true_params(PROCESS_KEY[name], period)
    theta = np.array([tp[k] for k in TRUE_PARAM_KEYS], dtype=float)
    g = None if tp["guard_lo"] is None else (tp["guard_lo"], tp["guard_hi"])
    proc = TrueProcess(theta, g, tp["slope_floor"],
                       label=f"{name} {period} (registered, {C.DESIGN_PROCESS_FILE})")
    if proc.as_dict() != tp:
        raise ValueError(f"{name} {period}: TrueProcess does not reproduce the registered "
                         "true_params")
    return proc


def process(name: str, period: str, design: dict | None = None) -> TrueProcess:
    """True process of a registered cell for one period.

    (iv) and (v) 'during' (contract 8): with design=None the REGISTERED chain
    (registered_process); an explicit design (the output of
    design_constants) is used only by the authors' registration script (not part of this archive) and the design tests, which build the
    chain the registry is written from.
    """
    if design is None and (name.startswith("iv_bistable_")
                           or (name == "v_posctrl" and period == "during")):
        return registered_process(name, period)
    guard, floor = _support(period), _ou_a1(period)
    if name == "i_ou":
        return TrueProcess(_fit(period, "ou"), label="OU (V1 fit)")
    if name == "ii_cubic":
        return TrueProcess(_fit(period, "cubic_constB"), guard, floor,
                           label="monostable cubic, constant B (V1 fit)")
    if name == "iii_mult":
        return TrueProcess(_fit(period, "full"), guard, floor,
                           label="cubic + multiplicative noise (V1 full fit)")
    if name.startswith("iv_bistable_"):
        k = float(name.rsplit("_", 1)[1])
        al = None if design is None else design[f"alpha_{period}"]
        return TrueProcess(bistable_params(period, k, al), label=f"symmetric cubic, 2dV={k:g}")
    if name == "v_posctrl":
        g, fl = _support("pre"), _ou_a1("pre")
        if period == "pre":
            return TrueProcess(_fit("pre", "cubic_constB"), g, fl, label="pos. control pre = (ii) pre")
        if design is None:
            raise ValueError("v_posctrl during needs the calibrated design (lam, beta)")
        return TrueProcess(posctrl_during_params(design["lam"], design["beta"]),
                           label="pos. control during (asymmetric monostable cubic)")
    raise KeyError(name)


def mask_content_sha256(m: pd.DataFrame) -> str:
    a = (m[["CountyCode", "Year", "ProvinceCode", "available"]]
         .astype("int64").sort_values(["CountyCode", "Year"]).to_numpy())
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


MASK_CONTENT_SHA256 = "32adc7ead9a2602e652949d43d18f5f6c8b4fd7b091b46c6bd63dfeb08c4a6b8"


def mask_provenance(m: pd.DataFrame) -> dict:
    return dict(m.attrs.get("k1_mask") or {"source": "in-memory",
                                           "file_sha256": None,
                                           "content_sha256": mask_content_sha256(m)})


def load_mask(path: str | None = None, allow_unpinned: bool = False) -> pd.DataFrame:
    cands = [path] if path else [os.environ.get("K1_MASK")] + [
        os.path.join(ROOT, "data_clean", f) for f in
        ("k1_mask.parquet", "k1_mask.csv.gz", "k1_mask.csv", "k1_mask.npz")]
    for c in cands:
        if not c or not os.path.exists(c):
            continue
        if c.endswith(".parquet"):
            m = pd.read_parquet(c)
        elif c.endswith(".npz"):
            z = np.load(c)
            m = pd.DataFrame({k: z[k] for k in ("CountyCode", "Year", "ProvinceCode", "available")})
        else:
            m = pd.read_csv(c)
        m = m.astype({"CountyCode": "int64", "Year": "int64", "ProvinceCode": "int64",
                      "available": "bool"})
        m = m[["CountyCode", "Year", "ProvinceCode", "available"]].reset_index(drop=True)
        with open(c, "rb") as f:
            file_sha = hashlib.sha256(f.read()).hexdigest()
        content = mask_content_sha256(m)
        if content != MASK_CONTENT_SHA256 and not allow_unpinned:
            raise ValueError(f"presence mask {c} has content sha256 {content[:16]}…, not the "
                             f"pinned {MASK_CONTENT_SHA256[:16]}… (MASK_CONTENT_SHA256)")
        m.attrs["k1_mask"] = {"source": os.path.basename(c), "file_sha256": file_sha,
                              "content_sha256": content}
        return m
    raise FileNotFoundError("k1 mask not found (data_clean/k1_mask.{parquet,csv.gz,npz} or $K1_MASK)")


class Window:
    """County set, province index and presence for one window of one cell."""

    def __init__(self, period, codes, prov, presence, offsets=None, targets=None):
        self.period = period
        self.years = np.arange(WINDOWS[period][0], WINDOWS[period][1] + 1)
        self.codes = np.asarray(codes, dtype=np.int64)
        self.prov = np.asarray(prov, dtype=np.int64)
        _, self.prov_idx = np.unique(self.prov, return_inverse=True)
        self.presence = None if presence is None else np.asarray(presence, dtype=bool)
        self.offsets = offsets or []
        self.targets = targets or {}

    @property
    def n(self):
        return len(self.codes)

    @property
    def n_prov(self):
        return int(self.prov_idx.max()) + 1 if self.n else 0


def _mask_matrix(mask: pd.DataFrame, period: str):
    a, b = WINDOWS[period]
    m = mask[(mask["Year"] >= a) & (mask["Year"] <= b) & mask["available"]]
    codes = np.sort(m["CountyCode"].unique())
    years = np.arange(a, b + 1)
    P = np.zeros((len(codes), len(years)), dtype=bool)
    ci = np.searchsorted(codes, m["CountyCode"].to_numpy())
    P[ci, m["Year"].to_numpy() - a] = True
    prov = (mask.drop_duplicates("CountyCode").set_index("CountyCode")
            .loc[codes, "ProvinceCode"].to_numpy())
    return codes, prov, P


def _largest_remainder(weights: np.ndarray, n: int) -> np.ndarray:
    q = weights / weights.sum() * n
    k = np.floor(q).astype(int)
    rem = n - k.sum()
    k[np.argsort(-(q - k), kind="stable")[:rem]] += 1
    return k


def open_base(mask: pd.DataFrame, period: str, n: int | None = None) -> Window:
    """Base county set of mechanisms (a)–(c): n synthetic counties allocated to
    the window's provinces in proportion to their available county-years.
    Default n = round(observed pairs / transitions) so that (a) reproduces the
    registered pair counts (5,860 / 1,734 → 488 × 12, 289 × 6)."""
    codes, prov, P = _mask_matrix(mask, period)
    if n is None:
        n = int(round((P[:, :-1] & P[:, 1:]).sum() / (P.shape[1] - 1)))
    up, inv = np.unique(prov, return_inverse=True)
    w = np.bincount(inv, weights=P.sum(1))
    k = _largest_remainder(w, n)
    pv = np.repeat(up, k)
    syn = 900_000_000 + pv * 100_000 + np.concatenate([np.arange(c) for c in k])
    return Window(period, syn, pv, np.ones((n, P.shape[1]), dtype=bool))


def observed_pairs(mask: pd.DataFrame, period: str) -> int:
    _, _, P = _mask_matrix(mask, period)
    return int((P[:, :-1] & P[:, 1:]).sum())


def exit_base_n(mask: pd.DataFrame, period: str) -> int:
    T = WINDOWS[period][1] - WINDOWS[period][0] + 1
    surv = sum((1.0 - EXIT_RATE) ** k for k in range(1, T))
    return int(round(observed_pairs(mask, period) / surv))


def block_d_window(mask: pd.DataFrame) -> Window:
    """(d): the 2016 cross-section of the during window, all present 2013–2016;
    the counties absent in 2017 leave at 2016→17 (192 of 385) and the rest
    stay to 2019.  One dated block shock (δ_d) at the window start for the
    leaving block."""
    codes, prov, P = _mask_matrix(mask, "during")
    j16 = 2016 - WINDOWS["during"][0]
    sel = P[:, j16]
    codes, prov, P = codes[sel], prov[sel], P[sel]
    leave = ~P[:, j16 + 1]
    pres = np.ones_like(P)
    pres[leave, j16 + 1:] = False
    tgt = calibration()["leaver_mean_by_transition"]["2016"]
    return Window("during", codes, prov, pres,
                  offsets=[(0, np.where(leave)[0], 2016)], targets={2016: tgt})


def replay_window(mask: pd.DataFrame, period: str, drop_breaks=()) -> Window:
    """(e): the mask itself; one dated block shock per break, applied to that
    break's leavers at the start of the spell that ends in it.
    drop_breaks (break-year sensitivity, report only): break years
    (last year present) whose leavers do NOT leave — they stay present to
    the end of the window, with no dated shock for that break.
    """
    codes, prov, P = _mask_matrix(mask, period)
    P = P.copy()
    for yb in drop_breaks or ():
        j = int(yb) - WINDOWS[period][0]
        if 0 <= j < P.shape[1] - 1:
            lv = np.where(P[:, j] & ~P[:, j + 1])[0]
            P[np.ix_(lv, np.arange(j + 1, P.shape[1]))] = True
    lm = calibration()["leaver_mean_by_transition"]
    offs, tg = [], {}
    for j in range(P.shape[1] - 1):
        lv = np.where(P[:, j] & ~P[:, j + 1])[0]
        if not len(lv):
            continue
        year = int(WINDOWS[period][0] + j)
        if lm.get(str(year)) is None:
            continue
        starts = np.empty(len(lv), dtype=int)
        for k, i in enumerate(lv):
            s = j
            while s > 0 and P[i, s - 1]:
                s -= 1
            starts[k] = s
        for s in np.unique(starts):
            offs.append((int(s), lv[starts == s], year))
        tg[year] = float(lm[str(year)])
    return Window(period, codes, prov, P, offsets=offs, targets=tg)


def null_window(win: Window) -> Window:
    return Window(win.period, win.codes, win.prov, np.ones((win.n, len(win.years)), dtype=bool))


def shocked_null_window(win: Window) -> Window:
    """Dated-shock no-exit cell: the null window of a (d)/(e) window
    WITH its dated block shocks (Window.offsets; δ_b from the K1a cell's
    calibration), every county present every year.  The δ_b of a spell that
    starts after the window start lands on a transition that the exit cell
    does not observe (the county is absent the year before); here that
    transition would be observed, so its pair is dropped (drop_pairs) — the
    county stays in every cross-section.  No targets: nothing is calibrated
    in this cell.
    """
    nw = Window(win.period, win.codes, win.prov, np.ones((win.n, len(win.years)), dtype=bool),
                offsets=list(win.offsets))
    nw.drop_pairs = [(int(s) - 1, np.asarray(idx)) for s, idx, _ in win.offsets if s > 0]
    return nw


def stationary_draw(proc: TrueProcess, size, rng, lo=-12.0, hi=12.0, n=6001):
    """Draws from the Fokker–Planck stationary density p ∝ exp(2∫A/B)/B on a
    grid (only a starting point; the burn-in then relaxes to the chain's law)."""
    g = np.linspace(lo, hi, n)
    f = proc.A(g) / proc.B(g)
    F = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(g))])
    lp = 2 * F - proc.logB(g)
    p = np.exp(lp - lp.max())
    c = np.cumsum(p)
    c /= c[-1]
    return np.interp(rng.random(size), c, g)


def _stepper(proc: TrueProcess, win: Window, placement: str, w: float, shape, rng):
    shock = placement == "shock" and w > 0

    def step(x):
        e = rng.standard_normal(shape)
        if shock:
            z = rng.standard_normal(shape[:-1] + (win.n_prov,))[..., win.prov_idx]
            e = np.sqrt(1.0 - w) * e + np.sqrt(w) * z
        return x + proc.A(x) + np.sqrt(proc.B(x)) * e
    return step


def burn_in(proc: TrueProcess, win: Window, rng, placement: str, w: float | None = None,
            n_rep: int | None = None, burn: int = BURN_IN) -> np.ndarray:
    w = _w_shock() if w is None else w
    shape = (win.n,) if n_rep is None else (n_rep, win.n)
    step = _stepper(proc, win, placement, w, shape, rng)
    x = stationary_draw(proc, shape, rng)
    for _ in range(burn):
        x = step(x)
    return x


def simulate_latent(proc: TrueProcess, win: Window, rng, placement: str = "shock",
                    deltas: dict | None = None, n_rep: int | None = None,
                    burn: int = BURN_IN, w: float | None = None, _burned=None):
    """(X, U): latent x[(rep,) county, year] for one window and the persistent
    observation offset u[(rep,) county] (zeros for 'shock').

    'shock': province-common innovation share w in the burn-in and the window;
    the dated block shocks deltas = {break year: δ} are added to x at the
    spell starts of Window.offsets.  'persistent': no province structure in
    x; u = s_u·g_p + Σ_b δ_b 1[i leaves at b], s_u² = ρ/(1 − ρ)·Var(x_0)."""
    if placement not in RE_PLACEMENTS:
        raise ValueError(f"unknown re_placement {placement!r}")
    w = _w_shock() if w is None else w
    shape = (win.n,) if n_rep is None else (n_rep, win.n)
    step = _stepper(proc, win, placement, w, shape, rng)
    x = burn_in(proc, win, rng, placement, w, n_rep, burn) if _burned is None else _burned.copy()
    U = np.zeros(shape)
    if placement == "persistent":
        rho = _rho_level()
        s_u = np.sqrt(rho / (1.0 - rho)) * x.std(axis=-1, keepdims=True)
        g = rng.standard_normal(shape[:-1] + (win.n_prov,))[..., win.prov_idx]
        U = s_u * g
        for (s, idx, yb) in win.offsets:
            if deltas and deltas.get(yb, 0.0) != 0.0:
                U[..., idx] += deltas[yb]
    T = len(win.years)
    X = np.empty(shape + (T,))
    for t in range(T):
        if t > 0:
            x = step(x)
        if placement == "shock" and deltas:
            for (s, idx, yb) in win.offsets:
                if s == t and deltas.get(yb, 0.0) != 0.0:
                    x[..., idx] += deltas[yb]
        X[..., t] = x
    return X, U


def observed_latent(X, U):
    """y = x + u: the quantity the output equation carries."""
    return X + U[..., None]


def _std_over(X, P):
    """Mean and s.d. (ddof = 0, as compute_saai) of X over present counties,
    per year; X, P shape (..., n, T)."""
    cnt = P.sum(-2)
    mu = np.where(P, X, 0).sum(-2) / np.maximum(cnt, 1)
    var = np.where(P, (X - mu[..., None, :]) ** 2, 0).sum(-2) / np.maximum(cnt, 1)
    return mu, np.sqrt(var)


def province_share(v: np.ndarray, prov_idx: np.ndarray) -> float:
    """Between-province share of the cross-sectional variance of v[n]."""
    n_p = np.bincount(prov_idx)
    m_p = np.bincount(prov_idx, weights=v) / np.maximum(n_p, 1)
    return float(np.var(m_p[prov_idx]) / np.var(v))


def leaver_means(Y, P, win: Window) -> dict:
    """Mean over leavers of the directly standardised ξ at each break (vectorised
    over a leading replicate axis)."""
    mu, sd = _std_over(Y, P)
    out = {}
    for yb in win.targets:
        j = yb - win.years[0]
        lv = P[:, j] & ~P[:, j + 1]
        xi = (Y[..., lv, j] - mu[..., j, None]) / sd[..., j, None]
        out[yb] = xi.mean(-1)
    return out


def calibrate_offsets(proc: TrueProcess, win: Window, seed: int, placement: str = "shock",
                      n_rep: int = 32, sweeps: int = 30, tol: float = OFFSET_TOL,
                      w: float | None = None) -> tuple[dict, dict]:
    """δ per break such that E[leaver mean ξ] = target (common random numbers,
    Brent per break, Gauss–Seidel sweeps).  Uses only the latent chain and
    the presence mask: no estimator is run.  Returns (deltas, diagnostics);
    diagnostics[year] = target, achieved, residual, n_leave, bracketed,
    converged (|residual| < tol), small_n (n_leave < SMALL_N_LEAVE).  Nothing
    fails silently: an unbracketed root keeps the bracket bound but is
    recorded as bracketed = False / converged = False."""
    if not win.targets:
        return {}, {}
    x0 = burn_in(proc, win, np.random.default_rng([seed, 98]), placement, w, n_rep)

    def lm(deltas):
        X, U = simulate_latent(proc, win, np.random.default_rng([seed, 97]), placement,
                               deltas, n_rep=n_rep, w=w, _burned=x0)
        return {k: float(v.mean())
                for k, v in leaver_means(observed_latent(X, U), win.presence, win).items()}

    deltas = {yb: 0.0 for yb in win.targets}
    bracketed = {yb: True for yb in win.targets}
    n_sweeps = 0
    for _ in range(sweeps):
        n_sweeps += 1
        for yb, tgt in sorted(win.targets.items()):
            def f(d):
                dd = dict(deltas)
                dd[yb] = d
                return lm(dd)[yb] - tgt
            lo, hi = -2.0, 2.0
            while f(lo) > 0 and lo > -64:
                lo *= 2
            while f(hi) < 0 and hi < 64:
                hi *= 2
            flo, fhi = f(lo), f(hi)
            bracketed[yb] = bool(flo < 0 < fhi)
            deltas[yb] = float(optimize.brentq(f, lo, hi, xtol=1e-5)) \
                if bracketed[yb] else (lo if flo > 0 else hi)
        got = lm(deltas)
        if max(abs(got[k] - win.targets[k]) for k in got) < tol:
            break
    got = lm(deltas)
    n_leave = {yb: int((win.presence[:, yb - win.years[0]]
                        & ~win.presence[:, yb - win.years[0] + 1]).sum()) for yb in win.targets}
    diag = {yb: {"delta": float(deltas[yb]), "target": float(win.targets[yb]),
                 "achieved": float(got[yb]), "residual": float(got[yb] - win.targets[yb]),
                 "n_leave": n_leave[yb], "bracketed": bracketed[yb],
                 "converged": bool(bracketed[yb] and abs(got[yb] - win.targets[yb]) < tol),
                 "small_n": bool(n_leave[yb] < SMALL_N_LEAVE), "tol": tol,
                 "n_rep_crn": n_rep, "sweeps": n_sweeps, "placement": placement}
            for yb in win.targets}
    return deltas, diag


@functools.lru_cache(maxsize=1)
def logit_c_params() -> tuple[float, float]:
    """(c): P(leave | ξ) = logistic(α + γ ξ) with E[P] = 10 %/yr and
    E[ξ | leave] − E[ξ | stay] = the 2010→11 gap (−0.630 − 0.196), ξ ~ N(0, 1)
    (Gauss–Hermite)."""
    c = calibration()["mech_c_2010"]
    gap = c["mean_leave_t"] - c["mean_stay_t"]
    z, w = np.polynomial.hermite_e.hermegauss(80)
    w = w / w.sum()

    def eqs(th):
        p = special.expit(th[0] + th[1] * z)
        pl = (w * p).sum()
        ml = (w * p * z).sum() / pl
        ms = (w * (1 - p) * z).sum() / (1 - pl)
        return [pl - EXIT_RATE, (ml - ms) - gap]

    sol = optimize.fsolve(eqs, [np.log(EXIT_RATE / (1 - EXIT_RATE)), -1.0], full_output=True)
    th = sol[0]
    assert np.max(np.abs(eqs(th))) < 1e-9, "logit (c) calibration failed"
    return float(th[0]), float(th[1])


def presence_for(mech: str, Y: np.ndarray, win: Window, rng) -> np.ndarray:
    """Presence for one window; (c) is decided on the observed ξ_t (info_t)."""
    n, T = Y.shape
    if mech in ("a", "d", "e"):
        return win.presence.copy()
    P = np.ones((n, T), dtype=bool)
    if mech == "b":
        for t in range(T - 1):
            P[:, t + 1] = P[:, t] & (rng.random(n) >= EXIT_RATE)
        return P
    if mech == "c":
        al, ga = logit_c_params()
        for t in range(T - 1):
            c = P[:, t]
            xi = (Y[:, t] - Y[c, t].mean()) / Y[c, t].std()
            leave = rng.random(n) < special.expit(al + ga * xi)
            P[:, t + 1] = c & ~leave
        return P
    raise KeyError(mech)


def output_inputs(win: Window, rng) -> tuple[np.ndarray, np.ndarray]:
    """log PopNum, log CultivateArea per county-year: county means ~ N(mean,
    cov) of the real county means, plus random walks with the real robust s.d.
    of within-county annual changes; independent of x."""
    oe = calibration()["output_equation"]
    m = rng.multivariate_normal(oe["county_mean_logx_mean"], oe["county_mean_logx_cov"], size=win.n)
    sd = np.array(oe["within_county_dlogx_robust_sd"])
    T = len(win.years)
    steps = rng.standard_normal((win.n, T, 2)) * sd
    steps[:, 0, :] = 0.0
    L = m[:, None, :] + np.cumsum(steps, axis=1)
    return L[..., 0], L[..., 1]


def saai_frame(Y, P, lp, ll, win: Window) -> pd.DataFrame:
    oe = calibration()["output_equation"]
    c0, b1, b2 = oe["coef_const_logpop_logland"]
    lam = oe["yearly_resid_sd_mean"]
    i, t = np.nonzero(P)
    logy = c0 + b1 * lp[i, t] + b2 * ll[i, t] + lam * Y[i, t]
    return pd.DataFrame({"CountyCode": win.codes[i], "Year": win.years[t],
                         "GrossOutputValue": np.exp(logy), "PopNum": np.exp(lp[i, t]),
                         "CultivateArea": np.exp(ll[i, t])})


def _xi_from_saai(sa: pd.DataFrame, shape, win: Window) -> np.ndarray:
    xi = np.full(shape, np.nan)
    if len(sa):
        ci = np.searchsorted(win.codes, sa["CountyCode"].to_numpy())
        xi[ci, sa["Year"].to_numpy() - win.years[0]] = sa["SAAI"].to_numpy()
    return xi


def xi_matrix_saai(Y, P, lp, ll, win: Window) -> np.ndarray:
    """ξ[n, T] (NaN where absent) from the real compute_saai."""
    from src.saai import compute_saai
    return _xi_from_saai(compute_saai(saai_frame(Y, P, lp, ll, win), **SAAI_SPEC), Y.shape, win)


def xi_matrix_direct(Y, P) -> np.ndarray:
    mu, sd = _std_over(Y, P)
    return np.where(P, (Y - mu) / sd, np.nan)


def pairs_from_matrix(xi: np.ndarray, win: Window) -> pd.DataFrame:
    """Same output as fokker_planck.build_transition_pairs on the long panel
    (tested equal): consecutive-year pairs within the window."""
    ok = ~np.isnan(xi[:, :-1]) & ~np.isnan(xi[:, 1:])
    for j, idx in getattr(win, "drop_pairs", ()):
        ok[idx, j] = False
    i, t = np.nonzero(ok)
    order = np.lexsort((t, win.codes[i]))
    i, t = i[order], t[order]
    return pd.DataFrame({"CountyCode": win.codes[i], "Year_t": win.years[t],
                         "xi_t": xi[i, t], "xi_t1": xi[i, t + 1],
                         "delta_xi": xi[i, t + 1] - xi[i, t]})


def v1_pairs(pipeline, Y, P, lp, ll, win: Window) -> pd.DataFrame:
    """China V1: per transition, restandardise (refit SAAI, or standardise y)
    on the stayers S_t only, then take that transition's pairs."""
    from src.saai import compute_saai
    out = []
    for j in range(len(win.years) - 1):
        S = P[:, j] & P[:, j + 1]
        if S.sum() < 10:
            continue
        Pj = np.zeros_like(P)
        Pj[S, j] = Pj[S, j + 1] = True
        if pipeline == "direct":
            xi = xi_matrix_direct(Y, Pj)
        else:
            xi = _xi_from_saai(compute_saai(saai_frame(Y, Pj, lp, ll, win), **SAAI_SPEC),
                               Y.shape, win)
        out.append(pairs_from_matrix(xi, win))
    return pd.concat(out, ignore_index=True) if out else pairs_from_matrix(
        np.full(Y.shape, np.nan), win)


def composition_table(xi, X, P, win: Window, re_placement: str,
                      exit_dating: str) -> pd.DataFrame:
    """Contract COMPOSITION_COLUMNS for one replicate/pipeline/period:
    mu_x/sigma_x = moments of the LATENT x over C_t (x_units 'latent_x');
    mech_term in ξ units (Lemma 1, asserted equal to the stayers' mean Δξ to
    1e-10); xi_bar_stay / var_stay = mean and variance (ddof 0) of the
    pipeline's own ξ_t over S_t."""
    mu, sd = _std_over(X, P)
    rows = []
    for j in range(len(win.years) - 1):
        a, b = P[:, j], P[:, j + 1]
        S, L, E = a & b, a & ~b, ~a & b
        nS, nL, nE = int(S.sum()), int(L.sum()), int(E.sum())
        mL = xi[L, j].mean() if nL else 0.0
        mE = xi[E, j + 1].mean() if nE else 0.0
        mech = (nL * mL - nE * mE) / nS if nS else np.nan
        obs = float(np.mean(xi[S, j + 1] - xi[S, j])) if nS else np.nan
        resid = obs - mech
        if nS:
            assert abs(resid) < 1e-10, f"Lemma 1 identity fails at {win.years[j]}: {resid:.2e}"
        rows.append({"period": win.period, "year_t": int(win.years[j]), "n_stay": nS,
                     "n_leave": nL, "n_enter": nE,
                     "mu_x_t": float(mu[j]), "sigma_x_t": float(sd[j]),
                     "mu_x_t1": float(mu[j + 1]), "sigma_x_t1": float(sd[j + 1]),
                     "x_units": X_UNITS, "mech_term": float(mech),
                     "xi_bar_stay": float(xi[S, j].mean()) if nS else np.nan,
                     "var_stay": float(xi[S, j].var()) if nS else np.nan,
                     "re_placement": re_placement, "exit_dating": exit_dating,
                     "mean_dxi_stay": obs, "identity_residual": float(resid),
                     "mean_leave_xi_t": float(mL) if nL else np.nan,
                     "mean_enter_xi_t1": float(mE) if nE else np.nan})
    return pd.DataFrame(rows)


def province_herfindahl(win: Window) -> float:
    """H = Σ_p s_p², s_p the share of the window's counties in province p."""
    s = np.bincount(win.prov_idx) / win.n
    return float(np.sum(s ** 2))


def common_shock_wH(win: Window, placement: str = "shock", w: float | None = None) -> float:
    w = _w_shock() if w is None else float(w)
    if placement != "shock" or w <= 0:
        return 0.0
    return float(w * province_herfindahl(win))


def standardisation_table(X, win: Window, placement: str = "shock",
                          w: float | None = None) -> pd.DataFrame:
    """Contract STANDARDISATION_COLUMNS: complete panel (every simulated county
    of the window, every year, no exit) of the same replicate, latent x, and
    the window's absorbed shock share w·H (common_shock_wH; `placement` and
    `w` as given to simulate_latent)."""
    mu, sd = _std_over(X, np.ones(X.shape, dtype=bool))
    return pd.DataFrame({"period": win.period, "year": win.years.astype(int), "n": win.n,
                         "mu_x": mu, "sigma_x": sd, "x_units": X_UNITS,
                         C.SHOCK_SHARE_COLUMN: common_shock_wH(win, placement, w)})


def observables(params: np.ndarray, xi_data: np.ndarray) -> dict:
    """src.k1_contract.observables (degenerate convention 'limit')."""
    return C.observables(params, xi_data)


def estimate_full(pairs: pd.DataFrame, x0=None) -> tuple[dict, np.ndarray, str]:
    """({obs_rule: {quantity: estimate}}, params, id_status)."""
    xi, dxi = pairs["xi_t"].to_numpy(), pairs["delta_xi"].to_numpy()
    if len(xi) < 20:
        return {r: {q: np.nan for q in QUANTITIES} for r in OBS_RULES}, None, "too_few_pairs"
    fit = fit_mle(xi, dxi, x0=x0)
    o = observables(fit["params"], xi)
    est = {q: o[q] for q in QUANTITIES if q != "rho1"}
    est["rho1"] = C.rho1_from_pairs(xi, dxi)
    return {OBS_RULES[0]: est}, fit["params"], o["id_status"]


def estimate(pairs: pd.DataFrame) -> dict:
    """Contract `estimate`: MLE point estimates of QUANTITIES for one period."""
    return estimate_full(pairs)[0][OBS_RULES[0]]


def truth(true_params, standardisation: pd.DataFrame, period: str,
          cell: dict | None = None) -> dict:
    """cell : the contract cell meta (contract_cell_meta).  With it (every
    truth that enters a gate) the table must be the replicate's complete
    panel over EVERY window of the cell (contract 9: n constant within a
    window, the window sizes summing to the cell's N), so the whole
    both-window table is passed and k1_theory checks it.  Without it (unit
    tests, design calibration) the period's rows are selected as before.
    """
    from src.k1_theory import truth as theory_truth
    tp = true_params.as_dict() if isinstance(true_params, TrueProcess) else dict(true_params)
    std = standardisation
    if cell is None and "period" in std.columns:
        std = std[std["period"] == period]
    if cell is not None:
        cell = {k: v for k, v in cell.items() if k != "valid_for_law"}
    out = dict(theory_truth(tp, std.reset_index(drop=True), period, cell=cell))
    return out


@functools.lru_cache(maxsize=None)
def theory_compat(process_name: str, period: str, design_key: str = "") -> dict:
    from src.k1_theory import process_spec, stationary_edge_mass, stationary_law
    try:
        proc = process(process_name, period)
    except ValueError as exc:
        return {"process": process_name, "period": period, "compatible": False,
                "reason": f"no registered chain: {exc}"}
    out = {"process": process_name, "period": period, "guard": proc.guard,
           "slope_floor": proc.slope_floor}
    try:
        theta, guard = process_spec(proc.as_dict())
        grid, p, m_th, s_th = stationary_law(theta, guard)
        edge = float(stationary_edge_mass(theta, guard))
    except Exception as exc:
        out.update(compatible=False, reason=f"stationary_law failed: {exc}")
        return out
    rng = np.random.default_rng([SEED_BASE, 4242, len(period)])
    x = stationary_draw(proc, 20_000, rng)
    act = []
    for k in range(400):
        x = x + proc.A(x) + np.sqrt(proc.B(x)) * rng.standard_normal(x.shape)
        if proc.guard is not None and k >= 350:
            act.append(float(((x < proc.guard[0]) | (x > proc.guard[1])).mean()))
    m_sim, s_sim = float(x.mean()), float(x.std())
    active = float(np.mean(act)) if act else 0.0
    ok = {"edge_mass": edge <= COMPAT_TOL["edge_mass"],
          "mean_sd": abs(m_sim - m_th) < COMPAT_TOL["mean_sd"] * s_th,
          "sd_rel": abs(s_sim / s_th - 1.0) < COMPAT_TOL["sd_rel"]}
    out.update(theory_mean=float(m_th), theory_sd=float(s_th), theory_edge_mass=edge,
               sim_mean=m_sim, sim_sd=s_sim, guard_active_frac=active, checks=ok,
               tol=COMPAT_TOL, theory_law="guarded", compatible=bool(all(ok.values())))
    return out


def _cell(process_name, mech, n=None, structure=None, re="shock", gated=True, B=B_REGISTERED,
          gate_use=None, dated_shocks=False):
    return {"process": process_name, "mechanism": mech, "n_override": n, "structure": structure,
            "re_placement": re, "gated": gated, "B": B, "gate_use": gate_use,
            "dated_shocks": dated_shocks}


SENS_NO2006_GATE_USE = "K1a_sens_no2006"
SENS_DROP_BREAKS = (2006,)


def registered_cells() -> list[dict]:
    cells = []
    for p in ("i_ou", "ii_cubic", "iii_mult"):
        for m in MECHANISMS:
            cells.append(_cell(p, m, gate_use="K1a/K1b"))
    for k in (0, 1, 2, 4):
        for m in ("a", "e"):
            cells.append(_cell(f"iv_bistable_{k}", m, gate_use="K2"))
    for m in ("a", "d", "e"):
        cells.append(_cell("v_posctrl", m, gate_use="K1c"))
    for p in PROCESS_NAMES:
        cells.append(_cell(p, "a", n=N_S0, B=0, gate_use="S0"))
    for p, m in K1A_CELLS:
        cells.append(_cell(p, "a", structure=m, B=0, gate_use="K1a_null"))
    for p, m in K1A_CELLS:
        cells.append(_cell(p, m, re="persistent", gated=False, B=0, gate_use="control"))
    for p, m in K1A_CELLS:
        cells.append(_cell(p, "a", structure=m, re="persistent", gated=False, B=0,
                           gate_use="control_null"))
    for p, m in K1A_CELLS:
        cells.append(_cell(p, "a", structure=m, B=0, gate_use="K1a_shocked_null",
                           dated_shocks=True))
    for p, m in K1A_CELLS:
        if m == "e":
            c = _cell(p, "e", B=0, gate_use=SENS_NO2006_GATE_USE)
            c["drop_breaks"] = SENS_DROP_BREAKS
            cells.append(c)
    for i, c in enumerate(cells):
        c["cell_index"] = i
    return cells


NEURAL_CELLS = {(p, m) for p, m in K1A_CELLS}
NEURAL_GATE_USES = ("K1a/K1b", "K1a_null")


def is_neural_cell(cell: dict) -> bool:
    """True for the K3 cells (i), (ii) × (d), (e) and their matched gated
    no-exit (a) null cells."""
    key = (cell["process"], cell.get("structure") or cell["mechanism"])
    return (cell.get("gate_use") in NEURAL_GATE_USES and bool(cell.get("gated"))
            and cell.get("re_placement") == GATED_RE_PLACEMENT
            and not cell.get("dated_shocks") and key in NEURAL_CELLS
            and (cell["mechanism"] == "a") == (cell.get("gate_use") == "K1a_null"))
REPORT_ONLY_GATE_USES = ("control", "control_null", "K1a_shocked_null", SENS_NO2006_GATE_USE)


def null_cell_for(cell: dict) -> dict:
    for c in registered_cells():
        if (c["mechanism"] == "a" and c["structure"] == cell["mechanism"]
                and c["process"] == cell["process"] and not c.get("dated_shocks")
                and c["re_placement"] == cell["re_placement"] and c["gated"] == cell["gated"]):
            return c
    raise KeyError(f"no null baseline for cell {cell.get('cell_index')}")


def shocked_null_for(cell: dict) -> dict:
    """The no-exit cell with the dated block shocks of a gated K1a cell."""
    for c in registered_cells():
        if (c.get("dated_shocks") and c["structure"] == cell["mechanism"]
                and c["process"] == cell["process"]):
            return c
    raise KeyError(f"no shocked null for cell {cell.get('cell_index')}")


def exit_source_cell(cell: dict) -> dict | None:
    """The exit cell whose design a cell uses: the cell itself for (b)–(e);
    for a dated-shock null, the gated K1a cell whose calibrated δ_b it
    applies (its guards are that cell's guards); None for a plain (a) cell."""
    if cell["mechanism"] != "a":
        return cell
    if cell.get("dated_shocks"):
        for c in registered_cells():
            if (c["process"] == cell["process"] and c["mechanism"] == cell["structure"]
                    and c["gate_use"] == "K1a/K1b"):
                return c
        raise KeyError(f"no K1a source cell for dated-shock cell {cell.get('cell_index')}")
    return None


def check_cell(cell: dict) -> None:
    rep, gated = cell.get("re_placement"), cell.get("gated")
    if cell.get("dated_shocks") and not (cell["mechanism"] == "a" and rep == GATED_RE_PLACEMENT
                                         and (cell["process"], cell.get("structure")) in K1A_CELLS):
        raise ValueError(f"cell {cell.get('cell_index')}: dated shocks without exit exist only "
                         "as the 'shock' no-exit cells of (i), (ii) × (d), (e)")
    if rep not in RE_PLACEMENTS:
        raise ValueError(f"cell {cell.get('cell_index')}: unknown re_placement {rep!r}")
    if not isinstance(gated, (bool, np.bool_)):
        raise ValueError(f"cell {cell.get('cell_index')}: gated must be a bool")
    if gated and rep != GATED_RE_PLACEMENT:
        raise ValueError(f"cell {cell.get('cell_index')}: gated cell with re_placement "
                         f"{rep!r} refused (amendment 2 (g): gated cells are 'shock' only)")
    if not gated:
        key = (cell["process"], cell["structure"] or cell["mechanism"])
        if rep != CONTROL_RE_PLACEMENT or key not in K1A_CELLS:
            raise ValueError(f"cell {cell.get('cell_index')}: ungated cells are only the "
                             "'persistent' controls of (i), (ii) × (d), (e) and their baselines")
    if hasattr(C, "check_cell_config"):
        for pl in PIPELINES:
            C.check_cell_config(contract_cell_meta(cell, pipeline=pl))


def cell_windows(cell: dict, mask: pd.DataFrame) -> dict:
    m, n, st = cell["mechanism"], cell["n_override"], cell.get("structure")
    if st is not None:
        if m != "a" or n is not None:
            raise ValueError("a null-baseline structure needs mechanism 'a' and no n_override")
        nw = shocked_null_window if cell.get("dated_shocks") else null_window
        return {p: nw(w) for p, w in
                cell_windows({**cell, "mechanism": st, "structure": None,
                              "dated_shocks": False}, mask).items()}
    if m == "a":
        return {p: open_base(mask, p, n) for p in PERIODS}
    if m in ("b", "c"):
        return {p: open_base(mask, p, n if n is not None else exit_base_n(mask, p))
                for p in PERIODS}
    if n is not None:
        raise ValueError("n_override only for open-base mechanisms")
    if m == "d":
        return {"pre": open_base(mask, "pre"), "during": block_d_window(mask)}
    if m == "e":
        return {p: replay_window(mask, p, cell.get("drop_breaks") or ()) for p in PERIODS}
    raise KeyError(m)


@functools.lru_cache(maxsize=None)
def _cell_window_sizes(cell_index: int) -> tuple:
    wins = cell_windows(registered_cells()[cell_index], load_mask())
    return tuple((p, int(wins[p].n)) for p in PERIODS)


def _cell_N(cell_index: int) -> int:
    return int(sum(n for _, n in _cell_window_sizes(cell_index)))


def contract_cell_meta(cell: dict, pipeline: str) -> dict:
    if cell.get("cell_index") is not None and cell.get("cell_index") < len(registered_cells()) \
            and registered_cells()[cell["cell_index"]] == cell:
        sizes = dict(_cell_window_sizes(cell["cell_index"]))
    else:
        wins = cell_windows(cell, load_mask())
        sizes = {p: int(wins[p].n) for p in PERIODS}
    N = int(sum(sizes.values()))
    return {"process": PROCESS_KEY[cell["process"]], "exit_mechanism": cell["mechanism"],
            "N": N, "N_windows": sizes, "windows": WINDOWS_LABEL, "pipeline": pipeline,
            "re_placement": cell["re_placement"],
            "exit_dating": EXIT_DATING[cell["mechanism"]], "gated": bool(cell["gated"]),
            "cell_index": cell.get("cell_index"), "sim_process": cell["process"],
            "structure": cell.get("structure"), "gate_use": cell.get("gate_use"),
            "report_only": cell.get("gate_use") in REPORT_ONLY_GATE_USES}


def _design_truths(proc: TrueProcess, period: str) -> list[dict]:
    """truth() of `proc` on N_DESIGN_SEEDS complete panels of the REGISTERED
    size (the (a) open base, 488 / 289 counties, 'shock' placement), fixed
    design seeds (common random numbers across the parameter values being
    calibrated)."""
    win = open_base(load_mask(), period)
    out = []
    for s in range(N_DESIGN_SEEDS):
        X, _ = simulate_latent(proc, win, np.random.default_rng([SEED_BASE, 555, s, len(period)]))
        out.append(truth(proc, standardisation_table(X, win), period))
    return out


def _median(vals) -> float:
    v = np.asarray(vals, dtype=float)
    if not np.isfinite(v).all():
        return np.nan
    return float(np.median(v))


def _bracket_root(f, lo, hi, step=0.7, max_expand=8):
    flo, fhi = f(lo), f(hi)
    k = 0
    while np.isfinite(flo) and np.isfinite(fhi) and flo * fhi > 0 and k < max_expand:
        lo, hi = lo - step, hi + step
        flo, fhi = f(lo), f(hi)
        k += 1
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
        raise RuntimeError(f"design calibration: root not bracketed (f={flo}, {fhi})")
    return optimize.brentq(f, lo, hi, xtol=1e-6)


V_BETA_BRACKET = (1e-4, 0.05)
V_LAM_BRACKET = (0.003, 0.08)
V_SCAN_NODES = 25
V_SEARCH_NOTE = (f"amendment 5 (b): lambda: brentq on {V_LAM_BRACKET}; beta: smallest root of "
                 f"shift = -0.31 on {V_BETA_BRACKET} (first sign change on a {V_SCAN_NODES}-node "
                 "log grid, then brentq xtol 1e-6; the shift is not monotone in beta)")


def _first_crossing(f, lo, hi, n=V_SCAN_NODES):
    """The smallest root of f on [lo, hi]: the first sign change of f on n
    equally spaced nodes, refined by brentq.  Raises if f is non-finite at a
    node before the first sign change, or if f never changes sign."""
    xs = np.linspace(lo, hi, n)
    prev = None
    for x in xs:
        fx = f(x)
        if not np.isfinite(fx):
            raise RuntimeError(f"design calibration: f non-finite at {x} before a sign change")
        if fx == 0:
            return float(x)
        if prev is not None and prev[1] * fx < 0:
            return float(optimize.brentq(f, prev[0], x, xtol=1e-6))
        prev = (x, fx)
    raise RuntimeError(f"design calibration: no sign change of f on [{lo}, {hi}]")


@functools.lru_cache(maxsize=None)
def design_constants(process_name: str) -> dict:
    if process_name.startswith("iv_bistable_"):
        k = float(process_name.rsplit("_", 1)[1])
        out = {"rule": f"median of {N_DESIGN_SEEDS} replicate truths at the registered N"}
        for p in PERIODS:
            al0 = -bistable_params(p, max(k, 1.0))[3]
            out[f"alpha_analytic_{p}"] = float(al0)
            if k == 0:
                out[f"alpha_{p}"] = float(al0)
                continue

            def f(log_al):
                proc = TrueProcess(bistable_params(p, k, float(np.exp(log_al))))
                return _median([t["two_delta_V"] for t in _design_truths(proc, p)]) - k
            al = float(np.exp(_bracket_root(f, np.log(al0) - 0.7, np.log(al0) + 0.7)))
            out[f"alpha_{p}"] = al
            tv = [t["two_delta_V"] for t in _design_truths(
                TrueProcess(bistable_params(p, k, al)), p)]
            out[f"design_truth_2dV_{p}"] = {"median": _median(tv), "min": float(np.min(tv)),
                                           "max": float(np.max(tv))}
        return out
    if process_name == "v_posctrl":
        pre = process("v_posctrl", "pre")
        tp = _design_truths(pre, "pre")

        def stats(lam, beta):
            td = _design_truths(TrueProcess(posctrl_during_params(lam, beta)), "during")
            sh = _median([d["xi_trap"] - q["xi_trap"] for d, q in zip(td, tp)])
            rt = _median([C._ratio(d, q, "tau_mfpt") for d, q in zip(td, tp)])
            return sh, rt

        def beta_for(lam):
            f = lambda lb: stats(lam, np.exp(lb))[0] + 0.31
            return float(np.exp(_first_crossing(f, np.log(V_BETA_BRACKET[0]),
                                                np.log(V_BETA_BRACKET[1]))))

        def g(ll):
            lam = np.exp(ll)
            return np.log(stats(lam, beta_for(lam))[1]) - np.log(4.0)
        ll = optimize.brentq(g, np.log(V_LAM_BRACKET[0]), np.log(V_LAM_BRACKET[1]), xtol=1e-6)
        lam = float(np.exp(ll))
        beta = beta_for(lam)
        sh, rt = stats(lam, beta)
        r = [sh + 0.31, np.log(rt) - np.log(4.0)]
        if not np.all(np.isfinite(r)) or abs(r[0]) > 0.01 or abs(r[1]) > 0.02:
            raise RuntimeError(f"(v) design calibration did not converge: residuals {r}")
        return {"lam": lam, "beta": beta, "search": V_SEARCH_NOTE,
                "residual_shift": float(r[0]),
                "residual_logtau": float(r[1]),
                "rule": f"median of {N_DESIGN_SEEDS} replicate truth contrasts at the registered N",
                "design_truth_pre_xi_trap_median": _median([q["xi_trap"] for q in tp]),
                "design_truth_pre_tau_median": _median([q["tau_mfpt"] for q in tp])}
    return {}


def cell_processes(cell: dict) -> dict:
    """The registered true chain of the cell per period ((iv) / (v) from the
    design registry, contract 8)."""
    return {p: process(cell["process"], p) for p in PERIODS}


@functools.lru_cache(maxsize=None)
def _offsets_cached(cell_index: int) -> tuple:
    """((period, (deltas, diagnostics)), ...) of a (d)/(e) cell."""
    cell = registered_cells()[cell_index]
    mask = load_mask()
    wins = cell_windows(cell, mask)
    procs = cell_processes(cell)
    res = {}
    for p in PERIODS:
        if wins[p].targets:
            d, diag = calibrate_offsets(procs[p], wins[p], seed=SEED_BASE + cell_index,
                                        placement=cell["re_placement"])
            res[p] = (d, diag)
    return tuple(sorted(res.items()))


class GuardRefused(RuntimeError):
    """An ordering guard refused (never turned into a NaN replicate)."""


def _check_guard(mechanism: str):
    if mechanism != "a":
        assert_theory_frozen = public_guard_skipped
        assert_theory_frozen()


def is_guard_refusal(exc: BaseException) -> bool:
    if isinstance(exc, GuardRefused) or getattr(exc, "guard_refusal", False):
        return True
    TheoryNotFrozen = public_guard_skipped(GuardRefused)
    return isinstance(exc, TheoryNotFrozen)


def neural_preflight(cell: dict, neural, estimate: bool = True) -> str:
    if neural is None or not estimate or not is_neural_cell(cell):
        return "n/a"
    pf = getattr(neural, "preflight", None)
    if pf is None:
        raise GuardRefused(f"cell {cell.get('cell_index')}: the neural callable has no "
                           "preflight(cell_meta) admissibility check (fail closed)")
    try:
        pf(contract_cell_meta(cell, "saai"))
    except Exception as exc:
        raise GuardRefused(f"cell {cell.get('cell_index')}: neural refused: "
                           f"{type(exc).__name__}: {exc}") from exc
    return "neural allowed"


COVERAGE_BAND = (0.92, 0.98)
COVERAGE_PRECHECK_DIR = "results/records/k1_precheck_a_registered_f2"
COVERAGE_PRECHECK_DIR_FREEZE1 = "results/records/k1_precheck_a_registered"
COVERAGE_PRECHECK_UNITS = {"xi_trap": (("contrast",), None),
                           "tau_mfpt": (("contrast",), None),
                           "two_delta_V": (("pre", "during"), (17, 19, 21))}
COVERAGE_PRECHECK_2DV_PROCESSES = ("iv_bistable_1", "iv_bistable_2", "iv_bistable_4")
COVERAGE_PRECHECK_GATE_USES = ("K1a/K1b", "K2", "K1c")
COVERAGE_PRECHECK_NOT_EVALUABLE = (("i_ou", "tau_mfpt", "contrast"),)
COVERAGE_PRECHECK_SCOPES = {"K1b": ("i_ou", "ii_cubic", "iii_mult"),
                            "K1c": ("v_posctrl",),
                            "K2": ("iv_bistable_0", "iv_bistable_1", "iv_bistable_2",
                                   "iv_bistable_4")}
ITEM12_UNITS_SECTION = "### amendment 5"
ITEM12_UNITS_APPROVAL = r"\*\*Item 12 units approved by the author:\*\* approved \(\d{4}-\d{2}-\d{2}, author\)"
S2_SECTION = "### amendment 7"
S2_APPROVAL = r"\*\*Approved by the author:\*\* approved \(\d{4}-\d{2}-\d{2}, author\)"


def coverage_precheck_scope(cell: dict) -> str:
    """The precheck scope (gate) that an estimator on `cell` needs: K1b for
    (i)–(iii), K1c for (v), K2 for (iv) (by the cell's process)."""
    for g, procs in COVERAGE_PRECHECK_SCOPES.items():
        if cell["process"] in procs:
            return g
    raise ValueError(f"process {cell['process']!r} has no precheck scope")


def item12_units_registered(frozen: str | None = None, plan_text: str | None = None) -> bool:
    import re
    import subprocess
    if plan_text is None:
        rel = 'plan.md'
        try:
            if frozen is not None:
                plan_text = subprocess.run(["git", "-C", ROOT, "show", f"{frozen}:{rel}"],
                                           capture_output=True, text=True,
                                           check=True).stdout
            else:
                with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
                    plan_text = f.read()
        except (OSError, subprocess.CalledProcessError):
            return False
    def section(head):
        a = plan_text.find(head)
        if a < 0:
            return None
        b = plan_text.find("\n### ", a + 1)
        return plan_text[a:b if b > 0 else None]
    s5, s7 = section(ITEM12_UNITS_SECTION), section(S2_SECTION)
    return bool(s5 is not None and re.search(ITEM12_UNITS_APPROVAL, s5)
                and s7 is not None and re.search(S2_APPROVAL, s7))


def coverage_precheck_a_cells() -> list:
    """The 8 registered (a) cells the coverage precheck covers."""
    return [c for c in registered_cells() if c["mechanism"] == "a"
            and c["gate_use"] in COVERAGE_PRECHECK_GATE_USES and c["n_override"] is None
            and c["B"] > 0]


def coverage_precheck_units(a_cells=None, include_not_evaluable: bool = False) -> list:
    a_cells = coverage_precheck_a_cells() if a_cells is None else a_cells
    by_idx = {c["cell_index"]: c for c in a_cells}
    units = []
    for q, (periods, cells) in COVERAGE_PRECHECK_UNITS.items():
        if cells is None:
            chosen = a_cells
        else:
            missing = [i for i in cells if i not in by_idx]
            if missing:
                raise ValueError(f"coverage-precheck {q} cells {missing} are not registered (a) cells")
            chosen = [by_idx[i] for i in cells]
            if q == "two_delta_V" and tuple(c["process"] for c in chosen) != \
                    COVERAGE_PRECHECK_2DV_PROCESSES:
                raise ValueError("coverage-precheck 2ΔV cells are not the (iv) members with nominal "
                                 f"2ΔV > 0: {[c['process'] for c in chosen]}")
        units += [(c, q, per) for c in chosen for per in periods]
    if not include_not_evaluable:
        units = [u for u in units
                 if (u[0]["process"], u[1], u[2]) not in COVERAGE_PRECHECK_NOT_EVALUABLE]
    return units


def coverage_precheck(tables_dir: str | None = None, frozen: str | None = None) -> dict:
    if not public_guard_skipped(True):
        return {"pass": False, "scopes": {},
                "why": "the coverage precheck units are not registered by the "
                       "author (amendment 5 (a) and amendment 7 (S2); read "
                       "at the freeze commit)"}
    d = tables_dir or os.path.join(ROOT, COVERAGE_PRECHECK_DIR)
    try:
        cov = pd.read_csv(os.path.join(d, "k1_sim_coverage.csv"))
        with open(os.path.join(d, "k1_sim_meta.json")) as f:
            meta = json.load(f)
    except (OSError, ValueError) as exc:
        return {"pass": False, "scopes": {},
                "why": f"no registered (a) coverage precheck in {d}: {type(exc).__name__}"}
    if meta.get("registered") is not True or meta.get("dev") is not False:
        return {"pass": False, "scopes": {},
                "why": "the (a) coverage precheck is not a registered run"}
    if frozen is not None and meta.get("theory_frozen") != frozen:
        return {"pass": False, "scopes": {},
                "why": f"the (a) coverage precheck ran under freeze "
                       f"{meta.get('theory_frozen')}, not {frozen}"}
    if meta.get("bootstrap_scheme") != BOOTSTRAP_SCHEME:
        return {"pass": False, "scopes": {},
                "why": "the (a) coverage precheck was not made with the amendment 7 (S2) "
                       f"bootstrap ({meta.get('bootstrap_scheme')!r})"}
    if meta.get("seed_stream") != "registered" or \
            meta.get("precheck_seed_base") != C.PRECHECK_SEED_BASE:
        return {"pass": False, "scopes": {},
                "why": "the (a) coverage precheck did not use the registered amendment 7 "
                       "precheck seed stream"}
    try:
        units = coverage_precheck_units()
        ne_units = [u for u in coverage_precheck_units(include_not_evaluable=True)
                    if u not in units]
    except ValueError as exc:
        return {"pass": False, "scopes": {}, "why": f"coverage-precheck units: {exc}"}

    def row(c, q, per):
        return cov[(cov["cell_index"] == c["cell_index"]) & (cov["pipeline"] == "saai")
                   & (cov["variant"] == "V1") & (cov["ci_nan_rule"] == "tol0")
                   & (cov["obs_rule"] == OBS_RULES[0]) & (cov["quantity"] == q)
                   & (cov["period"] == per)]
    out, n, scopes = [], 0, {g: {"units": 0, "out_of_band": []} for g in COVERAGE_PRECHECK_SCOPES}
    for c, q, per in units:
        r = row(c, q, per)
        n += 1
        if len(r) != 1 or int(r["R"].iloc[0]) != R_REGISTERED:
            return {"pass": False, "scopes": {}, "why": f"(a) precheck lacks cell "
                    f"{c['cell_index']} {q} {per} at R = {R_REGISTERED}"}
        v = float(r["coverage"].iloc[0])
        g = coverage_precheck_scope(c)
        scopes[g]["units"] += 1
        if not (COVERAGE_BAND[0] <= v <= COVERAGE_BAND[1]):
            msg = f"cell {c['cell_index']} {c['process']} {q} {per}: {v:.3f}"
            out.append(msg)
            scopes[g]["out_of_band"].append(msg)
    for g, sc in scopes.items():
        sc["pass"] = bool(sc["units"] > 0 and not sc["out_of_band"])
    ne = []
    for c, q, per in ne_units:
        r = row(c, q, per)
        ne.append({"cell_index": c["cell_index"], "process": c["process"], "quantity": q,
                   "period": per, "status": "not evaluable (amendment 7 (S2) item 4)",
                   "coverage": float(r["coverage"].iloc[0]) if len(r) == 1 else None})
    return {"pass": not out and all(sc["pass"] for sc in scopes.values()),
            "scopes": scopes, "out_of_band": out, "n_units": n, "not_evaluable": ne,
            "why": "" if not out else f"{len(out)} of {n} (a) coverage units outside "
                                      f"{list(COVERAGE_BAND)}"}


def _check_coverage_precheck(cell: dict | None = None) -> None:
    assert_theory_frozen = public_guard_skipped
    res = coverage_precheck(frozen=assert_theory_frozen())
    if cell is None:
        ok, why = res["pass"], res["why"]
    else:
        g = coverage_precheck_scope(cell)
        sc = (res.get("scopes") or {}).get(g)
        ok = bool(sc and sc["pass"])
        why = res["why"] if not sc else (f"scope {g}: {len(sc['out_of_band'])} of "
                                        f"{sc['units']} units outside {list(COVERAGE_BAND)}")
    if not ok:
        raise RuntimeError(f"coverage precheck: {why} — no estimator runs on an exit "
                           f"cell until the (a) coverage lies in {list(COVERAGE_BAND)}")


def exit_guards(cell: dict, estimate: bool, pipelines=PIPELINES, neural=None) -> str:
    src = exit_source_cell(cell)
    if src is None:
        if estimate and "saai" in pipelines:
            neural_preflight(cell, neural, estimate)
        return "no exit design"
    try:
        _check_guard(src["mechanism"])
        for pl in PIPELINES:
            C.assert_exit_simulation_allowed(contract_cell_meta(src, pl))
        if estimate:
            for pl in pipelines:
                C.assert_exit_estimation_allowed(contract_cell_meta(src, pl))
            _check_coverage_precheck(src)
    except Exception as exc:
        raise GuardRefused(f"cell {cell.get('cell_index')}: {type(exc).__name__}: {exc}") from exc
    if estimate and "saai" in pipelines:
        neural_preflight(cell, neural, estimate)
    return "estimation allowed" if estimate else "composition only"


SEED_STREAMS = ("registered", "implcheck")


def is_precheck_a_cell(cell: dict) -> bool:
    """One of the 8 registered (a) cells of the coverage-precheck coverage precheck."""
    return any(c["cell_index"] == cell.get("cell_index") for c in coverage_precheck_a_cells())


def replicate_seed(cell: dict, replicate: int, pipeline: str | None = None,
                   stream: str = "registered") -> int:
    if stream not in SEED_STREAMS:
        raise ValueError(f"unknown seed stream {stream!r}")
    if stream == "implcheck":
        if not is_precheck_a_cell(cell):
            raise ValueError("the implcheck seed stream exists only for the (a) precheck cells")
        return int(C.IMPLCHECK_SEED_BASE + cell["cell_index"] * 10_000 + int(replicate))
    if cell["gate_use"] != "S0" and is_precheck_a_cell(cell):
        return int(C.PRECHECK_SEED_BASE + cell["cell_index"] * 10_000 + int(replicate))
    if cell["gate_use"] == "S0":
        if pipeline not in PIPELINES:
            raise ValueError("an S0 replicate seed needs its pipeline")
        if not 0 <= int(replicate) < C.R_PER_CELL:
            raise ValueError(f"S0 replicate {replicate} outside the registered 0..{C.R_PER_CELL - 1}")
        return int(C.s0_seeds(PROCESS_KEY[cell["process"]], pipeline)[int(replicate)])
    return SEED_BASE + cell["cell_index"] * 10_000 + int(replicate)


def merge_s0_pipelines(parts: list[dict], pipelines) -> dict:
    out = dict(parts[0])
    for k in ("estimates", "composition", "truth", "standardisation"):
        dfs = []
        for pl, r in zip(pipelines, parts):
            df = r[k].copy()
            if "pipeline" not in df.columns:
                df.insert(2 if len(df.columns) >= 2 else 0, "pipeline", pl)
            dfs.append(df)
        out[k] = pd.concat(dfs, ignore_index=True)
    out["seed"] = {pl: r["seed"] for pl, r in zip(pipelines, parts)}
    out["status"] = ";".join(sorted({str(r["status"]) for r in parts}))
    return out


def _estimates_columns() -> list:
    return ["cell_index", "replicate", "pipeline", "saai_spec", "variant", "obs_rule", "B",
            "period", "quantity", "estimate", "truth", "ci_lo", "ci_hi", "n_nan_boot",
            "ci_lo_lit", "ci_hi_lit", "id_status", "replicate_status"]


def simulate_panels(cell: dict, seed: int, mask: pd.DataFrame) -> dict:
    wins = cell_windows(cell, mask)
    procs = cell_processes(cell)
    rep_pl = cell["re_placement"]
    src = exit_source_cell(cell)
    deltas = (dict(_offsets_cached(src["cell_index"]))
              if src is not None and src["mechanism"] in ("d", "e") else {})
    rng_lat = np.random.default_rng([seed, 0])
    rng_exit = np.random.default_rng([seed, 1])
    rng_out = np.random.default_rng([seed, 2])
    sim, std, diag = {}, {}, {}
    for p in PERIODS:
        w = wins[p]
        dl = deltas.get(p, ({}, {}))[0]
        X, U = simulate_latent(proc := procs[p], w, rng_lat, rep_pl, dl)
        Y = observed_latent(X, U)
        P = presence_for(cell["mechanism"], Y, w, rng_exit)
        lp, ll = output_inputs(w, rng_out)
        sim[p] = (X, Y, P, lp, ll)
        std[p] = standardisation_table(X, w, rep_pl)
        diag[p] = {"prov_share_x_first": province_share(X[:, 0], w.prov_idx),
                   "prov_share_x_last": province_share(X[:, -1], w.prov_idx),
                   "prov_share_y_first": province_share(Y[:, 0], w.prov_idx),
                   "prov_share_y_last": province_share(Y[:, -1], w.prov_idx),
                   "guard_active_frac": float(np.mean((X < proc.guard[0]) | (X > proc.guard[1])))
                   if proc.guard is not None else 0.0}
    return {"wins": wins, "procs": procs, "sim": sim, "std": std, "diag": diag,
            "deltas": deltas}


def xi_matrix(pipeline: str, sim_p: tuple, win: Window) -> np.ndarray:
    """ξ of one period's panel (X, Y, P, lp, ll) under `pipeline`."""
    X, Y, P, lp, ll = sim_p
    return xi_matrix_saai(Y, P, lp, ll, win) if pipeline == "saai" else xi_matrix_direct(Y, P)


def run_replicate(cell: dict, replicate: int, mask: pd.DataFrame, B: int | None = None,
                  neural=None, pipelines=PIPELINES, estimate: bool = True,
                  seed_stream: str = "registered") -> dict:
    src_cell = exit_source_cell(cell)
    if src_cell is not None:
        try:
            _check_guard(src_cell["mechanism"])
        except Exception as exc:
            raise GuardRefused(f"cell {cell.get('cell_index')}: {type(exc).__name__}: "
                               f"{exc}") from exc
    check_cell(cell)
    if cell["gate_use"] == "S0" and len(pipelines) != 1:
        parts = [run_replicate(cell, replicate, mask, B, neural, (pl,), estimate, seed_stream)
                 for pl in pipelines]
        return merge_s0_pipelines(parts, pipelines)
    guard = exit_guards(cell, estimate, pipelines, neural=neural)
    B = cell["B"] if B is None else B
    seed = replicate_seed(cell, replicate, pipelines[0], stream=seed_stream)
    pan = simulate_panels(cell, seed, mask)
    wins, procs, sim, std, diag, deltas = (pan[k] for k in (
        "wins", "procs", "sim", "std", "diag", "deltas"))
    rep_pl = cell["re_placement"]
    dating = EXIT_DATING[cell["mechanism"]]
    rng_boot = np.random.default_rng([seed, BOOT_STREAM])

    tmeta = contract_cell_meta(cell, pipelines[0])
    std_all = pd.concat([std[p] for p in PERIODS], ignore_index=True)
    tr = {p: truth(procs[p], std_all, p, cell=tmeta) for p in PERIODS}

    est_rows, comp_rows = [], []
    pairs_store = {}
    for pl in pipelines:
        for p in PERIODS:
            X, Y, P, lp, ll = sim[p]
            xi = xi_matrix(pl, sim[p], wins[p])
            c = composition_table(xi, X, P, wins[p], rep_pl, dating)
            c.insert(0, "pipeline", pl)
            comp_rows.append(c)
            if not estimate:
                continue
            v0 = pairs_from_matrix(xi, wins[p])
            v1 = v0 if cell["mechanism"] == "a" else v1_pairs(pl, Y, P, lp, ll, wins[p])
            pairs_store[(pl, "V0", p)] = v0
            pairs_store[(pl, "V1", p)] = v1
    if not estimate:
        comp = pd.concat(comp_rows, ignore_index=True)
        stdt = pd.concat([std[p] for p in PERIODS], ignore_index=True)
        for df in (comp, stdt):
            df.insert(0, "replicate", replicate)
            df.insert(0, "cell_index", cell["cell_index"])
        truth_rows = [{"cell_index": cell["cell_index"], "replicate": replicate, "period": p,
                       "obs_rule": OBS_RULES[0], "quantity": q, "truth": tr[p][q],
                       "id_status": tr[p].get("id_status", "")}
                      for p in PERIODS for q in QUANTITIES]
        return {"estimates": pd.DataFrame(columns=_estimates_columns()), "composition": comp,
                "truth": pd.DataFrame(truth_rows), "standardisation": stdt, "deltas": deltas,
                "seed": seed, "status": "ok", "mode": "composition_only", "guard": guard,
                "true_params": {p: procs[p].as_dict() for p in PERIODS},
                "true_process": {p: procs[p].describe() for p in PERIODS}}

    draws = province_block_draws(wins, B, rng_boot) if B > 0 else None

    for pl in pipelines:
        done = {}
        for var in ("V0", "V1"):
            if var == "V1" and cell["mechanism"] == "a":
                assert pairs_store[(pl, "V1", "pre")] is pairs_store[(pl, "V0", "pre")]
                _emit(est_rows, pl, var, *done["V0"], tr, B)
                continue
            pe, par, ids = {}, {}, {}
            for p in PERIODS:
                e_, par[p], ids[p] = estimate_full(pairs_store[(pl, var, p)])
                pe[p] = e_
            bd = _bootstrap(pl, var, cell["mechanism"], sim, wins, par, draws) \
                if B > 0 else None
            done[var] = (pe, ids, bd)
            _emit(est_rows, pl, var, pe, ids, bd, tr, B)
    if neural is not None and "saai" in pipelines and is_neural_cell(cell):
        pe, ids = {}, {}
        nmeta = contract_cell_meta(cell, "saai")
        for p in PERIODS:
            try:
                pe[p] = dict(neural(pairs_store[("saai", "V0", p)], cell=nmeta,
                                    replicate=int(replicate), period=p))
                ids[p] = str(pe[p].get("id_status") or "neural")
            except Exception as exc:
                if is_guard_refusal(exc):
                    raise GuardRefused(f"cell {cell.get('cell_index')} replicate {replicate} "
                                       f"{p}: neural refused: {type(exc).__name__}: "
                                       f"{exc}") from exc
                pe[p], ids[p] = {q: np.nan for q in QUANTITIES}, f"neural_failed:{type(exc).__name__}"
            pe[p] = {OBS_RULES[0]: {q: float(pe[p].get(q, np.nan)) for q in QUANTITIES}}
        _emit(est_rows, "saai", "V0_neural", pe, ids, None, tr, 0)

    est = pd.DataFrame(est_rows)
    est["replicate_status"] = "ok"
    comp = pd.concat(comp_rows, ignore_index=True)
    stdt = pd.concat([std[p] for p in PERIODS], ignore_index=True)
    for df in (est, comp, stdt):
        df.insert(0, "replicate", replicate)
        df.insert(0, "cell_index", cell["cell_index"])
    truth_rows = [{"cell_index": cell["cell_index"], "replicate": replicate, "period": p,
                   "obs_rule": OBS_RULES[0], "quantity": q, "truth": tr[p][q],
                   "id_status": tr[p].get("id_status", "")}
                  for p in PERIODS for q in QUANTITIES]
    truth_rows += [{"cell_index": cell["cell_index"], "replicate": replicate, "period": p,
                    "obs_rule": "", "quantity": f"diag_{k}", "truth": float(v), "id_status": ""}
                   for p in PERIODS for k, v in diag[p].items()]
    return {"estimates": est, "composition": comp, "truth": pd.DataFrame(truth_rows),
            "standardisation": stdt, "deltas": deltas, "seed": seed, "status": "ok",
            "mode": "full", "guard": guard,
            "true_params": {p: procs[p].as_dict() for p in PERIODS},
            "true_process": {p: procs[p].describe() for p in PERIODS}}


def failed_replicate(cell: dict, replicate: int, B: int, exc: BaseException,
                     neural=None, pipelines=PIPELINES, seed_stream: str = "registered") -> dict:
    """NaN rows of a replicate that raised: every row the replicate would have
    written, estimate/truth/CI NaN, n_nan_boot = B, replicate_status
    'failed:<Exception>'.  The replicate stays in every denominator."""
    status = f"failed:{type(exc).__name__}"
    variants = ["V0", "V1"] + (["V0_neural"] if neural is not None
                               and is_neural_cell(cell) else [])
    rows = []
    for pl in pipelines:
        for var in variants:
            if var == "V0_neural" and pl != "saai":
                continue
            for r in OBS_RULES:
                for p in PERIODS + ("contrast",):
                    for q in QUANTITIES:
                        rows.append({"pipeline": pl, "saai_spec": SAAI_SPEC_LABEL[pl],
                                     "variant": var, "obs_rule": r,
                                     "B": 0 if var == "V0_neural" else B, "period": p,
                                     "quantity": q, "estimate": np.nan, "truth": np.nan,
                                     "ci_lo": np.nan, "ci_hi": np.nan,
                                     "ci_lo_lit": np.nan, "ci_hi_lit": np.nan,
                                     "n_nan_boot": 0 if var == "V0_neural" else B,
                                     "id_status": status})
    est = pd.DataFrame(rows)
    est["replicate_status"] = status
    est.insert(0, "replicate", replicate)
    est.insert(0, "cell_index", cell["cell_index"])
    truth_rows = pd.DataFrame([{"cell_index": cell["cell_index"], "replicate": replicate,
                                "period": p, "obs_rule": r, "quantity": q, "truth": np.nan,
                                "id_status": status}
                               for p in PERIODS for r in OBS_RULES for q in QUANTITIES])
    if cell["gate_use"] == "S0":
        truth_rows = pd.concat([truth_rows.assign(pipeline=pl) for pl in pipelines],
                               ignore_index=True)
    empty = lambda: pd.DataFrame({"cell_index": pd.Series(dtype="int64"),
                                  "replicate": pd.Series(dtype="int64")})
    return {"estimates": est, "composition": empty(), "truth": truth_rows,
            "standardisation": empty(), "deltas": None,
            "seed": ({pl: replicate_seed(cell, replicate, pl) for pl in pipelines}
                     if cell["gate_use"] == "S0"
                     else replicate_seed(cell, replicate, stream=seed_stream)),
            "status": status,
            "true_params": None, "true_process": None}


def contrast_values(q, d, p, d_trap=None, p_trap=None):
    """during − pre (difference) or during / pre (ratio) with the contract's
    equal-zero convention (src.k1_contract._ratio: both 0 → 1, only the
    denominator 0 → ±inf, NaN only from NaN; a τ is 0 when its trap is
    within PATH_ZERO_TOL of the target).  Vectorised over draws."""
    d = np.asarray(d, dtype=float)
    p = np.asarray(p, dtype=float)
    if PERIOD_CONTRASTS[q] == "difference":
        return d - p

    def is_zero(v, trap):
        z = v == 0.0
        if q == "tau_mfpt" and trap is not None:
            t = np.asarray(trap, dtype=float)
            z = z | (np.isfinite(t) & (np.abs(t - C.XI_TARGET) <= C.PATH_ZERO_TOL))
        return z
    zd, zp = is_zero(d, d_trap), is_zero(p, p_trap)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(zd & zp, 1.0, np.where(zp, np.sign(d) * np.inf, np.where(zd, 0.0, d / p)))
    return np.where(np.isnan(d) | np.isnan(p), np.nan, r)


def province_block_draws(wins: dict, B: int, rng) -> dict:
    members = {p: [np.flatnonzero(wins[p].prov_idx == g) for g in range(wins[p].n_prov)]
               for p in PERIODS}
    out = {p: [] for p in PERIODS}
    for _ in range(B):
        for p in PERIODS:
            G = len(members[p])
            dp = rng.integers(0, G, G)
            out[p].append(np.concatenate([members[p][g] for g in dp]))
    return out


def bootstrap_draw_pairs(pipeline: str, variant: str, mechanism: str, sim_p: tuple,
                         win: Window, idx: np.ndarray) -> pd.DataFrame:
    if getattr(win, "drop_pairs", ()):
        raise ValueError("a bootstrap on a dated-shock null window is not registered (B = 0)")
    X, Y, P, lp, ll = sim_p
    idx = np.asarray(idx, dtype=np.int64)
    wd = Window(win.period, np.arange(len(idx)), win.prov[idx], P[idx])
    if variant == "V1" and mechanism != "a":
        return v1_pairs(pipeline, Y[idx], P[idx], lp[idx], ll[idx], wd)
    return pairs_from_matrix(xi_matrix(pipeline, (X[idx], Y[idx], P[idx], lp[idx], ll[idx]),
                                       wd), wd)


def _bootstrap(pipeline: str, variant: str, mechanism: str, sim: dict, wins: dict,
               warm: dict, draws: dict) -> dict | None:
    B = len(draws[PERIODS[0]])
    out = {r: {p: {q: np.full(B, np.nan) for q in QUANTITIES} for p in PERIODS}
           for r in OBS_RULES}
    r0 = OBS_RULES[0]
    for b in range(B):
        for p in PERIODS:
            if warm[p] is None:
                continue
            try:
                pr = bootstrap_draw_pairs(pipeline, variant, mechanism, sim[p], wins[p],
                                          draws[p][b])
                xi, dxi = pr["xi_t"].to_numpy(), pr["delta_xi"].to_numpy()
                if len(xi) < 20:
                    continue
                fit = fit_mle(xi, dxi, x0=warm[p])
                o = observables(fit["params"], xi)
            except Exception:
                continue
            for q in QUANTITIES:
                if q != "rho1":
                    out[r0][p][q][b] = o[q]
            out[r0][p]["rho1"][b] = C.rho1_from_pairs(xi, dxi)
    return out


def order_ci(v: np.ndarray) -> tuple[float, float]:
    """Symmetric order-statistic interval of the draws v (no NaN; ±inf are
    ordinary values): k = ceil(CI_TAIL · n), lo = k-th smallest, hi = k-th
    largest.  NaN for an empty v."""
    v = np.sort(np.asarray(v, dtype=float))
    n = len(v)
    if n == 0:
        return np.nan, np.nan
    k = max(1, int(np.ceil(CI_TAIL * n - 1e-12)))
    return float(v[k - 1]), float(v[n - k])


def _ci(v: np.ndarray, B: int):
    """(lo, hi, n_nan, lo_literal, hi_literal).  lo/hi: order_ci over the
    non-NaN draws (±inf kept); whether that CI is usable is decided
    downstream by the tol0 / tol5 rules from the NaN count, which is always
    written (tol0: usable only with no NaN draw, so then n = B).
    lo_literal/hi_literal: every draw in the denominator, a NaN draw at −∞ for
    the lower and +∞ for the upper bound (CI_NAN_RULES 'literal'; REPORT ONLY,
    anti-conservative: it covers any truth once n_nan ≥ k)."""
    v = np.asarray(v, dtype=float)
    n_nan = int(np.isnan(v).sum())
    if len(v) == 0:
        return np.nan, np.nan, 0, np.nan, np.nan
    lo_l = order_ci(np.where(np.isnan(v), -np.inf, v))[0]
    hi_l = order_ci(np.where(np.isnan(v), np.inf, v))[1]
    fin = v[~np.isnan(v)]
    if len(fin) == 0:
        return np.nan, np.nan, n_nan, lo_l, hi_l
    lo, hi = order_ci(fin)
    return lo, hi, n_nan, lo_l, hi_l


def _emit(rows, pl, var, pe, ids, bd, tr, B):
    """pe: {period: {rule: {q: v}}}; bd: {rule: {period: {q: draws}}}."""
    nan5 = (np.nan, np.nan, 0, np.nan, np.nan)
    for r in OBS_RULES:
        base = {"pipeline": pl, "saai_spec": SAAI_SPEC_LABEL[pl], "variant": var,
                "obs_rule": r, "B": B}
        for p in PERIODS:
            for q in QUANTITIES:
                lo, hi, nn, lol, hil = _ci(bd[r][p][q], B) if bd is not None else nan5
                rows.append({**base, "period": p, "quantity": q, "estimate": pe[p][r][q],
                             "truth": tr[p][q], "ci_lo": lo, "ci_hi": hi, "n_nan_boot": nn,
                             "ci_lo_lit": lol, "ci_hi_lit": hil, "id_status": ids[p]})
        for q in QUANTITIES:
            e = float(contrast_values(q, pe["during"][r][q], pe["pre"][r][q],
                                      pe["during"][r]["xi_trap"], pe["pre"][r]["xi_trap"]))
            t = float(contrast_values(q, tr["during"][q], tr["pre"][q],
                                      tr["during"]["xi_trap"], tr["pre"]["xi_trap"]))
            if bd is not None:
                v = contrast_values(q, bd[r]["during"][q], bd[r]["pre"][q],
                                    bd[r]["during"]["xi_trap"], bd[r]["pre"]["xi_trap"])
                lo, hi, nn, lol, hil = _ci(v, B)
            else:
                lo, hi, nn, lol, hil = nan5
            rows.append({**base, "period": "contrast", "quantity": q, "estimate": e, "truth": t,
                         "ci_lo": lo, "ci_hi": hi, "n_nan_boot": nn,
                         "ci_lo_lit": lol, "ci_hi_lit": hil,
                         "id_status": f"{ids['pre']}|{ids['during']}"})
