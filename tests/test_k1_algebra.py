"""Part A: every step of the derivation of the composition term (Supplementary Information) as a SymPy identity.
Part B: numerical checks — Lemma 2 reproduced by OLS on a large synthetic
single transition with chosen (μ, σ); Lemma 2' recovered by fit_mle; the
KL projection equal to the MLE limit; the truth recovered by pooled fit_mle
on no-exit panels (B11, B12); zero predicted bias under constant
composition; the shared conventions and guards of contract version 2.
"""
import numpy as np
import pandas as pd
import pytest
import sympy as sp

from src import k1_contract as C
from src import k1_theory as T
from src.k1_contract import (K1A_CELLS, K1A_GATE, NONFINITE_MAX_FRAC,
                             PATH_ZERO_TOL, QUANTITIES, R_PER_CELL, REPORT_ROWS,
                             TRUE_PARAM_KEYS, X_UNITS, cell_key, check_cell_config,
                             contrast_bias, k1a_cell_verdict, law_claims,
                             median_with_nonfinite, observables,
                             predicted_effect_sizes, rho1_from_pairs, s0_check)
from src.km_mle import _A, _logB, derived_observables, fit_mle

xi, eta, a0, a1, D = sp.symbols("xi eta a0 a1 D", real=True)
mu_t, mu_t1 = sp.symbols("mu_t mu_t1", real=True)
s_t, s_t1 = sp.symbols("sigma_t sigma_t1", positive=True)


def test_A1_lemma1_finite_identity():
    s0 = sp.symbols("s0:3")
    l0 = sp.symbols("l0:2")
    s1 = sp.symbols("u0:3")
    e1 = sp.symbols("e0:1")
    c_t = sp.solve(sp.Eq(sum(s0) + sum(l0), 0), s0[0])[0]
    c_t1 = sp.solve(sp.Eq(sum(s1) + sum(e1), 0), s1[0])[0]
    st0 = [c_t] + list(s0[1:])
    st1 = [c_t1] + list(s1[1:])
    lhs = sum(b - a for a, b in zip(st0, st1)) / 3
    n_L, n_E, n_S = 2, 1, 3
    m_L, m_E = sum(l0) / n_L, sum(e1) / n_E
    assert sp.simplify(lhs - (n_L * m_L - n_E * m_E) / n_S) == 0


def _dxi_linear():
    x = mu_t + s_t * xi
    x1 = (1 + a1) * x + a0 + eta
    return sp.expand((x1 - mu_t1) / s_t1 - xi)


def test_A2_lemma2_pathwise_decomposition():
    a0h = ((1 + a1) * mu_t + a0 - mu_t1) / s_t1
    a1h = (1 + a1) * s_t / s_t1 - 1
    assert sp.simplify(_dxi_linear() - (a0h + a1h * xi + eta / s_t1)) == 0


def test_A3_lemma2_population_ols():
    e, v = sp.symbols("e v", positive=True)
    dxi = _dxi_linear()
    c1 = dxi.coeff(eta)
    rest = sp.expand(dxi - c1 * eta)
    k1 = rest.coeff(xi)
    k0 = sp.expand(rest - k1 * xi)
    assert sp.diff(k0, xi) == 0 and sp.diff(k1, xi) == 0
    cov = k1 * ((v + e ** 2) - e ** 2)
    slope = sp.simplify(cov / v)
    intercept = sp.simplify((k0 + k1 * e) - slope * e)
    assert sp.simplify(slope - ((1 + a1) * s_t / s_t1 - 1)) == 0
    assert sp.simplify(intercept - ((1 + a1) * mu_t + a0 - mu_t1) / s_t1) == 0
    assert sp.simplify(c1 ** 2 * D - D / s_t1 ** 2) == 0


def test_A4_lemma2prime_closure_of_the_family():
    A = sp.symbols("A0:4", real=True)
    C = sp.symbols("C0:3", real=True)
    x = mu_t + s_t * xi
    Ax = sum(A[k] * x ** k for k in range(4))
    At = sp.expand((x + Ax - mu_t1) / s_t1 - xi)
    logBt = sp.expand(sum(C[k] * x ** k for k in range(3)) - 2 * sp.log(s_t1))
    assert sp.Poly(At, xi).degree() == 3
    assert sp.Poly(logBt, xi).degree() == 2
    rng = np.random.default_rng(0)
    vals = rng.normal(size=7)
    m_, s_, m1_, s1_ = 0.3, 1.4, -0.2, 0.9
    sub = {**{A[k]: vals[k] for k in range(4)}, **{C[k]: vals[4 + k] for k in range(3)},
           mu_t: m_, s_t: s_, mu_t1: m1_, s_t1: s1_}
    coef_A = [float(sp.Poly(At, xi).coeff_monomial(xi ** k).subs(sub)) for k in range(4)]
    coef_B = [float(sp.Poly(logBt, xi).coeff_monomial(xi ** k).subs(sub)) for k in range(3)]
    got = T.map_params(vals, m_, s_, m1_, s1_)
    np.testing.assert_allclose(got, coef_A + coef_B, rtol=1e-12, atol=1e-12)


def test_A5_truth_map_is_A_over_s():
    m, s = sp.symbols("m s", positive=True)
    A = sp.symbols("A0:4", real=True)
    x = m + s * xi
    Ax = sum(A[k] * x ** k for k in range(4))
    assert sp.simplify((x + Ax - m) / s - xi - Ax / s) == 0


def test_A6_ou_stationary_moments():
    m = -a0 / a1
    s2 = D / (1 - (1 + a1) ** 2)
    assert sp.simplify((1 + a1) * m + a0 - m) == 0
    assert sp.simplify((1 + a1) ** 2 * s2 + D - s2) == 0


def test_A7_pooled_ols_closed_form():
    w1, xb1, xb2, V1, V2, M1, M2, b1, b2 = sp.symbols(
        "w1 xb1 xb2 V1 V2 M1 M2 b1 b2", real=True)
    w2 = 1 - w1
    Ex = w1 * xb1 + w2 * xb2
    Exx = w1 * (V1 + xb1 ** 2) + w2 * (V2 + xb2 ** 2)
    Ey = w1 * M1 + w2 * M2
    Exy = w1 * (xb1 * M1 + b1 * V1) + w2 * (xb2 * M2 + b2 * V2)
    slope_raw = (Exy - Ex * Ey) / (Exx - Ex ** 2)
    xbar, Mbar = Ex, Ey
    Sxx = w1 * (V1 + (xb1 - xbar) ** 2) + w2 * (V2 + (xb2 - xbar) ** 2)
    Sxy = w1 * (b1 * V1 + (xb1 - xbar) * (M1 - Mbar)) + w2 * (
        b2 * V2 + (xb2 - xbar) * (M2 - Mbar))
    assert sp.simplify(slope_raw - Sxy / Sxx) == 0
    b = Sxy / Sxx
    root = sp.solve(sp.Eq(Mbar + b * (xi - xbar), 0), xi)[0]
    assert sp.simplify(root - (xbar - Mbar / b)) == 0


def test_A8_zeroth_order_law():
    Mb = sp.symbols("Mbar", real=True)
    a1n = sp.symbols("a1n", negative=True)
    b = (1 + a1n) * 1 - 1
    xi_star = 0 - Mb / b
    assert sp.simplify(xi_star - Mb / sp.Abs(a1n)) == 0


def test_A9_slope_first_order_in_sigma_ratio():
    r = sp.symbols("r", positive=True)
    assert sp.simplify(((1 + a1) * r - 1) - a1 - (1 + a1) * (r - 1)) == 0


def test_A10_ou_spurious_barrier():
    a = sp.symbols("a", negative=True)
    xs, xT, B = sp.symbols("xi_s xi_T", real=True) + (sp.symbols("B", positive=True),)
    V = -sp.integrate(a * (xi - xs) / B, xi)
    two_dV = 2 * (V.subs(xi, xT) - V.subs(xi, xs))
    assert sp.simplify(two_dV - sp.Abs(a) * (xs - xT) ** 2 / B) == 0
    assert sp.diff(two_dV, xs).subs(xs, xT) == 0


def test_A11_rho1():
    b = sp.symbols("b", real=True)
    expr = sp.expand(xi + (sp.symbols("c") + b * xi))
    assert expr.coeff(xi) == 1 + b


def test_A12_first_order_root_shift():
    d = sp.symbols("delta", real=True)
    k, r0 = sp.symbols("k r0", positive=True)
    g0, g1 = sp.symbols("g0 g1", real=True)
    A = -k * (xi - r0) - (xi - r0) ** 3
    g = g0 + g1 * xi
    F = A + d * g
    dxdd = -sp.diff(F, d) / sp.diff(F, xi)
    lhs = sp.simplify(dxdd.subs({d: 0, xi: r0}))
    rhs = sp.simplify((-g / sp.diff(A, xi)).subs(xi, r0))
    assert sp.simplify(lhs - rhs) == 0


def test_A13_envelope_for_a_true_barrier():
    B0, beta = sp.Rational(1, 2), sp.Rational(1, 10)
    A = -xi * (xi - 1) * (xi + 1)
    B = B0 * (1 + beta * xi)
    dA, dB = sp.Rational(1, 7) + sp.Rational(1, 5) * xi, sp.Rational(1, 9) * xi
    d = sp.symbols("delta", real=True)

    def barrier(dv):
        Ad = A + dv * dA
        Bd = B + dv * dB
        roots = sorted(float(r) for r in sp.Poly(sp.expand(Ad), xi).nroots()
                       if abs(sp.im(r)) < 1e-12)
        lo, saddle = roots[0], roots[1]
        return float(sp.Integral(-Ad / Bd, (xi, lo, saddle)).evalf(30))

    h = 1e-5
    num = (barrier(h) - barrier(-h)) / (2 * h)
    pred = float(sp.Integral(-(dA / B - A * dB / B ** 2), (xi, -1, 0)).evalf(30))
    assert abs(num - pred) < 1e-6


def test_A14_stayer_design_moments():
    nS, nL, mL, = sp.symbols("n_S n_L m_L", positive=True)
    xbar = -nL * mL / nS
    V = 1 - (nS * xbar ** 2 + nL * mL ** 2) / (nS + nL)
    mean = (nS * xbar + nL * mL) / (nS + nL)
    msq = (nS * (V + xbar ** 2) + nL * (V + mL ** 2)) / (nS + nL)
    assert sp.simplify(mean) == 0
    assert sp.simplify(msq - 1) == 0


def test_B1_lemma2_by_ols_on_large_single_transition():
    """Lemma 2 reproduced by OLS on N = 400,000 stayers with chosen (μ, σ)
    and a deliberately non-Gaussian design."""
    rng = np.random.default_rng(20261021)
    N = 400_000
    A0, A1, Dv = 0.05, -0.125, 0.4
    m_, s_, m1_, s1_ = 0.3, 1.7, -0.2, 1.3
    x = rng.gamma(2.0, 1.0, N) - 1.5
    x1 = (1 + A1) * x + A0 + rng.normal(0, np.sqrt(Dv), N)
    z, z1 = (x - m_) / s_, (x1 - m1_) / s1_
    dz = z1 - z
    X = np.column_stack([np.ones(N), z])
    beta, *_ = np.linalg.lstsq(X, dz, rcond=None)
    resid = dz - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var())
    pred = T.lemma2_linear(A0, A1, Dv, m_, s_, m1_, s1_)
    assert abs(beta[0] - pred["a0_hat"]) < 5 * se[0]
    assert abs(beta[1] - pred["a1_hat"]) < 5 * se[1]
    assert abs(resid.var() - pred["D_hat"]) < 5 * pred["D_hat"] * np.sqrt(2 / N)


def test_B2_lemma2prime_recovered_by_fit_mle():
    """Lemma 2' (cubic drift, state-dependent noise): fit_mle on one
    synthetic transition standardised with chosen (μ, σ) recovers map_params."""
    rng = np.random.default_rng(7)
    N = 200_000
    th = np.array([0.05, -0.2, 0.03, -0.04, np.log(0.3), -0.15, 0.02])
    m_, s_, m1_, s1_ = 0.1, 1.2, -0.05, 1.1
    x = rng.normal(0.2, 1.0, N)
    x1 = x + _A(th, x) + np.sqrt(np.exp(_logB(th, x))) * rng.normal(size=N)
    z, z1 = (x - m_) / s_, (x1 - m1_) / s1_
    fit = fit_mle(z, z1 - z)
    target = T.map_params(th, m_, s_, m1_, s1_)
    grid = np.linspace(-1.5, 1.5, 31)
    np.testing.assert_allclose(_A(fit["params"], grid), _A(target, grid), atol=0.01)
    np.testing.assert_allclose(_logB(fit["params"], grid), _logB(target, grid), atol=0.03)


def test_B3_projection_is_the_mle_limit_on_a_mixture():
    """The KL projection equals the limit of fit_mle on data drawn from a
    mixture of transition laws (arbitrary, uncalibrated numbers)."""
    rng = np.random.default_rng(11)
    lines = [(0.4, -0.10, 0.8, -0.30, 0.25), (-0.1, -0.15, 1.0, 0.02, 0.20),
             (0.0, -0.12, 0.9, 0.00, 0.22)]
    ns = [40_000, 80_000, 80_000]
    xs, ys, nodes, wts, At, Bt = [], [], [], [], [], []
    zq = np.sort(rng.normal(size=4000))
    zq = (zq - zq.mean()) / zq.std()
    for (xb, sl, V, M, B), n in zip(lines, ns):
        x = xb + np.sqrt(V) * rng.normal(size=n)
        xs.append(x)
        ys.append(M + sl * (x - xb) + np.sqrt(B) * rng.normal(size=n))
        q = xb + np.sqrt(V) * zq
        nodes.append(q)
        wts.append(np.full(q.size, n / q.size))
        At.append(M + sl * (q - xb))
        Bt.append(np.full(q.size, B))
    fit = fit_mle(np.concatenate(xs), np.concatenate(ys))
    th = T.project(np.concatenate(nodes), np.concatenate(wts),
                   np.concatenate(At), np.concatenate(Bt))
    grid = np.linspace(-1.5, 1.5, 31)
    np.testing.assert_allclose(_A(fit["params"], grid), _A(th, grid), atol=0.01)
    np.testing.assert_allclose(_logB(fit["params"], grid), _logB(th, grid), atol=0.04)


def test_B4_stationary_law_matches_ou_closed_form():
    tp = np.array([0.02, -0.125, 0, 0, np.log(0.3), 0, 0])
    _, _, m, s = T.stationary_law(tp)
    assert abs(m - (-0.02 / -0.125)) < 1e-6
    assert abs(s - np.sqrt(0.3 / (1 - 0.875 ** 2))) < 1e-6


def _comp(rows, period="during", design=None, re_placement="none",
          exit_dating="info_t", y0=None):
    """rows: (n_S, n_L, n_E, mu_x_t, sigma_x_t, mu_x_t1, sigma_x_t1, mech[,
    xi_bar_stay, var_stay]).  Without the last two, the observed stayer
    moments are filled with the values that are exact for exit-only rows
    under equal within-group variance (stayer_design)."""
    y0 = y0 if y0 is not None else T.period_transitions(period)[0]
    out = []
    for i, r in enumerate(rows):
        if len(r) == 10:
            xb, V = r[8], r[9]
        else:
            xb, V, _ = T.stayer_design(r[0], r[1], r[2], r[7])
        out.append(dict(replicate=0, period=period, pipeline="direct",
                        year_t=y0 + i, n_stay=r[0], n_leave=r[1], n_enter=r[2],
                        mu_x_t=r[3], sigma_x_t=r[4], mu_x_t1=r[5],
                        sigma_x_t1=r[6], x_units=X_UNITS, mech_term=r[7],
                        xi_bar_stay=xb, var_stay=V, re_placement=re_placement,
                        exit_dating=exit_dating))
    return pd.DataFrame(out)


OU = dict(a0=0.02, a1=-0.125, a2=0.0, a3=0.0, c0=np.log(0.3), c1=0.0, c2=0.0,
          **T.NO_GUARD)
CUBIC = dict(a0=0.05, a1=-0.2, a2=0.03, a3=-0.04, c0=np.log(0.3), c1=-0.1, c2=0.02,
             **T.NO_GUARD)


CELL_N = 2300
CELL_N_TOTAL = 2 * CELL_N


def _std(tp, n=CELL_N):
    return T.stationary_standardisation(tp, n=n)


@pytest.mark.parametrize("tp", [OU, CUBIC], ids=["ou", "cubic_multiplicative"])
@pytest.mark.parametrize("exchange", [False, True], ids=["closed", "neutral_exchange"])
def test_B5_zero_bias_under_constant_composition(tp, exchange):
    """No composition change (or an exchange of equal numbers of typical
    counties, mech = 0) at the complete-panel moments → zero predicted bias."""
    _, _, m, s = T.stationary_law(T.params_vector(tp))
    nl, ne = (30, 30) if exchange else (0, 0)
    comp = _comp([(300, nl, ne, m, s, m, s, 0.0)] * 6)
    bias = T.predict_bias(comp, tp, "during", _std(tp))
    for q in QUANTITIES:
        if q in T.SIMULATION_ONLY:
            assert np.isnan(bias[q])
        else:
            assert abs(bias[q]) < 1e-4, (q, bias[q])


def test_B6_contract_shape_and_guards():
    tp = OU
    _, _, m, s = T.stationary_law(T.params_vector(tp))
    comp = _comp([(300, 0, 0, m, s, m, s, 0.0)] * 5
                 + [(193, 192, 0, m, s, m, 0.9 * s, -0.45)])
    bias = T.predict_bias(comp, tp, "during", _std(tp))
    assert set(bias) == set(QUANTITIES)
    assert all(k in tp for k in TRUE_PARAM_KEYS)
    with pytest.raises(ValueError, match="lacks the pre transitions"):
        T.predict_bias(comp, tp, "pre", _std(tp))
    with pytest.raises(ValueError):
        T.predict_bias(pd.concat([comp, comp.assign(pipeline="saai")]), tp,
                       "during", _std(tp))
    with pytest.raises(ValueError):
        T.predict_bias(pd.concat([comp, comp.assign(replicate=1)]), tp,
                       "during", _std(tp))
    with pytest.raises(ValueError, match="outside"):
        T.predict_bias(comp.assign(year_t=comp["year_t"] + 10), tp, "during", _std(tp))
    with pytest.raises(ValueError, match="lacks years"):
        T.predict_bias(comp, tp, "during", _std(tp).iloc[:5])


def test_B7_ou_prediction_agrees_with_the_closed_form_law():
    """For OU the cubic projection's ρ1 equals the closed-form pooled slope
    (same lines, same designs) and its attractor is near ξ*_lin."""
    _, _, m, s = T.stationary_law(T.params_vector(OU))
    comp = _comp([(300, 0, 0, m, s, m, s, 0.0)] * 3
                 + [(193, 192, 0, m, s, m, 0.9 * s, -0.45)]
                 + [(193, 0, 0, m, 0.9 * s, m, 0.93 * s, 0.0)] * 2)
    out = T.predict(comp, OU, "during", _std(OU))
    cf = T.closed_form_ou(comp, OU["a1"], np.exp(OU["c0"]), "during")
    assert abs(out["estimate"]["rho1"] - cf["rho1_lin"]) < 1e-10
    assert abs(out["estimate"]["xi_trap"] - cf["xi_star_lin"]) < 0.05


def test_B8_sign_of_the_ou_law():
    """Bookkeeping only (signs, not the law): a poor block leaving moves the
    predicted attractor down; the zeroth-order law has the same sign."""
    _, _, m, s = T.stationary_law(T.params_vector(OU))
    comp = _comp([(300, 0, 0, m, s, m, s, 0.0)] * 5
                 + [(193, 192, 0, m, s, m, s, -0.45)])
    b = T.predict_bias(comp, OU, "during", _std(OU))
    cf = T.closed_form_ou(comp, OU["a1"], np.exp(OU["c0"]), "during")
    assert b["xi_trap"] < 0 and cf["zeroth_order_xi_star"] < 0
    assert b["two_delta_V"] > 0


def _block_exit_window(mu, s):
    return _comp([(300, 0, 0, mu, s, mu, s, 0.0)] * 3
                 + [(193, 192, 0, mu, s, mu, 0.9 * s, -0.45)]
                 + [(193, 0, 0, mu, 0.9 * s, mu, 0.93 * s, 0.0)] * 2)


def test_B9_linear_prediction_does_not_depend_on_the_latent_location():
    _, _, m, s = T.stationary_law(T.params_vector(OU))
    e_m = T.predict(_block_exit_window(m, s), OU, "during", _std(OU))["estimate"]
    e_0 = T.predict(_block_exit_window(m + 0.7 * s, s), OU, "during", _std(OU))["estimate"]
    for q in QUANTITIES:
        if q not in T.SIMULATION_ONLY:
            assert abs(e_m[q] - e_0[q]) < 1e-8, q
    _, _, mc, sc = T.stationary_law(T.params_vector(CUBIC))
    c_m = T.predict_bias(_block_exit_window(mc, sc), CUBIC, "during", _std(CUBIC))
    c_0 = T.predict_bias(_block_exit_window(mc + 0.7 * sc, sc), CUBIC, "during",
                         _std(CUBIC))
    assert abs(c_m["xi_trap"] - c_0["xi_trap"]) > 1e-3


def test_B10_location_guards():
    """x_units must be 'latent_x'; all-zero means are refused for a
    non-linear model (residual-unit input), allowed for OU."""
    _, _, _, sc = T.stationary_law(T.params_vector(CUBIC))
    with pytest.raises(ValueError, match="latent-x moments"):
        T.predict_bias(_block_exit_window(0.0, sc), CUBIC, "during", _std(CUBIC))
    _, _, m, s = T.stationary_law(T.params_vector(OU))
    T.predict_bias(_block_exit_window(0.0, s), OU, "during", _std(OU))
    for bad in ("residual", "log_y", "xi"):
        with pytest.raises(ValueError, match="x_units"):
            T.predict_bias(_block_exit_window(m, s).assign(x_units=bad), OU,
                           "during", _std(OU))


def _no_exit_panel(tp, N, infl, seed, years):
    th = T.params_vector(tp)
    _, _, m, s = T.stationary_law(th)
    rng = np.random.default_rng(seed)
    x = m + infl * s * rng.standard_normal(N)
    X = [x]
    for _ in years[1:]:
        x = x + _A(th, x) + np.sqrt(np.exp(_logB(th, x))) * rng.standard_normal(N)
        X.append(x)
    X = np.array(X)
    mu, sd = X.mean(1), X.std(1)
    std = pd.DataFrame({"year": years, "n": N, "mu_x": mu, "sigma_x": sd,
                        "x_units": X_UNITS, C.SHOCK_SHARE_COLUMN: 0.0})
    Z = (X - mu[:, None]) / sd[:, None]
    return std, Z[:-1].ravel(), (Z[1:] - Z[:-1]).ravel()


def test_B11_truth_recovered_by_fit_mle_on_a_nonstationary_no_exit_panel():
    std, xi, dxi = _no_exit_panel(OU, 20_000, 1.6, 3, list(range(2013, 2020)))
    fit = fit_mle(xi, dxi)
    est = observables(fit["params"], xi)
    tr = T.truth(OU, std, "during")
    cp = T._truth_full(T.params_vector(OU), std, "during")["rho1_complete_projection"]
    assert abs(rho1_from_pairs(xi, dxi) - cp) < 0.003
    assert tr["rho1"] == pytest.approx(1 + OU["a1"], abs=1e-12)
    assert abs(cp - tr["rho1"]) > 0.02
    assert abs(est["xi_trap"] - tr["xi_trap"]) < 0.02
    assert abs(est["mean_diffusion"] / tr["mean_diffusion"] - 1) < 0.02
    assert abs(est["two_delta_V"] - tr["two_delta_V"]) < 1e-3
    _, _, _, s_st = T.stationary_law(T.params_vector(OU))
    assert std["sigma_x"].iloc[0] > 1.4 * s_st


def test_B12_truth_recovered_by_fit_mle_cubic_stationary_start():
    std, xi, dxi = _no_exit_panel(CUBIC, 20_000, 1.0, 4, list(range(2013, 2020)))
    fit = fit_mle(xi, dxi)
    est = observables(fit["params"], xi)
    tr = T.truth(CUBIC, std, "during")
    assert abs(rho1_from_pairs(xi, dxi) - tr["rho1"]) < 0.005
    assert abs(est["xi_trap"] - tr["xi_trap"]) < 0.02
    assert abs(est["mean_diffusion"] / tr["mean_diffusion"] - 1) < 0.02
    assert abs(est["B_ratio_w"] / tr["B_ratio_w"] - 1) < 0.03


def test_B13_observed_stayer_moments_are_used():
    rng = np.random.default_rng(5)
    n_S, n_L = 600, 100
    x_s = rng.standard_normal(n_S)
    x_l = rng.normal(-1.2, 0.3, n_L)
    x_t = np.concatenate([x_s, x_l])
    mu_t, sd_t = x_t.mean(), x_t.std()
    z_s = (x_s - mu_t) / sd_t
    xb_obs, V_obs = z_s.mean(), z_s.var()
    m_L = ((x_l - mu_t) / sd_t).mean()
    assert abs(xb_obs + n_L * m_L / n_S) < 1e-12
    xb_fb, _, _ = T.stayer_design(n_S, n_L, 100, 0.0)
    assert xb_fb == 0.0 and xb_obs > 0.1
    _, _, m, s = T.stationary_law(T.params_vector(OU))
    rows = [(300, 0, 0, m, s, m, s, 0.0, 0.0, 1.0)] * 5 + [
        (n_S, n_L, 100, m, s, m, s, 0.0, xb_obs, V_obs)]
    out = T.predict(_comp(rows), OU, "during", _std(OU))
    assert out["diag"]["xbar_stay"][-1] == xb_obs
    assert out["diag"]["V_stay"][-1] == V_obs
    fb = T.predict(_comp(rows), OU, "during", _std(OU), design="fallback")
    assert fb["diag"]["xbar_stay"][-1] == 0.0 and fb["diag"]["design"] == "fallback"
    with pytest.raises(ValueError, match="observed"):
        T.predict(_comp(rows).drop(columns=["var_stay"]), OU, "during", _std(OU))


def test_B14_validity_flags():
    _, _, m, s = T.stationary_law(T.params_vector(OU))
    for rep, ex, ok in [("none", "info_t", True), ("initial_condition", "exogenous", True),
                        ("shock", "info_t", True), ("persistent", "info_t", False),
                        ("persistent", "exogenous", False),
                        ("shock", "info_t1", False)]:
        d = T.predict(_block_exit_window(m, s).assign(re_placement=rep, exit_dating=ex),
                      OU, "during", _std(OU))["diag"]
        assert d["valid_for_law"] is ok, (rep, ex)
    for old in ("county_fe", "innovation", "drift_offset", "observation_offset"):
        with pytest.raises(ValueError, match="unknown"):
            T.predict(_block_exit_window(m, s).assign(re_placement=old), OU,
                      "during", _std(OU))


def test_B15_degenerate_convention_default_unchanged_and_limit_continuous():
    xi_data = np.random.default_rng(0).standard_normal(20_000)
    lo, hi = np.percentile(xi_data, 1), np.percentile(xi_data, 99)
    h = (hi - lo) / 599
    base = np.array([0.0, -0.15, 0.0, 0.0, np.log(0.25), 0.0, 0.0])

    def at(root, mode):
        th = base.copy()
        th[0] = 0.15 * root
        return derived_observables(th, xi_data, degenerate=mode)

    d = at(0.3 * h, "nan")
    assert d["id_status"] == "degenerate_path" and np.isnan(d["xi_trap"])
    z = at(0.0, "limit")
    assert z["id_status"] == "subgrid_path" and z["delta_V"] < 1e-12 and z["tau_mfpt"] < 1e-12
    assert abs(z["xi_trap"]) < 1e-12
    below, above = at(-0.9 * h, "limit"), at(-2.6 * h, "limit")
    assert below["id_status"] == "subgrid_path" and above["id_status"] == "stable_zero"
    assert 0 < below["tau_mfpt"] < above["tau_mfpt"] < 4 * below["tau_mfpt"]
    assert 0 <= below["delta_V"] < above["delta_V"]
    o = observables(base, xi_data)
    assert o["two_delta_V"] < 1e-12 and o["id_status"] == "subgrid_path"


def test_B16_contrast_conventions():
    """Contract v3: difference contrast; ratio contrast with the equal-zero
    convention (OU truth τ = 0 in both periods → true ratio 1, so the τ check
    tests the spurious slow-down) and ±inf kept; NaN only when undefined."""
    est_p = dict(xi_trap=0.0, two_delta_V=0.0, tau_mfpt=0.0, mean_diffusion=1.0,
                 B_ratio_w=1.0, rho1=0.9)
    est_d = dict(xi_trap=-0.3, two_delta_V=0.1, tau_mfpt=2.0, mean_diffusion=0.8,
                 B_ratio_w=0.9, rho1=0.85)
    tru = dict(xi_trap=1e-17, two_delta_V=1e-33, tau_mfpt=5e-16,
               mean_diffusion=1.0, B_ratio_w=1.0, rho1=0.9)
    c = contrast_bias(est_p, est_d, tru, tru)
    assert c["xi_trap"] == pytest.approx(-0.3) and c["rho1"] == pytest.approx(-0.05)
    assert c["two_delta_V"] == pytest.approx(0.1)
    assert c["tau_mfpt"] == np.inf
    assert c["mean_diffusion"] == pytest.approx(-0.2)
    e2 = dict(est_p, tau_mfpt=1.0, xi_trap=-0.05)
    assert contrast_bias(e2, est_d, tru, tru)["tau_mfpt"] == pytest.approx(1.0)
    assert contrast_bias(est_p, est_p, tru, tru)["tau_mfpt"] == 0.0
    tru_d = dict(tru, xi_trap=-0.2, tau_mfpt=1.0)
    assert contrast_bias(e2, est_d, tru, tru_d)["tau_mfpt"] == -np.inf
    assert np.isnan(contrast_bias(dict(e2, tau_mfpt=np.nan), est_d, tru, tru)["tau_mfpt"])
    tr_d, tr_p = T.truth(OU, _std(OU), "during"), T.truth(OU, _std(OU), "pre")
    assert abs(tr_d["xi_trap"]) < 1e-3 * PATH_ZERO_TOL
    assert contrast_bias(tr_p, tr_d, tr_p, tr_d)["tau_mfpt"] == 0.0
    r = median_with_nonfinite([1.0, np.nan, 3.0, np.inf])
    assert r == {"median": 2.0, "n_nonfinite": 2, "n_nan": 1, "n_inf": 1, "n_total": 4}
    assert median_with_nonfinite([1.0, np.nan, 3.0, np.inf], allow_inf=True)[
        "median"] == 3.0
    assert K1A_GATE["pipeline"] == "saai" and K1A_GATE["row"] == "projection"


def _contrasts(R, xi, dV, rho, tau, nan_xi=0):
    out = []
    for r in range(R):
        out.append(dict(replicate=r, xi_trap=np.nan if r < nan_xi else xi,
                        two_delta_V=dV, rho1=rho, tau_mfpt=tau, mean_diffusion=0.0,
                        B_ratio_w=np.nan))
    return out


R500 = R_PER_CELL
RATIO_Q = [q for q, k in C.PERIOD_CONTRASTS.items() if k == "ratio"]


@pytest.fixture
def guards(monkeypatch):
    monkeypatch.setattr(C, "_require_frozen", lambda: {"frozen_commit": "TEST"})
    monkeypatch.setattr(C, "_require_s0_pass", lambda process, pipeline: {"pass": True})


def _cell(process="ou", mech="d", gated=True, rep="shock", ex="info_t", **kw):
    c = dict(process=process, exit_mechanism=mech, N=CELL_N_TOTAL,
             windows="pre:2000-2012;during:2013-2019", pipeline="saai",
             re_placement=rep, exit_dating=ex if mech != "a" else "none", gated=gated)
    return {**c, **kw}


def _fake_row(cell, row, r, per, est, tru, design=None):
    """A predict_row-shaped output with a valid digest (the gate rules are
    tested on chosen values; B17g ties predict_row itself to the gates)."""
    d = {"cell_checked": True, "cell_key": cell_key(check_cell_config(cell)),
         "row": row, "design": design or C.ROW_DESIGN[row], "replicate": r,
         "period": per, "theory_version": T.THEORY_VERSION,
         "contract_version": C.CONTRACT_VERSION,
         "true_params": C.registered_true_params(cell["process"], per),
         "true_params_check": C.REGISTERED_CHECK}
    d["digest"] = C.prediction_digest(d, est, tru)
    bias = {q: np.float64(est[q]) - np.float64(tru[q]) for q in QUANTITIES}
    return {"estimate": est, "truth": tru, "bias": bias, "diag": d}


def _preds(cell, contrasts, row="projection", design=None):
    """2R row outputs whose contrast_bias is exactly the given contrasts:
    difference quantities from a common base 0.5 (so no trap sits at the
    target), ratio quantities from 1 (an infinite ratio from a pre value 0)."""
    out = []
    for c in contrasts:
        r = c["replicate"]
        pre, dur = {}, {}
        tru_p, tru_d = {}, {}
        for q in QUANTITIES:
            v = np.float64(c[q])
            if q in RATIO_Q:
                tru_p[q] = tru_d[q] = 1.0
                if np.isinf(v):
                    pre[q], dur[q] = 0.0, float(np.sign(v))
                else:
                    pre[q], dur[q] = 1.0, 1.0 + v
            else:
                tru_p[q] = tru_d[q] = 0.5
                pre[q], dur[q] = 0.5, 0.5 + v
        out.append(_fake_row(cell, row, r, "pre", pre, tru_p, design))
        out.append(_fake_row(cell, row, r, "during", dur, tru_d, design))
    return out


def _null(R=R500, spread=0.005, seed=0):
    """No-exit contrasts centred on 0 with a small spread (band ≈ ±2 spread)."""
    z = np.random.default_rng(seed).standard_normal((R, 4)) * spread
    return [dict(replicate=i, xi_trap=a, two_delta_V=b, rho1=c, tau_mfpt=d,
                 mean_diffusion=0.0, B_ratio_w=np.nan) for i, (a, b, c, d) in enumerate(z)]


def _verdict(pred, sim, null=None, cell=None, null_cell=None, row="projection",
             preds=None, own_offset=None):
    """pred: R predicted contrasts (turned into 2R row outputs for the cell).
    For the zeroth_order row the row's own no-exit predictions are built on
    the null cell from own_offset (R contrasts; default all 0)."""
    cell = cell or _cell()
    null_cell = null_cell or {**cell, "exit_mechanism": "a", "exit_dating": "none"}
    preds = preds if preds is not None else _preds(cell, pred, row)
    npreds = None
    if row in C.OWN_OFFSET_ROWS:
        off = own_offset if own_offset is not None else _contrasts(R500, 0.0, 0.0, 0.0, 0.0)
        npreds = _preds(null_cell, off, row)
    return k1a_cell_verdict(preds, sim, _null() if null is None else null, cell,
                            null_cell, row=row, null_predictions=npreds)


def test_B17_k1a_cell_verdict_precedence(guards):
    R = R500
    sim = _contrasts(R, -0.30, 0.05, -0.02, 0.8)
    ok = _verdict(_contrasts(R, -0.36, 0.07, -0.022, np.inf), sim)
    assert ok["pass"] and ok["law_status"] == "pass"
    assert ok["reported"]["xi_trap"]["pred"]["median"] == pytest.approx(-0.36)
    assert ok["reported"]["tau_mfpt"]["pred"]["median"] == np.inf
    assert set(ok["checks"]) == {"xi_trap_rel", "rho1_rel", "two_delta_V_sign",
                                 "rho1_sign", "tau_mfpt_sign"}
    assert _verdict(_contrasts(R, -0.40, 0.07, -0.02, 1.0), sim)[
        "checks"]["xi_trap_rel"] == "fail"
    v = _verdict(_contrasts(R, -0.30, -0.01, -0.02, 1.0), sim)
    assert v["checks"]["two_delta_V_sign"] == "fail" and not v["pass"]
    k = int(NONFINITE_MAX_FRAC * R) + 1
    v = _verdict(_contrasts(R, -0.30, 0.07, -0.02, 1.0, nan_xi=k), sim)
    assert v["checks"]["xi_trap_rel"] == "fail (non-finite)"
    v = _verdict(_contrasts(R, -0.30, 0.07, -0.02, 1.0, nan_xi=k - 1), sim)
    assert v["checks"]["xi_trap_rel"] == "pass"
    v = _verdict(_contrasts(R, np.inf, 0.07, -0.02, np.inf), sim)
    assert v["checks"]["xi_trap_rel"] == "fail (non-finite)"
    assert v["checks"]["tau_mfpt_sign"] == "pass"
    v = _verdict(_contrasts(R, -0.3, 0.07, -0.02, 1.0),
                 _contrasts(R, -0.3, 0.0, -0.02, 0.8))
    assert v["checks"]["two_delta_V_sign"] == "not evaluable" and not v["pass"]
    v = _verdict(_contrasts(R, -0.36, 0.07, -0.022, 1.0), sim,
                 cell=_cell(ex="info_t1"))
    assert v["pass"] and v["law_status"] == "does not test the law"
    with pytest.raises(ValueError):
        _verdict(_contrasts(R, -0.3, 0.07, -0.02, 1.0), sim[:-1])


def test_B17_replicate_counts_are_the_registered_500(guards):
    sim = _contrasts(R500, -0.30, 0.05, -0.02, 0.8)
    pred = _contrasts(R500, -0.36, 0.07, -0.022, 1.0)
    v = _verdict(pred, sim)
    assert v["R"] == R500 and v["R_null"] == R500 and f"R={R500}" in v["label"]
    for r in (1, 40, 499, 501):
        with pytest.raises(ValueError, match="registers R = 500"):
            _verdict(_contrasts(r, -0.36, 0.07, -0.022, 1.0),
                     _contrasts(r, -0.30, 0.05, -0.02, 0.8))
    for r0 in (40, 200, 499):
        with pytest.raises(ValueError, match="registers R = 500"):
            _verdict(pred, sim, null=_null(R=r0))
    with pytest.raises(ValueError, match=r"R = 499, R0 = 500 \(need 500\)"):
        law_claims([{**v, "R": 499}])


def test_B17_predictions_must_be_made_for_the_judged_cell(guards):
    _, _, m, s = T.stationary_law(T.params_vector(OU))
    cell = _cell(pipeline="direct")
    comp = _block_exit_window(m, s).assign(re_placement="shock", exit_dating="info_t")
    d = T.predict(comp, OU, "during", _std(OU), cell=cell)["diag"]
    assert d["cell_checked"] is True and d["cell_key"] == cell_key(check_cell_config(cell))
    assert T.predict(comp, OU, "during", _std(OU))["diag"]["cell_checked"] is False
    for col, val in (("re_placement", "persistent"), ("exit_dating", "info_t1"),
                     ("pipeline", "saai")):
        with pytest.raises(ValueError, match="disagree"):
            T.predict(comp.assign(**{col: val}), OU, "during", _std(OU), cell=cell)
    with pytest.raises(ValueError, match="disagree"):
        T.zeroth_order_observables(comp.assign(pipeline="saai"), OU, "during",
                                   _std(OU), cell=cell)
    with pytest.raises(ValueError, match="gated cell"):
        T.predict(comp.assign(re_placement="none"), OU, "during", _std(OU),
                  cell=_cell(pipeline="direct", rep="none"))
    with pytest.raises(ValueError, match="needs the cell"):
        T.predict_row("projection", comp, OU, "during", _std(OU), None)
    sim = _contrasts(R500, -0.30, 0.05, -0.02, 0.8)
    pred = _contrasts(R500, -0.36, 0.07, -0.022, 1.0)
    good = _preds(_cell(), pred)
    with pytest.raises(ValueError, match="cell_key"):
        _verdict(pred, sim, preds=_preds(_cell(N=20000), pred))
    unchecked = [dict(p, diag=dict(p["diag"], cell_checked=False)) for p in good]
    with pytest.raises(ValueError, match="cell_key"):
        _verdict(pred, sim, preds=unchecked)
    with pytest.raises(ValueError, match="row"):
        _verdict(pred, sim, row="zeroth_order", preds=good)
    with pytest.raises(ValueError, match="design"):
        _verdict(pred, sim, row="fallback",
                 preds=_preds(_cell(), pred, row="fallback", design="observed"))
    edited = [dict(p, estimate=dict(p["estimate"], xi_trap=0.1)) if i == 7 else p
              for i, p in enumerate(good)]
    with pytest.raises(ValueError, match="digest"):
        _verdict(pred, sim, preds=edited)
    fake_bias = [dict(p, bias={q: 99.0 for q in QUANTITIES}) for p in good]
    v = _verdict(pred, sim, preds=fake_bias)
    assert v["reported"]["xi_trap"]["pred"]["median"] == pytest.approx(-0.36)
    with pytest.raises(ValueError, match="pre'/'during' pair"):
        _verdict(pred, sim, preds=good[:-1])
    with pytest.raises(ValueError, match="repeated"):
        _verdict(pred, sim, preds=good[:-1] + [good[-3]])
    shifted = []
    for p in good:
        if p["diag"]["period"] == "during":
            dd = dict(p["diag"], replicate=p["diag"]["replicate"] + R500)
            dd["digest"] = C.prediction_digest(dd, p["estimate"], p["truth"])
            p = dict(p, diag=dd)
        shifted.append(p)
    with pytest.raises(ValueError, match="pair"):
        _verdict(pred, sim, preds=shifted)
    with pytest.raises(ValueError, match="replicate ids"):
        _verdict(pred, [dict(x, replicate=x["replicate"] + 1) for x in sim])
    with pytest.raises(ValueError, match="replicate"):
        _verdict(pred, [{k: v for k, v in x.items() if k != "replicate"} for x in sim])
    with pytest.raises(ValueError, match="repeats"):
        _verdict(pred, sim, null=[dict(x, replicate=0) for x in _null()])


def _reg_window(process, rep="shock", ex="info_t", pipeline="direct", n=CELL_N):
    """Registered true processes of both periods, a complete-panel table at
    each period's stationary moments (2000–2012 pre, 2013–2023 during) and a
    window with no exit in 'pre' and the (d)-like block exit in 'during'."""
    tp = {p: C.registered_true_params(process, p) for p in PERIODS}
    ms = {p: T.stationary_law(*T.process_spec(tp[p]))[2:] for p in PERIODS}
    years = range(2000, 2024)
    std = pd.DataFrame({"year": list(years), "n": n,
                        "mu_x": [ms["pre" if y <= 2012 else "during"][0] for y in years],
                        "sigma_x": [ms["pre" if y <= 2012 else "during"][1] for y in years],
                        "x_units": X_UNITS, C.SHOCK_SHARE_COLUMN: 0.0})
    (mp, sp_), (md, sd) = ms["pre"], ms["during"]
    pre = _comp([(300, 0, 0, mp, sp_, mp, sp_, 0.0)] * 12, period="pre")
    comp = pd.concat([pre, _block_exit_window(md, sd)]).assign(
        re_placement=rep, exit_dating=ex, pipeline=pipeline)
    return tp, std, comp


def test_B17g_predict_row_outputs_feed_the_gates(guards):
    """predict_row is the only producer the gates accept: its outputs for
    every row carry the cell_key, row, replicate, period, the registered
    true process and a digest that pair_predictions verifies, and the
    verdict's contrasts equal contrast_bias of its estimates and truths."""
    cell = _cell(pipeline="direct")
    tp, std, comp = _reg_window("ou")
    for row in REPORT_ROWS:
        a = T.predict_row(row, comp, tp["pre"], "pre", std, cell)
        b = T.predict_row(row, comp, tp["during"], "during", std, cell)
        pairs = C.pair_predictions(cell, [a, b], row)
        assert [r for r, _, _ in pairs] == [0]
        assert a["diag"]["row"] == row and a["diag"]["design"] == C.ROW_DESIGN[row]
        assert a["diag"]["true_params"] == tp["pre"]
        assert a["diag"]["true_params_check"] == C.REGISTERED_CHECK
        cb = contrast_bias(a["estimate"], b["estimate"], a["truth"], b["truth"])
        if row == "projection":
            want = T.predict(comp, tp["during"], "during", std, cell=cell)["estimate"]
            assert b["estimate"]["xi_trap"] == want["xi_trap"]
        assert np.isfinite(cb["xi_trap"]), row
        with pytest.raises(ValueError, match="row"):
            C.pair_predictions(cell, [a, b], "fallback" if row != "fallback" else
                               "projection")


def test_B17h_predictions_bind_the_registered_true_process(guards):
    reg = {p: C.registered_true_params("cubic", p) for p in PERIODS}
    assert reg["pre"]["guard_lo"] is not None
    assert reg["pre"]["slope_floor"] == -abs(_calibration()["v1_mle_fits"]["pre"]["ou"][1])
    assert C.registered_true_params("ou", "pre")["guard_lo"] is None
    assert C.registered_true_params("cubic_mult", "during")["c1"] != 0.0
    saved = C._CAL.pop("design", None)
    C._CAL["design"] = {}
    try:
        for proc in ("cubic_bistable_0", "cubic_bistable_2", "cubic_shift"):
            with pytest.raises(ValueError, match="no registered constants"):
                C.registered_true_params(proc, "during")
    finally:
        C._CAL.pop("design", None)
        if saved is not None:
            C._CAL["design"] = saved
    with pytest.raises(ValueError, match="FAMILY"):
        C.registered_true_params("cubic_bistable", "pre")
    cell = _cell(process="cubic", pipeline="direct")
    tp, std, comp = _reg_window("cubic")
    a = T.predict_row("projection", comp, tp["pre"], "pre", std, cell)
    b = T.predict_row("projection", comp, tp["during"], "during", std, cell)
    assert len(C.pair_predictions(cell, [a, b], "projection")) == 1
    unguarded = dict(tp["pre"], **T.NO_GUARD)
    assert T.truth(unguarded, std, "pre")["two_delta_V"] == pytest.approx(0.01386, rel=0.02)
    assert a["truth"]["two_delta_V"] == pytest.approx(0.01498, rel=0.02)
    for bad in (unguarded, tp["during"], dict(tp["pre"], a3=tp["pre"]["a3"] * 1.01),
                CUBIC):
        with pytest.raises(ValueError, match="registered"):
            T.predict_row("projection", comp, bad, "pre", std, cell)
    d = dict(a["diag"], true_params=unguarded)
    d["digest"] = C.prediction_digest(d, a["estimate"], a["truth"])
    with pytest.raises(ValueError, match="registered"):
        C.pair_predictions(cell, [dict(a, diag=d), b], "projection")
    d = dict(a["diag"], true_params_check="unchecked")
    d["digest"] = C.prediction_digest(d, a["estimate"], a["truth"])
    with pytest.raises(ValueError, match="registered true process"):
        C.pair_predictions(cell, [dict(a, diag=d), b], "projection")
    d = dict(a["diag"], true_params=unguarded)
    with pytest.raises(ValueError, match="digest"):
        C.pair_predictions(cell, [dict(a, diag=d), b], "projection")


def test_B17c_null_baseline_sign_rule(guards):
    R = R500
    pred = _contrasts(R, -0.36, 0.07, -0.022, 1.0)
    wide = _null(spread=0.05, seed=1)
    biased = [dict(d, two_delta_V=w["two_delta_V"] + 0.05, tau_mfpt=w["tau_mfpt"] + 0.8)
              for d, w in zip(_null(), wide)]
    v = _verdict(pred, _contrasts(R, -0.30, 0.05, -0.02, 0.8), null=biased)
    assert v["checks"]["two_delta_V_sign"] == "indistinguishable from null"
    assert v["checks"]["tau_mfpt_sign"] == "indistinguishable from null"
    assert v["checks"]["rho1_sign"] == "pass" and not v["pass"]
    nb = v["reported"]["two_delta_V"]["null"]
    assert nb["lo"] < 0.05 < nb["hi"] and "null" in v["reported"]["xi_trap"]
    neg = [dict(d, rho1=d["rho1"] - 0.5) for d in _null()]
    v = _verdict(pred, _contrasts(R, -0.30, 0.05, -0.02, 0.8), null=neg)
    assert v["checks"]["rho1_sign"] == "indistinguishable from null"
    n_bad = int(NONFINITE_MAX_FRAC * R) + 1
    nan_null = [dict(d, tau_mfpt=np.nan) if i < n_bad else d
                for i, d in enumerate(_null())]
    v = _verdict(pred, _contrasts(R, -0.30, 0.05, -0.02, 0.8), null=nan_null)
    assert v["checks"]["tau_mfpt_sign"] == "fail (non-finite)"
    inf_null = [dict(d, tau_mfpt=np.inf) if i < 3 else d for i, d in enumerate(_null())]
    v = _verdict(pred, _contrasts(R, -0.30, 0.05, -0.02, 0.8), null=inf_null)
    assert np.isfinite(v["reported"]["tau_mfpt"]["null"]["hi"])
    assert v["checks"]["tau_mfpt_sign"] == "pass"
    sim = _contrasts(R, -0.30, 0.05, -0.02, 0.8)
    with pytest.raises(ValueError, match="no-exit"):
        _verdict(pred, sim, null_cell=_cell(mech="b", ex="exogenous"))
    for k, val in [("N", 20000), ("process", "cubic"),
                   ("windows", "pre:2000-2012;during:2013-2018")]:
        with pytest.raises(ValueError, match="differs"):
            _verdict(pred, sim, null_cell={**_cell(mech="a"), k: val})


def test_B17i_zeroth_order_must_exceed_its_own_no_exit_offset(guards):
    R = R500
    sim = _contrasts(R, -0.30, 0.05, -0.02, 0.8)
    pred = _contrasts(R, -0.33, 0.06, -0.021, 1.0)
    cell = _cell(process="cubic")
    base = _verdict(pred, sim, cell=cell, row="zeroth_order")
    assert base["pass"] and all(v == "pass" for v in base["checks"].values())
    off = _contrasts(R, 0.0, -4.63e-4, 0.0, -0.0399)
    v = _verdict(pred, sim, cell=cell, row="zeroth_order", own_offset=off)
    assert v["pass"] and v["reported"]["tau_mfpt"]["own_offset"]["median"] == pytest.approx(-0.0399)
    off = _contrasts(R, 0.0, 0.06, -0.021, 1.0)
    v = _verdict(pred, sim, cell=cell, row="zeroth_order", own_offset=off)
    for q in ("two_delta_V_sign", "rho1_sign", "tau_mfpt_sign"):
        assert v["checks"][q] == C.OWN_OFFSET_STATUS, q
    assert v["checks"]["xi_trap_rel"] == "pass" and not v["pass"]
    assert v["law_status"] == "fail"
    off = _contrasts(R, 0.0, -0.07, 0.0, 0.0)
    v = _verdict(pred, sim, cell=cell, row="zeroth_order", own_offset=off)
    assert v["checks"]["two_delta_V_sign"] == C.OWN_OFFSET_STATUS
    assert v["checks"]["tau_mfpt_sign"] == "pass"
    off = _contrasts(R, 0.0, 0.0599, 0.0, 0.0)
    assert _verdict(pred, sim, cell=cell, row="zeroth_order",
                    own_offset=off)["checks"]["two_delta_V_sign"] == "pass"
    v = _verdict(_contrasts(R, -0.33, -0.06, -0.021, 1.0), sim, cell=cell,
                 row="zeroth_order")
    assert v["checks"]["two_delta_V_sign"] == "fail"
    k = int(NONFINITE_MAX_FRAC * R) + 1
    off = _contrasts(R, 0.0, 0.0, 0.0, 0.0)
    for d in off[:k]:
        d["two_delta_V"] = np.nan
    v = _verdict(pred, sim, cell=cell, row="zeroth_order", own_offset=off)
    assert v["checks"]["two_delta_V_sign"] == "fail (non-finite)"
    null_cell = {**cell, "exit_mechanism": "a", "exit_dating": "none"}
    with pytest.raises(ValueError, match="needs null_predictions"):
        k1a_cell_verdict(_preds(cell, pred, "zeroth_order"), sim, _null(), cell,
                         null_cell, row="zeroth_order")
    with pytest.raises(ValueError, match="R0 = 500"):
        k1a_cell_verdict(_preds(cell, pred, "zeroth_order"), sim, _null(), cell,
                         null_cell, row="zeroth_order",
                         null_predictions=_preds(null_cell, pred[:10], "zeroth_order"))
    with pytest.raises(ValueError, match="cell_key"):
        k1a_cell_verdict(_preds(cell, pred, "zeroth_order"), sim, _null(), cell,
                         null_cell, row="zeroth_order",
                         null_predictions=_preds(cell, pred, "zeroth_order"))
    with pytest.raises(ValueError, match="used only by"):
        k1a_cell_verdict(_preds(cell, pred), sim, _null(), cell, null_cell,
                         null_predictions=_preds(null_cell, pred, "projection"))
    assert _verdict(pred, sim, cell=cell)["pass"]


def test_B17b_rho1_relative_error_gate(guards):
    R = R500
    sim = _contrasts(R, -0.30, 0.05, -0.02, 0.8)
    v = _verdict(_contrasts(R, -0.30, 0.05, -0.04, 0.8), sim)
    assert v["checks"]["rho1_sign"] == "pass" and v["checks"]["rho1_rel"] == "fail"
    assert not v["pass"]
    assert _verdict(_contrasts(R, -0.30, 0.05, -0.0259, 0.8), sim)["checks"][
        "rho1_rel"] == "pass"
    assert _verdict(_contrasts(R, -0.30, 0.05, -0.0261, 0.8), sim)["checks"][
        "rho1_rel"] == "fail"


def test_B17f_null_band_is_symmetric():
    for R0, k in ((40, 1), (200, 5), (500, 13), (1000, 25), (37, 1), (81, 3)):
        v = np.arange(R0, dtype=float)
        nb = C._null_band(np.random.default_rng(R0).permutation(v))
        assert nb["k"] == k
        assert nb["lo"] == k - 1 and nb["hi"] == R0 - k, (R0, nb)
        assert nb["lo"] - v.min() == v.max() - nb["hi"], R0
        lo_o, hi_o = np.quantile(v, (0.025, 0.975), method="inverted_cdf")
        if (0.025 * R0) == int(0.025 * R0):
            assert (lo_o - v.min()) != (v.max() - hi_o), R0
    x = np.random.default_rng(3).standard_normal(500)
    a, b = C._null_band(x), C._null_band(-x)
    assert a["lo"] == -b["hi"] and a["hi"] == -b["lo"]
    y = np.arange(500, dtype=float)
    y[:5] = np.nan
    nb = C._null_band(y)
    assert nb["k_eff"] == 8 and nb["lo"] == 5 + 7 and nb["hi"] == 499 - 7
    y[:13] = np.nan
    nb = C._null_band(y)
    assert nb["lo"] == -np.inf and nb["hi"] == np.inf


def test_B17d_cell_config_enforces_shock_placement(guards):
    assert check_cell_config(_cell())["valid_for_law"] is True
    for rep in ("persistent", "none", "initial_condition"):
        with pytest.raises(ValueError, match="gated cell"):
            check_cell_config(_cell(rep=rep))
    with pytest.raises(ValueError, match="ungated"):
        check_cell_config(_cell(gated=False))
    ctl = check_cell_config(_cell(gated=False, rep="persistent"))
    assert ctl["valid_for_law"] is False
    check_cell_config(_cell(process="cubic", mech="a", gated=False, rep="persistent"))
    with pytest.raises(ValueError, match="controls exist only"):
        check_cell_config(_cell(process="cubic_mult", gated=False, rep="persistent"))
    with pytest.raises(ValueError, match="controls exist only"):
        check_cell_config(_cell(mech="b", ex="exogenous", gated=False, rep="persistent"))
    with pytest.raises(ValueError, match="bool"):
        check_cell_config(_cell(gated="yes"))
    with pytest.raises(ValueError, match="lacks"):
        check_cell_config({k: v for k, v in _cell().items() if k != "windows"})
    with pytest.raises(ValueError, match="go together"):
        check_cell_config(_cell(mech="d", ex="none"))
    sim = _contrasts(R500, -0.30, 0.05, -0.02, 0.8)
    c = _cell(gated=False, rep="persistent")
    v = _verdict(_contrasts(R500, -0.36, 0.07, -0.022, 1.0), sim, cell=c)
    assert v["pass"] and v["law_status"] == "does not test the law"
    assert "control, not gated" in v["label"]


def test_B17e_closed_form_judged_on_ou_only_and_law_claims(guards):
    R = R500
    sim = _contrasts(R, -0.30, 0.05, -0.02, 0.8)
    good = _contrasts(R, -0.33, 0.06, -0.021, 1.0)
    bad = _contrasts(R, -0.50, 0.06, -0.021, 1.0)
    nan = _contrasts(R, np.nan, np.nan, np.nan, np.nan)

    def v(proc, mech, row, pred):
        return _verdict(pred, sim, cell=_cell(process=proc, mech=mech), row=row)

    cub_cf = v("cubic", "d", "closed_form", nan)
    assert cub_cf["judged"] is False and cub_cf["law_status"].startswith("not judged")
    assert v("ou", "d", "closed_form", good)["law_status"] == "pass"
    vs = []
    for proc, mech in K1A_CELLS:
        vs.append(v(proc, mech, "projection", good))
        vs.append(v(proc, mech, "fallback", bad))
        vs.append(v(proc, mech, "closed_form", good if proc == "ou" else nan))
        vs.append(v(proc, mech, "zeroth_order", bad if (proc, mech) == ("cubic", "e")
                    else good))
        vs.append(v(proc, mech, "zeroth_order_mbar", good))
    c = law_claims(vs)
    assert c["zeroth_order_mbar"] == "reported only"
    assert c["projection"] == "prediction confirmed"
    assert c["closed_form"] == "law under linear drift"
    assert c["zeroth_order"] == "not a law" and c["prl_condition_1"] is False
    vs2 = [x for x in vs if not (x["row"] == "zeroth_order"
                                 and x["cell"]["process"] == "cubic"
                                 and x["cell"]["exit_mechanism"] == "e")]
    vs2.append(v("cubic", "e", "zeroth_order", good))
    assert law_claims(vs2)["prl_condition_1"] is True
    with pytest.raises(ValueError, match="lacks"):
        law_claims(vs[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        law_claims(vs + [vs[0]])
    with pytest.raises(ValueError, match="not a K1a cell"):
        law_claims(vs + [v("ou", "b", "projection", good)])
    ctl = _verdict(good, sim, cell=_cell(process="cubic", mech="e", gated=False,
                                         rep="persistent"), row="zeroth_order")
    gated_missing = [x for x in vs if not (x["row"] == "zeroth_order"
                                           and x["cell"]["process"] == "cubic"
                                           and x["cell"]["exit_mechanism"] == "e")]
    with pytest.raises(ValueError, match="non-gated"):
        law_claims(gated_missing + [ctl])
    with pytest.raises(ValueError, match="non-gated"):
        law_claims(vs + [ctl])
    assert set(REPORT_ROWS) == {"projection", "fallback", "closed_form", "zeroth_order",
                                "zeroth_order_mbar"}


PERIODS = C.PERIODS


def _bias_contrasts(R=R500, nan_q=None, n_nan=0, **kw):
    """R per-replicate (bias_during − bias_pre) values, as contrasts for _preds
    (whose pre bias is 0 and during bias the value, ratio quantities too)."""
    out = []
    for r in range(R):
        b = {q: kw.get(q, 0.0) for q in QUANTITIES} | {"B_ratio_w": np.nan, "replicate": r}
        if nan_q and r < n_nan:
            b[nan_q] = np.nan
        out.append(b)
    return out


def _entry(process, mech, contr, pipeline="saai", **kw):
    cell = _cell(process=process, mech=mech, pipeline=pipeline, **kw)
    return {"cell": cell, "predictions": _preds(cell, contr)}


def _entries_for(process="cubic", pipeline="saai", dV=0.022, tau=1.5, xi=0.2, rho=0.05,
                 md=0.05):
    """The two K1a entries of (process, pipeline) whose predicted effect is
    (xi, dV, tau, rho, md) — what s0_check takes (contract 7)."""
    d1 = _bias_contrasts(xi_trap=-xi, two_delta_V=dV, tau_mfpt=tau, rho1=-rho,
                         mean_diffusion=md)
    d2 = _bias_contrasts(xi_trap=-2 * xi, two_delta_V=2 * dV, tau_mfpt=2 * tau,
                         rho1=-2 * rho, mean_diffusion=2 * md)
    return [_entry(process, "d", d1, pipeline), _entry(process, "e", d2, pipeline)]


def _effect_for(process="cubic", pipeline="saai", **kw):
    return predicted_effect_sizes(process, pipeline, _entries_for(process, pipeline, **kw))


def _s0cell(process="cubic", **kw):
    return _cell(process=process, mech="a",
                 **{"N": 2 * C.S0_N, "N_windows": C.s0_window_sizes(), **kw})


S0_TRU = dict(xi_trap=0.0, two_delta_V=0.0, tau_mfpt=0.0, mean_diffusion=0.2,
              B_ratio_w=1.0, rho1=0.875)
S0_EST = dict(xi_trap=0.002, two_delta_V=1e-4, tau_mfpt=0.02, mean_diffusion=0.2005,
              B_ratio_w=1.02, rho1=0.874)


def _s0(est=S0_EST, tru=S0_TRU, R=R500, during=None, pre=None, process="cubic",
        pipeline="saai"):
    """{'pre': [...], 'during': [...]} estimates (with replicate ids and the
    registered seeds) and truths (with the registered true_params);
    during / pre: a list of R estimate dicts replacing that period's."""
    seeds = C.s0_seeds(process, pipeline)

    def seeded(xs):
        return [dict(x, seed=seeds[x["replicate"]] if x["replicate"] < len(seeds) else 0)
                for x in xs]
    E = {"pre": seeded(pre or [dict(est, replicate=r) for r in range(R)]),
         "during": seeded(during or [dict(est, replicate=r) for r in range(R)])}
    Tr = {p: [dict(tru, replicate=r, true_params=_tp(process, p)) for r in range(R)]
          for p in PERIODS}
    return E, Tr


def _tp(process, period):
    try:
        return C.registered_true_params(process, period)
    except ValueError:
        return {}


def _rep(xs):
    return [dict(x, replicate=r) for r, x in enumerate(xs)]


def test_B18_s0_metric(guards):
    E, Tr = _s0()
    ent = _entries_for()
    r = s0_check(_s0cell(), E, Tr, ent)
    assert r["pass"], r
    for per in PERIODS:
        assert r[per]["two_delta_V"]["tol"] == pytest.approx(0.05 * 0.022)
        assert r[per]["tau_mfpt"]["tol"] == pytest.approx(0.05 * 1.5)
        assert r[per]["xi_trap"]["tol"] == pytest.approx(0.05 * 0.2)
        assert r[per]["B_ratio_w"]["basis"].startswith("physical only (simulation-only")
    assert f"R={R500}" in r["label"] and r["pipeline"] == "saai"
    assert r["predicted_effect"]["effect"]["two_delta_V"] == pytest.approx(0.022)
    r = s0_check(_s0cell(), *_s0(dict(S0_EST, two_delta_V=0.003)), ent)
    assert r["pre"]["two_delta_V"]["binding"] == "effect"
    assert r["pre"]["two_delta_V"]["status"] == "fail" and not r["pass"]
    zero = _entries_for(rho=0.0)
    assert predicted_effect_sizes("cubic", "saai", zero)["status"]["rho1"] == "zero"
    r = s0_check(_s0cell(), E, Tr, zero)
    assert r["pre"]["rho1"]["basis"].startswith("physical only (frozen prediction exactly 0")
    big = _entries_for(xi=10.0)
    assert s0_check(_s0cell(), E, Tr, big)["during"]["xi_trap"]["tol"] == pytest.approx(0.05)
    k = int(NONFINITE_MAX_FRAC * R500) + 1
    bad = _rep([dict(S0_EST, mean_diffusion=np.nan)] * k + [S0_EST] * (R500 - k))
    r = s0_check(_s0cell(), *_s0(during=bad), ent)
    assert r["during"]["mean_diffusion"]["status"] == "fail (non-finite)" and not r["pass"]
    bad = _rep([dict(S0_EST, rho1=np.nan)] * k + [S0_EST] * (R500 - k))
    r = s0_check(_s0cell(), *_s0(during=bad), ent)
    assert r["during"]["rho1"]["status"] == "report only" and r["pass"]
    assert r["during"]["rho1"]["status_if_gated"] == "fail (non-finite)"
    nanp = [_entry("cubic", "d", _bias_contrasts(nan_q="two_delta_V", n_nan=k,
                                                 two_delta_V=0.03)), ent[1]]
    with pytest.raises(ValueError, match="refused"):
        s0_check(_s0cell(), E, Tr, nanp)
    with pytest.raises(ValueError, match="K1a entries"):
        s0_check(_s0cell(), E, Tr, None)
    with pytest.raises(ValueError, match="exactly the K1a cells"):
        s0_check(_s0cell(process="ou"), *_s0(process="ou"), ent)
    Em, Trm = _s0(process="cubic_mult")
    r = s0_check(_s0cell(process="cubic_mult"), Em, Trm, None)
    assert all(r[p][q]["basis"].startswith("physical only (no K1a")
               for p in PERIODS for q in QUANTITIES)
    with pytest.raises(ValueError, match="no K1a cells"):
        s0_check(_s0cell(process="cubic_mult"), Em, Trm, ent)


def test_S1_s0_rho1_report_only_and_rederivation5_flag(guards):
    ent = _entries_for()
    shares = [0.5, 0.3, 0.2]
    ref = {"w": 0.2, "shares": {"pre": shares, "during": shares}, "N": {"pre": 20000,
                                                                         "during": 20000}}
    exp = C.rederivation5_rho1_bias(0.875, 0.2, shares, 20000)
    H, G = 0.38, 0.5 ** 3 + 0.3 ** 3 + 0.2 ** 3
    K = H - 2 * G + H * H
    Kt = 0.04 * K + (2 * 0.2 * 0.8 * (1 - H) + 0.64) / 20000 - 0.64 / 20000 ** 2
    assert exp == pytest.approx(-0.875 * (1 - 0.875 ** 2) / 2 * Kt
                                / (1 - 0.2 * H - 0.8 / 20000) ** 2, rel=1e-12)
    rng = np.random.default_rng(3)
    errs = exp + 0.01 * rng.standard_normal(R500)
    pre = _rep([dict(S0_EST, rho1=0.875 + e) for e in errs])
    r = s0_check(_s0cell(), *_s0(pre=pre, during=pre), ent, rho1_reference=ref)
    for per in PERIODS:
        row = r[per]["rho1"]
        assert row["status"] == "report only" and "status_if_gated" in row
        b = row['rederivation5']
        assert b["expected_bias"] == pytest.approx(exp, rel=1e-9)
        assert b["within_3se"] is True and abs(b["z"]) <= 3
    assert r["pass"]
    far = _rep([dict(S0_EST, rho1=0.875 + exp + 0.01 + 1e-4 * e) for e in errs])
    r = s0_check(_s0cell(), *_s0(pre=far, during=far), ent, rho1_reference=ref)
    assert r["pre"]["rho1"]['rederivation5']["within_3se"] is False and r["pass"]
    r0 = s0_check(_s0cell(), *_s0(pre=far, during=far), ent)
    assert r0["pre"]["rho1"]['rederivation5']["within_3se"] is None
    C.validate_s0_result(r, "cubic", "saai")
    assert "rho1" not in C.s0_gated_quantities("cubic")
    assert "rho1" in C.s0_gated_quantities("cubic_mult")
    bad = {**r, "pre": {**r["pre"], "rho1": {**r["pre"]["rho1"], "status": "pass"}}}
    with pytest.raises(RuntimeError, match="did not pass"):
        C.validate_s0_result(bad, "cubic", "saai")


def test_R8c_s0_rho1_flag_against_the_estimator_bias(guards):
    ref = {"w": 0.25, "shares": {"pre": [0.4, 0.35, 0.25], "during": [0.6, 0.4]},
                    "N": {"pre": 20000, "during": 20000}}
    for per in PERIODS:
        rho = 1 + C.registered_true_params("ou", per)["a1"]
        tp = C.registered_true_params("ou", per)
        std = T.stationary_standardisation(tp)
        tru = T.truth(tp, std, per)
        assert tru["rho1"] == pytest.approx(rho, abs=1e-13)
        exp = C.rederivation5_rho1_bias(rho, ref["w"], ref["shares"][per], ref["N"][per])
        assert exp < 0
        rng = np.random.default_rng(8)
        errs = 0.002 * rng.standard_normal(R500)
        errs = errs - errs.mean()
        at = [({"rho1": rho + exp + e}, {"rho1": tru["rho1"]}) for e in errs]
        rep = C.s0_rho1_report(at, ref, per)
        assert rep["truth_rho1_constant"] is True
        assert rep["rho_at"] == pytest.approx(rho, abs=1e-13)
        assert rep["expected_bias"] == pytest.approx(exp, rel=1e-12)
        assert rep["within_3se"] is True and abs(rep["z"]) < 1e-6
        assert "amendment 8 (c)" in rep["flag_rule"] and "−e/2" in rep["flag_rule"]
        off = [({"rho1": rho + e}, {"rho1": tru["rho1"]}) for e in errs]
        assert C.s0_rho1_report(off, ref, per)["within_3se"] is False
        moved = [(e, {"rho1": t["rho1"] + 1e-6 * i}) for i, (e, t) in enumerate(at)]
        assert C.s0_rho1_report(moved, ref, per)["truth_rho1_constant"] is False
    assert C.s0_rho1_report(at, None, per)["within_3se"] is None
    Em, Trm = _s0(dict(S0_EST, rho1=0.5), process="cubic_mult")
    r = s0_check(_s0cell(process="cubic_mult"), Em, Trm, None)
    assert r["pre"]["rho1"]["status"] == "fail" and not r["pass"]


def test_B18_s0_checks_both_periods(guards):
    ent = _entries_for()
    E, Tr = _s0()
    bad = _rep([dict(S0_EST, xi_trap=0.5)] * R500)
    r = s0_check(_s0cell(), *_s0(during=bad), ent)
    assert r["pre"]["pass"] and not r["during"]["pass"] and not r["pass"]
    assert r["during"]["xi_trap"]["status"] == "fail"
    r = s0_check(_s0cell(), *_s0(pre=bad), ent)
    assert not r["pre"]["pass"] and r["during"]["pass"] and not r["pass"]
    Em, Trm = _s0(process="cubic_mult")
    Tr2 = {"pre": Trm["pre"], "during": [dict(t, mean_diffusion=2.0) for t in Trm["during"]]}
    r = s0_check(_s0cell(process="cubic_mult"), Em, Tr2, None)
    assert r["during"]["mean_diffusion"]["tol_physical"] == pytest.approx(0.1)
    assert r["pre"]["mean_diffusion"]["tol_physical"] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="both"):
        s0_check(_s0cell(), E["pre"], Tr["pre"], ent)
    with pytest.raises(ValueError, match="both"):
        s0_check(_s0cell(), {"pre": E["pre"]}, {"pre": Tr["pre"]}, ent)
    with pytest.raises(ValueError, match="replicate"):
        s0_check(_s0cell(), {"pre": [S0_EST] * R500, "during": [S0_EST] * R500}, Tr, ent)
    shifted = [dict(S0_EST, replicate=r + 1) for r in range(R500)]
    with pytest.raises(ValueError, match="pre and during replicates differ"):
        s0_check(_s0cell(), {"pre": E["pre"], "during": shifted},
                 {"pre": Tr["pre"], "during": [dict(t, replicate=t["replicate"] + 1)
                                               for t in Tr["during"]]}, ent)
    with pytest.raises(ValueError, match="different replicates"):
        s0_check(_s0cell(), E, {"pre": Tr["pre"], "during": [
            dict(t, replicate=t["replicate"] + 7) for t in Tr["during"]]}, ent)


def test_B18i_s0_seeds_ids_and_truths_are_the_registered_ones(guards):
    ent = _entries_for()
    E, Tr = _s0()
    assert s0_check(_s0cell(), E, Tr, ent)["pass"]
    shifted = {p: [dict(e, replicate=e["replicate"] + 1) for e in E[p]] for p in PERIODS}
    shifted_t = {p: [dict(t, replicate=t["replicate"] + 1) for t in Tr[p]] for p in PERIODS}
    with pytest.raises(ValueError, match="exactly 0"):
        s0_check(_s0cell(), shifted, shifted_t, ent)
    noseed = {**E, "during": [{k: v for k, v in e.items() if k != "seed"}
                              for e in E["during"]]}
    with pytest.raises(ValueError, match="registered seed"):
        s0_check(_s0cell(), noseed, Tr, ent)
    other = {**E, "pre": [dict(e, seed=C.s0_seeds("cubic", "direct")[e["replicate"]])
                          for e in E["pre"]]}
    with pytest.raises(ValueError, match="registered seed"):
        s0_check(_s0cell(), other, Tr, ent)
    unguarded = dict(C.registered_true_params("cubic", "pre"), **T.NO_GUARD)
    for bad_tp in (unguarded, C.registered_true_params("cubic", "during"), {}):
        bad = {**Tr, "pre": [dict(t, true_params=bad_tp) for t in Tr["pre"]]}
        with pytest.raises(ValueError, match="registered"):
            s0_check(_s0cell(), E, bad, ent)
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(C._CAL, "design", {})
        with pytest.raises(ValueError, match="no registered constants"):
            s0_check(_s0cell(process="cubic_shift"), *_s0(process="cubic_shift"), None)


def test_B18_s0_runs_only_on_the_registered_s0_cell(guards):
    E, Tr = _s0()
    ent = _entries_for()
    for bad in (_cell(process="cubic", N=20000),
                _cell(process="cubic", mech="b", ex="exogenous", N=20000),
                _s0cell(N=2300), _s0cell(N=19999),
                _s0cell(N=20000, N_windows={"pre": 10000, "during": 10000}),
                _s0cell(N_windows={"pre": 30000, "during": 10000}),
                _s0cell(N_windows=None)):
        with pytest.raises(ValueError, match="S0 runs on cell"):
            s0_check(bad, E, Tr, ent)
    with pytest.raises(ValueError, match="S0 runs on cell"):
        s0_check(_s0cell(gated=False, rep="persistent"), E, Tr, ent)
    with pytest.raises(ValueError, match="gated cell"):
        s0_check(_s0cell(rep="none"), E, Tr, ent)
    with pytest.raises(ValueError, match="labelled"):
        s0_check(_s0cell(pipeline="raw"), E, Tr, ent)
    for r in (10, 499):
        with pytest.raises(ValueError, match="registers R = 500"):
            s0_check(_s0cell(), *_s0(R=r), ent)
    Ed, Trd = _s0(pipeline="direct")
    with pytest.raises(ValueError, match="pipeline"):
        s0_check(_s0cell(pipeline="direct"), Ed, Trd, ent)
    ent_d = _entries_for(pipeline="direct", dV=0.04)
    r = s0_check(_s0cell(pipeline="direct"), Ed, Trd, ent_d)
    assert r["pipeline"] == "direct"
    assert r["pre"]["two_delta_V"]["tol"] == pytest.approx(0.05 * 0.04)
    with pytest.raises(ValueError, match="pipeline"):
        s0_check(_s0cell(), E, Tr, ent_d)


def test_B18b_predicted_effect_sizes(guards):
    """|predicted contrast| per quantity = min over the process's two K1a
    cells of |median_r(bias_during − bias_pre)|, per pipeline, computed from
    the predict_row outputs (estimate − truth) themselves.  It refuses an
    empty list, a missing / extra / foreign cell, another pipeline, the wrong
    row / design / R, outputs not made for the cell, and more than
    NONFINITE_MAX_FRAC undefined replicates — which are counted, never
    dropped."""
    e = _effect_for()
    assert e["effect"]["xi_trap"] == pytest.approx(0.2)
    assert e["effect"]["two_delta_V"] == pytest.approx(0.022)
    assert e["effect"]["tau_mfpt"] == pytest.approx(1.5)
    assert np.isnan(e["effect"]["B_ratio_w"]) and e["status"]["B_ratio_w"] == "simulation only"
    assert e["process"] == "cubic" and e["pipeline"] == "saai" and e["frozen_commit"] == "TEST"
    d1 = _bias_contrasts(xi_trap=-0.3, two_delta_V=0.03, tau_mfpt=1.1)
    ed, ee = _entry("cubic", "d", d1), _entry("cubic", "e", d1)
    with pytest.raises(ValueError, match="two K1a cells"):
        predicted_effect_sizes("cubic", "saai", [])
    with pytest.raises(ValueError, match="exactly the K1a cells"):
        predicted_effect_sizes("cubic", "saai", [ed])
    with pytest.raises(ValueError, match="exactly the K1a cells"):
        predicted_effect_sizes("cubic", "saai", [ed, ed])
    with pytest.raises(ValueError, match="exactly the K1a cells"):
        predicted_effect_sizes("cubic", "saai", [ed, _entry("ou", "e", d1)])
    with pytest.raises(ValueError, match="exactly the K1a cells"):
        predicted_effect_sizes("cubic", "saai", [ed, ee, _entry("cubic", "b", d1,
                                                                 ex="exogenous")])
    with pytest.raises(ValueError, match="no K1a cells"):
        predicted_effect_sizes("cubic_mult", "saai", [ed, ee])
    with pytest.raises(ValueError, match="pipeline"):
        predicted_effect_sizes("cubic", "raw", [ed, ee])
    with pytest.raises(ValueError, match="row"):
        zc = _cell(process="cubic", mech="e")
        predicted_effect_sizes("cubic", "saai", [ed, {"cell": zc, "predictions": _preds(
            zc, d1, row="zeroth_order")}])
    with pytest.raises(ValueError, match="design"):
        predicted_effect_sizes("cubic", "saai", [ed, {"cell": ee["cell"], "predictions":
                                                      _preds(ee["cell"], d1,
                                                             design="fallback")}])
    with pytest.raises(ValueError, match="pipeline"):
        predicted_effect_sizes("cubic", "saai", [ed, _entry("cubic", "e", d1, "direct")])
    with pytest.raises(ValueError, match="gated 'shock'"):
        predicted_effect_sizes("cubic", "saai", [ed, _entry("cubic", "e", d1, gated=False,
                                                            rep="persistent")])
    with pytest.raises(ValueError, match="need R = 500"):
        predicted_effect_sizes("cubic", "saai", [ed, {**ee, "predictions":
                                                      ee["predictions"][:-2]}])
    with pytest.raises(ValueError, match="cell_key"):
        predicted_effect_sizes("cubic", "saai", [ed, {**ee, "predictions":
                                                      _preds(_cell(), d1)}])
    k = int(NONFINITE_MAX_FRAC * R500)
    ok = predicted_effect_sizes("cubic", "saai", [ed, _entry(
        "cubic", "e", _bias_contrasts(nan_q="two_delta_V", n_nan=k, two_delta_V=0.03))])
    assert ok["per_cell"]["cubic-e"]["two_delta_V"]["n_nonfinite"] == k
    with pytest.raises(ValueError, match="refused"):
        predicted_effect_sizes("cubic", "saai", [ed, _entry(
            "cubic", "e", _bias_contrasts(nan_q="two_delta_V", n_nan=k + 1,
                                          two_delta_V=0.03))])
    with pytest.raises(ValueError):
        s0_check(_s0cell(), *_s0(), [])


def _git_repo(path):
    g = ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t"]
    import subprocess
    try:
        subprocess.run(g[:3] + ["init", "-q"], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    return lambda *a: subprocess.run(g + list(a), check=True, capture_output=True,
                                     text=True).stdout.strip()


def _s0_repo(monkeypatch, tmp_path):
    """A git repo standing in for REPO, with the export-subst attribute and
    the archive-commit placeholder as in this repository."""
    import shutil
    pass
    monkeypatch.setattr(C, "_FROZEN", {})
    monkeypatch.setattr(C, "_require_frozen", lambda: {"frozen_commit": "F1"})
    pass
    monkeypatch.setattr(C, "REPO", tmp_path)
    (tmp_path / 'results/records').mkdir(parents=True)
    (tmp_path / "src").mkdir()
    shutil.copy(T.REPO / ".gitattributes", tmp_path / ".gitattributes")
    shutil.copy(T.REPO / C.ARCHIVE_COMMIT_FILE, tmp_path / C.ARCHIVE_COMMIT_FILE)
    git = _git_repo(tmp_path)
    git("config", "user.name", "t")
    git("config", "user.email", "t@t")
    git("add", "-A")
    git("commit", "-qm", "code")
    return git


def _s0_result(process="cubic", pipeline="saai", during=None):
    ent = _entries_for(process, pipeline) if process in ("ou", "cubic") else None
    return s0_check(_s0cell(process=process, pipeline=pipeline),
                    *_s0(during=during, process=process, pipeline=pipeline), ent)


def _start_and_record(git, res, process="cubic", pipeline="saai"):
    """start_s0_attempt (commits), the run's commit = the start commit,
    write_s0_record (commits)."""
    st = C.start_s0_attempt(process, pipeline)
    assert git("rev-parse", "HEAD") == st["start_commit"]
    return C.write_s0_record(res, C.s0_seeds(process, pipeline), st["start_commit"]), st


def test_B18f_s0_record_is_validated_stamped_and_append_only(monkeypatch, tmp_path):
    import json
    git = _s0_repo(monkeypatch, tmp_path)
    head = git("rev-parse", "HEAD")
    seeds = C.s0_seeds("cubic", "saai")
    assert len(seeds) == R500 and len(set(seeds)) == R500
    assert seeds != C.s0_seeds("cubic", "direct") != C.s0_seeds("ou", "saai")
    res = _s0_result()
    assert res["pass"]
    cell_e = _cell(process="cubic", mech="e")
    path = C.s0_record_path("cubic", "saai")
    path.write_text(json.dumps({"frozen_commit": "F1", "pass": True,
                                "contract_version": C.CONTRACT_VERSION}))
    git("add", "-A")
    git("commit", "-qm", "forged")
    with pytest.raises(RuntimeError):
        C.assert_exit_estimation_allowed(cell_e)
    git("rm", "-q", str(path))
    git("commit", "-qm", "drop forgery")
    with pytest.raises(RuntimeError, match="no open S0 attempt"):
        C.write_s0_record(res, seeds, head)
    with pytest.raises(RuntimeError, match="no open S0 attempt"):
        C.open_s0_attempt("cubic", "saai")
    (tmp_path / "scratch.txt").write_text("x")
    with pytest.raises(RuntimeError, match="dirty tree"):
        C.start_s0_attempt("cubic", "saai")
    (tmp_path / "scratch.txt").unlink()
    with monkeypatch.context() as mp:
        mp.setitem(C._CAL, "design", {})
        with pytest.raises(ValueError, match="no registered constants"):
            C.start_s0_attempt("cubic_shift", "saai")
    st = C.start_s0_attempt("cubic", "saai")
    assert st["attempt"] == 1 and st["stage"] == "started"
    assert git("rev-parse", "HEAD") == st["start_commit"] != head
    assert not git("status", "--porcelain")
    assert C.open_s0_attempt("cubic", "saai")["attempt"] == 1
    with pytest.raises(RuntimeError, match="already logged"):
        C.start_s0_attempt("cubic", "saai")
    with pytest.raises(RuntimeError, match="registered s0_seeds"):
        C.write_s0_record(res, seeds[::-1], st["start_commit"])
    with pytest.raises(RuntimeError, match="ancestor"):
        C.write_s0_record(res, seeds, "0" * 40)
    with pytest.raises(RuntimeError, match="estimates come from"):
        C.write_s0_record(res, seeds, head)
    (tmp_path / "scratch.txt").write_text("x")
    with pytest.raises(RuntimeError, match="dirty tree"):
        C.write_s0_record(res, seeds, st["start_commit"])
    (tmp_path / "scratch.txt").unlink()
    out = C.write_s0_record(res, seeds, st["start_commit"])
    assert not git("status", "--porcelain")
    rec = json.loads(out.read_text())
    assert rec["clean_tree"] is True and rec["code_commit"] == st["start_commit"]
    assert rec["attempt"] == 1 and rec["seeds"] == seeds
    assert rec["estimates_commit"] == rec["start_commit"] == st["start_commit"]
    log = C._read_attempts()
    assert [e["stage"] for e in log] == ["started", "result"]
    assert log[1]["result_sha256"] == rec["result_sha256"]
    got = C._require_s0_pass("cubic", "saai")
    assert got["verified_by"] == "git" and got["attempts_logged"] == 1
    C.assert_exit_estimation_allowed(cell_e)
    with pytest.raises(RuntimeError, match="no open S0 attempt"):
        C.open_s0_attempt("cubic", "saai")
    with pytest.raises(RuntimeError, match="never overwritten"):
        C.write_s0_record(res, seeds, st["start_commit"])
    git("rm", "-q", str(path))
    git("commit", "-qm", "drop record")
    with pytest.raises(RuntimeError, match="already logged"):
        C.write_s0_record(res, seeds, st["start_commit"])
    with pytest.raises(RuntimeError, match="already logged"):
        C.start_s0_attempt("cubic", "saai")
    git("revert", "--no-edit", "HEAD")
    C.assert_exit_estimation_allowed(cell_e)
    good = json.loads(path.read_text())

    def forge(mut, fix_hash=True):
        r = json.loads(json.dumps(good))
        mut(r)
        if fix_hash:
            r["result_sha256"] = C._sha(r["result"])
        path.write_text(json.dumps(r))
        git("add", "-A")
        git("commit", "-qm", "edit")

    def during_fail(r):
        r["result"]["during"]["xi_trap"]["status"] = "fail"
    for mut, match in [
            (lambda r: r["result"].pop("during"), "hash|attempt|during"),
            (during_fail, "attempt|did not pass"),
            (lambda r: r["result"]["cell"].update(pipeline="direct"), "attempt|S0 cell"),
            (lambda r: r["result"].update(R=499), "attempt|R = 499"),
            (lambda r: r.update(seeds=seeds[:-1] + [1]), "seeds"),
            (lambda r: r.update(clean_tree=False), "clean")]:
        forge(mut)
        with pytest.raises(RuntimeError, match=match):
            C.assert_exit_estimation_allowed(cell_e)
    forge(lambda r: r["result"]["pre"]["rho1"].update(median_bias=0.0), fix_hash=False)
    with pytest.raises(RuntimeError, match="hash"):
        C.assert_exit_estimation_allowed(cell_e)
    for mut, match in [(lambda r: r.pop("during"), "lacks the 'during'"),
                       (during_fail, "did not pass"),
                       (lambda r: r["cell"].update(N=2300), "S0 cell"),
                       (lambda r: r.update(replicates=[0] * R500), "replicate"),
                       (lambda r: r.update(predicted_effect=None), "predicted effect")]:
        body = json.loads(json.dumps(good["result"]))
        if mut is during_fail:
            body["during"]["xi_trap"]["status"] = "fail"
        else:
            mut(body)
        with pytest.raises(RuntimeError, match=match):
            C.validate_s0_result(body, "cubic", "saai")
    path.write_text(json.dumps(good, indent=2, ensure_ascii=False) + "\n")
    git("add", "-A")
    git("commit", "-qm", "restore")
    C.assert_exit_estimation_allowed(cell_e)
    lp = C.s0_attempt_log_path()
    lp.write_text(lp.read_text().replace('"attempt": 1', '"attempt": 1 '))
    git("add", "-A")
    git("commit", "-qm", "rewrite log")
    with pytest.raises(RuntimeError, match="append-only"):
        C.assert_exit_estimation_allowed(cell_e)


def test_B18g_failed_s0_is_logged_and_blocks(monkeypatch, tmp_path):
    git = _s0_repo(monkeypatch, tmp_path)
    bad = _rep([dict(S0_EST, xi_trap=0.5)] * R500)
    res = _s0_result(during=bad)
    assert not res["pass"]
    _start_and_record(git, res)
    assert not git("status", "--porcelain")
    with pytest.raises(RuntimeError, match="did not pass"):
        C.assert_exit_estimation_allowed(_cell(process="cubic", mech="d"))
    with pytest.raises(RuntimeError, match="already logged"):
        C.start_s0_attempt("cubic", "saai")
    C.start_s0_attempt("ou", "saai")
    with pytest.raises(RuntimeError, match="already logged"):
        C.start_s0_attempt("ou", "saai")
    monkeypatch.setattr(C, "_require_frozen", lambda: {"frozen_commit": "F2"})
    _start_and_record(git, _s0_result())
    got = C._require_s0_pass("cubic", "saai")
    assert got["attempt"] == 2 and got["attempts_logged"] == 2
    assert [e["pass"] for e in C._read_attempts() if e["stage"] == "result"] == [False, True]


def test_B18h_s0_record_outside_a_checkout_is_bound_to_the_archive(monkeypatch, tmp_path):
    import shutil
    import subprocess
    src = tmp_path / "repo"
    src.mkdir()
    git = _s0_repo(monkeypatch, src)
    cell_e = _cell(process="cubic", mech="e")

    def export(rev, name):
        out = tmp_path / name
        out.mkdir()
        tar = subprocess.run(["git", "-C", str(src), "archive", rev], check=True,
                             capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(out)], input=tar, check=True)
        return out
    st = C.start_s0_attempt("cubic", "saai")
    run_dir = export(st["start_commit"], "run")
    monkeypatch.setattr(C, "REPO", run_dir)
    assert C.open_s0_attempt("cubic", "saai")["attempt"] == 1
    monkeypatch.setattr(C, "REPO", src)
    C.write_s0_record(_s0_result(), C.s0_seeds("cubic", "saai"), st["start_commit"])
    code_commit = st["start_commit"]
    arc = export("HEAD", "archive")
    monkeypatch.setattr(C, "REPO", arc)
    assert C.archive_commit()["commit"] == git("rev-parse", "HEAD")
    assert C._require_s0_pass("cubic", "saai")["verified_by"].startswith("git archive")
    C.assert_exit_estimation_allowed(cell_e)
    with pytest.raises(RuntimeError, match="git checkout"):
        C.write_s0_record(_s0_result(), C.s0_seeds("cubic", "saai"), code_commit)
    plain = tmp_path / "plain"
    shutil.copytree(src, plain, ignore=shutil.ignore_patterns(".git"))
    monkeypatch.setattr(C, "REPO", plain)
    assert C.archive_commit() is None
    with pytest.raises(RuntimeError, match="git-archive export"):
        C.assert_exit_estimation_allowed(cell_e)
    with pytest.raises(RuntimeError, match="git-archive export"):
        C.open_s0_attempt("cubic", "saai")
    own = export(code_commit, "own")
    shutil.copytree(src / 'results', own / 'results', dirs_exist_ok=True)
    monkeypatch.setattr(C, "REPO", own)
    with pytest.raises(RuntimeError, match="own producing commit"):
        C.assert_exit_estimation_allowed(cell_e)


R10B_CELL = ('n/a (K2 not gated under amendment 10, re-derivation 8 shows this '
             "truth is not the estimator's large-sample limit for w > 0)")


def _sweep(roots, mode, xi_data, c1=0.0, n_grid=600):
    out = []
    for r in roots:
        th = np.array([0.15 * r, -0.15, 0.0, 0.0, np.log(0.25), c1, 0.0])
        out.append(derived_observables(th, xi_data, degenerate=mode, n_grid=n_grid))
    return out


def test_B20_limit_convention_continuous_across_the_threshold():
    xi_data = np.random.default_rng(0).standard_normal(20_000)
    lo, hi = np.percentile(xi_data, 1), np.percentile(xi_data, 99)
    h = (hi - lo) / 599
    roots = -np.linspace(0.2, 4.0, 191) * h
    obs = _sweep(roots, "limit", xi_data)
    st = {o["id_status"] for o in obs}
    assert st == {"subgrid_path", "stable_zero"}
    ell = np.abs(roots)
    dV = np.array([o["delta_V"] for o in obs])
    assert np.max(np.abs(dV / (0.15 * ell ** 2 / 0.5) - 1)) < 1e-9
    tau = np.array([o["tau_mfpt"] for o in obs])
    step = np.diff(tau)
    assert np.all(step > 0) and step.max() < 1.05 * step.min()
    fine = np.array([o["tau_mfpt"] for o in _sweep(roots[::19], "limit", xi_data,
                                                    n_grid=6000)])
    assert np.max(np.abs(tau[::19] / fine - 1)) < 2e-3
    win = _sweep(-np.array([1.8, 1.7, 1.6]) * h, "limit", xi_data)
    t = [o["tau_mfpt"] for o in win]
    assert abs((t[0] - t[1]) / (t[1] - t[2]) - 1) < 0.05
    gap = xi_data[np.abs(xi_data) > 0.2]
    md = np.array([o["mean_diffusion"] for o in _sweep(roots, "limit", gap, c1=0.4)])
    assert np.max(np.abs(np.diff(md))) < 1e-3 * md.min()
    assert abs(md[0] / 0.25 - 1) < 0.4 * h + 1e-3


def test_B22_composition_must_hold_each_transition_exactly_once():
    _, _, m, s = T.stationary_law(T.params_vector(OU))
    comp = _block_exit_window(m, s)
    T.predict_bias(comp, OU, "during", _std(OU))
    with pytest.raises(ValueError, match="lacks the during transitions"):
        T.predict_bias(comp.iloc[[3]], OU, "during", _std(OU))
    with pytest.raises(ValueError, match="repeated"):
        T.predict_bias(pd.concat([comp, comp.iloc[[3]]]), OU, "during", _std(OU))
    with pytest.raises(ValueError, match="repeated"):
        T.closed_form_ou(pd.concat([comp, comp.iloc[[0]]]), OU["a1"], 0.3, "during")
    z = comp.copy()
    z.loc[z.index[0], "n_stay"] = 0
    T.predict_bias(z, OU, "during", _std(OU))
    std = _std(OU)
    with pytest.raises(ValueError, match="repeats a year"):
        T.predict_bias(comp, OU, "during", pd.concat([std, std.iloc[[14]]]))


def test_B23_reported_rows_closed_form_and_zeroth_order():
    """The readable law (closed form) and the registered-inputs row (zeroth
    order) are computed as rows of their own.  With no exit at stationary
    moments both equal the truth (OU); the closed form is OU only; on the
    block-exit window they differ from the gated projection by an amount
    the verdict must report (bookkeeping of the rows, not of the law)."""
    _, _, m, s = T.stationary_law(T.params_vector(OU))
    none = _comp([(300, 0, 0, m, s, m, s, 0.0)] * 6)
    tr = T.truth(OU, _std(OU), "during")
    for f in (T.closed_form_observables, T.zeroth_order_observables):
        o = f(none, OU, "during", _std(OU))
        for q in ("xi_trap", "two_delta_V", "rho1", "mean_diffusion"):
            assert abs(o[q] - tr[q]) < 1e-6, (f.__name__, q, o[q], tr[q])
        assert abs(o["tau_mfpt"]) < 1e-6
    blk = _block_exit_window(m, s)
    cf = T.closed_form_observables(blk, OU, "during", _std(OU))
    lin = T.closed_form_ou(blk, OU["a1"], np.exp(OU["c0"]), "during")
    assert abs(cf["xi_trap"] - lin["xi_star_lin"]) < 1e-9
    assert abs(cf["two_delta_V"] - lin["two_delta_V_lin"]) < 1e-9
    z = T.zeroth_order_observables(blk, OU, "during", _std(OU))
    zm = T.zeroth_order_observables(blk, OU, "during", _std(OU), form="mbar")
    M = T.predict(blk, OU, "during", _std(OU))["diag"]["M_bar"]
    inp = T.zeroth_order_inputs(blk, OU, "during", _std(OU))
    assert abs(inp["M_bar"] - M) < 1e-12 and abs(inp["L_bar"] - inp["E_bar"] - M) < 1e-12
    assert abs(inp["L_bar"] + lin["xbar_stay_pooled"]) < 1e-12
    assert abs(z["xi_trap"] - ((1 + OU["a1"]) * inp["L_bar"] - inp["E_bar"]) / 0.125) < 1e-9
    assert abs(zm["xi_trap"] - M / 0.125) < 1e-9
    assert abs(lin["zeroth_order_xi_star"] - z["xi_trap"]) < 1e-9
    assert abs(lin["xi_star_exact_LE"] - lin["xi_star_lin"]) < 1e-12
    _, _, mc, sc = T.stationary_law(T.params_vector(CUBIC))
    assert np.isnan(T.closed_form_observables(_block_exit_window(mc, sc), CUBIC,
                                              "during", _std(CUBIC))["xi_trap"])
    assert np.isfinite(T.zeroth_order_observables(_block_exit_window(mc, sc), CUBIC,
                                                  "during", _std(CUBIC))["xi_trap"])


def test_B23b_zeroth_order_no_exit_offset_for_the_cubic_is_pinned():
    _, _, m, s = T.stationary_law(T.params_vector(CUBIC))
    res = {}
    for per in ("pre", "during"):
        c = _comp([(300, 0, 0, m, s, m, s, 0.0)] * len(T.period_transitions(per)),
                  period=per)
        z = T.zeroth_order_observables(c, CUBIC, per, _std(CUBIC))
        t = T.truth(CUBIC, _std(CUBIC), per)
        p = T.predict(c, CUBIC, per, _std(CUBIC))["estimate"]
        for q in ("xi_trap", "two_delta_V", "tau_mfpt", "rho1", "mean_diffusion"):
            assert p[q] == pytest.approx(t[q], rel=1e-9, abs=1e-12), (per, q)
        assert z["tau_mfpt"] == pytest.approx(0.044178, rel=1e-4)
        assert t["tau_mfpt"] == pytest.approx(0.04052, rel=5e-4)
        assert z["tau_mfpt"] / t["tau_mfpt"] - 1 == pytest.approx(0.090, abs=2e-3)
        assert z["two_delta_V"] == pytest.approx(2.4443e-5, rel=1e-4)
        assert t["two_delta_V"] == pytest.approx(2.4454e-5, rel=1e-4)
        assert z["xi_trap"] == pytest.approx(t["xi_trap"], abs=1e-12)
        assert z["rho1"] == pytest.approx(t["rho1"], abs=1e-12)
        res[per] = (z, t)
    cb = contrast_bias(res["pre"][0], res["during"][0], res["pre"][1], res["during"][1])
    assert cb["tau_mfpt"] == pytest.approx(2.445e-4, rel=2e-2)
    assert 0 < cb["two_delta_V"] < 1e-9
    assert abs(cb["xi_trap"]) < 1e-12 and abs(cb["rho1"]) < 1e-12


def _calibration():
    import json
    return json.loads((T.REPO / "src/k1_sim_calibration.json").read_text())


def _registered(process, period, guarded=True):
    """The registered (i) / (ii) true processes as k1_sim builds them
    (k1_sim.process: (i) the V1 OU fit, unguarded; (ii) the V1 cubic_constB
    fit with guard = the 1–99 % support and slope floor = −|a1 of the OU fit|)."""
    f = _calibration()["v1_mle_fits"][period]
    if process == "ou":
        return dict(zip(TRUE_PARAM_KEYS, f["ou"]), **T.NO_GUARD)
    tp = dict(zip(TRUE_PARAM_KEYS, f["cubic_constB"]), **T.NO_GUARD)
    if guarded:
        lo, hi = f["support_p01_p99"]
        tp.update(guard_lo=lo, guard_hi=hi, slope_floor=-abs(f["ou"][1]))
    return tp


def _sim_true_process(params, guard, floor):
    p = np.asarray(params, float)

    def pA(x):
        return p[0] + p[1] * x + p[2] * x ** 2 + p[3] * x ** 3

    def pdA(x):
        return p[1] + 2 * p[2] * x + 3 * p[3] * x ** 2
    lo, hi = guard
    fl = -abs(floor)
    s_lo, s_hi = min(pdA(lo), fl), min(pdA(hi), fl)

    def A(x):
        a = pA(np.clip(x, lo, hi))
        return a + np.where(x > hi, s_hi * (x - hi), 0.0) + np.where(x < lo, s_lo * (x - lo), 0.0)

    def logB(x):
        x = np.clip(x, lo, hi)
        return np.clip(p[4] + p[5] * x + p[6] * x ** 2, -20.0, 20.0)
    return A, logB


def test_A4b_pointwise_lemma2prime_equals_map_params():
    """transition_moments (the pointwise Lemma 2' used by _pooled for every
    chain) equals the coefficient map of map_params for an unguarded cubic."""
    th = T.params_vector(CUBIC)
    xi_ = np.linspace(-4, 4, 41)
    a, b = T.transition_moments(th, None, xi_, 0.3, 1.4, -0.2, 0.9)
    m = T.map_params(th, 0.3, 1.4, -0.2, 0.9)
    np.testing.assert_allclose(a, _A(m, xi_), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(b, np.exp(_logB(m, xi_)), rtol=1e-12)


def test_B24_unconfined_chain_is_refused_never_returned():
    cal = _calibration()
    th = np.array(cal["v1_mle_fits"]["during"]["cubic_constB"])
    assert th[3] > 0
    with pytest.raises(ValueError, match="not confined"):
        T.stationary_law(th)
    tp = _registered("cubic", "during", guarded=False)
    std = pd.DataFrame({"year": range(2000, 2024), "n": 1000, "mu_x": 0.0,
                        "sigma_x": 1.0, "x_units": X_UNITS, C.SHOCK_SHARE_COLUMN: 0.0})
    with pytest.raises(ValueError, match="not confined"):
        T.truth(tp, std, "during")
    with pytest.raises(ValueError, match="not confined"):
        T.stationary_standardisation(tp)
    gu = {k: v for k, v in T.GOLDEN_GUARDED.items()}
    with pytest.raises(ValueError, match="not confined"):
        T.stationary_law(T.params_vector(gu))
    for tp in (OU, CUBIC, T.GOLDEN_GUARDED, _registered("cubic", "pre"),
               _registered("cubic", "during"), _registered("ou", "pre"),
               _registered("ou", "during")):
        assert T.stationary_edge_mass(*T.process_spec(tp)) < 1e-12


def test_B24b_true_params_must_declare_the_guard():
    """Contract 6: every true_params dict declares guard_lo, guard_hi,
    slope_floor (None when unguarded), so a guarded simulation can never be
    paired silently with an unguarded theory."""
    seven = {k: OU[k] for k in TRUE_PARAM_KEYS}
    with pytest.raises(ValueError, match="lacks"):
        T.truth(seven, _std(OU), "pre")
    with pytest.raises(ValueError, match="lacks"):
        T.params_vector(seven)
    with pytest.raises(ValueError, match="partial guard"):
        T.process_spec({**OU, "guard_lo": -2.0})
    with pytest.raises(ValueError, match="slope_floor"):
        T.process_spec({**OU, "guard_lo": -2.0, "guard_hi": 2.0, "slope_floor": 0.1})
    with pytest.raises(ValueError, match="< guard_hi"):
        T.process_spec({**OU, "guard_lo": 2.0, "guard_hi": -2.0, "slope_floor": -0.1})
    assert T.process_spec(OU)[1] is None
    g = {**OU, "guard_lo": -2.0, "guard_hi": 2.0, "slope_floor": -0.1}
    _, _, m, s = T.stationary_law(*T.process_spec(g))
    none = _comp([(300, 0, 0, m, s, m, s, 0.0)] * 6)
    assert np.isnan(T.closed_form_observables(none, g, "during",
                                              T.stationary_standardisation(g))["xi_trap"])


def test_B24c_guard_semantics_are_the_simulators():
    """The theory's guarded A and log B equal k1_sim.TrueProcess's (an
    independent copy of its formulas) on and beyond the support, for the
    registered (ii) fits of both periods; the fixed points include the
    linear continuations."""
    xs = np.linspace(-12, 12, 2001)
    for per in ("pre", "during"):
        tp = _registered("cubic", per)
        th, g = T.process_spec(tp)
        A, logB = _sim_true_process(th, (g[0], g[1]), g[2])
        np.testing.assert_allclose(T.drift(th, g, xs), A(xs), rtol=0, atol=1e-14)
        np.testing.assert_allclose(T.log_diffusion(th, g, xs), logB(xs), rtol=0, atol=1e-14)
        for r in T.fixed_points(th, g):
            assert abs(A(np.array([r]))[0]) < 1e-10
    th = np.array([0.5, -0.1, 0.0, 0.0, np.log(0.1), 0.0, 0.0])
    fps = T.fixed_points(th, (-1.0, 1.0, -0.2))
    assert len(fps) == 1 and fps[0] == pytest.approx(1.0 + 0.4 / 0.2)


def test_B24d_theory_truth_is_the_truth_of_the_simulated_guarded_chain():
    """The theory's stationary law of the guarded registered (ii) chain
    matches a long run of that chain (k1_sim's formulas; 20,000 counties,
    400 Euler steps, no province structure): mean within 0.02 s.d., s.d.
    within 2 % — the k1_sim.theory_compat thresholds.  The guard is active
    on 2–3 % of the stationary mass in BOTH periods, so the unguarded
    polynomial is not the simulated chain even where it confines ('pre')."""
    rng = np.random.default_rng(20261022)
    for per in ("pre", "during"):
        tp = _registered("cubic", per)
        th, g = T.process_spec(tp)
        grid, p, m, s = T.stationary_law(th, g)
        A, logB = _sim_true_process(th, (g[0], g[1]), g[2])
        x = rng.normal(m, s, 20000)
        for _ in range(400):
            x = x + A(x) + np.sqrt(np.exp(logB(x))) * rng.standard_normal(x.size)
        assert abs(x.mean() - m) < 0.02 * s, (per, x.mean(), m)
        assert abs(x.std() / s - 1) < 0.02, (per, x.std(), s)
        active = p[(grid < g[0]) | (grid > g[1])].sum()
        assert 0.01 < active < 0.05, (per, active)
        tr = T.truth(tp, T.stationary_standardisation(tp), per)
        assert all(np.isfinite(tr[q]) for q in QUANTITIES), (per, tr)
        assert 0.8 < tr["rho1"] < 1.0 and 0.1 < tr["mean_diffusion"] < 0.3, (per, tr)
    a = T.truth(_registered("cubic", "pre", False),
                T.stationary_standardisation(_registered("cubic", "pre", False)), "pre")
    b = T.truth(_registered("cubic", "pre"),
                T.stationary_standardisation(_registered("cubic", "pre")), "pre")
    assert abs(a["two_delta_V"] / b["two_delta_V"] - 1) > 0.05


def test_B23c_zeroth_order_no_exit_offset_on_the_registered_processes():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "k1_zoff", T.REPO / "scripts/k1_zeroth_order_offset.py")
    Z = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Z)
    cal = _calibration()
    res = {}
    for proc in ("ou", "cubic"):
        for per in ("pre", "during"):
            r = Z.rows_for(Z.registered(cal, proc, per), per, linear=proc == "ou")
            for q in ("xi_trap", "two_delta_V", "tau_mfpt", "rho1", "mean_diffusion"):
                assert r["projection"][q] == pytest.approx(r["truth"][q], rel=1e-9,
                                                           abs=1e-12), (proc, per, q)
            res[proc, per] = r
            for q in ("xi_trap", "two_delta_V", "tau_mfpt", "rho1", "mean_diffusion"):
                assert r["zeroth_order_mbar"][q] == r["zeroth_order"][q], (proc, per, q)
    for per in ("pre", "during"):
        z, t = res["ou", per]["zeroth_order"], res["ou", per]["truth"]
        assert abs(z["two_delta_V"] - t["two_delta_V"]) < 1e-12
        assert abs(z["tau_mfpt"] - t["tau_mfpt"]) < 1e-12
    zp, tp_ = res["cubic", "pre"]["zeroth_order"], res["cubic", "pre"]["truth"]
    zd, td = res["cubic", "during"]["zeroth_order"], res["cubic", "during"]["truth"]
    assert zp["tau_mfpt"] == pytest.approx(2.6125, rel=1e-3)
    assert tp_["tau_mfpt"] == pytest.approx(2.3230, rel=1e-3)
    assert zd["tau_mfpt"] == pytest.approx(1.0767, rel=1e-3)
    assert td["tau_mfpt"] == pytest.approx(1.0502, rel=1e-3)
    assert zp["two_delta_V"] / tp_["two_delta_V"] - 1 == pytest.approx(0.0317, abs=1e-3)
    assert zd["two_delta_V"] / td["two_delta_V"] - 1 == pytest.approx(0.0036, abs=3e-4)
    cb = contrast_bias(zp, zd, tp_, td)
    assert cb["tau_mfpt"] == pytest.approx(-0.0399, abs=5e-4)
    assert cb["two_delta_V"] == pytest.approx(-4.63e-4, rel=1e-2)
    assert abs(cb["xi_trap"]) < 1e-12 and abs(cb["rho1"]) < 1e-12
    u = Z.registered(cal, "cubic_unguarded", "pre")
    ru = Z.rows_for(u, "pre", linear=False)
    assert ru["zeroth_order"]["tau_mfpt"] == pytest.approx(2.603, rel=1e-3)
    assert ru["truth"]["tau_mfpt"] == pytest.approx(2.296, rel=1e-3)
    assert ru["zeroth_order"]["two_delta_V"] == pytest.approx(0.01432, rel=1e-3)
    assert ru["truth"]["two_delta_V"] == pytest.approx(0.01386, rel=1e-3)


def _s0_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("k1_s0", T.REPO / "scripts/k1_s0.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    return S


def _stub_backend(name, xi_offset=0.0, unguarded=False, observed_n=None):
    import sys
    import types
    m = types.ModuleType(name)

    def tps(process):
        tp = {p: C.registered_true_params(process, p) for p in PERIODS}
        if unguarded:
            tp = {p: dict(v, **T.NO_GUARD) for p, v in tp.items()}
        return tp

    def cell_meta(process, mech, pipeline, s0):
        if s0:
            return _cell(process=process, mech="a", N=2 * C.S0_N, pipeline=pipeline,
                         N_windows=C.s0_window_sizes())
        return _cell(process=process, mech=mech, pipeline=pipeline)

    def simulate_no_exit(cell, r, seed):
        _, std, _ = _reg_window(cell["process"], pipeline=cell["pipeline"],
                                n=observed_n or cell["N"] // 2)
        tp = tps(cell["process"])
        est = {p: dict(T.truth(C.registered_true_params(cell["process"], p), std, p))
               for p in PERIODS}
        for p in PERIODS:
            est[p]["xi_trap"] += xi_offset
        return {"pairs": {p: {"est": est[p]} for p in PERIODS}, "standardisation": std,
                "true_params": tp}

    def simulate_composition(cell, r):
        _, std, comp = _reg_window(cell["process"], pipeline=cell["pipeline"],
                                   n=observed_n or cell["N"] // 2)
        if cell["exit_mechanism"] == "e":
            comp = comp.assign(mech_term=comp["mech_term"] * 1.5)
        return {"composition": comp.assign(replicate=r), "standardisation": std,
                "true_params": tps(cell["process"]), "seed": 7000 + r}

    m.cell_meta, m.simulate_no_exit = cell_meta, simulate_no_exit
    m.simulate_composition = simulate_composition
    m.estimate = lambda pairs: pairs["est"]
    sys.modules[name] = m
    return name


def test_B25_s0_script_runs_end_to_end(monkeypatch, tmp_path):
    import gzip
    import json
    S = _s0_script()
    repo = tmp_path / "repo"
    repo.mkdir()
    git = _s0_repo(monkeypatch, repo)
    monkeypatch.setattr(C, "R_PER_CELL", 4)
    raw = tmp_path / "raw"
    ok = _stub_backend("k1_s0_stub_ok")
    with pytest.raises(RuntimeError, match="no open S0 attempt"):
        S.run("cubic", "direct", range(0, 4), raw, backend_name=ok)
    with pytest.raises(RuntimeError, match="not importable"):
        S.load_backend("k1_s0_no_such_backend")
    st = C.start_s0_attempt("cubic", "direct")
    S.run("cubic", "direct", range(0, 2), raw, backend_name=ok)
    S.run("cubic", "direct", range(2, 4), raw, backend_name=ok)
    with gzip.open(sorted(raw.glob("*.json.gz"))[0], "rt") as f:
        chunk = json.load(f)
    assert chunk["meta"]["code_commit"] == st["start_commit"]
    assert [d["seed"] for d in chunk["s0"]] == C.s0_seeds("cubic", "direct")[:2]
    assert set(chunk["k1a"]) == {"d", "e"} and "estimate" not in chunk["k1a"]["d"]["replicates"][0]
    with pytest.raises(RuntimeError, match="COMMITTED S0 record"):
        S.record("cubic", "direct", raw, dry_run=True)
    C.start_s0_attempt("cubic", "saai")
    out = S.record("cubic", "direct", raw)
    res = out["result"]
    assert res["pass"], res["label"]
    assert res["predicted_effect"]["pipeline"] == "direct"
    assert res["k1a_composition_seeds"]["d"] == [7000, 7001, 7002, 7003]
    rec = json.loads(out["path"].read_text())
    assert rec["estimates_commit"] == rec["start_commit"] == st["start_commit"]
    assert not git("status", "--porcelain")
    dry = S.record("cubic", "direct", raw, dry_run=True)
    assert dry["path"] is None and dry["matches"] and not git("status", "--porcelain")
    assert dry["result_sha256"] == rec["result_sha256"] == out["result_sha256"]
    assert C._require_s0_pass("cubic", "direct")["pass"] is True
    with pytest.raises(RuntimeError, match="no open S0 attempt"):
        S.record("cubic", "direct", raw)
    (sorted(raw.glob("*.json.gz"))[1]).unlink()
    with pytest.raises(RuntimeError, match="exactly the replicates"):
        S.load_raw(raw, "cubic", "direct")
    monkeypatch.setattr(C, "_require_frozen", lambda: {"frozen_commit": "F2"})
    C.start_s0_attempt("cubic", "direct")
    bad = _stub_backend("k1_s0_stub_unguarded", unguarded=True)
    with pytest.raises(ValueError, match="registered"):
        S.run("cubic", "direct", range(0, 4), tmp_path / "raw2", backend_name=bad)
    obs = _stub_backend("k1_s0_stub_observed", observed_n=1000)
    S.run("cubic", "direct", range(0, 4), tmp_path / "raw_obs", backend_name=obs)
    with pytest.raises(ValueError, match="complete panel"):
        S.record("cubic", "direct", tmp_path / "raw_obs")
    off = _stub_backend("k1_s0_stub_off", xi_offset=0.5)
    S.run("cubic", "direct", range(0, 4), tmp_path / "raw3", backend_name=off)
    r = S.record("cubic", "direct", tmp_path / "raw3")
    assert not r["result"]["pass"] and r["result"]["pre"]["xi_trap"]["status"] == "fail"
    with pytest.raises(RuntimeError, match="did not pass"):
        C._require_s0_pass("cubic", "direct")
    assert [e["stage"] for e in C._read_attempts()
            if (e["process"], e["pipeline"]) == ("cubic", "direct")] == [
        "started", "result", "started", "result"]


def test_B25b_s0_code_is_pinned_to_the_attempts_start_commit(monkeypatch, tmp_path):
    import json
    S = _s0_script()
    repo = tmp_path / "repo"
    repo.mkdir()
    git = _s0_repo(monkeypatch, repo)
    monkeypatch.setattr(C, "R_PER_CELL", 4)
    (repo / "scripts").mkdir()
    (repo / "scripts/k1_s0.py").write_text("# stand-in for the S0 script\n")
    (repo / "src/k1_s0_backend.py").write_text("# stand-in for the backend\n")
    git("add", "-A")
    git("commit", "-qm", "backend and script")
    ok = _stub_backend("k1_s0_stub_pin")
    st = C.start_s0_attempt("cubic", "direct")
    raw = tmp_path / "raw"
    S.run("cubic", "direct", range(0, 4), raw, backend_name=ok)
    with pytest.raises(RuntimeError, match="COMMITTED S0 record"):
        S.record("cubic", "direct", raw, dry_run=True)
    for rel in ("src/k1_s0_backend.py", "scripts/k1_s0.py", "src/k1_theory_stub.py"):
        (repo / rel).write_text("# edited after start\n")
        git("add", "-A")
        git("commit", "-qm", f"edit {rel} after start")
        with pytest.raises(RuntimeError, match="differs from the S0 start commit"):
            S.record("cubic", "direct", raw)
        with pytest.raises(RuntimeError, match="S0 runs once per freeze"):
            S.run("cubic", "direct", range(0, 4), tmp_path / "raw_edit", backend_name=ok)
        git("revert", "--no-edit", "HEAD")
        edited = git("rev-parse", "HEAD~1")
        with pytest.raises(RuntimeError, match="estimates come from"):
            C.s0_code_binding(C.open_s0_attempt("cubic", "direct"), edited)
    assert [e["stage"] for e in C._read_attempts()] == ["started"]
    out = S.record("cubic", "direct", raw)
    rec = json.loads(out["path"].read_text())
    assert rec["start_commit"] == st["start_commit"]
    monkeypatch.setattr(C, "_require_frozen", lambda: {"frozen_commit": "F2"})
    C.start_s0_attempt("cubic", "direct")
    (repo / "src/k1_s0_backend.py").write_text("# edited again\n")
    git("add", "-A")
    git("commit", "-qm", "edit backend")
    with pytest.raises(RuntimeError, match="estimates come from"):
        C.write_s0_record(out["result"], C.s0_seeds("cubic", "direct"),
                          git("rev-parse", "HEAD"))


def test_B28_s0_pass_is_bound_to_the_validated_code(monkeypatch, tmp_path):
    import gzip
    import json
    import shutil
    import subprocess
    S = _s0_script()
    repo = tmp_path / "repo"
    repo.mkdir()
    git = _s0_repo(monkeypatch, repo)
    monkeypatch.setattr(C, "R_PER_CELL", 4)
    (repo / "scripts").mkdir()
    (repo / "scripts/k1_s0.py").write_text("from src import k1_contract as C\n")
    (repo / "src/k1_s0_backend.py").write_text(
        "from src import k1_sim as S\n\ndef estimate(pairs):\n    return S.estimate(pairs)\n")
    (repo / "src/k1_sim.py").write_text(
        "def estimate(pairs):\n    from src.k1_later import x  # lazy\n    return pairs\n")
    git("add", "-A")
    git("commit", "-qm", "backend, simulator and script")
    code = C.s0_code_digest()
    assert set(code) == {"scripts/k1_s0.py", "src/k1_s0_backend.py", "src/k1_sim.py",
                         "src/k1_later.py", "src/k1_contract.py"}
    assert code["src/k1_later.py"] is None and code["src/k1_contract.py"] is None
    ok = _stub_backend("k1_s0_stub_bind")
    st = C.start_s0_attempt("cubic", "direct")
    raw = tmp_path / "raw"
    S.run("cubic", "direct", range(0, 4), raw, backend_name=ok)
    with gzip.open(sorted(raw.glob("*.json.gz"))[0], "rt") as f:
        assert json.load(f)["meta"]["s0_code"] == code
    bad = tmp_path / "raw_bad"
    shutil.copytree(raw, bad)
    for fp in bad.glob("*.json.gz"):
        with gzip.open(fp, "rt") as f:
            d = json.load(f)
        d["meta"]["s0_code"] = {**code, "src/k1_sim.py": "0" * 40}
        with gzip.open(fp, "wt") as f:
            json.dump(d, f)
    with pytest.raises(RuntimeError, match="made by code that differs"):
        S.record("cubic", "direct", bad)
    out = S.record("cubic", "direct", raw)
    rec = json.loads(out["path"].read_text())
    assert rec["pass"] and rec["s0_code"] == code and rec["start_commit"] == st["start_commit"]
    cell_e = _cell(process="cubic", mech="e", pipeline="direct")
    C.assert_exit_estimation_allowed(cell_e)
    assert S.record("cubic", "direct", raw, dry_run=True)["matches"]
    for rel, text in (("src/k1_s0_backend.py", "def estimate(pairs):\n    return {}\n"),
                      ("src/k1_sim.py", "def estimate(pairs):\n    return dict(pairs)\n"),
                      ("scripts/k1_s0.py", "# refactored\n"),
                      ("src/k1_later.py", "x = 1\n")):
        before = (repo / rel).read_text() if (repo / rel).exists() else None
        (repo / rel).write_text(text)
        git("add", "-A")
        git("commit", "-qm", f"edit {rel} after S0")
        with pytest.raises(RuntimeError, match="S0-validated code changed"):
            C.assert_exit_estimation_allowed(cell_e)
        with pytest.raises(RuntimeError, match=rel.replace(".", "\\.")):
            C._require_s0_pass("cubic", "direct")
        arc = tmp_path / f"arc_{rel.replace('/', '_')}"
        arc.mkdir()
        tar = subprocess.run(["git", "-C", str(repo), "archive", "HEAD"], check=True,
                             capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(arc)], input=tar, check=True)
        monkeypatch.setattr(C, "REPO", arc)
        assert C.archive_commit() is not None
        with pytest.raises(RuntimeError, match="S0-validated code changed"):
            C.assert_exit_estimation_allowed(cell_e)
        monkeypatch.setattr(C, "REPO", repo)
        if before is None:
            (repo / rel).unlink()
        else:
            (repo / rel).write_text(before)
        git("add", "-A")
        git("commit", "-qm", f"restore {rel}")
        C.assert_exit_estimation_allowed(cell_e)
    (repo / "src/k1_sim.py").write_text("# dirty\n")
    with pytest.raises(RuntimeError, match="S0-validated code changed"):
        C._require_s0_pass("cubic", "direct")
    git("checkout", "--", "src/k1_sim.py")
    C.assert_exit_estimation_allowed(cell_e)
    assert "s0_code" in rec
    noc = {k: v for k, v in rec.items() if k not in ("s0_code", "s0_code_sha256")}
    with pytest.raises(RuntimeError, match="no s0_code digest"):
        C._check_s0_code(noc)


def test_B28b_s0_code_closure_follows_imports():
    code = C.s0_code_digest()
    for rel in ("scripts/k1_s0.py", "src/k1_s0_backend.py", "src/k1_contract.py",
                "src/k1_theory.py", "src/km_mle.py", "src/config.py"):
        assert rel in code, rel
    files = {"src/a.py": b"from . import b\nimport importlib\nM = 'src.c'\n",
             "src/b.py": b"def f():\n    from .d import g\n",
             "src/c.py": b"", "src/d.py": b"from src.e import h\n"}
    read = files.get
    got = C._imports_of("src/a.py", files["src/a.py"], read)
    assert "src/b.py" in got and "src/c.py" in got
    assert C._imports_of("src/b.py", files["src/b.py"], read) == ["src/d.py"]
    assert C._imports_of("src/d.py", files["src/d.py"], read) == ["src/e.py"]


def test_B28c_start_checks_the_backend_before_it_commits_an_attempt(monkeypatch, tmp_path):
    import sys
    S = _s0_script()
    repo = tmp_path / "repo"
    repo.mkdir()
    git = _s0_repo(monkeypatch, repo)
    name = _stub_backend("k1_s0_stub_pre")
    m = sys.modules[name]

    def blocked(process, pipeline):
        raise RuntimeError(f"blocked {process}/{pipeline}")
    m.s0_precheck = blocked
    with pytest.raises(RuntimeError, match="blocked cubic/saai"):
        S.start("cubic", "saai", backend_name=name)
    m.s0_precheck = lambda process, pipeline: None
    meta = m.cell_meta
    for bad in ({"N_windows": {"pre": C.S0_N + 5000, "during": C.S0_N - 5000}},
                {"N": C.S0_N, "N_windows": {"pre": C.S0_N // 2, "during": C.S0_N // 2}},
                {"N_windows": None}):
        m.cell_meta = lambda p, mech, pl, s0, bad=bad: {**meta(p, mech, pl, s0), **bad} \
            if s0 else meta(p, mech, pl, s0)
        with pytest.raises(RuntimeError, match="not the registered S0 cell"):
            S.start("cubic", "saai", backend_name=name)
    assert C._read_attempts() == [] and not git("status", "--porcelain")
    m.cell_meta = meta
    e = S.start("cubic", "saai", backend_name=name)
    assert e["attempt"] == 1 and [x["stage"] for x in C._read_attempts()] == ["started"]


def test_B26_standardisation_must_be_the_cells_complete_panel(guards):
    cell = _cell(process="cubic", pipeline="direct")
    tp, std, comp = _reg_window("cubic")
    assert (std["n"] == CELL_N).all() and cell["N"] == 2 * CELL_N
    T.predict_row("projection", comp, tp["during"], "during", std, cell)
    T.truth(tp["pre"], std, "pre", cell=cell)
    obs = std.assign(n=np.where(std["year"] >= 2016, CELL_N - 192, CELL_N))
    for call in (lambda: T.truth(tp["during"], obs, "during", cell=cell),
                 lambda: T.predict(comp, tp["during"], "during", obs, cell=cell),
                 lambda: T.predict_bias(comp, tp["during"], "during", obs, cell=cell),
                 lambda: T.zeroth_order_observables(comp, tp["during"], "during", obs,
                                                    cell=cell),
                 *[(lambda r=r: T.predict_row(r, comp, tp["during"], "during", obs, cell))
                   for r in C.REPORT_ROWS]):
        with pytest.raises(ValueError, match="complete panel"):
            call()
    with pytest.raises(ValueError, match="complete panel"):
        T.predict_row("projection", comp, tp["pre"], "pre", std, {**cell, "N": 20000})
    with pytest.raises(ValueError, match="complete panel"):
        T.predict_row("projection", comp, tp["pre"], "pre", std, {**cell, "N": CELL_N})
    with pytest.raises(ValueError, match="one row per year"):
        T.truth(tp["pre"], std[std["year"] != 2005], "pre", cell=cell)
    small = _reg_window("cubic", n=200)[1]
    with pytest.raises(ValueError, match="exceed the complete"):
        T.predict_row("projection", comp, tp["pre"], "pre", small, {**cell, "N": 400})
    uneq = std.assign(n=np.where(std["year"] <= 2012, CELL_N + 400, CELL_N))
    T.predict_row("projection", comp, tp["during"], "during", uneq,
                  {**cell, "N": 2 * CELL_N + 400})
    T.truth(tp["during"], obs, "during")


def _design_entry(tp):
    return {k: tp[k] for k in C.TRUE_PROCESS_KEYS}


def test_B27_processes_are_member_keyed(monkeypatch, guards):
    members = ("cubic_bistable_0", "cubic_bistable_1", "cubic_bistable_2",
               "cubic_bistable_4")
    assert set(members) | {"ou", "cubic", "cubic_mult", "cubic_shift"} == set(C.PROCESSES)
    assert "cubic_bistable" not in C.PROCESSES
    for call in (lambda: C.registered_true_params("cubic_bistable", "pre"),
                 lambda: C.s0_seeds("cubic_bistable", "saai"),
                 lambda: C.check_cell_config(_cell(process="cubic_bistable", mech="a"))):
        with pytest.raises(ValueError, match="FAMILY"):
            call()
    base = C.S0_SEED_BASE
    for i, p in enumerate(("ou", "cubic", "cubic_mult")):
        assert C.s0_seeds(p, "direct")[:2] == [base + 10000 * i + 1000, base + 10000 * i + 1001]
    assert C.s0_seeds("cubic_shift", "saai")[0] == base + 40000
    assert all(C.s0_seeds(m, "saai") == C.s0_seeds(members[0], "saai") for m in members)
    assert len({C.s0_record_path(m, "saai") for m in members}) == 4
    monkeypatch.setitem(C._CAL, "design", {})
    for m in members:
        with pytest.raises(ValueError, match="no registered constants"):
            C.registered_true_params(m, "pre")
    assert C.registered_true_params("cubic_shift", "pre") == C.registered_true_params(
        "cubic", "pre")
    with pytest.raises(ValueError, match="no registered constants"):
        C.registered_true_params("cubic_shift", "during")
    ou = C.registered_true_params("ou", "pre")
    sym = dict(ou, a0=0.0, a1=0.2, a2=0.0, a3=-0.1)
    monkeypatch.setitem(C._CAL, "design", {"cubic_bistable_2": {
        "pre": _design_entry(sym), "during": _design_entry(sym)}})
    assert C.registered_true_params("cubic_bistable_2", "during")["a3"] == -0.1
    for m in ("cubic_bistable_1", "cubic_bistable_4"):
        with pytest.raises(ValueError, match="no registered constants"):
            C.registered_true_params(m, "during")
    with pytest.raises(ValueError, match="registered"):
        C.check_registered_true_params("cubic_bistable_4", "pre", sym)
    for bad in ({k: v for k, v in sym.items() if k != "guard_lo"},
                dict(sym, label="x"), dict(sym, guard_lo=-1.0)):
        monkeypatch.setitem(C._CAL, "design", {"cubic_bistable_1": {"pre": bad}})
        with pytest.raises(ValueError):
            C.registered_true_params("cubic_bistable_1", "pre")
    monkeypatch.setitem(C._CAL, "design", None)
    monkeypatch.delitem(C._CAL, "design")
    import json
    d = json.loads((T.REPO / C.DESIGN_PROCESS_FILE).read_text())
    assert C.DESIGN_PROCESS_FILE in T.FROZEN_FILES
    assert set(d["members"]) <= set(C.DESIGN_MEMBERS)
    for m, ent in d["members"].items():
        assert tuple(ent) == C.DESIGN_MEMBERS[m]
        for per, tp in ent.items():
            assert C.registered_true_params(m, per) == {
                k: (None if v is None else float(v)) for k, v in tp.items()}
    for m in set(C.DESIGN_MEMBERS) - set(d["members"]):
        assert m in d.get("unregistered", {})
        with pytest.raises(ValueError, match="no registered constants"):
            C.registered_true_params(m, C.DESIGN_MEMBERS[m][0])
    assert "scripts/k1_s0.py" in T.FROZEN_FILES


def test_D2_absorbed_diffusion_formula():
    assert T.TRUTH_ABSORBS_PROVINCE_SHOCK_SHARE is True
    pq = np.full(5, 0.2)
    b = np.full(5, 0.3)
    assert np.allclose(T.absorbed_diffusion(b, pq, 0.04), 0.96 * 0.3)
    assert T.absorbed_diffusion(b, pq, 0.0) is b
    bb = np.array([0.1, 0.2, 0.5, 1.0, 2.0])
    assert (T.absorbed_diffusion(bb, pq, 0.5) > 0).all()
    for bad in (-0.1, 1.0, np.nan):
        with pytest.raises(ValueError, match="shock share"):
            T.absorbed_diffusion(bb, pq, bad)
    rng = np.random.default_rng(0)
    n, w, reps = 2000, 0.3, 400
    sizes = np.array([400, 300, 300, 200, 200, 200, 150, 150, 100])
    prov = np.repeat(np.arange(len(sizes)), sizes)
    H = float(np.sum((sizes / n) ** 2))
    x = rng.standard_normal(n)
    B = np.exp(-1.0 + 0.6 * x)
    r2 = np.zeros(n)
    for _ in range(reps):
        e = np.sqrt(1 - w) * rng.standard_normal(n) + np.sqrt(w) * rng.standard_normal(
            len(sizes))[prov]
        u = np.sqrt(B) * e
        r2 += (u - u.mean()) ** 2
    r2 /= reps
    pred = T.absorbed_diffusion(B, np.full(n, 1.0 / n), w * H)
    edges = np.quantile(x, np.linspace(0, 1, 6))
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (x >= lo) & (x <= hi)
        assert abs(r2[k].mean() / pred[k].mean() - 1) < 0.02
    assert abs(r2.mean() / B.mean() - 1) > 0.5 * w * H


def test_D2_truth_and_prediction_carry_the_term(guards):
    tp, std, comp = _reg_window("ou")
    base = T.truth(tp["during"], std, "during")
    wH = 0.038
    t1 = T.truth(tp["during"], std.assign(**{C.SHOCK_SHARE_COLUMN: wH}), "during")
    rho, D = 1 + tp["during"]["a1"], np.exp(tp["during"]["c0"])
    s2 = float(std.loc[std["year"] == 2014, "sigma_x"].iloc[0]) ** 2
    e0, e1 = rho ** 2 * s2 + D, rho ** 2 * s2 + D * (1 - wH)
    assert np.isclose(e0 / s2, 1.0, rtol=1e-4)
    assert np.isclose(base["mean_diffusion"], D / s2, rtol=1e-4)
    assert np.isclose(t1["mean_diffusion"] / base["mean_diffusion"],
                      (1 - wH) * e0 / e1, rtol=1e-6)
    assert np.isclose(t1["mean_diffusion"], D * (1 - wH) / e1, rtol=1e-4)
    assert np.isclose(t1["xi_trap"], base["xi_trap"], atol=1e-9)
    th_d = T.params_vector(tp["during"])
    cp0 = T._truth_full(th_d, std, "during")["rho1_complete_projection"]
    cp1 = T._truth_full(th_d, std.assign(**{C.SHOCK_SHARE_COLUMN: wH}),
                        "during")["rho1_complete_projection"]
    assert np.isclose(cp1 / cp0, np.sqrt(e0 / e1), rtol=1e-9)
    assert t1["rho1"] == base["rho1"] == pytest.approx(rho, abs=1e-12)
    for bad in (std.drop(columns=C.SHOCK_SHARE_COLUMN),):
        with pytest.raises(ValueError, match="lacks columns"):
            T.truth(tp["during"], bad, "during")
    with pytest.raises(ValueError, match="share"):
        T.truth(tp["during"], std.assign(**{C.SHOCK_SHARE_COLUMN: 1.2}), "during")
    pre = comp[comp["period"] == "pre"]
    s1 = std.assign(**{C.SHOCK_SHARE_COLUMN: wH})
    out = T.predict(pre, tp["pre"], "pre", s1)
    assert all(abs(out["bias"][q]) < 1e-6 for q in ("xi_trap", "mean_diffusion", "rho1"))
    rho_p, D_p = 1 + tp["pre"]["a1"], np.exp(tp["pre"]["c0"])
    s2_p = float(std.loc[std["year"] == 2005, "sigma_x"].iloc[0]) ** 2
    assert np.isclose(out["truth"]["mean_diffusion"]
                      / T.truth(tp["pre"], std, "pre")["mean_diffusion"],
                      (1 - wH) * (rho_p ** 2 * s2_p + D_p)
                      / (rho_p ** 2 * s2_p + D_p * (1 - wH)), rtol=1e-6)


def _s1_table(tp, wH=0.03, seed=5, n=CELL_N):
    """A complete-panel table whose realised (μ_t, σ_t) wander around the
    stationary moments (as a finite panel's do)."""
    m, sd = T.stationary_law(*T.process_spec(tp))[2:]
    rng = np.random.default_rng(seed)
    years = range(2000, 2024)
    return pd.DataFrame({"year": list(years), "n": n,
                         "mu_x": m + 0.05 * sd * rng.standard_normal(len(years)),
                         "sigma_x": sd * (1 + 0.03 * rng.standard_normal(len(years))),
                         "x_units": X_UNITS, C.SHOCK_SHARE_COLUMN: wH})


def test_S1_truth_slope_is_population_rho_ou(guards):
    tp = C.registered_true_params("ou", "pre")
    std = _s1_table(tp)
    s = std.set_index("year")
    rho, D, wH = 1 + tp["a1"], np.exp(tp["c0"]), 0.03
    ts = T.period_transitions("pre")
    pop = np.mean([rho * s.at[t, "sigma_x"]
                   / np.sqrt(rho ** 2 * s.at[t, "sigma_x"] ** 2 + D * (1 - wH)) for t in ts])
    old = np.mean([rho * s.at[t, "sigma_x"] / s.at[t + 1, "sigma_x"] for t in ts])
    cp = T._truth_full(T.params_vector(tp), std, "pre")["rho1_complete_projection"]
    assert cp == pytest.approx(pop, abs=1e-12)
    assert abs(cp - old) > 1e-3
    assert T.truth(tp, std, "pre")["rho1"] == pytest.approx(rho, abs=1e-12)


def test_R8a_rho1_truth_is_the_stationary_slope(guards):
    for per in PERIODS:
        tp = C.registered_true_params("ou", per)
        assert T.stationary_rho1(*T.process_spec(tp)) == pytest.approx(1 + tp["a1"],
                                                                      abs=1e-13)
    for proc in ("ou", "cubic", "cubic_mult", "cubic_bistable_1", "cubic_shift"):
        for per in PERIODS:
            tp = C.registered_true_params(proc, per)
            th, g = T.process_spec(tp)
            grid, p, m, sd = T.stationary_law(th, g)
            nxt = grid + T.drift(th, g, grid)
            direct = np.sum(p * (grid - m) * nxt) / np.sum(p * (grid - m) ** 2)
            r = T.stationary_rho1(th, g)
            assert r == pytest.approx(direct, abs=1e-12)
            vals = []
            for wH, seed in ((0.0, 1), (0.04, 2)):
                rng = np.random.default_rng(seed)
                st = pd.DataFrame({"year": list(range(2000, 2024)), "n": 1000,
                                   "mu_x": m + 0.2 * sd * rng.standard_normal(24),
                                   "sigma_x": sd * (1 + 0.1 * rng.standard_normal(24)),
                                   "x_units": X_UNITS, C.SHOCK_SHARE_COLUMN: wH})
                vals.append(T.truth(tp, st, per)["rho1"])
            assert vals[0] == vals[1] == r, (proc, per)
    tp = C.registered_true_params("cubic", "pre")
    th, g = T.process_spec(tp)
    grid, p, _, _ = T.stationary_law(th, g)
    rng = np.random.default_rng(31)
    slopes = []
    for _ in range(20):
        x = rng.choice(grid, size=5000, p=p)
        for _ in range(100):
            x = x + T.drift(th, g, x) + np.sqrt(np.exp(T.log_diffusion(th, g, x))) \
                * rng.standard_normal(5000)
        y = x + T.drift(th, g, x) + np.sqrt(np.exp(T.log_diffusion(th, g, x))) \
            * rng.standard_normal(5000)
        slopes.append(rho1_from_pairs(x, y - x))
    se = np.std(slopes, ddof=1) / np.sqrt(len(slopes))
    assert abs(np.mean(slopes) - T.stationary_rho1(th, g)) < 4 * se
    tp = C.registered_true_params("ou", "during")
    std = T.stationary_standardisation(tp)
    full = T._truth_full(T.params_vector(tp), std, "during")
    assert full["obs"]["rho1"] == pytest.approx(full["rho1_complete_projection"], abs=1e-10)


def test_S1_no_exit_prediction_equals_truth(guards):
    for proc in ("ou", "cubic"):
        tp = C.registered_true_params(proc, "pre")
        std = _s1_table(tp)
        s = std.set_index("year")
        ts = T.period_transitions("pre")
        rows = [(CELL_N, 0, 0, s.at[t, "mu_x"], s.at[t, "sigma_x"], s.at[t + 1, "mu_x"],
                 s.at[t + 1, "sigma_x"], 0.0, 0.0, 1.0) for t in ts]
        comp = _comp(rows, period="pre", re_placement="shock")
        out = T.predict(comp, tp, "pre", std)
        for q in ("xi_trap", "two_delta_V", "mean_diffusion", "rho1"):
            assert abs(out["bias"][q]) < 1e-9, (proc, q, out["bias"][q])
        if proc == "ou":
            cf = T.closed_form_observables(comp, tp, "pre", std)
            assert cf["rho1"] == pytest.approx(out["truth"]["rho1"], abs=1e-12)
            sh = _comp([r[:7] + (0.01, 0.0, 1.0) for r in rows], period="pre",
                       re_placement="shock")
            o2 = T.predict(sh, tp, "pre", std)
            assert o2["estimate"]["rho1"] == pytest.approx(out["estimate"]["rho1"], abs=1e-12)
            assert abs(o2["estimate"]["xi_trap"] - out["estimate"]["xi_trap"]) > 1e-3


def test_R8b_rho1_prediction_is_truth_plus_composition_effect(guards, monkeypatch):
    for proc in ("ou", "cubic"):
        tp, std, comp = _reg_window(proc)
        out = T.predict(comp, tp["during"], "during", std)
        d = out["diag"]
        assert out["estimate"]["rho1"] - out["truth"]["rho1"] == pytest.approx(
            d["rho1_open_projection"] - d["rho1_complete_projection"], abs=1e-14)
        assert abs(d["rho1_open_projection"] - d["rho1_complete_projection"]) > 1e-4
    real = T._truth_full

    def moved(*a, **k):
        out = real(*a, **k)
        out["obs"] = {**out["obs"], "rho1": out["obs"]["rho1"] + 0.0123}
        return out
    monkeypatch.setattr(T, "_truth_full", moved)
    for proc in ("ou", "cubic"):
        tp = C.registered_true_params(proc, "pre")
        std = _s1_table(tp)
        s = std.set_index("year")
        rows = [(CELL_N, 0, 0, s.at[t, "mu_x"], s.at[t, "sigma_x"], s.at[t + 1, "mu_x"],
                 s.at[t + 1, "sigma_x"], 0.0, 0.0, 1.0) for t in T.period_transitions("pre")]
        comp = _comp(rows, period="pre", re_placement="shock")
        for design in ("observed", "fallback"):
            out = T.predict(comp, tp, "pre", std, design=design)
            assert out["estimate"]["rho1"] == pytest.approx(out["truth"]["rho1"], abs=1e-12)
            assert out["truth"]["rho1"] == pytest.approx(
                T.stationary_rho1(*T.process_spec(tp)) + 0.0123, abs=1e-12)
        if proc == "ou":
            cf = T.closed_form_observables(comp, tp, "pre", std)
            assert cf["rho1"] == pytest.approx(out["truth"]["rho1"], abs=1e-12)


def _ai_table(m, sd, mu=None, sig=None, wH=0.0):
    years = range(2000, 2024)
    return pd.DataFrame({"year": list(years), "n": 1000,
                         "mu_x": m if mu is None else mu, "sigma_x": sd if sig is None else sig,
                         "x_units": X_UNITS, C.SHOCK_SHARE_COLUMN: wH})


def _registered_rows(s, period):
    return [{"n_stay": float(min(s.at[t, "n"], s.at[t + 1, "n"])),
             "mu_t": float(s.at[t, "mu_x"]), "sigma_t": float(s.at[t, "sigma_x"]),
             "mu_t1": float(s.at[t + 1, "mu_x"]), "sigma_t1": float(s.at[t + 1, "sigma_x"]),
             "mech": 0.0, "xbar": 0.0, "V": 1.0, "wH": T._wH_of(s, t + 1),
             "cp": T._complete_moments(s, t)} for t in T.period_transitions(period)]


def test_S2_affine_invariant_two_delta_V(guards):
    for proc in ("cubic_bistable_1", "cubic_bistable_2", "cubic_bistable_4"):
        for per in PERIODS:
            tp = C.registered_true_params(proc, per)
            th, g = T.process_spec(tp)
            assert T.is_bistable(th, g)
            _, _, m, sd = T.stationary_law(th, g)
            for wH in (0.0, 0.03):
                st = _ai_table(m, sd, wH=wH)
                reg = T._pooled(th, _registered_rows(st.set_index("year"), per), g)["obs"]
                assert T.truth(tp, st, per)["two_delta_V"] == pytest.approx(
                    reg["two_delta_V"], rel=1e-10)
            rng = np.random.default_rng(11)
            mu = m + 0.15 * sd * rng.standard_normal(24)
            sig = sd * (1 + 0.05 * rng.standard_normal(24))
            t1 = T.truth(tp, _ai_table(m, sd, mu, sig), per)
            ts = T.period_transitions(per)
            mu2 = mu.copy()
            mu2[[t - 2000 for t in ts]] = mu[[t - 2000 for t in ts]].mean()
            t2 = T.truth(tp, _ai_table(m, sd, mu2, sd * np.ones(24)), per)
            assert t2["two_delta_V"] == pytest.approx(t1["two_delta_V"], rel=1e-9)
            reg = T._pooled(th, _registered_rows(_ai_table(m, sd, mu, sig).set_index("year"),
                                                 per), g)["obs"]
            for q in ("xi_trap", "tau_mfpt", "mean_diffusion"):
                assert t1[q] == pytest.approx(reg[q], rel=1e-12, abs=1e-15)
            assert t1["rho1"] == T.stationary_rho1(th, g)
            lo = min(f for f in T.fixed_points(th, g)
                     if T.drift_slope(th, g, np.array([f]))[0] < 0)
            xs = np.linspace(lo, float(np.mean(mu[[t - 2000 for t in ts]])), 4001)
            r = T.drift(th, g, xs) / np.exp(T.log_diffusion(th, g, xs))
            V = -np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(xs))])
            assert t1["two_delta_V"] == pytest.approx(2 * V.max(), rel=2e-3)
    for proc, per in (("ou", "pre"), ("cubic", "during"), ("cubic_mult", "pre"),
                      ("cubic_bistable_0", "pre"), ("cubic_shift", "during")):
        tp = C.registered_true_params(proc, per)
        th, g = T.process_spec(tp)
        assert not T.is_bistable(th, g)
        _, _, m, sd = T.stationary_law(th, g)
        rng = np.random.default_rng(2)
        st = _ai_table(m, sd, m + 0.1 * sd * rng.standard_normal(24),
                       sd * (1 + 0.05 * rng.standard_normal(24)))
        reg = T._pooled(th, _registered_rows(st.set_index("year"), per), g)["obs"]
        assert T.truth(tp, st, per)["two_delta_V"] == reg["two_delta_V"]


def test_D3_reflecting_kernel_at_the_grid_edge():
    from scipy import stats as st
    grid = np.linspace(-5.0, 5.0, 201)
    mean_next = grid.copy()
    mean_next[:3] = 400.0
    mean_next[-3:] = -1e4
    sb = np.full_like(grid, 0.2)
    K = T._reflecting_kernel(grid, mean_next, sb)
    assert np.isfinite(K).all() and np.allclose(K.sum(1), 1.0)
    f = T._fold_into(np.array([400.0, -1e4, 5.5, -5.5, 0.3]), -5.0, 5.0)
    assert ((f >= -5.0) & (f <= 5.0)).all() and np.isclose(f[2], 4.5) and np.isclose(f[3], -4.5)
    old = st.norm.pdf(grid[None, :], mean_next[:, None], sb[:, None])
    i = 100
    assert np.allclose(K[i], old[i] / old[i].sum(), atol=1e-12)
    with pytest.raises(ValueError, match="non-finite"):
        T._reflecting_kernel(grid, np.where(grid > 4, np.nan, grid), sb)
    sim = pytest.importorskip("src.k1_sim")
    al0 = -sim.bistable_params("pre", 1.0)[3]
    th = sim.bistable_params("pre", 1.0, float(al0 * np.exp(0.7)))
    grid, p, m, s = T.stationary_law(th)
    assert np.isfinite(p).all() and np.isclose(p.sum(), 1.0) and s > 0


def test_D5_exact_linear_attractor_in_leaver_entrant_form():
    L, E = sp.symbols("Lbar Ebar", real=True)
    bn = sp.symbols("b", negative=True)
    xi_lin = (-L) - (L - E) / bn
    assert sp.simplify(xi_lin - ((1 + bn) * L - E) / sp.Abs(bn)) == 0
    assert sp.simplify(xi_lin.subs(E, 0) - (1 + bn) * (L - 0) / sp.Abs(bn)) == 0
    assert sp.simplify(xi_lin.subs(L, 0) - (L - E).subs(L, 0) / sp.Abs(bn)) == 0
    a1n = -0.3
    Lb = -0.182 * abs(a1n) / (1 + a1n)
    assert ((1 + a1n) * Lb - 0) / abs(a1n) == pytest.approx(-0.182)
    assert Lb / abs(a1n) == pytest.approx(-0.26, abs=0.005)
    _, _, m, s = T.stationary_law(T.params_vector(OU))
    for leave, enter, mech, xbar in ((40, 0, -0.06, 0.06), (0, 40, -0.05, 0.0)):
        comp = _comp([(300, leave, enter, m, s, m, s, mech)] * 6).assign(
            xi_bar_stay=xbar, var_stay=1.0)
        inp = T.zeroth_order_inputs(comp, OU, "during", _std(OU))
        z = T.zeroth_order_observables(comp, OU, "during", _std(OU))
        zm = T.zeroth_order_observables(comp, OU, "during", _std(OU), form="mbar")
        a = inp["a1"]
        assert z["xi_trap"] == pytest.approx(
            inp["xi_star_true"] + ((1 + a) * inp["L_bar"] - inp["E_bar"]) / abs(a), abs=1e-9)
        if enter == 0:
            assert (z["xi_trap"] - inp["xi_star_true"]) == pytest.approx(
                (1 + a) * (zm["xi_trap"] - inp["xi_star_true"]), abs=1e-9)
        else:
            assert z["xi_trap"] == pytest.approx(zm["xi_trap"], abs=1e-9)
    with pytest.raises(ValueError, match="form"):
        T.zeroth_order_observables(comp, OU, "during", _std(OU), form="x")


def test_D5_rows_and_prl_inputs():
    assert "zeroth_order_mbar" in REPORT_ROWS and C.ROW_DESIGN["zeroth_order_mbar"] == "observed"
    assert set(C.OWN_OFFSET_ROWS) == {"zeroth_order", "zeroth_order_mbar"}
    assert "PRL" not in C.REPORT_ROWS["zeroth_order_mbar"].replace("never PRL", "")
    assert "prl_condition_1_inputs" not in C.DIGEST_FIELDS


def test_s0_failed_replicate_counts_and_never_voids_the_attempt(monkeypatch, tmp_path):
    import sys
    S = _s0_script()
    repo = tmp_path / "repo"
    repo.mkdir()
    _s0_repo(monkeypatch, repo)
    monkeypatch.setattr(C, "R_PER_CELL", 4)
    name = _stub_backend("k1_s0_stub_flaky")
    m = sys.modules[name]
    est = m.estimate
    sim = m.simulate_no_exit

    def flaky_sim(cell, r, seed):
        out = sim(cell, r, seed)
        out["pairs"] = {p: {**v, "r": r} for p, v in out["pairs"].items()}
        return out

    def flaky(pairs):
        if pairs.get("r") == 1:
            raise FloatingPointError("fit diverged")
        return est(pairs)
    m.simulate_no_exit, m.estimate = flaky_sim, flaky
    S.start("cubic", "saai", backend_name=name)
    raw = tmp_path / "raw"
    S.run("cubic", "saai", range(0, 4), raw, backend_name=name)
    out = S.record("cubic", "saai", raw)
    res = out["result"]
    assert set(res["replicate_errors"]) == {"1"}
    assert "fit diverged" in res["replicate_errors"]["1"]["pre"]
    assert not res["pass"] and res["pre"]["xi_trap"]["n_nonfinite"] == 1
    assert out["path"].exists()

    class Refusal(RuntimeError):
        guard_refusal = True

    def refuse(pairs):
        raise Refusal("guard")
    m.estimate = refuse
    monkeypatch.setattr(C, "_require_frozen", lambda: {"frozen_commit": "F2"})
    S.start("cubic", "saai", backend_name=name)
    with pytest.raises(Refusal):
        S.run("cubic", "saai", range(0, 4), tmp_path / "raw2", backend_name=name)


def test_expi_bootstrap_degenerate_convention_default_is_unchanged():
    import inspect
    from src.km_mle import bootstrap_mle_quantities
    assert inspect.signature(bootstrap_mle_quantities).parameters["degenerate"].default == "nan"
    rng = np.random.default_rng(0)
    pairs = {}
    for p in ("pre", "during"):
        x = rng.standard_normal(600)
        pairs[p] = pd.DataFrame({"CountyCode": np.repeat(np.arange(100), 6), "xi_t": x,
                                 "delta_xi": -0.12 * x + 0.3 * rng.standard_normal(600)})
    d0, s0 = bootstrap_mle_quantities(None, n_boot=3, seed=1, pairs=pairs)
    d1, s1 = bootstrap_mle_quantities(None, n_boot=3, seed=1, pairs=pairs, degenerate="nan")
    pd.testing.assert_frame_equal(d0, d1)
    pd.testing.assert_frame_equal(s0, s1)
    d2, _ = bootstrap_mle_quantities(None, n_boot=3, seed=1, pairs=pairs, degenerate="limit")
    assert len(d2) == len(d0)
