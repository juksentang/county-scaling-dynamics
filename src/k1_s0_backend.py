from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from src import k1_contract as C
from src import k1_sim as S

SHOCK_SHARE_FLAG = "TRUTH_ABSORBS_PROVINCE_SHOCK_SHARE"


class S0Blocked(RuntimeError):
    pass


def theory_has_shock_share_term() -> bool:
    try:
        from src import k1_theory as T
    except Exception:
        return False
    return bool(getattr(T, SHOCK_SHARE_FLAG, False) or getattr(C, SHOCK_SHARE_FLAG, False))


@functools.lru_cache(maxsize=1)
def _mask() -> pd.DataFrame:
    return S.load_mask()


def _cells(process: str, gate_use: str, mech: str) -> list[dict]:
    return [c for c in S.registered_cells()
            if S.PROCESS_KEY[c["process"]] == process and c["gate_use"] == gate_use
            and c["mechanism"] == mech]


def s0_precheck(process: str, pipeline: str, w: float | None = None,
                compat=None) -> dict:
    if pipeline not in S.PIPELINES:
        raise S0Blocked(f"unknown pipeline {pipeline!r}")
    try:
        C._check_member(process)
    except ValueError as exc:
        raise S0Blocked(str(exc)) from exc
    cells = _cells(process, "S0", "a")
    if len(cells) != 1:
        raise S0Blocked(f"process {process!r} has {len(cells)} S0 cells in k1_sim; the "
                        "contract keeps one S0 record per (member) process")
    for per in S.PERIODS:
        try:
            C.registered_true_params(process, per)
        except ValueError as exc:
            raise S0Blocked(str(exc)) from exc
    if process in C.DESIGN_MEMBERS and not design_rule_signed():
        raise S0Blocked(f"{process}: its registered constants were calibrated under the "
                        "draft (II) calibration rule, which the authors have not signed "
                        f"({C.DESIGN_PROCESS_FILE} design_rule.signed is not true): S0 of a "
                        "design member is not simulated until it is (fail closed)")
    cell = cells[0]
    compat = compat or S.theory_compat
    for per in S.PERIODS:
        tc = compat(cell["process"], per)
        if not tc.get("compatible"):
            raise S0Blocked(f"{cell['process']} {per}: the simulated chain is not the one the "
                            f"theory computes (k1_sim.theory_compat: {tc.get('reason') or tc.get('checks')})")
    w = S._w_shock() if w is None else float(w)
    if w > 0 and cell["re_placement"] == "shock" and not theory_has_shock_share_term():
        raise S0Blocked(f"the truth lacks the province shock-share term at w = {w:.4f} "
                        f"(mean_diffusion biased by ≈ −w·H; k1_theory.{SHOCK_SHARE_FLAG} not "
                        "set): S0 is not simulated until the author / theory line decides")
    return cell


def design_rule_signed() -> bool:
    """design_rule.signed of the committed design registry (src/k1_design_processes.json)."""
    import json
    try:
        d = json.loads((C.REPO / C.DESIGN_PROCESS_FILE).read_text())
    except (OSError, ValueError):
        return False
    return (d.get("design_rule") or {}).get("signed") is True


def cell_meta(process: str, exit_mechanism: str, pipeline: str, s0: bool) -> dict:
    if s0:
        if exit_mechanism != "a":
            raise ValueError("the S0 cell has exit mechanism 'a'")
        cell = s0_precheck(process, pipeline)
    else:
        cells = _cells(process, "K1a/K1b", exit_mechanism)
        if len(cells) != 1 or (cells[0]["process"], exit_mechanism) not in S.K1A_CELLS:
            raise ValueError(f"no single K1a cell {process} × ({exit_mechanism})")
        cell = cells[0]
    return S.contract_cell_meta(cell, pipeline)


def _sim_cell(meta: dict) -> tuple[dict, str]:
    """The registered k1_sim cell of a contract meta (by cell_index) and its
    pipeline; refuses a meta that is not exactly that cell's."""
    ci, pl = meta.get("cell_index"), meta.get("pipeline")
    cells = S.registered_cells()
    if not isinstance(ci, (int, np.integer)) or not 0 <= int(ci) < len(cells):
        raise ValueError(f"cell meta without a registered cell_index: {ci!r}")
    cell = cells[int(ci)]
    want = S.contract_cell_meta(cell, pl)
    got = {k: v for k, v in meta.items() if k != "valid_for_law"}
    if got != want:
        bad = {k: (got.get(k), want.get(k)) for k in set(got) | set(want)
               if got.get(k) != want.get(k)}
        raise ValueError(f"cell meta is not registered cell {ci}'s: {bad}")
    return cell, pl


def _std_both(std: dict) -> pd.DataFrame:
    return pd.concat([std[p] for p in S.PERIODS], ignore_index=True)


def simulate_no_exit(cell: dict, replicate: int, seed: int) -> dict:
    sc, pl = _sim_cell(cell)
    if sc["gate_use"] != "S0":
        raise ValueError(f"cell {sc['cell_index']} is not an S0 cell")
    s0_precheck(S.PROCESS_KEY[sc["process"]], pl)
    if int(seed) != S.replicate_seed(sc, replicate, pl):
        raise ValueError(f"seed {seed} is not the registered S0 seed of replicate {replicate}")
    S.check_cell(sc)
    S.exit_guards(sc, estimate=True, pipelines=(pl,))
    pan = S.simulate_panels(sc, int(seed), _mask())
    pairs = {p: S.pairs_from_matrix(S.xi_matrix(pl, pan["sim"][p], pan["wins"][p]),
                                    pan["wins"][p]) for p in S.PERIODS}
    return {"pairs": pairs, "standardisation": _std_both(pan["std"]),
            "true_params": {p: pan["procs"][p].as_dict() for p in S.PERIODS}}


def simulate_composition(cell: dict, replicate: int) -> dict:
    """One replicate of a K1a exit cell, compositions only (no estimator);
    the seed is the Step 2 grid seed of that cell and replicate."""
    sc, pl = _sim_cell(cell)
    if sc["mechanism"] == "a":
        raise ValueError("simulate_composition is for the K1a exit cells")
    res = S.run_replicate(sc, int(replicate), _mask(), B=0, pipelines=(pl,), estimate=False)
    if res.get("mode") != "composition_only" or len(res["estimates"]):
        raise RuntimeError("composition-only replicate ran an estimator")
    comp = res["composition"]
    return {"composition": comp[comp["pipeline"] == pl].reset_index(drop=True),
            "standardisation": res["standardisation"].drop(
                columns=["cell_index", "replicate"]).reset_index(drop=True),
            "true_params": res["true_params"], "seed": int(res["seed"])}


def rho1_reference(process: str) -> dict:
    cells = _cells(process, "S0", "a")
    if len(cells) != 1:
        raise ValueError(f"process {process!r} has {len(cells)} S0 cells")
    wins = S.cell_windows(cells[0], _mask())
    shares, N = {}, {}
    for p in S.PERIODS:
        cnt = np.bincount(wins[p].prov_idx).astype(float)
        shares[p] = [float(v) for v in cnt[cnt > 0] / cnt.sum()]
        N[p] = int(wins[p].n)
    return {"w": float(S._w_shock()), "shares": shares, "N": N,
            "source": "k1_s0_backend.rho1_reference: S0 cell windows, k1_sim._w_shock"}


def estimate(pairs: pd.DataFrame) -> dict:
    return S.estimate(pairs)
