import json
import os
import sys
import types

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src import k1_contract as C
from src import k1_sim as S
import k1_exploratory as X
import k1_gates as G
import k1_run_cells as RC

APPROVED = ("## x\n\n### amendment 10 (…)\n\n**Approved by the author:** approved (2026-10-23, author)\n\n"
            "### amendment 11 (2026-10-24; approved by the author)\n\n**Approved by the author:** approved (2026-10-24, author)\n\n- (b) …\n")


def test_cells_are_the_registered_k1a_k1b_exit_cells_and_matched_nulls():
    cells = X.exploratory_cells()
    idx = [c["cell_index"] for c in cells]
    assert idx == [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 34, 35, 36, 37, 46, 47, 48, 49]
    allc = S.registered_cells()
    for c in cells:
        assert c == allc[c["cell_index"]]
    exits = [c for c in cells if c["mechanism"] != "a"]
    assert {(c["process"], c["mechanism"]) for c in exits} == {
        (p, m) for p in ("i_ou", "ii_cubic", "iii_mult") for m in "bcde"}
    assert all(c["gate_use"] == "K1a/K1b" and c["B"] == S.B_REGISTERED for c in exits)
    nulls = [c for c in cells if c["mechanism"] == "a"]
    assert {(c["process"], c["structure"], bool(c.get("dated_shocks"))) for c in nulls} == {
        (p, m, d) for p, m in S.K1A_CELLS for d in (False, True)}
    for ci in (0, 5, 10, 15, 22, 23, 24, 25, 26, 38, 42, 50, 51):
        assert not X.is_exploratory_cell(allc[ci])
    for c in cells:
        assert S.replicate_seed(c, 7) == S.SEED_BASE + c["cell_index"] * 10_000 + 7


def test_modified_cell_is_not_exploratory():
    c = dict(S.registered_cells()[3], B=2)
    assert not X.is_exploratory_cell(c)


def test_output_paths():
    j = lambda *p: os.path.join(ROOT, *p)
    for ok in (j("results", "tables", "k1_exploratory"),
               j("results", "tables", "k1_exploratory", "k1_exploratory_k1a.csv"),
               j('results', 'records', "k1_exploratory_chunks", "k1exp_c03.meta.json"),
               j('results', 'records', "k1_exploratory_partial")):
        assert X.allowed_output(ok), ok
    for bad in (j("results", "tables"), j("results", "tables", "k1_sim_bias.csv"),
                j("results", "tables", "k1_exploratory_x", "a.csv"),
                j("results", "tables", "k1_s0_scope_f2", "k1_exploratory.csv"),
                j('results', 'records'), j('results', 'records', "k1_precheck_a_registered_f2"),
                j('results', 'plan.md'), "/tmp/k1_exploratory"):
        assert not X.allowed_output(bad), bad
        with pytest.raises(SystemExit, match="writes only"):
            X.check_output(bad)


def test_run_refuses_non_exploratory_cells_and_paths(monkeypatch):
    called = []
    monkeypatch.setattr(X, "require_preconditions", lambda: called.append(1) or X.FREEZE_COMMIT)
    allc = S.registered_cells()
    with pytest.raises(SystemExit, match="not an exploratory cell"):
        X.run_task(allc[24], 0, 1)
    with pytest.raises(SystemExit, match="writes only"):
        X.run_task(allc[3], 0, 1, out=os.path.join(ROOT, "results", "tables"))
    a = types.SimpleNamespace(out=X.CHUNK_DIR, task=None, cells="3,24", reps="0:1",
                              reps_per_chunk=5)
    with pytest.raises(SystemExit, match="not exploratory cells"):
        X.cmd_run(a)
    assert not os.path.exists(os.path.join(X.CHUNK_DIR, "k1exp_c24_r0000-0001.meta.json"))


def test_task_plan_covers_every_cell_and_replicate_once():
    plan = X.task_plan()
    seen = {(c["cell_index"], r) for c, a, b in plan for r in range(a, b)}
    assert len(seen) == sum(b - a for _, a, b in plan)
    assert seen == {(ci, r) for ci in X.EXPLORATORY_CELLS for r in range(S.R_REGISTERED)}


def test_every_output_frame_is_labelled():
    df = X.label_frame(pd.DataFrame({"cell_index": [3], "x": [1.0]}), "CAV")
    assert list(df.columns[:3]) == ["run_label", "exploratory", "caveat"]
    assert (df["exploratory"] == "exploratory = true, not a verdict (amendment 11)").all()
    assert (df["caveat"] == "CAV").all()


def test_gate_rows_are_relabelled_never_gating():
    rows = [{"gate": "K1a", "status": "pass", "gating": True, "law_status": "pass", "value": 0.1},
            {"gate": "K1b", "status": "fail", "gating": False, "law_status": None, "value": 0.8}]
    df = X.label_gate_rows(rows, "CAV")
    assert "gate" not in df.columns and "status" not in df.columns
    assert list(df["exploratory_outcome"]) == ["exploratory:pass", "exploratory:fail"]
    assert not df["gating"].any()
    assert list(df["would_gate_if_registered"]) == [True, False]
    assert df["law_status"].iloc[0] == "exploratory:pass"
    assert not (df.astype(str) == "pass").any().any()


def test_exploratory_guard_scope_and_restore(monkeypatch):
    calls, checks = [], []

    def fake(cell, estimate, pipelines=S.PIPELINES, neural=None):
        calls.append((cell["cell_index"], estimate))
        return "composition only"
    monkeypatch.setattr(S, "exit_guards", fake)
    monkeypatch.setattr(X, "exploratory_estimation_checks",
                        lambda cell, pl, require=None: checks.append(cell["cell_index"])
                        or "S0 cubic/saai: registered refusal not enforced")
    rec = {}
    allc = S.registered_cells()
    with X.relaxed_estimation_guards(rec):
        g = S.exit_guards
        assert g is not fake
        out = g(allc[8], True)
        assert "exploratory" in out and "not enforced" in out
        g(allc[8], True)
        assert calls == [(8, False), (8, False)] and checks == [8]
        for ci in (0, 24, 15, 26, 38):
            with pytest.raises(S.GuardRefused, match="not an exploratory cell"):
                g(allc[ci], True)
        with pytest.raises(S.GuardRefused, match="MLE only"):
            g(allc[3], True, neural=object())
    assert S.exit_guards is fake
    assert "not enforced" in rec[8]


def test_plain_a_null_cells_keep_the_registered_estimation_guard(monkeypatch):
    def fake(cell, estimate, pipelines=S.PIPELINES, neural=None):
        if estimate:
            raise S.GuardRefused("neural preflight refused")
        return "no exit design"
    monkeypatch.setattr(S, "exit_guards", fake)
    with X.relaxed_estimation_guards({}):
        with pytest.raises(S.GuardRefused, match="neural preflight refused"):
            S.exit_guards(S.registered_cells()[34], True)


def test_unexpected_check_error_is_a_refusal(monkeypatch):
    monkeypatch.setattr(S, "exit_guards", lambda c, e, p=S.PIPELINES, n=None: "composition only")

    def boom(cell, pl, require=None):
        raise KeyError("x")
    monkeypatch.setattr(X, "exploratory_estimation_checks", boom)
    with X.relaxed_estimation_guards({}):
        with pytest.raises(X.ExploratoryRefused, match="KeyError"):
            S.exit_guards(S.registered_cells()[3], True)


def test_exploratory_guard_still_enforces_freeze_and_simulation(monkeypatch):
    def fake(cell, estimate, pipelines=S.PIPELINES, neural=None):
        raise S.GuardRefused("theory not frozen")
    monkeypatch.setattr(S, "exit_guards", fake)
    with X.relaxed_estimation_guards({}):
        with pytest.raises(S.GuardRefused, match="theory not frozen"):
            S.exit_guards(S.registered_cells()[3], True)
    assert S.exit_guards is fake


def test_guard_restored_after_an_exception(monkeypatch):
    orig = S.exit_guards
    with pytest.raises(ZeroDivisionError):
        with X.relaxed_estimation_guards({}):
            1 / 0
    assert S.exit_guards is orig


def test_s0_relaxation_tolerates_only_the_cubic_saai_pass_failure(monkeypatch):
    msg = {"m": "S0 did not pass for cubic/saai: fix the truth first"}

    def refuse(process, pipeline):
        raise RuntimeError(msg["m"].replace("cubic/saai", f"{process}/{pipeline}"))
    after = []
    monkeypatch.setattr(C, "_require_s0_pass", refuse)
    monkeypatch.setattr(C, "_require_frozen", lambda: {"ok": 1})
    monkeypatch.setattr(X, "s0_checks_except_pass", lambda p, q: after.append((p, q)) or {})
    rec = {}
    with X.s0_not_enforced(rec):
        assert C._require_s0_pass("cubic", "saai") is None
        assert after == [("cubic", "saai")]
        for p, q in (("ou", "saai"), ("cubic", "direct"), ("cubic_mult", "saai")):
            with pytest.raises(X.ExploratoryRefused, match="refused"):
                C._require_s0_pass(p, q)
        for other in ("S0 record was made under another CONTRACT_VERSION",
                      "no S0 record at results/records/k1_s0_f2_cubic_saai.json",
                      "the S0-validated code changed since the S0 record in ['src/k1_sim.py']",
                      "S0 record seeds are not the registered s0_seeds"):
            msg["m"] = other
            with pytest.raises(X.ExploratoryRefused, match="refused"):
                C._require_s0_pass("cubic", "saai")
    assert "not enforced" in rec["cubic/saai"]
    assert C._require_s0_pass is refuse

    def nofreeze():
        raise RuntimeError("not frozen")
    monkeypatch.setattr(C, "_require_frozen", nofreeze)
    with X.s0_not_enforced({}):
        with pytest.raises(RuntimeError, match="not frozen"):
            C._require_s0_pass("cubic", "saai")


def test_s0_relaxation_returns_a_passing_record(monkeypatch):
    monkeypatch.setattr(C, "_require_s0_pass", lambda p, q: {"pass": True, "p": p})
    monkeypatch.setattr(C, "_require_frozen", lambda: {"ok": 1})
    rec = {}
    with X.s0_not_enforced(rec):
        assert C._require_s0_pass("ou", "saai") == {"pass": True, "p": "ou"}
    assert rec == {}


@pytest.fixture
def frozen_env(monkeypatch):
    monkeypatch.setenv("K1_THEORY_FROZEN", X.FREEZE_COMMIT)
    monkeypatch.setattr(S, "exit_guards", S.exit_guards)


def test_loader_refuses_unlabelled_chunks(tmp_path):
    stem = tmp_path / "k1exp_c03_r0000-0001"
    for k in RC.TABLES:
        pd.DataFrame({"cell_index": [3]}).to_csv(f"{stem}.{k}.csv.gz", index=False)
    meta = {"cell": S.registered_cells()[3], "files": {}}
    (tmp_path / "k1exp_c03_r0000-0001.meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="exploratory label"):
        X.load_chunks([str(tmp_path)])
    meta[X.MARKER] = X.EXPLORATORY_LABEL
    (tmp_path / "k1exp_c03_r0000-0001.meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="table without the exploratory label"):
        X.load_chunks([str(tmp_path)])
    for k in RC.TABLES:
        X.label_frame(pd.DataFrame({"cell_index": [3]}), "CAV").to_csv(
            f"{stem}.{k}.csv.gz", index=False)
    tabs, metas = X.load_chunks([str(tmp_path)])
    assert list(tabs["estimates"].columns) == ["cell_index"]


def test_registered_aggregator_refuses_exploratory_chunks_and_paths():
    m = {"code_commit": "a" * 40, "dev": False, X.MARKER: X.EXPLORATORY_LABEL}
    with pytest.raises(SystemExit, match="exploratory"):
        RC.check_chunks([m])
    with pytest.raises(SystemExit, match="without the exploratory marker"):
        RC.check_chunks([{"code_commit": "a" * 40, "dev": False}], exploratory=True)
    a = types.SimpleNamespace(dev=False, out=os.path.join(ROOT, "results", "tables"),
                              non_registered=False, inputs=[X.CHUNK_DIR])
    RC.tree_dirty_orig = RC.tree_dirty
    try:
        RC.tree_dirty = lambda: False
        with pytest.raises(SystemExit, match="exploratory"):
            RC.cmd_aggregate(a)
        a.inputs, a.out = ["/tmp/x"], X.TABLES_DIR
        with pytest.raises(SystemExit, match="exploratory"):
            RC.cmd_aggregate(a)
    finally:
        RC.tree_dirty = RC.tree_dirty_orig
        del RC.tree_dirty_orig
    r = types.SimpleNamespace(out=X.CHUNK_DIR)
    with pytest.raises(SystemExit, match="exploratory"):
        RC.cmd_run(r)


def test_k1_gates_refuses_exploratory_paths():
    for p in (X.TABLES_DIR, X.CHUNK_DIR, X.PARTIAL_DIR):
        with pytest.raises(SystemExit, match="exploratory"):
            G.main(["--tables", p, "--dev"])


def test_k1_gates_refuses_exploratory_tables_meta_and_rows(tmp_path):
    lab = X.label_frame(pd.DataFrame({"cell_index": [3], "process": ["i_ou"]}), "CAV")
    with pytest.raises(SystemExit, match="exploratory"):
        G._std_cells(lab)
    with pytest.raises(SystemExit, match="exploratory"):
        G.gate_k1b(lab)
    with pytest.raises(SystemExit, match="exploratory"):
        G.evaluate(pd.DataFrame(), lab, pd.DataFrame(), None)
    with pytest.raises(SystemExit, match="exploratory"):
        G.refuse_exploratory({X.MARKER: X.EXPLORATORY_LABEL, "rows": []})
    sub = tmp_path / "tables"
    sub.mkdir()
    (sub / "k1_sim_meta.json").write_text(json.dumps({X.MARKER: X.EXPLORATORY_LABEL,
                                                      "registered": True}))
    with pytest.raises(SystemExit, match="exploratory"):
        G.main(["--tables", str(sub), "--dev"])
    (sub / "k1_sim_meta.json").write_text(json.dumps({"registered": False}))
    lab.to_csv(sub / "k1_sim_bias.csv", index=False)
    with pytest.raises(SystemExit, match="exploratory"):
        G.main(["--tables", str(sub), "--dev"])


def test_driver_is_outside_the_s0_closure_and_the_frozen_files():
    from src import k1_theory as T
    closure = C.s0_code_digest()
    for f in ("scripts/k1_exploratory.py", "scripts/k1_gates.py", "scripts/k1_run_cells.py",
              "tests/test_k1_exploratory.py"):
        assert f not in closure and f not in T.FROZEN_FILES


def test_control_set_cells_are_the_registered_report_only_variants():
    cells = X.control_cells()
    idx = [c["cell_index"] for c in cells]
    assert idx == [35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 50, 51]
    allc = S.registered_cells()
    for c in cells:
        assert c == allc[c["cell_index"]] and c["B"] == 0
        assert S.replicate_seed(c, 11) == S.SEED_BASE + c["cell_index"] * 10_000 + 11
    assert {allc[i]["gate_use"] for i in X.CONTROL_CELLS} == {"control"}
    assert {allc[i]["gate_use"] for i in X.CONTROL_NULL_CELLS} == {"control_null"}
    assert {allc[i]["gate_use"] for i in X.SENS_CELLS} == {S.SENS_NO2006_GATE_USE}
    for ci in X.CONTROL_CELLS:
        assert S.null_cell_for(allc[ci])["cell_index"] in X.CONTROL_NULL_CELLS
    assert set(X.EXPLORATORY_CELLS) & set(X.CONTROLS_SET_CELLS) == set(X.SENS_NULL_CELLS)
    assert [c["cell_index"] for c in X.set_cells("main")] == list(X.EXPLORATORY_CELLS)
    for ci in (3, 8, 23, 24, 25, 46, 0):
        assert not X.is_exploratory_cell(allc[ci], "controls")
    for ci in (38, 42, 50, 51):
        assert X.is_exploratory_cell(allc[ci], "controls")
        assert not X.is_exploratory_cell(allc[ci])
    with pytest.raises(SystemExit, match="unknown cell set"):
        X.cell_set("all")


def test_control_task_plan_covers_every_cell_and_replicate_once():
    plan = X.task_plan(set_name="controls")
    seen = {(c["cell_index"], r) for c, a, b in plan for r in range(a, b)}
    assert len(seen) == sum(b - a for _, a, b in plan) == 12 * S.R_REGISTERED
    assert seen == {(ci, r) for ci in X.CONTROLS_SET_CELLS for r in range(S.R_REGISTERED)}
    assert X.task_plan() == X.task_plan(set_name="main")


def test_control_label_and_paths():
    df = X.label_frame(pd.DataFrame({"cell_index": [38]}), "CAV", "controls")
    assert (df["exploratory"] == "report-only control, exploratory, not a verdict (amendment 11)").all()
    assert "REPORT-ONLY CONTROL" in df["run_label"].iloc[0]
    rows = X.label_gate_rows([{"gate": "K1a", "status": "pass", "gating": False}], "CAV",
                             "controls")
    assert rows["exploratory"].iloc[0] == X.CONTROL_LABEL and not rows["gating"].any()
    cs = X.cell_set("controls")
    for p in (cs["chunk_dir"], cs["partial_dir"],
              os.path.join(X.TABLES_DIR, "k1_exploratory_controls_k1a.csv")):
        assert X.allowed_output(p), p
    assert cs["chunk_dir"] != X.CHUNK_DIR and cs["chunk_prefix"] != X.CHUNK_PREFIX


def test_control_run_refuses_cells_outside_the_set(monkeypatch):
    monkeypatch.setattr(X, "require_preconditions", lambda: X.FREEZE_COMMIT)
    allc = S.registered_cells()
    with pytest.raises(SystemExit, match="not an exploratory cell"):
        X.run_task(allc[3], 0, 1, set_name="controls")
    with pytest.raises(SystemExit, match="not an exploratory cell"):
        X.run_task(allc[38], 0, 1)
    a = types.SimpleNamespace(out=None, task=None, cells="38,3", reps="0:1",
                              reps_per_chunk=5, set="controls")
    with pytest.raises(SystemExit, match="not exploratory cells"):
        X.cmd_run(a)


def test_control_guard_scope(monkeypatch):
    monkeypatch.setattr(S, "exit_guards", lambda c, e, p=S.PIPELINES, n=None: "composition only")
    monkeypatch.setattr(X, "exploratory_estimation_checks",
                        lambda cell, pl, require=None: "tolerated")
    allc = S.registered_cells()
    with X.relaxed_estimation_guards({}, "controls"):
        assert "tolerated" in S.exit_guards(allc[40], True)
        for ci in (3, 8, 24, 46):
            with pytest.raises(S.GuardRefused, match="set controls"):
                S.exit_guards(allc[ci], True)


def test_control_loader_refuses_main_labelled_chunks(tmp_path):
    stem = tmp_path / "k1expc_c38_r0000-0001"
    for k in RC.TABLES:
        X.label_frame(pd.DataFrame({"cell_index": [38]}), "CAV").to_csv(
            f"{stem}.{k}.csv.gz", index=False)
    meta = {"cell": S.registered_cells()[38], "files": {}, X.MARKER: X.EXPLORATORY_LABEL}
    (tmp_path / "k1expc_c38_r0000-0001.meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="exploratory label"):
        X.load_chunks([str(tmp_path)], "controls")
    meta[X.MARKER] = X.CONTROL_LABEL
    (tmp_path / "k1expc_c38_r0000-0001.meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="table without the exploratory label"):
        X.load_chunks([str(tmp_path)], "controls")
    for k in RC.TABLES:
        X.label_frame(pd.DataFrame({"cell_index": [38]}), "CAV", "controls").to_csv(
            f"{stem}.{k}.csv.gz", index=False)
    tabs, metas = X.load_chunks([str(tmp_path)], "controls")
    assert list(tabs["estimates"].columns) == ["cell_index"]
    with pytest.raises(SystemExit, match="no exploratory chunks"):
        X.load_chunks([str(tmp_path)])


def test_control_rows_keep_only_the_control_and_no2006_variants():
    rows = [
        {"cell_index": 38, "status": "pass", "gating": False, "statistic": "projection:x"},
        {"cell_index": 51, "status": "fail", "gating": False, "statistic": "projection:x"},
        {"cell_index": None, "status": "missing", "gating": True,
         "statistic": "K1a/K1b cell or its K1a_null null not in the aggregate"},
        {"cell_index": None, "status": "missing", "gating": False,
         "statistic": "K1a/K1b cell or its K1a_shocked_null null not in the aggregate"},
        {"cell_index": None, "status": "missing", "gating": False,
         "statistic": "control cell or its control_null null not in the aggregate"},
        {"cell_index": None, "status": "missing", "gating": False,
         "statistic": "K1a_sens_no2006 cell or its K1a_null null not in the aggregate"},
    ]
    keep, n = X.control_rows(rows)
    assert n == 2 and [r["statistic"][:7] for r in keep] == [
        "project", "project", "control", "K1a_sen"]
    with pytest.raises(SystemExit, match="gating"):
        X.control_rows([{"cell_index": 38, "status": "pass", "gating": True}])


def test_k1_gates_refuses_control_labelled_tables():
    lab = X.label_frame(pd.DataFrame({"cell_index": [38], "process": ["i_ou"]}), "CAV",
                        "controls")
    with pytest.raises(SystemExit, match="exploratory"):
        G._std_cells(lab)
    with pytest.raises(SystemExit, match="exploratory"):
        G.refuse_exploratory({X.MARKER: X.CONTROL_LABEL})
    m = {"code_commit": "a" * 40, "dev": False, X.MARKER: X.CONTROL_LABEL}
    with pytest.raises(SystemExit, match="exploratory"):
        RC.check_chunks([m])


def test_no2006_cell_survives_the_meta_json_round_trip():
    c50 = S.registered_cells()[50]
    rt = json.loads(json.dumps(c50))
    assert rt != c50
    got = X._registered_cell_from_json(rt)
    assert got == c50 and got["drop_breaks"] == (2006,)
    assert X.is_exploratory_cell(X._registered_cell_from_json(rt), "controls")
    bad = dict(rt, drop_breaks=[2007])
    assert X._registered_cell_from_json(bad) is bad
    assert not X.is_exploratory_cell(bad, "controls")
