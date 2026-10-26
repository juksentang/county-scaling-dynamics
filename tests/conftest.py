import glob
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Tests that simulate on the county-by-year presence mask of the Chinese panel
# (data_clean/k1_mask.*, derived from the licensed CSMAR database, not shipped).
MASK = [p for p in glob.glob(str(ROOT / "data_clean" / "k1_mask.*"))]
MASK_REASON = "needs data_clean/k1_mask.* (presence mask derived from CSMAR; DATA_ACCESS.md)"
NEEDS_CSMAR = {"test_k1_sim.py": MASK_REASON}
NEEDS_MASK = {
    "test_k1_algebra.py::test_B25_s0_script_runs_end_to_end",
    "test_k1_algebra.py::test_B25b_s0_code_is_pinned_to_the_attempts_start_commit",
    "test_k1_algebra.py::test_B28_s0_pass_is_bound_to_the_validated_code",
    "test_k1_algebra.py::test_s0_failed_replicate_counts_and_never_voids_the_attempt",
}
collect_ignore = [] if MASK else sorted(NEEDS_CSMAR)

# Tests that run a script's refusal of an uncommitted working tree (git status):
# they need this archive as a git checkout (git clone), not an unpacked zip.
NEEDS_GIT = {
    "test_expi_paired.py::test_dev_main_end_to_end_on_synthetic_panel",
    "test_k1_exploratory.py::test_k1_gates_refuses_exploratory_tables_meta_and_rows",
}


def _is_git_checkout() -> bool:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True)
    except OSError:
        return False
    return r.returncode == 0 and Path(r.stdout.strip()).resolve() == ROOT


GIT = _is_git_checkout()


def pytest_collection_modifyitems(config, items):
    for item in items:
        key = f"{Path(str(item.fspath)).name}::{item.originalname or item.name}"
        if not MASK and key in NEEDS_MASK:
            item.add_marker(pytest.mark.skip(reason=MASK_REASON))
        if not GIT and key in NEEDS_GIT:
            item.add_marker(pytest.mark.skip(reason="needs the archive as a git checkout (git clone)"))


def pytest_report_header(config):
    out = [f"not collected: {f}: {NEEDS_CSMAR[f]}" for f in collect_ignore]
    if not MASK:
        out.append(f"skipped: {len(NEEDS_MASK)} further tests that {MASK_REASON}")
    if not GIT:
        out.append(f"skipped: {len(NEEDS_GIT)} tests that need a git checkout of the archive")
    return out
