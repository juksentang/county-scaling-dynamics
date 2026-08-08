"""
Guard the figure build: exactly one generator may write each figure, and
`make figures` must leave every figure newer than the build.

This exists because both conditions had failed silently: two generators wrote
the same figure, the later one used a superseded estimator, and every rebuild
quietly restored a figure carrying a claim that had been withdrawn.

Usage:
    python scripts/check_figure_owners.py            # ownership only
    python scripts/check_figure_owners.py --since T  # also require mtime > T
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIG_DIR = "results/figures"

# Every producer the figures target runs, and what it is allowed to write.
# Producers the `figures` target runs; their outputs must be rewritten by
# every build.
FIGURES_TARGET = (
    "scripts/generate_figures.py",
    "scripts/make_fig_kramers.py",
    "scripts/make_si_figures.py",
)
# Producers that live in `make numbers` but also emit a manuscript figure.
# These must have a single owner and be present, but a figures-only build
# does not rewrite them.
NUMBERS_TARGET = (
    "scripts/run_attribution_extras.py",
    "scripts/run_km_robustness.py",
)
GENERATORS = FIGURES_TARGET + NUMBERS_TARGET
WRITE = re.compile(r"[\"'/]((?:fig|attr)_[a-z0-9_]+\.(?:pdf|png))")


def written_by(path: str) -> set:
    """Figure filenames a generator writes."""
    if path.endswith(".ipynb"):
        nb = json.load(open(path))
        text = "\n".join("".join(c["source"]) for c in nb["cells"])
    else:
        text = open(path).read()
    return set(WRITE.findall(text))


def referenced() -> set:
    """Every figure a generator claims to write."""
    out = set()
    for gen in GENERATORS:
        out |= {f for f in written_by(gen) if f.endswith(".pdf")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, default=None,
                    help="epoch seconds; every referenced figure must be newer")
    args = ap.parse_args()

    owners = {}
    problems = []
    for gen in GENERATORS:
        for fig in written_by(gen):
            owners.setdefault(fig, []).append(gen)

    for fig, gens in sorted(owners.items()):
        if len(gens) > 1:
            problems.append(f"{fig}: written by {len(gens)} generators -- "
                            + ", ".join(gens) + " (last one wins silently)")

    refs = referenced()
    for fig in sorted(refs):
        path = os.path.join(FIG_DIR, fig)
        if not os.path.exists(path):
            problems.append(f"{fig}: referenced by the manuscript but missing")
        elif (args.since is not None
              and all(g in NUMBERS_TARGET for g in owners.get(fig, FIGURES_TARGET))
              is False and os.path.getmtime(path) < args.since):
            problems.append(f"{fig}: not rewritten by this build "
                            "(no generator owns it, or the build skipped it)")

    orphans = sorted(refs - set(owners))
    for fig in orphans:
        problems.append(f"{fig}: expected but no generator writes it")

    print(f"{len(owners)} figures with a generator, {len(refs)} PDF outputs")
    for p in problems:
        print(f"  FAIL  {p}")
    if not problems:
        print("  ok: single ownership, all expected figures present"
              + (" and freshly written" if args.since is not None else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
