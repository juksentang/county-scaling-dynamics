"""
Compare the outputs of `make numbers` and `make figures` with the sha256
recorded in expected_outputs.sha256 (written when this archive was built).

    python3 scripts/verify_outputs.py      (or: make verify)

Entries under build/tex/ refer to the directory named by PAPER_K1_TEX_DIR
(default build/tex). Exits non-zero, listing each file, if any output is
missing or differs. Standard library only.
"""
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = os.path.join(ROOT, "expected_outputs.sha256")


def main():
    tex = os.environ.get("PAPER_K1_TEX_DIR", "build/tex")
    tex = tex if os.path.isabs(tex) else os.path.join(ROOT, tex)
    bad, n = [], 0
    for line in open(RECORD, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        want, rel = line.rstrip("\n").split("  ", 1)
        p = os.path.join(tex, rel[len("build/tex/"):]) if rel.startswith("build/tex/") else os.path.join(ROOT, rel)
        n += 1
        if not os.path.isfile(p):
            bad.append((rel, "missing (run make numbers and make figures first)"))
            continue
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if got != want:
            bad.append((rel, f"sha256 {got} != {want}"))
    for rel, why in bad:
        print(f"DIFFERS  {rel}: {why}", file=sys.stderr)
    if bad:
        print(f"{len(bad)} of {n} outputs differ from expected_outputs.sha256", file=sys.stderr)
        return 1
    print(f"all {n} outputs equal expected_outputs.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
