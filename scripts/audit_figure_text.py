"""
Extract the numbers printed inside each figure, so a figure that still
carries a withdrawn or superseded value is caught by the build.

Figures produced by matplotlib embed subsetted fonts whose glyph codes are
shifted from ASCII, so `pdftotext` returns mojibake and a plain text search
finds nothing. The shift is constant within a file, so it is recovered by
trying each offset and keeping the one that yields the most word-like text.

Usage:
    python scripts/audit_figure_text.py            # report figure numbers
    python scripts/audit_figure_text.py --expect   # also check the manifest
"""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIG_DIR = "results/figures"

# Words that appear in these figures' titles, axis labels and legends; used to
# score a candidate glyph offset.
CRIB = ("Pre", "During", "Post", "Year", "SAAI", "Population", "GDP",
        "potential", "policy", "output", "exponent", "urban", "Grain",
        "Primary", "Secondary", "Tertiary", "Non", "yr", "ratio", "quantile")

# Values a figure must NOT contain: withdrawn claims, and values the audit
# replaced. Checked against every figure.
FORBIDDEN = {
    "0.112": "withdrawn post-policy barrier (Table 2 reads 'not identified')",
    "0.041": "pre-policy barrier before the escape-path correction (now 0.023)",
    "0.173": "during-policy barrier before the correction (now 0.154)",
    "16.2": "during-policy MFPT before the correction (now 11.7)",
    "6.65e5": "withdrawn astronomical escape time",
}


def decode(raw: str) -> str:
    """Undo the constant glyph-code shift, choosing the offset by crib words.

    The shift maps printable ASCII onto the subset's glyph indices, so most of
    the alphabet lands in the control range and the digits land lowest of all
    -- an offset search that only touches printable input decodes the labels
    and silently drops every number.
    """
    # A figure whose fonts carry a correct ToUnicode map needs no decoding,
    # and searching shifts on already-readable text finds spurious offsets:
    # characters that fall outside the printable window are left untouched,
    # so the original words survive and score as if the shift were real.
    if sum(raw.count(w) for w in CRIB) >= 2:
        return raw

    best, best_score = raw, -1e9
    for shift in range(-8, 97):
        cand = "".join(
            chr(ord(c) + shift) if 0x20 <= ord(c) + shift < 0x7F else c
            for c in raw)
        # A shift that leaves much of the document unprintable is not the
        # encoding: the untouched characters merely leak through and score
        # accidental crib hits.
        junk = sum(1 for c in cand if not c.isprintable() and not c.isspace())
        score = (sum(cand.count(w) * len(w) for w in CRIB)
                 - 50.0 * junk / max(1, len(cand)))
        if score > best_score:
            best, best_score = cand, score
    return best


def numbers(text: str) -> list:
    """Numeric tokens printed in a figure, normalised for comparison."""
    out = []
    for tok in re.findall(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", text):
        tok = tok.replace(",", "")
        if tok not in ("", "-"):
            out.append(tok)
    return out


def figure_text(path: str) -> str:
    raw = subprocess.run(["pdftotext", "-f", "1", "-l", "1", path, "-"],
                         capture_output=True, text=True).stdout
    if not raw.strip():
        return ""
    # Subsetted CID fonts interleave NULs; drop them before scoring offsets.
    return decode(raw.replace("\x00", ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", action="store_true",
                    help="fail on withdrawn or superseded values")
    args = ap.parse_args()

    problems, report = [], []
    for name in sorted(os.listdir(FIG_DIR)):
        if not name.endswith(".pdf"):
            continue
        path = os.path.join(FIG_DIR, name)
        text = figure_text(path)
        if not text:
            report.append(f"{name}: no extractable text (raster figure)")
            continue
        nums = numbers(text)
        report.append(f"{name}: {len(nums)} numeric tokens")
        for bad, why in FORBIDDEN.items():
            if bad in nums or bad in text:
                problems.append(f"{name}: contains {bad} -- {why}")

    for line in report:
        print(f"  {line}")
    for p in problems:
        print(f"  FAIL  {p}")
    if not problems:
        print("  ok: no figure carries a withdrawn or superseded value")
    return 1 if (problems and args.expect) else 0


if __name__ == "__main__":
    sys.exit(main())
