"""Publication lint for the K1 paper figures (scripts/paper_k1_figures.py).

* every committed figure PDF passes the file-level lint (width 89/183 mm,
  height <= 170 mm, only Arial embedded, no Type 3 fonts);
* the generator, run into a temporary directory, passes the full in-figure
  lint (font >= 5 pt at final size, Arial, no clipped or overlapping text,
  greyscale-separable series, contrast) and reproduces the committed PDFs
  byte for byte (a committed figure must be regenerable);
* the lint itself rejects a figure that breaks the rules.

PAPER_K1_FIG_DIR (default results/figures) names the directory of the
committed PDFs. Where an expected_outputs.sha256 file sits at the root (the
public archive), the regenerated PDFs are compared with the sha256 recorded
there instead of with the working copy, so overwriting the shipped PDFs first
cannot hide a difference. The caption check runs only where the paper sources are present.
"""
import glob
import hashlib
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIGDIR = os.path.join(ROOT, os.environ.get("PAPER_K1_FIG_DIR", os.path.join('results', 'figures')))
EXPECTED = ["fig1_mechanism", "fig2_simulation", "fig3_dose_ladder", "fig4_china_coverage",
            "figS1_k1a_exploratory", "figS2_k1b_controls", "figS3_china_ci", "figS4_us_detail",
            "figS5_guarded", "figS6_brazil_detail", "fig5_masking", "figS7_mask_cells", "figS8_mask_guards"]
TEXDIR = os.path.join(ROOT, "paper_k1", "latex")

from src import paper_k1_style as S


EXPECTED_SHA = os.path.join(ROOT, "expected_outputs.sha256")


def _md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def _recorded_sha256():
    """{relative path: sha256} from expected_outputs.sha256, or None where the file is absent."""
    if not os.path.isfile(EXPECTED_SHA):
        return None
    out = {}
    for line in open(EXPECTED_SHA, encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            h, rel = line.rstrip("\n").split("  ", 1)
            out[rel] = h
    return out


def test_palette_is_greyscale_separable():
    assert S.palette_findings() == []


@pytest.mark.parametrize("stem", EXPECTED)
def test_committed_pdf_passes_file_lint(stem):
    p = os.path.join(FIGDIR, stem + ".pdf")
    assert os.path.isfile(p), f"{stem}.pdf missing; run make figures"
    assert S.lint_pdf(p) == []


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    out = tmp_path_factory.mktemp("figs")
    env = dict(os.environ, PAPER_K1_FIG_OUTDIR=str(out),
               PAPER_K1_FIG_PANELS=str(out / "panels.jsonl"))
    r = subprocess.run([sys.executable, "scripts/paper_k1_figures.py"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    return r, out


def test_generator_passes_lint(regenerated):
    r, _ = regenerated
    assert r.returncode == 0, r.stderr[-3000:]


@pytest.mark.parametrize("stem", EXPECTED)
def test_committed_pdf_is_regenerable(regenerated, stem):
    _, out = regenerated
    new = os.path.join(out, stem + ".pdf")
    assert os.path.isfile(new)
    rec = _recorded_sha256()
    if rec is not None:
        got = hashlib.sha256(open(new, "rb").read()).hexdigest()
        assert got == rec[f"results/figures/{stem}.pdf"], \
            f"{stem}.pdf as generated here differs from the recorded sha256 (expected_outputs.sha256)"
        return
    assert _md5(new) == _md5(os.path.join(FIGDIR, stem + ".pdf")), \
        f"{stem}.pdf differs from what the current code generates"


def test_lint_rejects_bad_figure():
    S.apply()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(S.mm(120), S.mm(80)))
    ax = fig.add_subplot(111)
    ax.set_title("a title")
    ax.text(0.5, 0.5, "tiny", fontsize=4)
    ax.text(0.5, 0.5, "overlap", fontsize=6)
    a, = ax.plot([0, 1], [0, 1], color="#808080", marker="o")
    b, = ax.plot([0, 1], [1, 0], color="#7F8580", marker="o")
    S.tag(a, "s1")
    S.tag(b, "s2")
    codes = {f.code for f in S.lint(fig, "bad")}
    plt.close(fig)
    assert {"width", "panel_title", "font_too_small", "text_overlap", "grayscale_collision"} <= codes


def _caption_panels():
    """{figure stem: set of panel letters named in bold in its caption} for main.tex and si.tex."""
    import re
    out = {}
    for tex in ("main.tex", "si.tex"):
        src = open(os.path.join(TEXDIR, tex), encoding="utf-8").read()
        for env in re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", src, flags=re.S):
            g = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", env)
            if not g:
                continue
            stem = os.path.splitext(os.path.basename(g.group(1)))[0]
            cap = env[env.index("\\caption{"):]
            letters = set(re.findall(r"\\textbf\{([a-z])\}", cap))
            for a, b in re.findall(r"\\textbf\{([a-z])\}--\\textbf\{([a-z])\}", cap):
                letters |= {chr(c) for c in range(ord(a), ord(b) + 1)}
            out[stem] = letters
    return out


def test_caption_panels_match_generated_panels(regenerated):
    """Every figure legend names exactly the panel letters the generator draws."""
    import json
    if not os.path.isfile(os.path.join(TEXDIR, "main.tex")):
        pytest.skip("paper sources not present")
    _, out = regenerated
    drawn = {}
    for line in open(out / "panels.jsonl", encoding="utf-8"):
        d = json.loads(line)
        drawn[d["stem"]] = set(d["panels"])
    caps = _caption_panels()
    assert set(caps) == set(EXPECTED), sorted(set(caps) ^ set(EXPECTED))
    bad = {k: (sorted(caps[k]), sorted(drawn.get(k, set()))) for k in caps if caps[k] != drawn.get(k, set())}
    assert not bad, f"caption panels vs drawn panels: {bad}"
