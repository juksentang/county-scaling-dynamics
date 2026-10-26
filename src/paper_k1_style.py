"""House style and publication lint for the K1 paper figures.

Nature-style conventions (author requirement: dense at first glance, ordered
on close reading):

* widths 89 mm (single column) or 183 mm (double); height <= 170 mm;
* Arial, 5-7 pt at final size (never below 5 pt); panel labels 8 pt bold
  lowercase, top left;
* axis lines 0.5 pt, ticks outward 2 pt, no top/right spines, no gridlines,
  no titles inside panels, frameless legends;
* one palette safe for colour-vision deficiency, with fixed semantics across all figures
  (China, US, Brazil; V0 vs V1), separable in greyscale (luminance gaps
  >= 0.1 or a different marker/line style);
* vector PDF only, with TrueType (Type 42) fonts (no raster output).

The lint mirrors the publication checks of StatsPAI
(``statspai.plots._editor_check``: minimum font size, greyscale collision,
low contrast) and adds figure-size, clipping and text-overlap checks.  It is
re-implemented here; StatsPAI is not a dependency.  ``save()`` runs the lint
and raises ``FigureLintError`` on any violation, so a figure that fails is
never written silently.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.text import Text

MM = 1.0 / 25.4
WIDTH_SINGLE_MM = 89.0
WIDTH_DOUBLE_MM = 183.0
ALLOWED_WIDTHS_MM = (WIDTH_SINGLE_MM, WIDTH_DOUBLE_MM)
MAX_HEIGHT_MM = 170.0
MIN_FONT_PT = 5.0
MAX_FONT_PT = 8.0
LABEL_PT = 8.0
BASE_PT = 6.0
SMALL_PT = 5.5
AXIS_LW = 0.5
TICK_LEN = 2.0
MATH_SCRIPT_SHRINK = 0.92


def mm(x: float) -> float:
    """Millimetres to inches."""
    return x * MM


def figure(width_mm: float, height_mm: float):
    """A figure at final print size (so every pt size is a final size)."""
    if width_mm not in ALLOWED_WIDTHS_MM:
        raise ValueError(f"width {width_mm} mm not in {ALLOWED_WIDTHS_MM}")
    if height_mm > MAX_HEIGHT_MM:
        raise ValueError(f"height {height_mm} mm > {MAX_HEIGHT_MM}")
    return plt.figure(figsize=(mm(width_mm), mm(height_mm)))


def ax_mm(fig, left, bottom, width, height):
    """Axes placed in absolute mm from the bottom-left corner of the figure."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


ARIAL_DIR = os.environ.get("PAPER_K1_ARIAL_DIR", '')
_ARIAL_FILES = ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")
FONT_FAMILY = "Arial"


def _register_fonts() -> str:
    found = False
    for f in _ARIAL_FILES:
        p = os.path.join(ARIAL_DIR, f)
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            found = True
    if not found:
        try:
            fm.findfont(fm.FontProperties(family=FONT_FAMILY), fallback_to_default=False)
        except ValueError:
            raise FigureLintError(
                f"Arial not found under {ARIAL_DIR} (set PAPER_K1_ARIAL_DIR) nor among the system fonts; "
                "the house style requires Arial") from None
    return FONT_FAMILY


def apply() -> None:
    """Install the house rcParams (idempotent)."""
    import logging
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    fam = _register_fonts()
    plt.rcdefaults()
    import matplotlib._mathtext as _mt
    _mt.SHRINK_FACTOR = MATH_SCRIPT_SHRINK
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [fam],
        "font.size": BASE_PT,
        "mathtext.fontset": "custom",
        "mathtext.rm": fam,
        "mathtext.it": f"{fam}:italic",
        "mathtext.bf": f"{fam}:bold",
        "mathtext.sf": fam,
        "mathtext.default": "it",
        "axes.titlesize": BASE_PT,
        "axes.labelsize": BASE_PT,
        "axes.linewidth": AXIS_LW,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.facecolor": "white",
        "axes.edgecolor": "#000000",
        "axes.labelpad": 2.0,
        "axes.unicode_minus": True,
        "xtick.labelsize": SMALL_PT,
        "ytick.labelsize": SMALL_PT,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": TICK_LEN,
        "ytick.major.size": TICK_LEN,
        "xtick.minor.size": 1.2,
        "ytick.minor.size": 1.2,
        "xtick.major.width": AXIS_LW,
        "ytick.major.width": AXIS_LW,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "lines.linewidth": 0.8,
        "lines.markersize": 3.0,
        "lines.markeredgewidth": 0.6,
        "errorbar.capsize": 0,
        "legend.fontsize": SMALL_PT,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.2,
        "legend.labelspacing": 0.25,
        "patch.linewidth": 0.5,
        "hatch.linewidth": 0.4,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


COUNTRY_COLOR = {"CN": "#D55E00", "US": "#0F4C81", "BR": "#6DB35A"}
COUNTRY_MARKER = {"CN": "o", "US": "s", "BR": "^"}
COUNTRY_NAME = {"CN": "China counties", "US": "US metropolitan areas",
                "BR": "Brazilian municipalities"}
COUNTRY_SHORT = {"CN": "China", "US": "US", "BR": "Brazil"}
V0_COLOR = "#8F8F8F"
V1_COLOR = "#3D2B8F"
V0_STYLE = dict(color=V0_COLOR, mfc="white", mec=V0_COLOR, marker="o", ls="--")
V1_STYLE = dict(color=V1_COLOR, mfc=V1_COLOR, mec=V1_COLOR, marker="o", ls="-")
TRUTH_COLOR = "#000000"
REF_COLOR = "#B0B0B0"
SUPERSEDED_COLOR = "#BDBDBD"
PROJ_COLOR = "#000000"
STATUS_COLOR = {"pass": "#E3E3E3", "fail": "#D55E00",
                "not_evaluable": "#4D4D4D", "report": "#FFFFFF"}
STAYER_COLOR = "#CFCFCF"
LEAVER_COLOR = "#2B2B2B"
ENTRANT_COLOR = "#8A8A8A"
WINDOW_SHADE = "#F1F1F1"
SEQ_CMAP = "Greys"


def tint(color, f: float = 0.5) -> str:
    """Lighter tint of a colour (mix with white by fraction ``f``).

    Used for the SECOND outcome of a country (GDP in the US, agricultural value
    added in Brazil): filled, lighter, edged in the country colour.  Open
    (white-filled) markers are reserved for V0 in every figure.
    """
    r, g, b = mcolors.to_rgb(color)
    return mcolors.to_hex((r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f))


def luminance(color) -> float:
    """Rec. 601 luma, the same weights StatsPAI uses."""
    r, g, b = mcolors.to_rgb(color)
    return 0.299 * r + 0.587 * g + 0.114 * b


def panel_label(fig, ax, letter: str, dx_mm: float = -7.0, dy_mm: float = 1.0):
    """Bold lowercase 8 pt label at the top-left corner of ``ax``."""
    bb = ax.get_position()
    W, H = fig.get_size_inches() / MM
    x = bb.x0 + dx_mm / W
    y = bb.y1 + dy_mm / H
    t = fig.text(x, y, letter, fontsize=LABEL_PT, fontweight="bold",
                 ha="left", va="bottom")
    t.set_gid("panel-label")
    if not hasattr(fig, "_k1_panels"):
        fig._k1_panels = []
    fig._k1_panels.append(letter)
    return t


def tag(artists, name: str):
    """Mark artists as one data series for the greyscale/contrast lint."""
    if artists is None:
        return artists
    if not isinstance(artists, (list, tuple)):
        items = [artists]
    else:
        items = list(artists)
    for a in items:
        if hasattr(a, "lines"):
            for sub in (a.lines[0],) + tuple(a.lines[2]):
                if sub is not None:
                    sub.set_gid(f"series:{name}")
        elif hasattr(a, "set_gid"):
            a.set_gid(f"series:{name}")
    return artists


def minus(s: str) -> str:
    """Typographic minus for text drawn on figures."""
    return s.replace("-", "−")


def despine(ax, left=True, bottom=True):
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.tick_params(left=False, labelleft=False)
    if not bottom:
        ax.tick_params(bottom=False, labelbottom=False)


class FigureLintError(RuntimeError):
    pass


@dataclass
class Finding:
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


def _drawn_tick_labels(fig):
    """Tick labels matplotlib will actually draw (inside the view limits)."""
    all_ids, drawn = set(), []
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                all_ids.update({id(tk.label1), id(tk.label2)})
            if not ax.get_visible() or not ax.axison:
                continue
            for tk in axis._update_ticks():
                for lab in (tk.label1, tk.label2):
                    if lab.get_visible() and tk.get_visible():
                        drawn.append(lab)
    return all_ids, drawn


def _visible_texts(fig) -> List[Text]:
    out = []
    tick_ids, drawn = _drawn_tick_labels(fig)
    drawn_ids = {id(t) for t in drawn}
    seen = set()
    for t in [t for t in fig.findobj(Text) if id(t) not in tick_ids] + drawn:
        if id(t) in seen:
            continue
        seen.add(id(t))
        if not t.get_visible():
            continue
        s = (t.get_text() or "").strip()
        if not s:
            continue
        ax = t.axes
        if ax is not None and not ax.get_visible():
            continue
        out.append(t)
    return out


def _series_signature(artist):
    """(colour, marker, linestyle, filled) as drawn."""
    color = None
    marker = None
    ls = None
    filled = None
    try:
        from matplotlib.lines import Line2D
        from matplotlib.collections import Collection
        from matplotlib.patches import Patch
        if isinstance(artist, Line2D):
            color = artist.get_color()
            marker = str(artist.get_marker())
            ls = str(artist.get_linestyle())
            mfc = artist.get_markerfacecolor()
            filled = not (isinstance(mfc, str) and mfc in ("none", "white")) and \
                mcolors.to_hex(mfc) != "#ffffff"
            if ls in ("None", "none", "", " ") and marker not in ("None", "none", ""):
                color = mfc if filled else artist.get_markeredgecolor()
        elif isinstance(artist, Collection):
            fc = artist.get_facecolor()
            ec = artist.get_edgecolor()
            if len(fc) and fc[0][3] > 0 and mcolors.to_hex(fc[0]) != "#ffffff":
                color = mcolors.to_hex(fc[0])
                filled = True
            elif len(ec):
                color = mcolors.to_hex(ec[0])
                filled = False
            ls = str(artist.get_linestyle()[0]) if artist.get_linestyle() else None
            paths = getattr(artist, "get_paths", lambda: [])()
            marker = f"coll{len(paths[0].vertices) if paths else 0}"
        elif isinstance(artist, Patch):
            fc = artist.get_facecolor()
            if fc[3] > 0 and mcolors.to_hex(fc) != "#ffffff":
                color = mcolors.to_hex(fc)
                filled = True
            else:
                color = mcolors.to_hex(artist.get_edgecolor())
                filled = False
            marker = "patch"
            ls = str(artist.get_linestyle())
    except Exception:
        return None
    if color is None:
        return None
    return (mcolors.to_hex(color), marker, ls, filled)


def _bbox_overlap(a, b) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    if w <= 0 or h <= 0:
        return 0.0
    return w * h


def lint(fig, name: str = "figure", allow_overlap: Iterable[str] = ()) -> List[Finding]:
    """Run every publication check on a drawn figure; return the findings."""
    findings: List[Finding] = []
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w_mm, h_mm = (fig.get_size_inches() / MM)

    if not any(abs(w_mm - a) < 0.25 for a in ALLOWED_WIDTHS_MM):
        findings.append(Finding("width", f"{name}: width {w_mm:.1f} mm not 89 or 183 mm"))
    if h_mm > MAX_HEIGHT_MM + 0.25:
        findings.append(Finding("height", f"{name}: height {h_mm:.1f} mm > 170 mm"))

    texts = _visible_texts(fig)
    fig_bb = fig.bbox
    pt_per_px = 72.0 / fig.dpi
    boxes = []
    for t in texts:
        fs = float(t.get_fontsize())
        s = t.get_text()
        if fs < MIN_FONT_PT - 1e-6:
            findings.append(Finding("font_too_small", f"{name}: '{s[:30]}' at {fs:.2f} pt < {MIN_FONT_PT} pt"))
        if fs > MAX_FONT_PT + 1e-6:
            findings.append(Finding("font_too_large", f"{name}: '{s[:30]}' at {fs:.2f} pt > {MAX_FONT_PT} pt"))
        try:
            path = fm.findfont(t.get_fontproperties(), fallback_to_default=False)
            if "arial" not in os.path.basename(path).lower():
                findings.append(Finding("font_family", f"{name}: '{s[:30]}' resolves to {os.path.basename(path)}"))
        except Exception as e:
            findings.append(Finding("font_family", f"{name}: '{s[:30]}' font lookup failed: {e}"))
        if abs(luminance(t.get_color()) - 1.0) < 0.35 and t.get_gid() != "on-dark":
            findings.append(Finding("text_contrast", f"{name}: text '{s[:30]}' colour {t.get_color()} too light"))
        bb = Text.get_window_extent(t, renderer)
        tol = 0.5 / pt_per_px
        if (bb.x0 < fig_bb.x0 - tol or bb.y0 < fig_bb.y0 - tol or
                bb.x1 > fig_bb.x1 + tol or bb.y1 > fig_bb.y1 + tol):
            findings.append(Finding("clipped", f"{name}: text '{s[:30]}' extends outside the figure"))
        boxes.append((t, bb))

    allow = set(allow_overlap)
    for (t1, b1), (t2, b2) in combinations(boxes, 2):
        ov = _bbox_overlap(b1, b2) * pt_per_px * pt_per_px
        if ov > 1.5:
            if t1.get_gid() in allow or t2.get_gid() in allow:
                continue
            findings.append(Finding(
                "text_overlap",
                f"{name}: '{t1.get_text()[:25]}' overlaps '{t2.get_text()[:25]}' ({ov:.1f} pt^2)"))

    for ax in fig.get_axes():
        if not ax.get_visible() or not ax.axison:
            continue
        if ax.get_title():
            findings.append(Finding("panel_title", f"{name}: axes title '{ax.get_title()}' (put it in the caption)"))
        for side in ("left", "bottom"):
            sp = ax.spines[side]
            if sp.get_visible() and abs(sp.get_linewidth() - AXIS_LW) > 1e-6:
                findings.append(Finding("spine_width", f"{name}: {side} spine {sp.get_linewidth()} pt"))
        for side in ("top", "right"):
            if ax.spines[side].get_visible() and not getattr(ax, "_k1_box_ok", False):
                findings.append(Finding("spine_box", f"{name}: {side} spine visible"))
        leg = ax.get_legend()
        if leg is not None and leg.get_frame_on():
            findings.append(Finding("legend_frame", f"{name}: legend has a frame"))

    for ax in fig.get_axes():
        series = {}
        for art in ax.get_children():
            gid = getattr(art, "get_gid", lambda: None)()
            if not gid or not str(gid).startswith("series:"):
                continue
            sig = _series_signature(art)
            if sig is None:
                continue
            series.setdefault(gid[7:], set()).add(sig)
        bg = luminance(ax.get_facecolor())
        reps = {}
        for k, sigs in series.items():
            sigs = sorted(sigs, key=str)
            reps[k] = sigs
            for sig in sigs:
                if abs(luminance(sig[0]) - bg) < 0.08:
                    findings.append(Finding("low_contrast", f"{name}: series '{k}' colour {sig[0]} vs background"))
        for (k1, s1), (k2, s2) in combinations(reps.items(), 2):
            c1 = {s[0] for s in s1}
            c2 = {s[0] for s in s2}
            if c1 == c2:
                continue
            close = any(abs(luminance(a) - luminance(b)) < 0.1 for a in c1 for b in c2)
            st1 = {s[1:] for s in s1}
            st2 = {s[1:] for s in s2}
            if close and st1 == st2:
                findings.append(Finding(
                    "grayscale_collision",
                    f"{name}: series '{k1}' and '{k2}' share marker/line/fill and differ in luminance < 0.1"))
    return findings


def palette_findings() -> List[Finding]:
    """Global palette checks: semantic colours separable in greyscale."""
    out = []
    keys = list(COUNTRY_COLOR)
    for a, b in combinations(keys, 2):
        d = abs(luminance(COUNTRY_COLOR[a]) - luminance(COUNTRY_COLOR[b]))
        if d < 0.1:
            out.append(Finding("palette", f"country colours {a}/{b} luminance gap {d:.3f} < 0.1"))
    d = abs(luminance(V0_COLOR) - luminance(V1_COLOR))
    if d < 0.1:
        out.append(Finding("palette", f"V0/V1 luminance gap {d:.3f} < 0.1"))
    for k, c in COUNTRY_COLOR.items():
        if abs(luminance(c) - 1.0) < 0.2:
            out.append(Finding("palette", f"{k} colour too light on white"))
    return out


def save(fig, outdir: str, stem: str, allow_overlap: Iterable[str] = ()):
    """Lint, then write ``stem.pdf`` (vector, Type 42 fonts). No raster output.

    Raises FigureLintError (nothing written) when any check fails.
    """
    findings = palette_findings() + lint(fig, stem, allow_overlap)
    dbg = os.environ.get("PAPER_K1_FIG_DEBUGDIR")
    if findings and dbg:
        os.makedirs(dbg, exist_ok=True)
        fig.savefig(os.path.join(dbg, f"{stem}_LINTFAIL.pdf"))
    if findings:
        raise FigureLintError(f"{stem}: {len(findings)} lint violation(s):\n  " +
                              "\n  ".join(str(f) for f in findings))
    os.makedirs(outdir, exist_ok=True)
    reg = os.environ.get("PAPER_K1_FIG_PANELS")
    if reg:
        import json
        with open(reg, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"stem": stem, "panels": sorted(getattr(fig, "_k1_panels", []))}) + "\n")
    pdf = os.path.join(outdir, f"{stem}.pdf")
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None,
                               "Producer": None, "Creator": "paper_k1_figures.py"})
    plt.close(fig)
    bad = lint_pdf(pdf)
    if bad:
        os.remove(pdf)
        raise FigureLintError(f"{stem}: written PDF failed lint:\n  " + "\n  ".join(str(f) for f in bad))
    return pdf


_MEDIABOX = re.compile(rb"/MediaBox\s*\[\s*([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s*\]")
_FONTNAME = re.compile(rb"/BaseFont\s*/([A-Za-z0-9+\-_,]+)")
_SUBTYPE = re.compile(rb"/Subtype\s*/(Type3|Type1|TrueType|Type0|CIDFontType2)")


def lint_pdf(path: str) -> List[Finding]:
    data = open(path, "rb").read()
    out = []
    m = _MEDIABOX.search(data)
    if not m:
        return [Finding("pdf", f"{path}: no MediaBox")]
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    w_mm = (x1 - x0) / 72 * 25.4
    h_mm = (y1 - y0) / 72 * 25.4
    if not any(abs(w_mm - a) < 0.3 for a in ALLOWED_WIDTHS_MM):
        out.append(Finding("width", f"{os.path.basename(path)}: {w_mm:.1f} mm"))
    if h_mm > MAX_HEIGHT_MM + 0.3:
        out.append(Finding("height", f"{os.path.basename(path)}: {h_mm:.1f} mm"))
    fonts = {f.decode() for f in _FONTNAME.findall(data)}
    bad = [f for f in fonts if "Arial" not in f]
    if bad:
        out.append(Finding("pdf_font", f"{os.path.basename(path)}: non-Arial fonts {sorted(bad)}"))
    if b"/Type3" in data or b"/Subtype /Type3" in data:
        out.append(Finding("pdf_type3", f"{os.path.basename(path)}: Type 3 font embedded"))
    n_img = len(re.findall(rb"/Subtype\s*/Image", data))
    if n_img:
        out.append(Finding("pdf_image", f"{os.path.basename(path)}: {n_img} embedded raster image(s)"))
    return out
