"""
Every number the K1 methods paper prints, pulled from committed result tables.

Reads ONLY CSV/JSON files under results/tables/ and writes

  results/tables/paper_k1_values.csv   one row per value (key, macro, value,
                                       formatted, source_file,
                                       source_row_or_column, transform, note)
  results/tables/paper_k1_values_meta.json   sha256 of every input
  <tex dir>/values.tex                 one \\newcommand per value
  <tex dir>/si_tables/*.tex            Supplementary tables and si_values.tex
  results/tables/paper_k1_si_tables_meta.json

<tex dir> is paper_k1/latex unless the environment variable PAPER_K1_TEX_DIR
names another directory.

Rules:
  * a value is either read from one cell of one committed file, or is a
    deterministic re-expression (ratio, difference, product, count, min/max,
    rounding) of values that were; the transform column says which;
  * a missing file, column or row, or a row filter that matches anything but
    exactly one row, raises: the build fails rather than printing a stale or
    guessed number;
  * no analysis is run here; nothing is estimated.

The output depends only on the input files (no timestamps, no git hash), so a
second run on the same tree is byte-identical.

Usage: python3 scripts/paper_k1_numbers.py
"""
import hashlib
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

import numpy as np
import pandas as pd

OUT_CSV = "results/tables/paper_k1_values.csv"
TEX_DIR = os.environ.get("PAPER_K1_TEX_DIR", 'build/tex')
OUT_TEX = f"{TEX_DIR}/values.tex"
OUT_META = "results/tables/paper_k1_values_meta.json"

ALLOWED_ROOTS = ("results/tables/",)
MACRO_RE = re.compile(r"^[A-Za-z]+$")


class SourceError(RuntimeError):
    pass


def _dec(v, n, signed=False):
    s = f"{v:+.{n}f}" if signed else f"{v:.{n}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    return s


def _sig_decimals(v, n):
    if v == 0:
        return n - 1
    return max(0, n - 1 - int(math.floor(math.log10(abs(v)))))


def fmt(v, spec):
    """spec: dec<n>, sdec<n> (signed), sig<n>, ssig<n>, sci<n>, int, pct<n>,
    str (verbatim text)."""
    if spec == "str":
        return str(v)
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        raise SourceError(f"non-finite value {v!r} cannot be formatted ({spec})")
    m = re.fullmatch(r"(s?)(dec|sig|sci|pct)(\d)", spec)
    if spec == "int":
        if abs(v - round(v)) > 1e-9:
            raise SourceError(f"value {v!r} is not an integer")
        iv = int(round(v))
        s = f"{iv:,}".replace(",", "{,}") if abs(iv) >= 10000 else str(iv)
        return f"\\ensuremath{{{s}}}"
    if not m:
        raise SourceError(f"unknown format {spec}")
    signed, kind, n = m.group(1) == "s", m.group(2), int(m.group(3))
    if kind == "dec":
        s = _dec(v, n, signed)
    elif kind == "sig":
        s = _dec(v, _sig_decimals(v, n), signed)
    elif kind == "pct":
        return f"\\ensuremath{{{_dec(100 * v, n, signed)}}}\\%"
    else:
        if v == 0:
            s = "0"
        else:
            e = int(math.floor(math.log10(abs(v))))
            mant = v / 10 ** e
            if round(abs(mant), n - 1) >= 10:
                e += 1
                mant = v / 10 ** e
            ms = _dec(mant, n - 1, signed)
            s = f"{ms}\\times10^{{{e}}}"
    return f"\\ensuremath{{{s}}}"


_SHA = {}


def _check_path(path):
    if not path.startswith(ALLOWED_ROOTS):
        raise SourceError(f"{path}: only results/tables/ may be read")
    if not os.path.isfile(path):
        raise SourceError(f"missing source file {path}")
    if path not in _SHA:
        with open(path, "rb") as fh:
            _SHA[path] = hashlib.sha256(fh.read()).hexdigest()


class Csv:
    def __init__(self, path):
        _check_path(path)
        self.path = path
        self.df = pd.read_csv(path, low_memory=False, keep_default_na=False, na_values=[""])

    def _need(self, cols):
        miss = [c for c in cols if c not in self.df.columns]
        if miss:
            raise SourceError(f"{self.path}: missing column(s) {miss}")

    def rows(self, **flt):
        self._need(list(flt))
        d = self.df
        for k, v in flt.items():
            d = d[d[k] == v]
        return d

    def row(self, **flt):
        d = self.rows(**flt)
        if len(d) != 1:
            raise SourceError(f"{self.path}: filter {flt} matched {len(d)} rows, need exactly 1")
        return Row(self, d.iloc[0], flt)

    def col(self, name, **flt):
        self._need([name])
        d = self.rows(**flt)
        if not len(d):
            raise SourceError(f"{self.path}: filter {flt} matched no row")
        return d[name], flt


class Row:
    def __init__(self, src, s, flt):
        self.src, self.s, self.flt = src, s, flt

    def where(self, col):
        f = ";".join(f"{k}={v}" for k, v in self.flt.items())
        return f"[{f}] col={col}"

    def get(self, col):
        self.src._need([col])
        return self.s[col]


class Json:
    def __init__(self, path):
        _check_path(path)
        self.path = path
        with open(path, encoding="utf-8") as fh:
            self.d = json.load(fh)

    def get(self, dotted):
        cur = self.d
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise SourceError(f"{self.path}: missing key {dotted}")
            cur = cur[part]
        return cur


class Registry:
    def __init__(self):
        self.rows = []
        self.by_key = {}
        self.by_macro = {}

    def _put(self, macro, key, value, spec, source_file, where, transform, note):
        if not MACRO_RE.match(macro):
            raise SourceError(f"macro name {macro!r} must be letters only")
        if macro in self.by_macro:
            raise SourceError(f"duplicate macro {macro}")
        if key in self.by_key:
            raise SourceError(f"duplicate key {key}")
        if spec != "str":
            value = float(value)
        rec = dict(key=key, macro=macro, value=value, formatted=fmt(value, spec),
                   source_file=source_file, source_row_or_column=where,
                   transform=transform, note=note)
        self.rows.append(rec)
        self.by_key[key] = rec
        self.by_macro[macro] = rec
        return value

    def cell(self, macro, key, row, col, spec, note="", transform="read"):
        v = row.get(col)
        if spec != "str":
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise SourceError(f"{row.src.path} {row.where(col)}: not numeric: {v!r}")
            if not math.isfinite(v):
                raise SourceError(f"{row.src.path} {row.where(col)}: non-finite {v!r}")
        return self._put(macro, key, v, spec, row.src.path, row.where(col), transform, note)

    def json(self, macro, key, js, dotted, spec, note=""):
        return self._put(macro, key, js.get(dotted), spec, js.path, f"json:{dotted}", "read", note)

    def derived(self, macro, key, value, spec, inputs, transform, note=""):
        """inputs: list of keys already registered (their files are listed)."""
        files = []
        for k in inputs:
            if k not in self.by_key:
                raise SourceError(f"derived {key}: input {k} not registered")
            for f in self.by_key[k]["source_file"].split(" | "):
                if f not in files:
                    files.append(f)
        where = "inputs: " + ", ".join(inputs)
        return self._put(macro, key, value, spec, " | ".join(files), where,
                         transform, note)

    def agg(self, macro, key, series_flt, src, col, how, spec, note=""):
        """Aggregate (min/max/count/sum) over a filtered column of one file."""
        series, flt = series_flt
        f = ";".join(f"{k}={v}" for k, v in flt.items())
        if how == "min":
            v = float(series.min())
        elif how == "max":
            v = float(series.max())
        elif how == "maxabs":
            v = float(series.abs().max())
        elif how == "count":
            v = float(len(series))
        elif how == "sum":
            v = float(series.sum())
        else:
            raise SourceError(how)
        if not math.isfinite(v):
            raise SourceError(f"{src.path}: {how} of {col} [{f}] not finite")
        return self._put(macro, key, v, spec, src.path, f"[{f}] col={col}",
                         f"{how} over rows", note)

    def v(self, key):
        return self.by_key[key]["value"]


R = Registry()


def _need_ok(r):
    """A paired-addendum row may be cited only with its registered tol0 interval in status ok."""
    if str(r.get("ci_rule")) != "tol0" or str(r.get("ci_status")) != "ok":
        raise SourceError(f"{r.src.path}: {r.flt} has no usable tol0 interval "
                          f"(rule {r.get('ci_rule')!r}, status {r.get('ci_status')!r})")


def china():
    by_year = Csv("results/tables/expi_attrition_by_year.csv")
    t = by_year.row(saai_spec="multi", year_t=2016)
    R.cell("CnBreakNS", "cn.t2016.n_S", t, "n_stay", "int", "stayers 2016->17, multi SAAI")
    R.cell("CnBreakNL", "cn.t2016.n_L", t, "n_leave", "int")
    R.cell("CnBreakNE", "cn.t2016.n_E", t, "n_enter", "int")
    R.cell("CnBreakML", "cn.t2016.m_L", t, "mean_leave_t", "dec2", "mean xi_t of leavers")
    R.cell("CnBreakMt", "cn.t2016.M_t", t, "mechanical_term", "dec2",
           "mechanical term M_t = mean stayer increment (Lemma 1)")
    R.derived("CnBreakDose", "cn.t2016.dose",
              (R.v("cn.t2016.n_E") + R.v("cn.t2016.n_L")) / R.v("cn.t2016.n_S"), "dec2",
              ["cn.t2016.n_E", "cn.t2016.n_L", "cn.t2016.n_S"], "(n_E+n_L)/n_S",
              "composition shock intensity of the 2016->17 transition")
    R.agg("CnMaxIdentityResidual", "cn.identity.max_abs_residual",
          by_year.col("identity_residual", saai_spec="multi"), by_year,
          "identity_residual", "maxabs", "sci2", "Lemma 1 residual, all multi transitions")

    per = Csv("results/tables/expi_attrition_by_period.csv")
    for p, P in (("pre", "Pre"), ("during", "During")):
        r = per.row(period=p)
        R.cell(f"CnPairs{P}", f"cn.{p}.n_pairs", r, "n_pairs", "int")
        R.cell(f"CnCounties{P}", f"cn.{p}.n_counties", r, "n_counties", "int")
        R.cell(f"CnBalanced{P}", f"cn.{p}.n_balanced", r, "n_balanced", "int",
               "counties observed in every year of the window")
        R.cell(f"CnMbar{P}", f"cn.{p}.M_bar", r, "mechanical_term_weighted", "sdec3",
               "pair-weighted mean mechanical term per year")
        R.cell(f"CnTransitions{P}", f"cn.{p}.n_transitions", r, "n_transitions", "int")

    cov = Csv("results/tables/k2br_coverage_summary.csv")
    cn_label = "China counties multi-SAAI (expi_attrition_by_year.csv)"
    for p, P in (("pre", "Pre"), ("during", "During")):
        r = cov.row(dataset=cn_label, window=p)
        R.cell(f"CnDose{P}", f"cn.{p}.dose", r, "nS_weighted_comp_shock", "dec2",
               "n_S-weighted (n_E+n_L)/n_S over the window")
        R.cell(f"CnDoseMax{P}", f"cn.{p}.dose_max", r, "max_comp_shock", "dec2",
               "largest single-transition (n_E+n_L)/n_S")
    if abs(R.v("cn.during.dose_max") - R.v("cn.t2016.dose")) > 1e-12:
        raise SourceError("China: during max dose != 2016->17 dose; the two tables disagree")

    lp = Csv("results/tables/k1_leaver_provinces.csv")
    sc = lp.row(year_t=2016, ProvinceCode=51, reason="missing GrossOutputValue (not in the scaling panel in year t+1)")
    R.cell("CnSichuanLeave", "cn.t2016.sichuan.n_leave_province", sc, "n_leave_province", "int",
           "Sichuan counties leaving 2016->17 (all reasons)")
    R.cell("CnSichuanCounties", "cn.t2016.sichuan.n_province", sc, "n_province_t", "int")
    R.cell("CnSichuanMissingGOV", "cn.t2016.sichuan.n_missing_gov", sc, "n_leave", "int",
           "of which: missing GrossOutputValue")
    gs = lp.row(year_t=2016, ProvinceCode=62, reason="missing CultivateArea")
    R.cell("CnGansuLeave", "cn.t2016.gansu.n_leave", gs, "n_leave", "int", "Gansu: missing CultivateArea")
    R.cell("CnGansuCounties", "cn.t2016.gansu.n_province", gs, "n_province_t", "int")

    pe = Csv("results/tables/expi_paired_estimands.csv")
    for p, P in (("pre", "Pre"), ("during", "During")):
        for V, Vm in (("V0", "Vzero"), ("V1", "Vone")):
            for est, M, spec, desc in ((f"rho1_{V}_{p}", f"CnRho{Vm}{P}", "dec3", "rho1 = 1 + pooled OLS slope"),
                                       (f"a1_{V}_{p}", f"CnAone{Vm}{P}", "sdec3", "a1 = rho1 - 1 (pooled OLS)")):
                r = pe.row(estimand=est, convention="none", role='dose_ladder')
                _need_ok(r)
                k = f"cn.{p}.{V}.ols_{est.split('_')[0]}"
                R.cell(M, k, r, "point_estimate", spec, f"{desc}, {V}, full sample")
                R.cell(f"{M}Lo", f"{k}.lo", r, "ci_lo", spec, "stratified province block, tol0 rank CI")
                R.cell(f"{M}Hi", f"{k}.hi", r, "ci_hi", spec, "stratified province block, tol0 rank CI")
        r = pe.row(estimand=f"amp_V1_{p}", convention="none", role='dose_ladder')
        _need_ok(r)
        R.cell(f"CnAmpVone{P}", f"cn.{p}.amp_V1", r, "point_estimate", "dec1",
               "amplification 1/|a1_V1| (pooled OLS; same definition as US/Brazil)")
        R.cell(f"CnAmpVone{P}Lo", f"cn.{p}.amp_V1.lo", r, "ci_lo", "dec1", "tol0 CI")
        R.cell(f"CnAmpVone{P}Hi", f"cn.{p}.amp_V1.hi", r, "ci_hi", "dec1", "tol0 CI")
        if abs(R.v(f"cn.{p}.amp_V1") - 1 / abs(R.v(f"cn.{p}.V1.ols_a1"))) > 1e-9:
            raise SourceError(f"China {p}: amp_V1 != 1/|a1_V1| in expi_paired_estimands.csv")
        r = pe.row(estimand=f"dose_{p}", convention="none", role='dose_ladder')
        if abs(float(r.get("point_estimate")) - R.v(f"cn.{p}.dose")) > 1e-12:
            raise SourceError(f"China {p}: paired-addendum dose != k2br_coverage_summary dose")
        r = pe.row(estimand=f"dose_x_amp_V1_{p}", convention="none", role='dose_ladder')
        _need_ok(r)
        R.cell(f"CnDoseTimesAmp{P}", f"cn.{p}.dose_x_amp_V1", r, "point_estimate", "dec2",
               "dose-ladder abscissa for China: dose / |a1_V1| (pooled OLS, as US/Brazil)")
        R.cell(f"CnDoseTimesAmp{P}Lo", f"cn.{p}.dose_x_amp_V1.lo", r, "ci_lo", "dec2", "tol0 CI")
        R.cell(f"CnDoseTimesAmp{P}Hi", f"cn.{p}.dose_x_amp_V1.hi", r, "ci_hi", "dec2", "tol0 CI")
        if abs(R.v(f"cn.{p}.dose_x_amp_V1") - R.v(f"cn.{p}.dose") * R.v(f"cn.{p}.amp_V1")) > 1e-9:
            raise SourceError(f"China {p}: dose_x_amp_V1 != dose * amp_V1")

    meta = Json("results/tables/expi_rerun_ci_variants.meta.json")
    R.json("CnRerunProvPre", "cn.rerun.n_provinces.pre", meta, "provinces.pre", "int")
    R.json("CnRerunProvDuring", "cn.rerun.n_provinces.during", meta, "provinces.during", "int",
           "province clusters in the during window")
    R.json("CnRerunCountiesPre", "cn.rerun.n_counties.pre", meta, "counties.pre", "int")
    R.json("CnRerunCountiesDuring", "cn.rerun.n_counties.during", meta, "counties.during", "int")
    R.json("CnRerunSeed", "cn.rerun.seed", meta, "seed", "str")
    R.json("CnRerunB", "cn.rerun.n_boot", meta, "n_boot", "int")

    sbs = Csv("results/tables/expi_rerun_ci_side_by_side.csv")
    Q = [("xi_trap (during-pre)", "XiShift", "sdec2", "xi_trap shift during-pre"),
         ("tau_mfpt (during/pre)", "TauRatio", "sig2", "tau_MFPT ratio during/pre"),
         ("gamma (during-pre)", "GammaDiff", "sdec2", "Gamma difference during-pre"),
         ("mean_diffusion (during/pre)", "DRatio", "dec2", "mean diffusion ratio during/pre"),
         ("B_ratio_w (during/pre)", "BRatio", "dec2", "B_ratio_w ratio during/pre (noise compression; crosses 1)")]
    for var, V in (("V0_baseline", "Vzero"), ("V1_stayer_restandardised", "Vone")):
        for q, Qn, spec, desc in Q:
            r = sbs.row(variant=var, degenerate="limit", quantity=q)
            base = f"cn.{var.split('_')[0]}.{Qn}"
            R.cell(f"Cn{V}{Qn}Point", f"{base}.point", r, "point_estimate", spec,
                   f"{desc}; full-sample point estimate")
            R.cell(f"Cn{V}{Qn}Lo", f"{base}.prov.lo", r, "rerun_ci_lo", spec,
                   "PRIMARY: stratified province block, per-draw pipeline re-run, 95% percentile, limit")
            R.cell(f"Cn{V}{Qn}Hi", f"{base}.prov.hi", r, "rerun_ci_hi", spec,
                   "PRIMARY: stratified province block, per-draw pipeline re-run, 95% percentile, limit")
            R.cell(f"Cn{V}{Qn}Median", f"{base}.prov.median", r, "rerun_median", spec,
                   "bootstrap median under the primary scheme")
            R.cell(f"Cn{V}{Qn}CountyLo", f"{base}.county.lo", r, "county_block_ci_lo", spec,
                   "SUPERSEDED (SI only, trap 2): county block, seed 106, no re-run")
            R.cell(f"Cn{V}{Qn}CountyHi", f"{base}.county.hi", r, "county_block_ci_hi", spec,
                   "SUPERSEDED (SI only, trap 2): county block, seed 106, no re-run")
            R.cell(f"Cn{V}{Qn}WidthRatio", f"{base}.width_ratio", r,
                   "width_ratio_rerun_over_county", "dec1", "CI width primary / county block")
    R.agg("CnWidthRatioMin", "cn.width_ratio.min",
          sbs.col("width_ratio_rerun_over_county", degenerate="limit"), sbs,
          "width_ratio_rerun_over_county", "min", "dec1", "limit convention, V0 and V1, five quantities")
    R.agg("CnWidthRatioMax", "cn.width_ratio.max",
          sbs.col("width_ratio_rerun_over_county", degenerate="limit"), sbs,
          "width_ratio_rerun_over_county", "max", "dec1", "limit convention, V0 and V1, five quantities")
    R.derived("CnVzeroMinusVoneXiShift", "cn.V0_minus_V1.xi_shift",
              R.v("cn.V0.XiShift.point") - R.v("cn.V1.XiShift.point"), "sdec2",
              ["cn.V0.XiShift.point", "cn.V1.XiShift.point"], "difference",
              "observed V0-V1 difference of the during-pre attractor shift (dose ladder, China)")

    for var, V in (("V0_baseline", "Vzero"), ("V1_stayer_restandardised", "Vone")):
        r = sbs.row(variant=var, degenerate="nan", quantity="xi_trap (during-pre)")
        R.cell(f"Cn{V}XiShiftNanLo", f"cn.{var.split('_')[0]}.XiShift.prov.nan.lo", r, "rerun_ci_lo", "sdec2",
               "nan convention, primary scheme")
        R.cell(f"Cn{V}XiShiftNanHi", f"cn.{var.split('_')[0]}.XiShift.prov.nan.hi", r, "rerun_ci_hi", "sdec2",
               "nan convention, primary scheme")
        R.cell(f"Cn{V}XiShiftNanFailed", f"cn.{var.split('_')[0]}.XiShift.prov.nan.n_failed", r,
               "rerun_n_failed", "int", "degenerate-path draws under nan")

    lev = Csv("results/tables/expi_rerun_ci_levels.csv")
    for var, V in (("V0_baseline", "Vzero"), ("V1_stayer_restandardised", "Vone")):
        for p, P in (("pre", "Pre"), ("during", "During")):
            r = lev.row(variant=var, degenerate="limit", period=p, quantity="xi_trap")
            R.cell(f"Cn{V}Xi{P}", f"cn.{var.split('_')[0]}.xi.{p}.point", r, "point_estimate", "sdec2",
                   "attractor level, full sample")
    for p, P in (("pre", "Pre"), ("during", "During")):
        R.derived(f"CnVzeroMinusVoneXi{P}", f"cn.V0_minus_V1.xi.{p}",
                  R.v(f"cn.V0.xi.{p}.point") - R.v(f"cn.V1.xi.{p}.point"), "sdec2",
                  [f"cn.V0.xi.{p}.point", f"cn.V1.xi.{p}.point"], "difference of committed levels",
                  "within-window V0-V1 attractor level difference (the quantity plotted for China in the "
                  "dose ladder); paired interval from the China paired addendum below")

    pe = Csv("results/tables/expi_paired_estimands.csv")
    for est, base, M in (("xi_trap_V0_minus_V1_pre", "cn.V0_minus_V1.xi.pre", "CnVzeroMinusVoneXiPre"),
                         ("xi_trap_V0_minus_V1_during", "cn.V0_minus_V1.xi.during", "CnVzeroMinusVoneXiDuring"),
                         ("xi_shift_V0_minus_V1", "cn.V0_minus_V1.xi_shift", "CnVzeroMinusVoneXiShift")):
        r = pe.row(estimand=est, convention="limit", role='paired_V0_V1')
        _need_ok(r)
        if abs(float(r.get("point_estimate")) - R.v(base)) > 1e-12:
            raise SourceError(f"{est}: paired point estimate != difference of committed levels")
        R.cell(f"{M}Lo", f"{base}.paired.lo", r, "ci_lo", "sdec2",
               "PAIRED V0-V1, stratified province block, pipeline re-run per draw, tol0 rank CI, limit")
        R.cell(f"{M}Hi", f"{base}.paired.hi", r, "ci_hi", "sdec2",
               "PAIRED V0-V1, stratified province block, pipeline re-run per draw, tol0 rank CI, limit")
        rn = pe.row(estimand=est, convention="nan", role='paired_V0_V1')
        if str(rn.get("ci_status")) == "ok":
            raise SourceError(f"{est} nan: expected not_evaluable_nonfinite under tol0")
        R.cell(f"{M}NanNonfinite", f"{base}.paired.nan.n_nonfinite", rn, "n_nonfinite", "int",
               "nan convention: non-finite draws (tol0 interval not evaluable)")
        R.cell(f"{M}NanDropLo", f"{base}.paired.nan.droplo", rn, "supplement_rule_lo", "sdec2",
               "nan convention, non-finite draws discarded (supplement rule), report only")
        R.cell(f"{M}NanDropHi", f"{base}.paired.nan.drophi", rn, "supplement_rule_hi", "sdec2",
               "nan convention, non-finite draws discarded (supplement rule), report only")
    for p, P in (("pre", "Pre"), ("during", "During")):
        r = pe.row(estimand=f"zeroth_order_Mbar_over_abs_a1_V1_{p}", convention="none", role="report")
        R.cell(f"CnZerothVone{P}", f"cn.{p}.zeroth_Mbar_over_abs_a1_V1", r, "point_estimate", "sdec2",
               "REPORT ONLY first-order value M-bar/|a1,V1| (no China projection test registered); "
               "interval not cited")
    r = pe.row(estimand="tau_ratio_V0_over_V1", convention="limit", role="report")
    _need_ok(r)
    R.cell("CnTauRatioVzeroOverVone", "cn.V0_over_V1.tau_ratio", r, "point_estimate", "sig2", "report only")
    R.cell("CnTauRatioVzeroOverVoneLo", "cn.V0_over_V1.tau_ratio.lo", r, "ci_lo", "sig2", "report only, tol0")
    R.cell("CnTauRatioVzeroOverVoneHi", "cn.V0_over_V1.tau_ratio.hi", r, "ci_hi", "sig3", "report only, tol0")

    old = Csv("results/tables/expi_attrition_variants_limit.csv")
    for var, V in (("V0_baseline", "Vzero"), ("V1_stayer_restandardised", "Vone"),
                   ("V2_mechanical_offset", "Vtwo")):
        r = old.row(variant=var, degenerate="limit", quantity="xi_trap (during-pre)")
        R.cell(f"CnOld{V}XiShiftMedian", f"cn.old.{var.split('_')[0]}.xi_shift.median", r, "median", "sdec2",
               "SUPERSEDED county block (seed 106), SI only")
        R.cell(f"CnOld{V}XiShiftLo", f"cn.old.{var.split('_')[0]}.xi_shift.lo", r, "ci_lo", "sdec2",
               "SUPERSEDED county block, SI only")
        R.cell(f"CnOld{V}XiShiftHi", f"cn.old.{var.split('_')[0]}.xi_shift.hi", r, "ci_hi", "sdec2",
               "SUPERSEDED county block, SI only")
        R.cell(f"CnOld{V}Verdict", f"cn.old.{var.split('_')[0]}.verdict", r, "verdict", "str",
               "registered China outcome on the county-block table (robust/attenuated/overturned)")
        r = old.row(variant=var, degenerate="limit", quantity="tau_mfpt (during/pre)")
        R.cell(f"CnOld{V}TauRatioMedian", f"cn.old.{var.split('_')[0]}.tau_ratio.median", r, "median", "sig2",
               "SUPERSEDED county block, SI only")
        R.cell(f"CnOld{V}TauRatioLo", f"cn.old.{var.split('_')[0]}.tau_ratio.lo", r, "ci_lo", "sig2",
               "SUPERSEDED county block, SI only")
        R.cell(f"CnOld{V}TauRatioHi", f"cn.old.{var.split('_')[0]}.tau_ratio.hi", r, "ci_hi", "sig3",
               "SUPERSEDED county block, SI only")
    rank = {"robust": 0, "attenuated": 1, "overturned": 2}
    v1, v2 = R.v("cn.old.V1.verdict"), R.v("cn.old.V2.verdict")
    if v1 not in rank or v2 not in rank:
        raise SourceError(f"unknown China verdict label {v1!r}/{v2!r}")
    R.derived("CnOldOverallVerdict", "cn.old.overall_verdict", max((v1, v2), key=rank.get), "str",
              ["cn.old.V1.verdict", "cn.old.V2.verdict"],
              "registered rule: the worse of the V1 and V2 verdicts",
              "SUPERSEDED county-block evaluation, SI only")


def k1():
    s0 = Csv("results/tables/k1_s0_scope_f2/k1_s0_scope_s0_rows.csv")
    r = s0.row(gate="S0", process="cubic", pipeline="saai", period="during",
               quantity="mean_diffusion", rule_set="limit")
    R.cell("KoneSzeroFailBias", "k1.s0.cubic_saai.during.mean_diffusion.bias", r, "value", "sdec6",
           "median bias of the one failing S0 record (second freeze)")
    R.cell("KoneSzeroFailTol", "k1.s0.cubic_saai.during.mean_diffusion.tol", r, "threshold", "dec6",
           "registered tolerance = 5% of the predicted contrast")
    recs = s0.df.groupby(["process", "pipeline"]).size()
    nfail = int(s0.df[s0.df["status"] == "fail"].groupby(["process", "pipeline"]).ngroups)
    R._put("KoneSzeroRecords", "k1.s0.n_records", float(len(recs)), "int", s0.path,
           "distinct (process, pipeline)", "count of groups", "MLE S0 records, second freeze")
    R._put("KoneSzeroRecordsFail", "k1.s0.n_records_fail", float(nfail), "int", s0.path,
           "[status=fail] distinct (process, pipeline)", "count of groups", "")
    R.derived("KoneSzeroRecordsPass", "k1.s0.n_records_pass",
              R.v("k1.s0.n_records") - R.v("k1.s0.n_records_fail"), "int",
              ["k1.s0.n_records", "k1.s0.n_records_fail"], "difference")

    pe = Csv("results/tables/k1_predicted_effects_f2/k1_predicted_effect_sizes.csv")
    r = pe.row(process="cubic", pipeline="saai", quantity="mean_diffusion")
    R.cell("KonePredContrastDiff", "k1.pred.cubic_saai.mean_diffusion", r, "effect", "dec6",
           "|predicted contrast| of mean diffusion, cubic/saai")
    R.derived("KoneSzeroFailShare", "k1.s0.fail_share",
              abs(R.v("k1.s0.cubic_saai.during.mean_diffusion.bias")) / R.v("k1.pred.cubic_saai.mean_diffusion"),
              "pct1", ["k1.s0.cubic_saai.during.mean_diffusion.bias", "k1.pred.cubic_saai.mean_diffusion"],
              "|bias| / |predicted contrast|")
    R.derived("KoneSzeroTolShare", "k1.s0.tol_share",
              R.v("k1.s0.cubic_saai.during.mean_diffusion.tol") / R.v("k1.pred.cubic_saai.mean_diffusion"),
              "pct0", ["k1.s0.cubic_saai.during.mean_diffusion.tol", "k1.pred.cubic_saai.mean_diffusion"],
              "tol / |predicted contrast|", "registered effect cap")
    for proc, P in (("ou", "Ou"), ("cubic", "Cubic")):
        r = pe.row(process=proc, pipeline="saai", quantity="xi_trap")
        R.cell(f"KonePredXi{P}", f"k1.pred.{proc}_saai.xi_trap", r, "effect", "dec2",
               "|predicted xi* contrast| (projection)")

    ver = Csv("results/tables/k1_k1c_f2/k1_gates_verdict.csv")
    r = ver.row(gate="S0", rule_set="limit")
    R.cell("KoneSzeroGatingRows", "k1.s0.n_gating_rows", r, "n_gating_rows", "int")
    R.cell("KoneSzeroPassRows", "k1.s0.n_pass_rows", r, "n_pass", "int")
    R.cell("KoneSzeroFailRows", "k1.s0.n_fail_rows", r, "n_fail", "int")
    R.cell("KoneSzeroNotEvalRows", "k1.s0.n_not_evaluable_rows", r, "n_not_evaluable", "int",
           "read defect (report-only rho1 rows read as record_inconsistent; marked F2-1 in the S0 SI table)")
    r = ver.row(gate="K1c", rule_set="limit/tol0")
    R.cell("KonecGatingRows", "k1.k1c.n_gating_rows", r, "n_gating_rows", "int")
    R.cell("KonecPassRows", "k1.k1c.n_pass_rows", r, "n_pass", "int")

    pc = Csv("results/tables/k1_s0_scope_f2/k1_s0_scope_precheck_units.csv")
    band = pc.row(cell_index=10, quantity="tau_mfpt", period="contrast").get("band")
    m = re.fullmatch(r"\[([0-9.]+), ([0-9.]+)\]", str(band))
    if not m:
        raise SourceError(f"precheck band {band!r} unparsable")
    R._put("KonePrecheckBandLo", "k1.precheck.band.lo", float(m.group(1)), "dec2", pc.path,
           "[cell_index=10;quantity=tau_mfpt] col=band", "regex parse of '[lo, hi]'", "")
    R._put("KonePrecheckBandHi", "k1.precheck.band.hi", float(m.group(2)), "dec2", pc.path,
           "[cell_index=10;quantity=tau_mfpt] col=band", "regex parse of '[lo, hi]'", "")
    R.cell("KonePrecheckCellTenTau", "k1.precheck.cell10.tau", pc.row(cell_index=10, quantity="tau_mfpt",
           period="contrast"), "coverage", "dec3", "K1b scope precheck failure that blocks K1a/K1b")
    R.cell("KonePrecheckKonecXi", "k1.precheck.cell23.xi", pc.row(cell_index=23, quantity="xi_trap",
           period="contrast"), "coverage", "dec3")
    R.cell("KonePrecheckKonecTau", "k1.precheck.cell23.tau", pc.row(cell_index=23, quantity="tau_mfpt",
           period="contrast"), "coverage", "dec3")
    g = pc.rows(status="gated")
    R._put("KonePrecheckGated", "k1.precheck.n_gated", float(len(g)), "int", pc.path,
           "[status=gated]", "count of rows", "")
    R._put("KonePrecheckOutside", "k1.precheck.n_outside", float((~g["in_band"].astype(bool)).sum()), "int",
           pc.path, "[status=gated;in_band=False]", "count of rows", "")

    gates = Csv("results/tables/k1_k1c_f2/k1_gates.csv")
    power = Csv("results/tables/k1_k1c_f2/k1_power.csv")
    names = {23: "A", 24: "B", 25: "C"}
    mech = {23: "a", 24: "d", 25: "e"}
    for cell, L in names.items():
        for pipe, Pp, gated in (("saai", "", True), ("direct", "Direct", False)):
            base = dict(gate="K1c", rule_set="limit/tol0", cell_index=float(cell), pipeline=pipe)
            r = gates.row(**base, statistic="V1_relative_bias_xi_shift")
            R.cell(f"Konec{Pp}RelBias{L}", f"k1.k1c.cell{cell}.{pipe}.rel_bias", r, "value", "sig2",
                   f"cell {cell} (v) x ({mech[cell]}); {'GATED' if gated else 'report only'}")
            r = gates.row(**base, statistic="V1_ci95_coverage_xi_shift")
            R.cell(f"Konec{Pp}Cov{L}", f"k1.k1c.cell{cell}.{pipe}.coverage", r, "value", "dec3",
                   f"cell {cell}; {'GATED' if gated else 'report only'}")
            r = gates.row(**base, statistic="V1_power_ci_excludes_0")
            R.cell(f"Konec{Pp}Power{L}", f"k1.k1c.cell{cell}.{pipe}.power", r, "value", "dec3",
                   f"cell {cell}; report only")
        r = power.row(cell_index=cell, pipeline="saai", variant="V1", period="contrast",
                      quantity="xi_trap", ci_nan_rule="tol0")
        R.cell(f"KonecTruth{L}", f"k1.k1c.cell{cell}.truth_median", r, "median_truth", "sdec2",
               f"median true xi* shift, cell {cell}")
    r = gates.row(gate="K1c", rule_set="limit/tol0", cell_index=23.0, pipeline="saai",
                  statistic="V1_relative_bias_xi_shift")
    R.cell("KonecBiasThreshold", "k1.k1c.threshold.rel_bias", r, "threshold", "dec2")
    r = gates.row(gate="K1c", rule_set="limit/tol0", cell_index=23.0, pipeline="saai",
                  statistic="V1_ci95_coverage_xi_shift")
    R.cell("KonecCovThreshold", "k1.k1c.threshold.coverage", r, "threshold", "dec2")
    keys = [f"k1.k1c.cell{c}.saai" for c in names]
    for stat, Mn, spec in (("rel_bias", "RelBias", "sig2"), ("coverage", "Cov", "dec3"),
                           ("power", "Power", "dec3")):
        vals = [R.v(f"{k}.{stat}") for k in keys]
        ins = [f"{k}.{stat}" for k in keys]
        R.derived(f"Konec{Mn}Min", f"k1.k1c.{stat}.min", min(vals), spec, ins, "min over cells 23-25")
        R.derived(f"Konec{Mn}Max", f"k1.k1c.{stat}.max", max(vals), spec, ins, "max over cells 23-25")
    truths = [R.v(f"k1.k1c.cell{c}.truth_median") for c in names]
    tins = [f"k1.k1c.cell{c}.truth_median" for c in names]
    R.derived("KonecTruthMin", "k1.k1c.truth.most_negative", min(truths), "sdec2", tins, "min")
    R.derived("KonecTruthMax", "k1.k1c.truth.least_negative", max(truths), "sdec2", tins, "max")
    rr = Json("results/tables/k1_k1c_f2/k1_k1c_run_record.json")
    R.json("KoneR", "k1.R", rr, "R", "int", "replicates per cell")
    R.json("KoneB", "k1.B", rr, "B", "int", "bootstrap draws per replicate")

    ex = Csv("results/tables/k1_exploratory/k1_exploratory_k1a.csv")
    tag = "EXPLORATORY, not a verdict (amendment 11); SI only"
    for outc, Mn in (("exploratory:pass", "Pass"), ("exploratory:fail", "Fail"),
                     ("exploratory:fail_(non-finite)", "FailNonfinite"),
                     ("exploratory:indistinguishable_from_null", "Indist")):
        R._put(f"KoneaExpl{Mn}", f"k1.k1a.expl.n_{Mn.lower()}",
               float((ex.df["exploratory_outcome"] == outc).sum()), "int", ex.path,
               f"[exploratory_outcome={outc}]", "count of rows", tag)
    R._put("KoneaExplRows", "k1.k1a.expl.n_rows", float(len(ex.df)), "int", ex.path, "all rows",
           "count of rows", tag)
    cellL = {3: "Three", 4: "Four", 8: "Eight", 9: "Nine"}
    for c, L in cellL.items():
        base = dict(pipeline="saai", null_variant="registered", cell_index=c)
        r = ex.row(**base, row="projection", statistic="projection:xi_trap_rel")
        R.cell(f"KoneaXiRelErr{L}", f"k1.k1a.cell{c}.xi_rel", r, "value", "dec3", tag)
        r = ex.row(**base, row="projection", statistic="projection:rho1_rel")
        R.cell(f"KoneaRhoRelErr{L}", f"k1.k1a.cell{c}.rho1_rel", r, "value", "dec2", tag)
        R.cell(f"KoneaRhoPred{L}", f"k1.k1a.cell{c}.rho1_pred", r, "median_pred", "ssig2", tag)
        R.cell(f"KoneaRhoSim{L}", f"k1.k1a.cell{c}.rho1_sim", r, "median_sim", "ssig2", tag)
        r = ex.row(**base, row="zeroth_order", statistic="zeroth_order:xi_trap_rel")
        R.cell(f"KoneaZerothXiRelErr{L}", f"k1.k1a.cell{c}.zeroth_xi_rel", r, "value", "dec3", tag)
        r = ex.row(**base, row="zeroth_order_mbar", statistic="zeroth_order_mbar:xi_trap_rel")
        R.cell(f"KoneaZerothMbarXiRelErr{L}", f"k1.k1a.cell{c}.zeroth_mbar_xi_rel", r, "value", "dec3",
               tag + "; old M-bar/|a1| row, report only")
    for c, L in ((3, "Three"), (4, "Four")):
        r = ex.row(pipeline="saai", null_variant="registered", cell_index=c, row="closed_form",
                   statistic="closed_form:xi_trap_rel")
        R.cell(f"KoneaClosedXiRelErr{L}", f"k1.k1a.cell{c}.closed_xi_rel", r, "value", "dec3",
               tag + "; closed form judged on OU only")
    xs = [R.v(f"k1.k1a.cell{c}.xi_rel") for c in cellL]
    R.derived("KoneaXiRelErrMin", "k1.k1a.xi_rel.min", min(xs), "dec2",
              [f"k1.k1a.cell{c}.xi_rel" for c in cellL], "min over cells 3,4,8,9", tag)
    R.derived("KoneaXiRelErrMax", "k1.k1a.xi_rel.max", max(xs), "dec2",
              [f"k1.k1a.cell{c}.xi_rel" for c in cellL], "max over cells 3,4,8,9", tag)
    sign = ex.rows(pipeline="saai", null_variant="registered", row="projection",
                   statistic="projection:rho1_sign")
    R._put("KoneaRhoSignFail", "k1.k1a.rho1_sign.n_fail",
           float((sign["exploratory_outcome"] == "exploratory:fail").sum()), "int", ex.path,
           "[pipeline=saai;null_variant=registered;row=projection;statistic=projection:rho1_sign;"
           "exploratory_outcome=exploratory:fail]", "count of rows", tag + "; the rho1 sign is wrong")
    R._put("KoneaRhoSignCells", "k1.k1a.rho1_sign.n_cells", float(len(sign)), "int", ex.path,
           "[pipeline=saai;null_variant=registered;row=projection;statistic=projection:rho1_sign]",
           "count of rows", tag)

    kb = Csv("results/tables/k1_exploratory/k1_exploratory_k1b.csv")
    for outc, Mn in (("exploratory:pass", "Pass"), ("exploratory:fail", "Fail"),
                     ("exploratory:report_not_evaluable_amendment_7", "NotEval")):
        R._put(f"KonebExpl{Mn}", f"k1.k1b.expl.n_{Mn.lower()}",
               float((kb.df["exploratory_outcome"] == outc).sum()), "int", kb.path,
               f"[exploratory_outcome={outc}]", "count of rows", tag)
    R._put("KonebExplRows", "k1.k1b.expl.n_rows", float(len(kb.df)), "int", kb.path, "all rows",
           "count of rows", tag)
    ev = kb.rows(pipeline="saai", ci_nan_rule="tol0", would_gate_if_registered=True)
    R._put("KonebEvalRows", "k1.k1b.eval.n_rows", float(len(ev)), "int", kb.path,
           "[pipeline=saai;ci_nan_rule=tol0;would_gate_if_registered=True]", "count of rows", tag)
    R._put("KonebEvalAboveNinety", "k1.k1b.eval.n_ge_090", float((ev["value"] >= 0.90).sum()), "int",
           kb.path, "[same filter] value>=0.90", "count of rows", tag)
    for q, Mn in (("xi_trap", "Xi"), ("tau_mfpt", "Tau")):
        eq = ev[ev["quantity"] == q]
        R._put(f"KonebEval{Mn}Rows", f"k1.k1b.eval.{q}.n_rows", float(len(eq)), "int", kb.path,
               f"[same filter;quantity={q}]", "count of rows", tag)
        R._put(f"KonebEval{Mn}AboveNinety", f"k1.k1b.eval.{q}.n_ge_090", float((eq["value"] >= 0.90).sum()),
               "int", kb.path, f"[same filter;quantity={q}] value>=0.90", "count of rows", tag)
    if len(ev) != len(ev[ev["quantity"].isin(["xi_trap", "tau_mfpt"])]):
        raise SourceError(f"{kb.path}: evaluable K1b rows other than xi_trap/tau_mfpt")
    pv = ev[ev["exploratory_outcome"] == "exploratory:pass"]["value"]
    R._put("KonebPassCovMin", "k1.k1b.pass_cov.min", float(pv.min()), "dec3", kb.path,
           "[same filter;exploratory_outcome=exploratory:pass] col=value", "min over rows", tag)
    R._put("KonebPassCovMax", "k1.k1b.pass_cov.max", float(pv.max()), "dec3", kb.path,
           "[same filter;exploratory_outcome=exploratory:pass] col=value", "max over rows", tag)
    r = kb.row(pipeline="saai", ci_nan_rule="tol0", cell_index=12, quantity="xi_trap")
    R.cell("KonebCellTwelveCov", "k1.k1b.cell12.xi_cov", r, "value", "dec3",
           tag + "; the one failing evaluable row, must be reported")

    ctl = Csv("results/tables/k1_exploratory/k1_exploratory_controls_k1a.csv")
    tagc = "report-only control, exploratory, not a verdict (amendment 11); SI only"
    R._put("KoneCtlRows", "k1.controls.n_rows", float(len(ctl.df)), "int", ctl.path, "all rows",
           "count of rows", tagc)
    for outc, Mn in (("exploratory:pass", "Pass"), ("exploratory:fail", "Fail"),
                     ("exploratory:fail_(non-finite)", "FailNonfinite"),
                     ("exploratory:indistinguishable_from_null", "Indist")):
        R._put(f"KoneCtl{Mn}", f"k1.controls.n_{Mn.lower()}",
               float((ctl.df["exploratory_outcome"] == outc).sum()), "int", ctl.path,
               f"[exploratory_outcome={outc}]", "count of rows", tagc)
    sn = Csv("results/tables/k1_exploratory/k1_exploratory_shocked_null.csv")
    R._put("KoneShockedNullRows", "k1.shocked_null.n_rows", float(len(sn.df)), "int", sn.path,
           "all rows", "count of rows", tag + "; reported as run")


def us():
    tag_g = "GUARDED pooled projection (guard: 1-99% support of V1 xi, slope floor -|a1|)"
    h3b = Csv("results/tables/k2/k2_h3b.csv")
    specs = [("pi", "main", "overlap", "PiMain"), ("pi", "main", "code", "PiMainCode"),
             ("pi", "secondary", "overlap", "PiSec"), ("gdp", "main", "overlap", "GdpMain"),
             ("gdp", "secondary", "overlap", "GdpSec")]
    for outc, per, link, S in specs:
        r = h3b.row(scope="metro", outcome=outc, period=per, link=link)
        k = f"us.{outc}.{per}.{link}"
        R.cell(f"Us{S}Obs", f"{k}.obs", r, "observed", "sdec4", "observed xi*(V0)-xi*(V1), Panel B")
        R.cell(f"Us{S}HalfWidth", f"{k}.halfwidth", r, "half_width_V1", "dec4",
               "half width of the V1 xi* 95% CI (lineage, tol0)")
        R.derived(f"Us{S}Gate", f"{k}.gate", 0.5 * R.v(f"{k}.halfwidth"), "dec4",
                  [f"{k}.halfwidth"], "0.5 * half width (registered resolution gate)")
        R.cell(f"Us{S}Zeroth", f"{k}.zeroth", r, "zeroth_order_xi_trap", "sdec4",
               "zeroth-order M-bar/|a1|, report only")
        R.cell(f"Us{S}Mbar", f"{k}.M_bar", r, "M_bar", "ssig2", "n_S-weighted mechanical term")
        R.cell(f"Us{S}Aone", f"{k}.a1_V1", r, "a1_V1", "sdec4", "a1 of V1 (OLS rho1-1), Panel B")
        R.derived(f"Us{S}Amp", f"{k}.inv_abs_a1", 1 / abs(R.v(f"{k}.a1_V1")), "dec0",
                  [f"{k}.a1_V1"], "1/|a1|", "amplification factor")
        R.cell(f"Us{S}Verdict", f"{k}.verdict", r, "verdict", "str")
        st = str(r.get("status"))
        if st == "ok":
            R.cell(f"Us{S}Pred", f"{k}.pred", r, "pred_xi_trap", "sdec4", tag_g)
            R.cell(f"Us{S}PredRho", f"{k}.pred_rho1", r, "pred_rho1", "ssig2", tag_g)
        else:
            mm = re.search(r"stationary mass ([0-9.eE+-]+)", st)
            if not mm:
                raise SourceError(f"{h3b.path}: cannot parse guarded stationary mass for {k}")
            R._put(f"Us{S}GuardMass", f"{k}.guarded_mass", float(mm.group(1)), "sig3", h3b.path,
                   r.where("status"), "regex 'stationary mass (x)'",
                   "guarded projection refused: latent chain not confined")
        us_ = str(r.get("unguarded_status"))
        mm = re.search(r"stationary mass ([0-9.eE+-]+)", us_)
        if not mm:
            raise SourceError(f"{h3b.path}: cannot parse unguarded stationary mass for {k}")
        R._put(f"Us{S}UnguardMass", f"{k}.unguarded_mass", float(mm.group(1)), "sig3", h3b.path,
               r.where("unguarded_status"), "regex 'stationary mass (x)'",
               "UNGUARDED projection refused in every US spec (guard disclosure)")
    k = "us.pi.main.overlap"
    R.derived("UsPiMainErr", f"{k}.abs_err", abs(R.v(f"{k}.obs") - R.v(f"{k}.pred")), "dec4",
              [f"{k}.obs", f"{k}.pred"], "|obs - pred|")
    meta = Json("results/tables/k2br_meta.json")
    R.json("HthreebRelTol", "h3b.rel_tol", meta, "constants.H3B_REL_TOL", "dec2",
           "registered H3b tolerance 0.30*|pred| (same for the US and Brazil)")
    R.json("HthreebGateFactor", "h3b.gate_factor", meta, "constants.GATE_FACTOR", "dec1",
           "gate = factor * V1 half width (same for the US and Brazil)")
    R.derived("UsPiMainAllow", f"{k}.allowance", R.v("h3b.rel_tol") * abs(R.v(f"{k}.pred")), "dec4",
              ["h3b.rel_tol", f"{k}.pred"], "0.30 * |pred|")
    us_unguarded = h3b.rows(scope="metro")["unguarded_status"].astype(str)
    R._put("UsUnguardedRefused", "us.n_specs_unguarded_refused",
           float(us_unguarded.str.startswith("projection_undefined").sum()), "int", h3b.path,
           "[scope=metro] col=unguarded_status startswith projection_undefined", "count of rows",
           "metro specs (incl. code link) where the unguarded projection is undefined")
    R._put("UsSpecsMetro", "us.n_specs_metro", float(len(h3b.rows(scope="metro"))), "int", h3b.path,
           "[scope=metro]", "count of rows", "")
    R._put("UsSpecsRegistered", "us.n_specs_registered", float(len(h3b.rows(scope="metro", link="overlap"))),
           "int", h3b.path, "[scope=metro;link=overlap]", "count of rows",
           "registered specifications (metro, overlap link, PI/GDP x main/secondary)")
    R._put("UsSpecsCodeLink", "us.n_specs_metro_code", float(len(h3b.rows(scope="metro", link="code"))),
           "int", h3b.path, "[scope=metro;link=code]", "count of rows", "report-only code-link variants")

    obs = Csv("results/tables/k2/k2_observables.csv")
    for panel, link, var, P in (("A", "none", "V0", "A"), ("B", "overlap", "V0", "VzeroB"),
                                ("B", "overlap", "V1", "VoneB")):
        r = obs.row(scope="metro", outcome="pi", period="main", panel=panel, link=link, variant=var,
                    conv="limit", quantity="xi_trap")
        R.cell(f"UsPiMainXi{P}", f"us.pi.main.xi.{P}", r, "estimate", "sdec3", "attractor level")
        if P == "VoneB":
            R.cell("UsPiMainXiVoneBLo", "us.pi.main.xi.VoneB.lo", r, "lineage_tol0_lo", "sdec4")
            R.cell("UsPiMainXiVoneBHi", "us.pi.main.xi.VoneB.hi", r, "lineage_tol0_hi", "sdec4")
    for outc, O in (("pi", "Pi"), ("gdp", "Gdp")):
        r = obs.row(scope="metro", outcome=outc, period="main", panel="B", link="overlap", variant="V0",
                    conv="limit", quantity="rho1")
        R.cell(f"Us{O}MainRhoVzero", f"us.{outc}.main.rho1.V0B", r, "estimate", "dec3", "persistence rho1")
        r = obs.row(scope="metro", outcome=outc, period="main", panel="B", link="overlap", variant="V1",
                    conv="limit", quantity="rho1")
        R.cell(f"Us{O}MainRhoVone", f"us.{outc}.main.rho1.V1B", r, "estimate", "dec3",
               "persistence rho1 under V1 (1 + a1 of V1)")
    rho = obs.col("estimate", scope="metro", outcome="gdp", conv="limit", quantity="rho1")
    R.agg("UsGdpRhoMin", "us.gdp.rho1.min", rho, obs, "estimate", "min", "dec3", "all metro GDP panels/variants")
    R.agg("UsGdpRhoMax", "us.gdp.rho1.max", rho, obs, "estimate", "max", "dec3", "all metro GDP panels/variants")

    pdiff = Csv("results/tables/k2/k2_paired_differences.csv")
    want = [("pi", "main", "V0-V1", "xi_trap", "PiMainVzeroVoneXi", "sdec3"),
            ("pi", "main", "V0(B)-A", "xi_trap", "PiMainVzeroAXi", "sdec3"),
            ("pi", "main", "V1(B)-A", "xi_trap", "PiMainHfourXi", "sdec3"),
            ("pi", "main", "V1(B)-A", "rho1", "PiMainHfourRho", "ssig2"),
            ("pi", "main", "V0-V1", "rho1", "PiMainHthreecRho", "ssig2"),
            ("pi", "main", "V1-V1d", "rho1", "PiMainDefRho", "ssig2"),
            ("pi", "secondary", "V1(B)-A", "xi_trap", "PiSecHfourXi", "sdec3"),
            ("pi", "secondary", "V1(B)-A", "rho1", "PiSecHfourRho", "ssig2"),
            ("pi", "secondary", "V1-V1d", "rho1", "PiSecDefRho", "ssig2"),
            ("gdp", "main", "V1(B)-A", "xi_trap", "GdpMainHfourXi", "sdec3"),
            ("gdp", "main", "V1(B)-A", "rho1", "GdpMainHfourRho", "ssig2"),
            ("gdp", "main", "V1-V1d", "rho1", "GdpMainDefRho", "ssig2"),
            ("gdp", "main", "V1d(B)-A", "rho1", "GdpMainVonedARho", "ssig2"),
            ("gdp", "secondary", "V1(B)-A", "xi_trap", "GdpSecHfourXi", "sdec3"),
            ("gdp", "secondary", "V1(B)-A", "rho1", "GdpSecHfourRho", "ssig2"),
            ("gdp", "secondary", "V1-V1d", "rho1", "GdpSecDefRho", "ssig2")]
    for outc, per, con, q, M, spec in want:
        r = pdiff.row(scope="metro", outcome=outc, period=per, link="overlap", contrast=con,
                      conv="limit", quantity=q)
        k = f"us.{outc}.{per}.{con}.{q}"
        R.cell(f"Us{M}", f"{k}.est", r, "estimate", spec, "paired difference, Panel B / A")
        R.cell(f"Us{M}Lo", f"{k}.lo", r, "lineage_tol0_lo", spec, "lineage block, tol0, limit (primary)")
        R.cell(f"Us{M}Hi", f"{k}.hi", r, "lineage_tol0_hi", spec, "lineage block, tol0, limit (primary)")
        if (outc, per, con, q) == ("pi", "main", "V0-V1", "xi_trap"):
            R.cell("UsPiMainVzeroVoneXiStateLo", f"{k}.state.lo", r, "state_tol0_lo", spec,
                   "state block (sensitivity); quote with the lineage CI")
            R.cell("UsPiMainVzeroVoneXiStateHi", f"{k}.state.hi", r, "state_tol0_hi", spec,
                   "state block (sensitivity); quote with the lineage CI")

    v = Csv("results/tables/k2/k2_verdicts.csv")
    r = v.row(hypothesis="H3c", scope="metro", outcome="pi", period="main", link="overlap")
    R.cell("UsPiMainHthreecPred", "us.pi.main.h3c.pred", r, "predicted", "ssig2", tag_g)
    for outc, per, S in (("pi", "main", "PiMain"), ("pi", "secondary", "PiSec"),
                         ("gdp", "main", "GdpMain"), ("gdp", "secondary", "GdpSec")):
        r = v.row(hypothesis="H4", scope="metro", outcome=outc, period=per, link="overlap")
        R.cell(f"Us{S}HfourVerdict", f"us.{outc}.{per}.h4.verdict", r, "verdict", "str")
    h1 = v.rows(hypothesis="H1", registered=True)
    R._put("UsHoneMaxResid", "us.h1.max_resid", float(h1["value"].max()), "sci2", v.path,
           "[hypothesis=H1;registered=True] col=value", "max over rows", "point estimates")
    km = Json("results/tables/k2/k2_meta.json")
    R.json("UsHoneMaxResidBoot", "us.h1.max_resid_boot", km, "h1_max_bootstrap", "sci2",
           "maximum over all bootstrap draws")
    R.json("UsSeed", "us.seed", km, "registration.seed", "str")
    R.json("UsB", "us.n_boot", km, "n_boot", "int")
    h2 = Csv("results/tables/k2/k2_h2.csv")
    R.agg("UsHtwoMaxDiff", "us.h2.max_pair_diff", h2.col("max_pair_diff_V0_V1"), h2,
          "max_pair_diff_V0_V1", "max", "dec0", "V0 vs V1 pairs on Panel A")

    cov = Csv("results/tables/k2_coverage_panelB_summary.csv")
    for per, P in (("2003-2024", "Main"), ("1993-2024", "Sec")):
        r = cov.row(scope="metro", link_rule="overlap", period=per)
        R.cell(f"UsDose{P}", f"us.dose.{per}", r, "nS_weighted_mech_bound_per_unit_m", "dec3",
               "n_S-weighted (n_E+n_L)/n_S, metro, overlap link")
        R.cell(f"UsDoseMax{P}", f"us.dose_max.{per}", r, "max_mech_bound_per_unit_m", "dec3",
               "largest single-transition (n_E+n_L)/n_S")
    R.derived("UsDoseTimesAmp", "us.pi.main.dose_x_amp",
              R.v("us.dose.2003-2024") * R.v("us.pi.main.overlap.inv_abs_a1"), "dec2",
              ["us.dose.2003-2024", "us.pi.main.overlap.inv_abs_a1"], "dose * 1/|a1|",
              "dose-ladder abscissa, US personal income main")

    lin = Csv("results/tables/k2/k2_lineage_summary.csv")
    r = lin.row(scope="metro")
    R.cell("UsLineages", "us.metro.n_lineages", r, "n_lineages", "int")
    R.cell("UsStates", "us.metro.n_states", r, "n_states", "int", "state blocks (not 49+DC)")
    R.cell("UsAreasA", "us.metro.n_panelA", r, "n_panelA_areas", "int")
    beta = Csv("results/tables/k2/k2_beta_by_year.csv")
    b = beta.col("beta", scope="metro")
    R.agg("UsBetaMin", "us.metro.beta.min", b, beta, "beta", "min", "dec3")
    R.agg("UsBetaMax", "us.metro.beta.max", b, beta, "beta", "max", "dec3")


def brazil():
    tag_g = "GUARDED pooled projection"
    ver = Csv("results/tables/k2br_verdicts.csv")
    pred = Csv("results/tables/k2br_predictions.csv")
    con = Csv("results/tables/k2br_contrasts.csv")
    cells = [("ref2010", "pib", "TenPib"), ("ref2010", "vab_agro", "TenVab"),
             ("ref2002", "pib", "TwoPib"), ("ref2002", "vab_agro", "TwoVab")]
    for w, o, C in cells:
        k = f"br.{w}.{o}"
        r = ver.row(window=w, outcome=o)
        R.cell(f"Br{C}HoneResid", f"{k}.h1.max_resid", r, "H1_max_residual", "sci2")
        R.cell(f"Br{C}Obs", f"{k}.h3b.obs", r, "H3b_observed", "ssig3", "observed xi*(V0)-xi*(V1), Panel B")
        R.cell(f"Br{C}Pred", f"{k}.h3b.pred", r, "H3b_predicted", "ssig3", tag_g)
        R.cell(f"Br{C}Unguarded", f"{k}.h3b.unguarded", r, "H3b_unguarded_predicted", "ssig3",
               "UNGUARDED projection, report only (SI)")
        R.cell(f"Br{C}Zeroth", f"{k}.h3b.zeroth", r, "H3b_zeroth_order_predicted", "ssig3",
               "zeroth order M-bar/|a1|, report only")
        R.cell(f"Br{C}Gate", f"{k}.h3b.gate", r, "H3b_gate_threshold", "sig3")
        R.cell(f"Br{C}HthreebVerdict", f"{k}.h3b.verdict", r, "H3b", "str")
        R.cell(f"Br{C}HthreecEst", f"{k}.h3c.est", r, "H3c_estimate", "ssig2", "rho1(V0)-rho1(V1)")
        R.cell(f"Br{C}HthreecLo", f"{k}.h3c.lo", r, "H3c_ci_lo", "ssig2")
        R.cell(f"Br{C}HthreecHi", f"{k}.h3c.hi", r, "H3c_ci_hi", "ssig2")
        R.cell(f"Br{C}HthreecPred", f"{k}.h3c.pred", r, "H3c_predicted_projection", "ssig2", tag_g)
        R.cell(f"Br{C}HfourVerdict", f"{k}.h4.verdict", r, "H4", "str")
        R.cell(f"Br{C}HfourImmediate", f"{k}.h4.immediate", r, "H4_immediate", "str")
        R.cell(f"Br{C}HfourUf", f"{k}.h4.uf", r, "H4_uf", "str")
        p = pred.row(window=w, outcome=o)
        R.cell(f"Br{C}Mbar", f"{k}.M_bar", p, "M_bar", "ssig2")
        R.cell(f"Br{C}Aone", f"{k}.a1_V1", p, "a1_V1_ols", "sdec4", "a1 of V1 (OLS)")
        R.derived(f"Br{C}Amp", f"{k}.inv_abs_a1", 1 / abs(R.v(f"{k}.a1_V1")), "dec0",
                  [f"{k}.a1_V1"], "1/|a1|")
        R.cell(f"Br{C}GuardLo", f"{k}.guard.lo", p, "guard_lo", "sdec2", "support of the projection guard, SI")
        R.cell(f"Br{C}GuardHi", f"{k}.guard.hi", p, "guard_hi", "sdec2", "support of the projection guard, SI")
        R.cell(f"Br{C}GuardSlope", f"{k}.guard.slope", p, "guard_slope_floor", "sdec4",
               "boundary slope floor -|a1| of the projection guard, SI")
        contr = [("V0B_minus_V1B", "VzeroVone"), ("V1B_minus_A", "Hfour"), ("V0B_minus_A", "VzeroA"),
                 ("V1pB_minus_A", "VonepA"), ("V1mB_minus_A", "VonemA"), ("V1B_minus_V1mB", "VoneVonem")]
        for cname, Cn in contr:
            for q, Q in (("xi_trap", "Xi"), ("rho1", "Rho")):
                rr = con.row(window=w, outcome=o, contrast=cname, convention="limit", quantity=q)
                kk = f"{k}.{cname}.{q}"
                note = "AMC block, tol0, limit" + ("; report-only variant" if Cn in ("VonepA", "VonemA", "VoneVonem", "VzeroA") else "")
                R.cell(f"Br{C}{Cn}{Q}", f"{kk}.est", rr, "estimate", "ssig2", note)
                R.cell(f"Br{C}{Cn}{Q}Lo", f"{kk}.lo", rr, "amc_tol0_lo", "ssig2", note)
                R.cell(f"Br{C}{Cn}{Q}Hi", f"{kk}.hi", rr, "amc_tol0_hi", "ssig2", note)
    R.derived("BrTwoPibGuardRatio", "br.ref2002.pib.guard_over_unguard",
              R.v("br.ref2002.pib.h3b.pred") / R.v("br.ref2002.pib.h3b.unguarded"), "dec0",
              ["br.ref2002.pib.h3b.pred", "br.ref2002.pib.h3b.unguarded"], "ratio guarded/unguarded",
              "two orders of magnitude; guard disclosure")

    rr = con.row(window="ref2002", outcome="pib", contrast="V1B_minus_A", convention="limit", quantity="xi_trap")
    R.cell("BrTwoPibHfourXiImmLo", "br.ref2002.pib.V1B_minus_A.xi_trap.immediate.lo", rr, "immediate_tol0_lo",
           "ssig2", "immediate-region block (sensitivity)")
    R.cell("BrTwoPibHfourXiImmHi", "br.ref2002.pib.V1B_minus_A.xi_trap.immediate.hi", rr, "immediate_tol0_hi",
           "ssig2", "immediate-region block (sensitivity)")

    dec = Csv("results/tables/k2br_decomposition.csv")
    r = dec.row(window="ref2002", outcome="pib", panel="B", family="V0", year_t=2000)
    R.cell("BrTwoPibBigNE", "br.ref2002.pib.t2000.n_E", r, "n_E", "int", "2000->01 entrants")
    R.cell("BrTwoPibBigNS", "br.ref2002.pib.t2000.n_S", r, "n_S", "int")
    R.cell("BrTwoPibBigME", "br.ref2002.pib.t2000.m_E", r, "m_E", "dec2")
    R.cell("BrTwoPibBigM", "br.ref2002.pib.t2000.M", r, "M", "ssig2")

    cov = Csv("results/tables/k2br_coverage_summary.csv")
    for lab, W in (("GDP ref2002 1999-2012", "Two"), ("GDP ref2010 2002-2023", "Ten")):
        r = cov.row(dataset="Brazil municipalities (panel B)", window=lab)
        R.cell(f"BrDose{W}", f"br.dose.{lab}", r, "nS_weighted_comp_shock", "sig2",
               "n_S-weighted (n_E+n_L)/n_S")
        R.cell(f"BrDoseMax{W}", f"br.dose_max.{lab}", r, "max_comp_shock", "sig2")
    R.derived("BrDoseTimesAmpTen", "br.ref2010.pib.dose_x_amp",
              R.v("br.dose.GDP ref2010 2002-2023") * R.v("br.ref2010.pib.inv_abs_a1"), "sig2",
              ["br.dose.GDP ref2010 2002-2023", "br.ref2010.pib.inv_abs_a1"], "dose * 1/|a1|")
    R.derived("BrDoseTimesAmpTwo", "br.ref2002.pib.dose_x_amp",
              R.v("br.dose.GDP ref2002 1999-2012") * R.v("br.ref2002.pib.inv_abs_a1"), "sig2",
              ["br.dose.GDP ref2002 1999-2012", "br.ref2002.pib.inv_abs_a1"], "dose * 1/|a1|")

    blk = Csv("results/tables/k2br_coverage_blocks.csv")
    for idx, M in (("uf.n_blocks", "BrStates"),):
        r = blk.row(**{"Unnamed: 0": idx}) if "Unnamed: 0" in blk.df.columns else blk.row(**{blk.df.columns[0]: idx})
        R.cell(M, f"br.{idx}", r, "value", "int")
    yr = Csv("results/tables/k2br_yearly.csv")
    b = yr.col("beta", outcome="pib")
    R.agg("BrBetaMin", "br.pib.beta.min", b, yr, "beta", "min", "dec3")
    R.agg("BrBetaMax", "br.pib.beta.max", b, yr, "beta", "max", "dec3")


def diagnostics():
    """Report-only diagnostics.

    US xi* stopping noise (summary of the committed literal-draw check; largest OBSERVED, not a bound).
    Brazilian xi* optimizer-tolerance noise (warm / cold / tight refits of fixed AMC draws).
    Complete-panel (no exit) finite-province rho1 term vs the predicted K1a rho1 contrast; NOT a bound
         for the observed panel with exits (not modelled), so no bound is printed.
    """
    lit = Csv("results/tables/k2_literal_noise_summary.csv")
    r = lit.row(level="overall", quantity="xi_trap")
    R.cell("UsLiteralXiMax", "us.literal.xi_trap.max", r, "max_abs_diff", "sci2",
           "largest observed |diff| of xi* over all 696 literal-check rows (3 draws per cell); not a bound")
    R.cell("UsLiteralDraws", "us.literal.n_draws", r, "n_draws", "int", "draws per cell in the literal check")
    r = lit.row(level="primary", quantity="xi_trap")
    R.cell("UsLiteralXiMaxPrimary", "us.literal.xi_trap.max.primary", r, "max_abs_diff", "sci2",
           "largest observed |diff| of xi*, metro / lineage / limit rows")

    opt = Csv("results/tables/k2br_optimizer_noise_summary.csv")
    cols = ["max_abs_warm_cold", "max_abs_warm_tight", "max_abs_cold_tight"]
    for w, W in (("all", "All"), ("ref2010", "Ten"), ("ref2002", "Two")):
        r = opt.row(window=w)
        for c in cols:
            R.cell(f"BrOpt{W}{''.join(x.title() for x in c.split('_')[2:])}", f"br.optnoise.{w}.{c}", r, c,
                   "sci2", "report only")
        R.derived(f"BrOptNoiseMax{W}", f"br.optnoise.{w}.max", max(R.v(f"br.optnoise.{w}.{c}") for c in cols),
                  "sci2", [f"br.optnoise.{w}.{c}" for c in cols], "max over the three refit pairs",
                  "largest observed xi* difference between warm, cold and tight-tolerance refits")
        R.cell(f"BrOptCells{W}", f"br.optnoise.{w}.n_cells", r, "n_cells", "int")
    R.cell("BrOptWarmCommitted", "br.optnoise.all.max_abs_warm_committed", opt.row(window="all"),
           "max_abs_warm_committed", "sci1", "warm-start refit vs committed draw value")
    for M, k in (("BrTwoPibHfourXiLo", "br.ref2002.pib.V1B_minus_A.xi_trap.lo"),
                 ("BrTwoVabHfourXiLo", "br.ref2002.vab_agro.V1B_minus_A.xi_trap.lo")):
        R.derived(f"{M}OverNoise", f"{k}.over_optnoise", R.v(k) / R.v("br.optnoise.ref2002.max"), "dec0",
                  [k, "br.optnoise.ref2002.max"], "ratio", "margin of the ref2002 H4 lower end over the noise")

    sh = Csv("results/tables/k1_rho1_term_share.csv")
    R.agg("KoneRhoTermShareMin", "k1.rho1_term.complete_panel.share.min", sh.col("share_abs_term"), sh,
          "share_abs_term", "min", "pct0", "complete panel, NOT a bound for the observed panel with exits")
    R.agg("KoneRhoTermShareMax", "k1.rho1_term.complete_panel.share.max", sh.col("share_abs_term"), sh,
          "share_abs_term", "max", "pct0", "complete panel, NOT a bound for the observed panel with exits")
    R.agg("KoneRhoTermHalfShareMin", "k1.rho1_term.complete_panel.share_half.min", sh.col("share_estimator_half"),
          sh, "share_estimator_half", "min", "pct0",
          "estimator's half of |e| (amendment 7), complete panel, NOT a bound")
    R.agg("KoneRhoTermHalfShareMax", "k1.rho1_term.complete_panel.share_half.max", sh.col("share_estimator_half"),
          sh, "share_estimator_half", "max", "pct0",
          "estimator's half of |e| (amendment 7), complete panel, NOT a bound")


def posthoc():
    tag = "POST-HOC diagnostic; report only"
    fr = Csv("results/tables/expi_identify_fractions.csv")
    R._put("CnIdFits", "cn.id.n_fits", float(fr.df["n_draws"].sum()), "int", fr.path, "all rows col=n_draws",
           "sum over rows", tag)
    R._put("CnIdUnidentified", "cn.id.n_not_identified", float(fr.df["n_not_identified"].sum()), "int",
           fr.path, "all rows col=n_not_identified", "sum over rows", tag)
    for ver, V in (("V0", "Vzero"), ("V1", "Vone")):
        for win, W_ in (("pre", "Pre"), ("during", "During")):
            r = fr.row(version=ver, window=win)
            R.cell(f"CnIdFrac{V}{W_}", f"cn.id.{ver}.{win}.frac_identified", r, "frac_identified", "dec3", tag)
    sh = fr.df["n_subgrid_path"] / fr.df["n_draws"]
    if sh.isna().any():
        raise SourceError(f"{fr.path}: short-path share not finite")
    R._put("CnIdShortShareMin", "cn.id.short_path_share.min", float(sh.min()), "pct1", fr.path,
           "all rows col=n_subgrid_path/n_draws", "min over rows", tag + "; all such draws are identified")
    R._put("CnIdShortShareMax", "cn.id.short_path_share.max", float(sh.max()), "pct1", fr.path,
           "all rows col=n_subgrid_path/n_draws", "max over rows", tag)
    r = fr.row(version="V0", window="during")
    R.cell("CnIdSubgridIdentifiedVzeroDuring", "cn.id.V0.during.n_subgrid_identified", r,
           "n_subgrid_and_identified", "int", tag)

    dr = Csv("results/tables/expi_identify_draws.csv.gz")
    st = "V0_baseline|root_status_during"
    xc = "V0_baseline|limit|xi_trap_during"
    dr._need([st, xc])
    bad = dr.df[dr.df[st] != "stable_zero"]
    if len(bad) != int(R.v("cn.id.n_not_identified")):
        raise SourceError(f"{dr.path}: {len(bad)} unidentified V0-during draws, fractions table says "
                          f"{R.v('cn.id.n_not_identified')}")
    R._put("CnIdPlaceholderMin", "cn.id.V0.during.placeholder.min", float(bad[xc].min()), "sdec2", dr.path,
           f"[{st}!=stable_zero] col={xc}", "min over rows", tag + "; argmin|A| placeholder")
    R._put("CnIdPlaceholderMax", "cn.id.V0.during.placeholder.max", float(bad[xc].max()), "sdec2", dr.path,
           f"[{st}!=stable_zero] col={xc}", "max over rows", tag)

    iv = Csv("results/tables/expi_identify_intervals.csv")
    vn = {"V0": "V0_baseline", "V1": "V1_stayer_restandardised"}
    n_mz = n_below = n_inside = n_above = 0
    for ver in ("V0", "V1"):
        for win in ("pre", "during"):
            ns, xc_ = f"{vn[ver]}|n_stable_{win}", f"{vn[ver]}|limit|xi_trap_{win}"
            fs = f"{vn[ver]}|first_stable_{win}"
            dr._need([ns, xc_, fs])
            m = dr.df[dr.df[ns] >= 2]
            if ((m[xc_] - m[fs]).abs() > 1e-12).any():
                raise SourceError(f"{dr.path}: multi-zero draws do not use the first stable zero ({ver} {win})")
            if not len(m):
                continue
            r = iv.row(estimand=f"xi_trap_{ver}_{win}", convention="limit", subset="all")
            lo_, hi_ = float(r.get("tol0_lo")), float(r.get("tol0_hi"))
            x = m[xc_]
            n_mz += len(m)
            n_below += int((x < lo_).sum())
            n_above += int((x > hi_).sum())
            n_inside += int(((x >= lo_) & (x <= hi_)).sum())
    if n_mz != int(Csv("results/tables/expi_identify_fractions.csv").df["n_multiple_stable_zeros"].sum()):
        raise SourceError("multi-zero draw count disagrees with expi_identify_fractions.csv")
    for Mn, val, how in (("", n_mz, "count n_stable>=2"), ("Below", n_below, "below all-draw tol0 interval"),
                         ("Inside", n_inside, "inside all-draw tol0 interval"),
                         ("Above", n_above, "above all-draw tol0 interval")):
        R._put(f"CnIdMultiZero{Mn}", f"cn.id.multizero.n{('_' + Mn.lower()) if Mn else ''}", float(val), "int",
               dr.path, "[*|n_stable_*>=2] col=*|limit|xi_trap_* vs expi_identify_intervals tol0 (all, limit)",
               how, tag + "; xi* = first (lowest) stable zero")
    for est, Mn in (("xi_trap_V0_during", "VzeroDuring"), ("xi_shift_V0_minus_V1", "Shift")):
        r = iv.row(estimand=est, convention="limit", subset="not_identified")
        n_ = float(r.get("n_subset"))
        for side in ("below", "inside", "above"):
            R._put(f"CnIdDrop{Mn}{side.capitalize()}", f"cn.id.dropped.{est}.n_{side}",
                   round(n_ * float(r.get(f"share_{side}_all_supp_interval"))), "int", iv.path,
                   r.where(f"n_subset*share_{side}_all_supp_interval"), "count", tag)
    for est, M in (("xi_shift_V0_minus_V1", "Shift"), ("xi_trap_V0_during", "VzeroDuring"),
                   ("xi_trap_V0_minus_V1_during", "DiffDuring")):
        for sub, S_ in (("all", "All"), ("identified", "Id")):
            r = iv.row(estimand=est, convention="limit", subset=sub)
            if str(r.get("tol0_status")) != "ok":
                raise SourceError(f"{iv.path}: {est} {sub} tol0 not ok")
            R.cell(f"CnId{M}{S_}Lo", f"cn.id.{est}.{sub}.tol0_lo", r, "tol0_lo", "sdec2", tag)
            R.cell(f"CnId{M}{S_}Hi", f"cn.id.{est}.{sub}.tol0_hi", r, "tol0_hi", "sdec2", tag)
    lim = iv.rows(convention="limit")
    est = sorted(set(lim["estimand"]))
    changed = 0
    for e in est:
        a = lim[(lim["estimand"] == e) & (lim["subset"] == "all")]["tol0_contains_null"]
        b = lim[(lim["estimand"] == e) & (lim["subset"] == "identified")]["tol0_contains_null"]
        if len(a) != 1 or len(b) != 1:
            raise SourceError(f"{iv.path}: {e}: all/identified rows missing")
        changed += int(str(a.iloc[0]) != str(b.iloc[0]))
    R._put("CnIdEstimands", "cn.id.n_estimands", float(len(est)), "int", iv.path,
           "[convention=limit] distinct estimand", "count", tag)
    R._put("CnIdSubsetCallChanges", "cn.id.n_subset_call_changes", float(changed), "int", iv.path,
           "[convention=limit] tol0_contains_null all vs identified", "count of differing estimands", tag)
    cv = Csv("results/tables/expi_identify_conventions.csv")
    flags = pd.concat([cv.df[c] for c in ("tol0_agree", "tol5_agree", "supp_agree", "median_sign_agree")])
    flags = flags.dropna().astype(str)
    R._put("CnIdConvComparisons", "cn.id.n_convention_comparisons", float(len(flags)), "int", cv.path,
           "non-empty *_agree cells", "count", tag)
    R._put("CnIdConvDisagree", "cn.id.n_convention_disagree", float((flags != "True").sum()), "int", cv.path,
           "non-empty *_agree cells != True", "count", tag)
    r = cv.row(estimand="xi_shift_V0_minus_V1", subset="all")
    R._put("CnIdNanShiftNonfiniteShare", "cn.id.nan.shift.nonfinite_share",
           float(r.get("n_nan_nonfinite_only")) / float(r.get("n_subset")), "pct1", cv.path,
           r.where("n_nan_nonfinite_only/n_subset"), "ratio", tag)

    gs = Csv("results/tables/k1_guard_sensitivity.csv")
    us_ = gs.rows(dataset="US", cell="metro|pi|main|B|overlap")
    if not len(us_):
        raise SourceError(f"{gs.path}: US primary rows missing")
    R._put("GsGuardsPerCell", "gs.n_guards_per_cell", float(len(us_)), "int", gs.path,
           "[dataset=US;cell=metro|pi|main|B|overlap]", "count of rows", tag)
    R._put("GsCells", "gs.n_cells", float(gs.df["cell"].nunique()), "int", gs.path, "distinct cell",
           "count", tag)
    ok = us_[us_["status"] == "ok"]["pred_over_registered"]
    R._put("GsUsRatioMin", "gs.us.ratio.min", float(ok.min()), "dec2", gs.path,
           "[US primary;status=ok] col=pred_over_registered", "min over rows", tag)
    R._put("GsUsRatioMax", "gs.us.ratio.max", float(ok.max()), "dec2", gs.path,
           "[US primary;status=ok] col=pred_over_registered", "max over rows", tag)
    okr = us_[us_["status"] == "ok"]
    tol_ = R.v("h3b.rel_tol")
    within = ((okr["observed_V0_minus_V1"] - okr["pred_xi_trap"]).abs() <= tol_ * okr["pred_xi_trap"].abs())
    R._put("GsUsDefined", "gs.us.n_defined", float(len(okr)), "int", gs.path, "[US primary;status=ok]",
           "count of rows", tag)
    R._put("GsUsWithinTol", "gs.us.n_within_tol", float(within.sum()), "int", gs.path,
           "[US primary;status=ok] |observed_V0_minus_V1 - pred_xi_trap| <= H3B_REL_TOL*|pred_xi_trap|",
           "count of rows", tag + "; post hoc, per-guard registered relative tolerance; H3b judged on the "
           "registered guard only")
    und = us_[us_["status"] != "ok"]
    R._put("GsUsUndefined", "gs.us.n_undefined", float(len(und)), "int", gs.path,
           "[US primary;status!=ok]", "count of rows", tag)
    for ff, Mn in ((1.0, "One"), (0.5, "Half")):
        g = und[und["floor_factor"] == ff]
        if len(g) != 1:
            raise SourceError(f"{gs.path}: undefined US guard with floor {ff}: {len(g)} rows")
        m = _parse_mass(g.iloc[0]["status"])
        R._put(f"GsUsMass{Mn}", f"gs.us.undefined.floor{ff}.mass", m, "sci2", gs.path,
               f"[US primary;status!=ok;floor_factor={ff}] col=status", "regex parse of stationary mass", tag)
    for cell, Mn in (("ref2002|pib", "BrTwoPib"), ("ref2010|pib", "BrTenPib")):
        g = gs.rows(dataset="Brazil", cell=cell)
        R._put(f"Gs{Mn}RatioMin", f"gs.{cell}.ratio.min", float(g["pred_over_registered"].min()), "dec2",
               gs.path, f"[cell={cell}] col=pred_over_registered", "min over rows", tag)
        R._put(f"Gs{Mn}RatioMax", f"gs.{cell}.ratio.max", float(g["pred_over_registered"].max()), "dec2",
               gs.path, f"[cell={cell}] col=pred_over_registered", "max over rows", tag)
    g = gs.rows(dataset="Brazil", cell="ref2002|pib")["pred_over_unguarded"]
    R._put("GsBrTwoPibUngMin", "gs.ref2002|pib.over_unguarded.min", float(round(g.min())), "int", gs.path,
           "[cell=ref2002|pib] col=pred_over_unguarded", "min over rows, rounded", tag)
    R._put("GsBrTwoPibUngMax", "gs.ref2002|pib.over_unguarded.max", float(round(g.max())), "int", gs.path,
           "[cell=ref2002|pib] col=pred_over_unguarded", "max over rows, rounded", tag)
    g = gs.df[(gs.df["dataset"] == "Brazil") & gs.df["cell"].str.endswith("vab_agro")]["pred_over_registered"]
    R._put("GsBrVabRatioMin", "gs.vab_agro.ratio.min", float(g.min()), "dec2", gs.path,
           "[cell=*|vab_agro] col=pred_over_registered", "min over rows", tag)
    R._put("GsBrVabRatioMax", "gs.vab_agro.ratio.max", float(g.max()), "dec2", gs.path,
           "[cell=*|vab_agro] col=pred_over_registered", "max over rows", tag)

    xs = [f"k1.k1a.cell{c}.xi_rel" for c in (3, 4, 8, 9)]
    R.derived("KoneaXiRelErrMinPct", "k1.k1a.xi_rel.min_pct", min(R.v(k) for k in xs), "pct1", xs,
              "min over cells 3,4,8,9", "EXPLORATORY, not a verdict (amendment 11)")
    R.derived("KoneaXiRelErrMaxPct", "k1.k1a.xi_rel.max_pct", max(R.v(k) for k in xs), "pct1", xs,
              "max over cells 3,4,8,9", "EXPLORATORY, not a verdict (amendment 11)")

    by = Csv("results/tables/expi_attrition_by_year.csv")
    w = by.df[(by.df["saai_spec"] == "multi") & by.df["period"].isin(["pre", "during"])]
    mx = max(float(w["mean_leave_t"].abs().max()), float(w["mean_enter_t1"].abs().max()))
    R._put("CnBlockMeanMaxAbs", "cn.block_mean.max_abs", mx, "dec2", by.path,
           "[saai_spec=multi;period in pre,during] cols=mean_leave_t,mean_enter_t1", "max |.| over rows",
           "largest |m_L| or |m_E| in the two windows")
    R._put("CnBlockMeanAboveOne", "cn.block_mean.n_above_one",
           float((w["mean_leave_t"].abs() > 1).sum() + (w["mean_enter_t1"].abs() > 1).sum()), "int", by.path,
           "[same filter] |m_L|>1 or |m_E|>1", "count of block means", "")


def ladder():
    v =R.v("cn.during.dose") / R.v("br.dose.GDP ref2010 2002-2023")
    R.derived("DoseRatioCnBr", "ladder.dose_ratio.cn_during_over_br_ref2010", float(round(v, -2)), "int",
              ["cn.during.dose", "br.dose.GDP ref2010 2002-2023"], "ratio, rounded to hundreds",
              "China during vs Brazil ref2010 dose")
    v = R.v("cn.during.dose") / R.v("us.dose.2003-2024")
    R.derived("DoseRatioCnUs", "ladder.dose_ratio.cn_during_over_us_main", float(round(v)), "int",
              ["cn.during.dose", "us.dose.2003-2024"], "ratio, rounded to integer")
    v = R.v("us.dose.2003-2024") / R.v("br.dose.GDP ref2010 2002-2023")
    R.derived("DoseRatioUsBr", "ladder.dose_ratio.us_main_over_br_ref2010", float(round(v)), "int",
              ["us.dose.2003-2024", "br.dose.GDP ref2010 2002-2023"], "ratio, rounded to integer")


MASK_DIR = "results/tables/k4_mask/confirmatory"
MASK_PILOT_DIR = "results/tables/k4_mask/pilot"
MASK_SYS = (("US", "Us"), ("BR", "Br"))
MASK_DOSES = (("us_like", "UsLike"), ("intermediate", "Inter"), ("china_like", "ChinaLike"))
MASK_VERDICT_TEXT = {"pass": "pass", "fail": "fail", "recovers": "recovers",
                     "closer_than_V0_only": "closer than V0 only",
                     "not_evaluable_availability": "not evaluable (availability)",
                     "not_evaluable": "not evaluable", "True": "powered", "False": "underpowered"}


def mask_points():
    """The committed per-mask table, with the gate flags of the registered rule
    (|estimate| > gate_factor x half-width of its tol0 interval)."""
    c = Csv(f"{MASK_DIR}/k4_points.csv")
    p = c.df.copy()
    gf = float(Json(f"{MASK_DIR}/k4_meta.json").get("criteria.gate_factor"))
    for q in ("T1_obs", "T3"):
        hw = (p[f"{q}_ci_hi"] - p[f"{q}_ci_lo"]) / 2.0
        p[f"gate_{q}"] = p[q].abs() > gf * hw
    p["point_ok"] = (p["pred_point_status"] == "ok") & (p["id_0"] == "stable_zero") & (p["id_1"] == "stable_zero")
    p["covered"] = (p["T1_obs"] >= p["range_lo"]) & (p["T1_obs"] <= p["range_hi"])
    p["width_rel"] = (p["range_hi"] - p["range_lo"]) / p["T1_obs"].abs()
    return c, p


def _ls_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([np.ones_like(x), x]).T
    (b0, b1), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(b0), float(b1)


def mask_summary(q):
    """Re-expressions of one subset of k4_points.csv (one system, stratum, dose or position)."""
    pt = q[q["point_ok"]]
    g3 = q[q["gate_T3"]]
    g1 = q[q["gate_T1_obs"]]
    out = dict(n=len(q), n_point=len(pt), point_share=len(pt) / len(q) if len(q) else float("nan"),
               med_obs_point=float(pt["T1_obs"].abs().median()) if len(pt) else float("nan"),
               med_err_point=float((pt["T1_obs"] - pt["pred_point"]).abs().median()) if len(pt) else float("nan"),
               med_err_zeroth=float((pt["T1_obs"] - pt["zeroth_order"]).abs().median()) if len(pt) else float("nan"),
               coverage=float(q["covered"].mean()),
               width_gated=float(q.loc[q["gate_T1_obs"], "width_rel"].median()) if len(g1) else float("nan"),
               n_gated_T3=len(g3),
               v1_share=float((g3["T2"].abs() < g3["T3"].abs()).mean()) if len(g3) else float("nan"),
               ratio=float(g3["T2"].abs().median() / g3["T3"].abs().median()) if len(g3) else float("nan"),
               naive_share=float((g3["T2"].abs() < g3["T_naive"].abs()).mean()) if len(g3) else float("nan"),
               med_T2=float(g3["T2"].abs().median()) if len(g3) else float("nan"),
               med_T3=float(g3["T3"].abs().median()) if len(g3) else float("nan"),
               med_naive=float(g3["T_naive"].abs().median()) if len(g3) else float("nan"),
               med_sel=float(g3["T3_selection"].abs().median()) if len(g3) else float("nan"),
               med_event=float(g3["T3_event"].abs().median()) if len(g3) else float("nan"),
               med_T2FS=float(g3["T2_vs_FS"].abs().median()) if len(g3) else float("nan"),
               med_T2_signed=float(q["T2"].median()), med_sel_signed=float(q["T3_selection"].median()))
    if len(pt) >= 3:
        out["slope_point"] = _ls_slope(pt["pred_point"], pt["T1_obs"])[1]
        out["slope_zeroth"] = _ls_slope(pt["zeroth_order"], pt["T1_obs"])[1]
    if len(q) >= 3:
        out["slope_T2_on_sel"] = _ls_slope(q["T3_selection"], q["T2"])[1]
        out["corr_T2_sel"] = float(np.corrcoef(q["T3_selection"], q["T2"])[0, 1])
    return out


def _mask_eval():
    ev = Csv(f"{MASK_DIR}/k4_evaluation.csv")
    d = ev.df
    if set(d["scope"].unique()) != {"active", "balanced", "null", "pooled"}:
        raise SourceError(f"{ev.path}: unexpected scope labels {sorted(map(str, d['scope'].unique()))}")
    v = d[d["kind"] == "verdict"]
    fz = v[v["criteria_set"] == "frozen"].reset_index(drop=True)
    pv = v[v["criteria_set"] == "provisional"].reset_index(drop=True)
    key = ["system", "scope", "name", "value"]
    if len(fz) != len(pv) or not (fz[key].astype(str).values == pv[key].astype(str).values).all():
        raise SourceError(f"{ev.path}: frozen and provisional verdict rows differ")
    return ev, fz, pv


def masking():
    tag = "controlled masking, confirmatory stage"
    ev, fz, pv = _mask_eval()
    meta = Json(f"{MASK_DIR}/k4_meta.json")
    pc, p = mask_points()
    mk = Csv(f"{MASK_DIR}/k4_masks.csv")
    R._put("MaskVerdictRows", "mask.verdict_rows", float(len(fz)), "int", ev.path, "[kind=verdict;criteria_set=frozen]",
           "count of rows; provisional rows identical", tag)
    R.json("MaskReps", "mask.n_rep", meta, "n_rep", "int", tag)
    R.json("MaskB", "mask.n_boot", meta, "n_boot", "int", tag)
    R.json("MaskClusterR", "mask.cluster_R", meta, "cluster_R", "int", tag)
    R._put("MaskGuards", "mask.n_guards", float(len(meta.get("guards"))), "int", meta.path, "json:guards",
           "count of entries", tag)
    crit = [("MaskCritAvail", "point_availability_min", "dec2"), ("MaskCritSign", "sign_agreement_min", "dec2"),
            ("MaskCritIntercept", "intercept_band_rel", "dec2"), ("MaskCritLevel", "level_abs_error_rel", "dec2"),
            ("MaskCritLevelMin", "level_min_masks", "int"), ("MaskCritCoverage", "range_coverage_min", "dec2"),
            ("MaskCritWidth", "range_width_rel_max", "dec1"), ("MaskCritRecShare", "recovery_share_min", "dec2"),
            ("MaskCritRecCiLo", "recovery_share_ci_lo_min", "dec2"),
            ("MaskCritRecRatio", "recovery_median_ratio_max", "dec2"), ("MaskCritGate", "gate_factor", "dec1"),
            ("MaskCritPower", "power_factor", "int"), ("MaskCritNull", "null_rel_max", "dec2"),
            ("MaskCritNaiveCiLo", "comparator_share_ci_lo_min", "dec2"),
            ("MaskCritGateAvail", "gate_availability_min", "dec2")]
    for m, k, s in crit:
        R.json(m, f"mask.crit.{k}", meta, f"criteria.{k}", s, tag)
    sb = meta.get("criteria.slope_band")
    R._put("MaskCritSlopeLo", "mask.crit.slope_band.lo", float(sb[0]), "dec2", meta.path, "json:criteria.slope_band[0]",
           "read", tag)
    R._put("MaskCritSlopeHi", "mask.crit.slope_band.hi", float(sb[1]), "dec2", meta.path, "json:criteria.slope_band[1]",
           "read", tag)
    for lv, L in MASK_DOSES:
        t = mk.df[mk.df["fraction_level"] == lv]["delta_event_target"].unique()
        fr = mk.df[mk.df["fraction_level"] == lv]["fraction"].unique()
        if len(t) != 1 or len(fr) != 1:
            raise SourceError(f"{mk.path}: dose level {lv} has {len(t)} targets")
        R._put(f"MaskDose{L}", f"mask.dose_target.{lv}", float(t[0]), "dec2", mk.path,
               f"[fraction_level={lv}] col=delta_event_target", "unique value", tag + "; event dose n_L/n_S")
        R._put(f"MaskFrac{L}", f"mask.fraction.{lv}", float(fr[0]), "dec3", mk.path,
               f"[fraction_level={lv}] col=fraction", "unique value", tag)
    for S_, P in MASK_SYS:
        m = mk.df[mk.df["system"] == S_]
        if (m["mask_status"] != "ok").any():
            raise SourceError(f"{mk.path}: {S_} has infeasible masks; the text says all were feasible")
        R._put(f"Mask{P}Masks", f"mask.{S_}.n_masks", float(len(m)), "int", mk.path, f"[system={S_}]",
               "count of rows", tag)
        R._put(f"Mask{P}Cells", f"mask.{S_}.n_cells", float(m["cell"].nunique()), "int", mk.path, f"[system={S_}] col=cell",
               "count of distinct values", tag)
        u = m["n_units"].unique()
        if len(u) != 1:
            raise SourceError(f"{mk.path}: {S_} n_units not unique")
        R._put(f"Mask{P}Units", f"mask.{S_}.n_units", float(u[0]), "int", mk.path, f"[system={S_}] col=n_units",
               "unique value", tag)
        R._put(f"Mask{P}Redrawn", f"mask.{S_}.n_attempt_gt0", float((m["attempt"] > 0).sum()), "int", mk.path,
               f"[system={S_}] col=attempt>0", "count of rows", tag)
        R._put(f"Mask{P}Duplicates", f"mask.{S_}.n_duplicate", float(m["mask_hash"].duplicated().sum()), "int",
               mk.path, f"[system={S_}] col=mask_hash", "count of repeated identities", tag)
        xf = p[p["system"] == S_]["xi_F"].unique()
        if len(xf) != 1:
            raise SourceError(f"{pc.path}: {S_} xi_F not unique")
        R._put(f"Mask{P}XiF", f"mask.{S_}.xi_F", float(xf[0]), "sdec3", pc.path, f"[system={S_}] col=xi_F",
               "unique value", tag + "; complete-panel reference")
    rp = Csv(f"{MASK_DIR}/k4_reproduction.csv")
    for S_, P in MASK_SYS:
        R.cell(f"Mask{P}ReproDiff", f"mask.{S_}.repro.xi.abs_diff", rp.row(system=S_, quantity="xi_trap"), "abs_diff",
               "sci2", tag + "; kill rule 2")
    eq = Csv(f"{MASK_DIR}/k4_equivalence.csv")
    R._put("MaskEquivMax", "mask.equivalence.max_diff", float(eq.df["max_diff"].max()), "int", eq.path,
           "all rows col=max_diff", "max over rows", tag + "; kill rule 4")
    it = Csv(f"{MASK_DIR}/k4_integrity.csv")
    if not it.df["pass"].astype(str).eq("True").all():
        raise SourceError(f"{it.path}: an integrity row failed")
    R._put("MaskIntegrityFiles", "mask.integrity.n_files", float(len(it.df)), "int", it.path, "all rows",
           "count of rows (all pass)", tag + "; kill rule 1")

    for S_, P in MASK_SYS:
        def cellm(macro, name, spec, scope="active", ci=False, note=""):
            r = _mrow(ev, S_, scope, name)
            R.cell(macro, f"mask.{S_}.{scope}.{name}", r, "value", spec, note or tag)
            if ci:
                R.cell(macro + "Lo", f"mask.{S_}.{scope}.{name}.lo", r, "cluster_ci_lo", spec, "cell-cluster bootstrap")
                R.cell(macro + "Hi", f"mask.{S_}.{scope}.{name}.hi", r, "cluster_ci_hi", spec, "cell-cluster bootstrap")
        cellm(f"Mask{P}Active", "n_masks", "int")
        cellm(f"Mask{P}PointN", "n_point_available", "int")
        cellm(f"Mask{P}PointAvail", "point_availability", "pct0")
        cellm(f"Mask{P}PointUndef", "point_projection_undefined_share", "pct0")
        cellm(f"Mask{P}AllGuards", "all_guards_defined_share", "pct0")
        cellm(f"Mask{P}Slope", "calibration_slope", "dec2", ci=True)
        cellm(f"Mask{P}IntS", "intercept_over_s", "sdec3", ci=True)
        cellm(f"Mask{P}Scale", "median_abs_obs", "dec2")
        cellm(f"Mask{P}Err", "median_abs_err_point", "dec3", ci=True)
        cellm(f"Mask{P}ErrNinety", "p90_abs_err_point", "dec3")
        cellm(f"Mask{P}GatedOne", "n_gated_T1", "int")
        cellm(f"Mask{P}Sign", "sign_agreement_gated", "dec2")
        cellm(f"Mask{P}Cov", "range_coverage", "dec2", ci=True)
        cellm(f"Mask{P}Width", "range_width_rel_median_gated", "dec2")
        cellm(f"Mask{P}GatedThree", "n_gated_T3", "int")
        cellm(f"Mask{P}VOne", "recovery_share_v1_closer", "dec2", ci=True)
        cellm(f"Mask{P}Ratio", "recovery_median_ratio", "dec3", ci=True)
        cellm(f"Mask{P}MedTTwo", "median_abs_T2", "dec3")
        cellm(f"Mask{P}MedTThree", "median_abs_T3", "dec2")
        cellm(f"Mask{P}Naive", "recovery_share_v1_beats_naive", "dec2", ci=True)
        cellm(f"Mask{P}MedNaive", "median_abs_T_naive", "dec3")
        cellm(f"Mask{P}MedSel", "median_abs_T3_selection", "dec3")
        cellm(f"Mask{P}MedEvent", "median_abs_T3_event", "dec2")
        cellm(f"Mask{P}MedTTwoFS", "median_abs_T2_vs_FS", "dec3")
        cellm(f"Mask{P}TTwoCiZero", "T2_ci_contains_0_share", "pct0")
        cellm(f"Mask{P}PcLow", "pc_low_median_T3", "sdec2")
        cellm(f"Mask{P}PcHigh", "pc_high_median_T3", "sdec2")
        hw_lo = _mrow(ev, S_, "active", "pc_low_median_halfwidth_T3")
        hw_hi = _mrow(ev, S_, "active", "pc_high_median_halfwidth_T3")
        R.cell(f"Mask{P}PcLowHw", f"mask.{S_}.pc_low_hw", hw_lo, "value", "dec2", tag)
        R.cell(f"Mask{P}PcHighHw", f"mask.{S_}.pc_high_hw", hw_hi, "value", "dec2", tag)
        pf = R.v("mask.crit.power_factor")
        R.derived(f"Mask{P}PcLowBar", f"mask.{S_}.pc_low_bar", -pf * R.v(f"mask.{S_}.pc_low_hw"), "sdec2",
                  [f"mask.{S_}.pc_low_hw", "mask.crit.power_factor"], "-power_factor x median half-width")
        R.derived(f"Mask{P}PcHighBar", f"mask.{S_}.pc_high_bar", pf * R.v(f"mask.{S_}.pc_high_hw"), "sdec2",
                  [f"mask.{S_}.pc_high_hw", "mask.crit.power_factor"], "+power_factor x median half-width")
        R.derived(f"Mask{P}CovN", f"mask.{S_}.active.n_covered",
                  round(R.v(f"mask.{S_}.active.range_coverage") * R.v(f"mask.{S_}.active.n_masks")), "int",
                  [f"mask.{S_}.active.range_coverage", f"mask.{S_}.active.n_masks"], "coverage x n_masks")
        R.derived(f"Mask{P}CovNeed", f"mask.{S_}.active.n_covered_needed",
                  math.ceil(R.v("mask.crit.range_coverage_min") * R.v(f"mask.{S_}.active.n_masks") - 1e-9), "int",
                  ["mask.crit.range_coverage_min", f"mask.{S_}.active.n_masks"], "ceil(threshold x n_masks)")
        R.derived(f"Mask{P}Removed", f"mask.{S_}.removed", 1.0 - R.v(f"mask.{S_}.active.recovery_median_ratio"),
                  "pct0", [f"mask.{S_}.active.recovery_median_ratio"], "1 - median|T2|/median|T3|",
                  "share of the median uncorrected shift removed by V1")
        rn = _mrow(ev, S_, "null", "median_abs_obs")
        R.cell(f"Mask{P}NullObs", f"mask.{S_}.null.median_abs_obs", rn, "value", "dec3", tag + "; null stratum")
        rn = _mrow(ev, S_, "null", "median_abs_pred_point")
        R.cell(f"Mask{P}NullPred", f"mask.{S_}.null.median_abs_pred_point", rn, "value", "dec3", tag)
        R.derived(f"Mask{P}NullBar", f"mask.{S_}.null.bar",
                  R.v("mask.crit.null_rel_max") * R.v(f"mask.{S_}.active.median_abs_obs"), "dec3",
                  ["mask.crit.null_rel_max", f"mask.{S_}.active.median_abs_obs"], "null_rel_max x active median |O|")
        for nm, M in (("T1_point", "VerdictPoint"), ("T1_range", "VerdictRange"), ("T2", "VerdictRecovery"),
                      ("T3_powered", "VerdictPower"), ("N_null_control", "VerdictNull")):
            r = fz[(fz["system"] == S_) & (fz["scope"] == "active") & (fz["name"] == nm)]
            if len(r) != 1:
                raise SourceError(f"{ev.path}: verdict {S_} {nm}: {len(r)} rows")
            val = str(r.iloc[0]["value"])
            R._put(f"Mask{P}{M}", f"mask.{S_}.verdict.{nm}", MASK_VERDICT_TEXT.get(val, val.replace("_", " ")), "str",
                   ev.path, f"[system={S_};scope=active;name={nm};criteria_set=frozen] col=value", "label", tag)
        q = p[(p["system"] == S_) & (p["stratum"] == "active")]
        s = mask_summary(q)
        checks = [("coverage", "range_coverage"), ("width_gated", "range_width_rel_median_gated"),
                  ("v1_share", "recovery_share_v1_closer"), ("ratio", "recovery_median_ratio"),
                  ("naive_share", "recovery_share_v1_beats_naive"), ("point_share", "point_availability"),
                  ("slope_point", "calibration_slope"), ("med_T2", "median_abs_T2"), ("med_T3", "median_abs_T3"),
                  ("med_err_point", "median_abs_err_point"), ("med_sel", "median_abs_T3_selection"),
                  ("med_T2FS", "median_abs_T2_vs_FS")]
        for a, b in checks:
            ref = float(_mrow(ev, S_, "active", b).get("value"))
            if abs(s[a] - ref) > 1e-9 * max(1.0, abs(ref)):
                raise SourceError(f"{pc.path}: re-expressed {a} = {s[a]!r} for {S_} != evaluation {b} = {ref!r}")
        src = pc.path
        def put(macro, key, val, spec, how):
            R._put(macro, key, float(val), spec, src, f"[system={S_};stratum=active]", how, tag + "; re-expression")
        put(f"Mask{P}SlopeZeroth", f"mask.{S_}.active.slope_zeroth", s["slope_zeroth"], "dec2",
            "least-squares slope of T1_obs on zeroth_order, point-available masks")
        put(f"Mask{P}ErrZeroth", f"mask.{S_}.active.med_err_zeroth", s["med_err_zeroth"], "dec3",
            "median |T1_obs - zeroth_order|, point-available masks")
        g1 = q[q["gate_T1_obs"] & q["point_ok"]]
        put(f"Mask{P}ZerothSign", f"mask.{S_}.active.zeroth_sign", float((np.sign(g1["zeroth_order"]) == np.sign(g1["T1_obs"])).mean()),
            "dec2", "share sign(zeroth_order) = sign(T1_obs), T1-gated point-available masks")
        qn = p[(p["system"] == S_) & (p["stratum"] == "null")]
        R._put(f"Mask{P}NullObsAll", f"mask.{S_}.null.median_abs_obs_all", float(qn["T1_obs"].abs().median()), "dec3",
               src, f"[system={S_};stratum=null] col=T1_obs", "median |T1_obs| over all null masks", tag)
        sg = q[q["position"].isin(["low", "high"])]
        want = np.where(sg["position"] == "low", -1.0, 1.0)
        put(f"Mask{P}SignForcing", f"mask.{S_}.active.sign_T3_forcing", float((np.sign(sg["T3"]) == want).sum()), "int",
            "count of masks with sign(T3) = forcing direction (low -, high +)")
        for lv, L in MASK_DOSES:
            sd = mask_summary(q[q["fraction_level"] == lv])
            put(f"Mask{P}Avail{L}", f"mask.{S_}.active.{lv}.point_share", sd["point_share"], "pct0",
                "share of point-available masks")
            put(f"Mask{P}Cov{L}", f"mask.{S_}.active.{lv}.coverage", sd["coverage"], "dec2", "share O in envelope")
            put(f"Mask{P}Width{L}", f"mask.{S_}.active.{lv}.width", sd["width_gated"], "dec2",
                "median width/|O|, T1-gated masks")
            put(f"Mask{P}Naive{L}", f"mask.{S_}.active.{lv}.naive", sd["naive_share"], "dec2",
                "share |T2|<|T_naive|, T3-gated masks")
            put(f"Mask{P}PointN{L}", f"mask.{S_}.active.{lv}.n_point", sd["n_point"], "int", "count")
        for pos, Ps in (("low", "Low"), ("high", "High")):
            sp = mask_summary(q[q["position"] == pos])
            put(f"Mask{P}Ratio{Ps}", f"mask.{S_}.active.{pos}.ratio", sp["ratio"], "sig2",
                "median|T2|/median|T3|, T3-gated masks")
            put(f"Mask{P}SelSlope{Ps}", f"mask.{S_}.active.{pos}.slope_T2_on_sel", sp["slope_T2_on_sel"], "dec2",
                "least-squares slope of T2 on T3_selection")
            put(f"Mask{P}MedTTwo{Ps}", f"mask.{S_}.active.{pos}.med_T2_signed", sp["med_T2_signed"], "sdec3",
                "median T2 (signed)")
            put(f"Mask{P}MedSel{Ps}", f"mask.{S_}.active.{pos}.med_sel_signed", sp["med_sel_signed"], "sdec3",
                "median T3_selection (signed)")
            put(f"Mask{P}MedAbsTTwo{Ps}", f"mask.{S_}.active.{pos}.med_abs_T2", sp["med_T2"], "dec2",
                "median |T2|, T3-gated masks")
            put(f"Mask{P}MedTTwoFS{Ps}", f"mask.{S_}.active.{pos}.med_abs_T2_vs_FS", sp["med_T2FS"], "dec2",
                "median |xi_V1 - xi_FS|, T3-gated masks")
            qp = q[q["position"] == pos]
            put(f"Mask{P}Between{Ps}", f"mask.{S_}.active.{pos}.v1_between_F_FS",
                float(((qp["xi_1"] - qp["xi_F"]) * (qp["xi_1"] - qp["xi_FS"]) < 0).mean()), "dec2",
                "share of masks with xi_V1 strictly between xi_F and xi_FS")
        put(f"Mask{P}DoseN", f"mask.{S_}.active.n_per_dose", float(q.groupby("fraction_level").size().iloc[0]), "int",
            "count of active masks per dose (equal across doses)")
        if q.groupby("fraction_level").size().nunique() != 1:
            raise SourceError(f"{pc.path}: {S_} active masks differ in number across doses")
        put(f"Mask{P}RefusedN", f"mask.{S_}.active.n_refused", float((~q["point_ok"]).sum()), "int",
            "count of active masks without a computable point projection")
        if (~q["point_ok"]).any():
            put(f"Mask{P}ObsRefused", f"mask.{S_}.active.med_abs_obs_refused",
                float(q.loc[~q["point_ok"], "T1_obs"].abs().median()), "dec2", "median |T1_obs|, refused masks")
        put(f"Mask{P}SlopeZerothAll", f"mask.{S_}.active.slope_zeroth_all",
            _ls_slope(q["zeroth_order"], q["T1_obs"])[1], "dec2",
            "least-squares slope of T1_obs on zeroth_order, all active masks")
        for sc, Sc in (("null", "Null"), ("balanced", "Bal")):
            ss = mask_summary(p[(p["system"] == S_) & (p["stratum"] == sc)])
            R._put(f"Mask{P}{Sc}TThree", f"mask.{S_}.{sc}.med_abs_T3", ss["med_T3"], "dec3", src,
                   f"[system={S_};stratum={sc}]", "median |T3|, T3-gated masks", tag + "; re-expression")
            R._put(f"Mask{P}{Sc}Sel", f"mask.{S_}.{sc}.med_abs_sel", ss["med_sel"], "dec3", src,
                   f"[system={S_};stratum={sc}]", "median |T3_selection|, T3-gated masks", tag + "; re-expression")
    gj = Csv(f"{MASK_DIR}/k4_projections.csv")
    act_us = p[(p["system"] == "US") & (p["stratum"] == "active")][["mask_id", "T1_obs"]]
    gg = gj.df.merge(act_us, on="mask_id", how="inner")
    if gg["mask_id"].nunique() != len(act_us):
        raise SourceError(f"{gj.path}: US active masks missing from the projection table")
    full, reg_err, full_lo, reg_lo = [], None, [], None
    for _k, gq in gg.groupby(["support_lo_pct", "support_hi_pct", "floor_factor"], sort=True):
        ok = gq[gq["status"] == "ok"]
        err = float((ok["T1_obs"] - ok["pred"]).abs().median()) if len(ok) else float("nan")
        if gq["registered_guard"].astype(str).eq("True").all():
            reg_err, reg_lo = err, float(_k[0])
        if len(ok) == len(gq):
            full.append((float((ok["pred"].abs() > ok["T1_obs"].abs()).mean()), err))
            full_lo.append(float(_k[0]))
    if not full or reg_err is None:
        raise SourceError(f"{gj.path}: no US guard defined on every active mask, or no registered guard")
    if not all(lo > reg_lo for lo in full_lo):
        raise SourceError(f"{gj.path}: a US guard defined on every mask does not have a narrower support")
    gtag = tag + "; post-hoc re-expression of the guard table, report-only"
    gw = "[stratum=active;system=US] guards with status=ok on every mask"
    R._put("MaskUsGuardAllN", "mask.US.active.guards_all_defined.n", float(len(full)), "int", gj.path, gw,
           "count of guards", gtag)
    R._put("MaskUsGuardAllOverMin", "mask.US.active.guards_all_defined.over.min", min(a for a, _ in full), "dec2",
           gj.path, gw, "min share |pred|>|O|", gtag)
    R._put("MaskUsGuardAllOverMax", "mask.US.active.guards_all_defined.over.max", max(a for a, _ in full), "dec2",
           gj.path, gw, "max share |pred|>|O|", gtag)
    R._put("MaskUsGuardAllErrMin", "mask.US.active.guards_all_defined.err.min", min(e for _, e in full), "dec3",
           gj.path, gw, "min median |O-pred|", gtag)
    R._put("MaskUsGuardAllErrMax", "mask.US.active.guards_all_defined.err.max", max(e for _, e in full), "dec3",
           gj.path, gw, "max median |O-pred|", gtag)
    R._put("MaskUsGuardPointErr", "mask.US.active.guard_point.err", reg_err, "dec3", gj.path,
           "[stratum=active;system=US;registered_guard=True;status=ok]", "median |O-pred|", gtag)
    pp = Csv(f"{MASK_PILOT_DIR}/k4_points.csv")
    R._put("MaskPilotMasks", "mask.pilot.n_masks", float(len(pp.df)), "int", pp.path, "all rows", "count of rows",
           "pilot stage (US only), exploratory")
    pa = pp.df[pp.df["stratum"] == "active"]
    R._put("MaskPilotActive", "mask.pilot.n_active", float(len(pa)), "int", pp.path, "[stratum=active]",
           "count of rows", "pilot")
    R._put("MaskPilotActivePoint", "mask.pilot.n_active_point", float((pa["pred_point_status"] == "ok").sum()), "int",
           pp.path, "[stratum=active] col=pred_point_status=ok", "count of rows", "pilot")
    pj = Csv(f"{MASK_PILOT_DIR}/k4_projections.csv")
    R._put("MaskPilotGuardRows", "mask.pilot.n_guard_rows", float(len(pj.df)), "int", pj.path, "all rows",
           "count of rows", "pilot")
    R._put("MaskPilotGuardUndef", "mask.pilot.n_guard_undefined", float((pj.df["status"] != "ok").sum()), "int",
           pj.path, "col=status!=ok", "count of rows", "pilot; every refusal is the confinement check")
    if not pj.df.loc[pj.df["status"] != "ok", "status"].str.contains("not confined").all():
        raise SourceError(f"{pj.path}: an undefined pilot projection is not a confinement refusal")
    cj = Csv(f"{MASK_DIR}/k4_projections.csv")
    und = cj.df[cj.df["status"] != "ok"]
    if not und["status"].str.contains("not confined").all():
        raise SourceError(f"{cj.path}: an undefined projection is not a confinement refusal")
    R._put("MaskGuardRowsUndef", "mask.n_guard_undefined", float(len(und)), "int", cj.path, "col=status!=ok",
           "count of rows", tag + "; every refusal is the confinement check")


def _mrow(ev, S_, scope, name):
    d = ev.df
    r = d[(d["system"] == S_) & (d["scope"] == scope) & (d["name"] == name) & (d["kind"] == "metric")]
    if len(r) != 1:
        raise SourceError(f"{ev.path}: metric {S_}/{scope}/{name}: {len(r)} rows")
    return Row(ev, r.iloc[0], dict(system=S_, scope=scope, name=name, kind="metric"))


def write():
    df = pd.DataFrame(R.rows, columns=["key", "macro", "value", "formatted", "source_file",
                                       "source_row_or_column", "transform", "note"])
    df["value"] = [v if isinstance(v, str) else repr(float(v)) for v in df["value"]]
    df.to_csv(OUT_CSV, index=False, lineterminator="\n")
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    lines = ["% values.tex -- GENERATED by scripts/paper_k1_numbers.py; do not edit.",
             "% Every macro is one committed value; provenance in results/tables/paper_k1_values.csv.",
             ""]
    for rec in R.rows:
        src = rec["source_file"].split(" | ")[0]
        lines.append(f"\\newcommand{{\\{rec['macro']}}}{{{rec['formatted']}}}% {rec['key']} <- {src}")
    with open(OUT_TEX, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(OUT_META, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"script": "scripts/paper_k1_numbers.py", "n_values": len(R.rows),
                   "inputs_sha256": dict(sorted(_SHA.items()))},
                  fh, indent=1, ensure_ascii=False)
        fh.write("\n")


SI_DIR = f"{TEX_DIR}/si_tables"
SI_META = "results/tables/paper_k1_si_tables_meta.json"
GENERATED_TAG = "GENERATED by scripts/paper_k1_numbers.py"

_DIGIT_WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
                "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
_NUM_WORDS = {10: "Ten", 11: "Eleven", 12: "Twelve"}


def _letters(token):
    """Event id -> letters only (A11 -> AEleven, I0 -> IZero)."""
    m = re.fullmatch(r"([A-Za-z]+)(\d*)", token)
    if not m:
        raise SourceError(f"bad id {token!r}")
    head, num = m.group(1), m.group(2)
    if not num:
        return head
    n = int(num)
    return head + (_NUM_WORDS.get(n) or "".join(_DIGIT_WORDS[c] for c in num))


def _esc(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                 ("_", r"\_"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\~{}"),
                 ("^", r"\^{}")):
        s = s.replace(a, b)
    return s


_AMEND_LABEL = re.compile(r"\((?:[^\x00-\x7f]+|amendment)\s*(\d+)")


def _amendment_label(text):
    return _AMEND_LABEL.sub(r"(amendment \1", str(text))


def _isnan(v):
    try:
        return v is None or (isinstance(v, str) and v.strip() == "") or not math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _n(v, spec, na="--"):
    if _isnan(v):
        return na
    return fmt(float(v), spec)


def _ci(lo, hi, spec):
    if _isnan(lo) or _isnan(hi):
        return "--"
    return f"[{_n(lo, spec)}, {_n(hi, spec)}]"


class _SiWriter:
    def __init__(self):
        self.macros = []
        self.files = {}
        self.reads = set()

    def csv(self, path):
        c = Csv(path)
        self.reads.add(path)
        return c

    def json(self, path):
        j = Json(path)
        self.reads.add(path)
        return j

    def macro(self, name, tex, source, note=""):
        if not MACRO_RE.match(name) or not name.startswith("Si"):
            raise SourceError(f"SI macro {name!r} must be letters only and start with 'Si'")
        if name in R.by_macro or any(m[0] == name for m in self.macros):
            raise SourceError(f"duplicate macro {name}")
        self.macros.append((name, tex, source, note))

    def table(self, name, colspec, header, rows, note=None):
        lines = [f"% {name} -- GENERATED by scripts/paper_k1_numbers.py (si_tables); do not edit.",
                 f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + r" \\", "\\midrule"]
        for r in rows:
            if r == "MID":
                lines.append("\\midrule")
            elif isinstance(r, str):
                lines.append(r)
            else:
                lines.append(" & ".join(r) + r" \\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        if note:
            lines.append(note)
        self.files[f"{SI_DIR}/{name}.tex"] = "\n".join(lines) + "\n"

_PROC = {"i_ou": "(i) OU", "ii_cubic": "(ii) cubic", "iii_mult": "(iii) cubic, mult.\\ noise",
         "v_posctrl": "(v) true shift", "ou": "(i) OU", "cubic": "(ii) cubic",
         "cubic_mult": "(iii) cubic, mult.\\ noise", "cubic_shift": "(v) true shift"}


def _proc(p):
    p = str(p)
    m = re.fullmatch(r"(?:iv_bistable_|cubic_bistable_)(\d)", p)
    if m:
        return f"(iv) bistable, $2\\Delta V={m.group(1)}$"
    return _PROC.get(p, _esc(p))


def _si_k1(W):
    roles = {"K1c": "K1c, gated", "K1a/K1b": "K1a/K1b, exploratory (amendment 11)",
             "K1a_null": "null band for K1a (no exit)", "K1a_shocked_null": "dated-shock null, exploratory",
             "control": "persistent-offset control, report only", "control_null": "null band for the controls",
             "K1a_sens_no2006": "no-2006 sensitivity, report only"}
    frames = []
    for p in ("results/tables/k1_k1c_f2/k1_sim_bias.csv",
              "results/tables/k1_exploratory/k1_exploratory_sim_bias.csv",
              "results/tables/k1_exploratory/k1_exploratory_controls_sim_bias.csv"):
        frames.append(W.csv(p).df[["cell_index", "process", "mechanism", "gate_use"]])
    pc2 = W.csv("results/tables/k1_s0_scope_f2/k1_s0_scope_precheck_units.csv")
    pre = pc2.df[["cell_index", "process"]].drop_duplicates().assign(mechanism="a", gate_use="precheck")
    frames.append(pre)
    cells = pd.concat(frames).drop_duplicates(["cell_index"]).sort_values("cell_index")
    rows = []
    for _, r in cells.iterrows():
        role = roles.get(r["gate_use"], "no-exit coverage precheck" if r["gate_use"] == "precheck" else r["gate_use"])
        rows.append([fmt(r["cell_index"], "int"), _proc(r["process"]), f"({r['mechanism']})", _esc(role)])
    W.table("tab_k1_cells", "rlll", ["Cell", "Process", "Exit mechanism", "Role in this paper"], rows)

    def s0_status(path):
        d = W.csv(path).df
        d = d[d["gating"].astype(str) == "True"]
        out = {}
        for (proc, pipe), g in d.groupby(["process", "pipeline"]):
            f = g[g["status"] == "fail"]
            if len(f):
                qn_ = {"rho1": "$\\rho_1$", "mean_diffusion": "$\\bar D$", "xi_trap": "$\\xi^*$",
                       "two_delta_V": "$2\\Delta V$", "tau_mfpt": "$\\tau$"}
                q = sorted({f"{qn_.get(a, _esc(a))} ({_esc(b)})" for a, b in zip(f["quantity"], f["period"])})
                out[(proc, pipe)] = "fail: " + ", ".join(q)
            else:
                extra = []
                ne = (g["status"] == "not_evaluable_record_inconsistent").sum()
                ro = (g["status"] == "report only").sum()
                if ne:
                    extra.append("$\\rho_1$ rows read as inconsistent (F2-1)")
                if ro:
                    extra.append("$\\rho_1$ report only")
                out[(proc, pipe)] = "pass" + (" (" + "; ".join(extra) + ")" if extra else "")
        return out
    s1 = s0_status("results/tables/k1_s0_scope_s0_rows.csv")
    s2 = s0_status("results/tables/k1_s0_scope_f2/k1_s0_scope_s0_rows.csv")
    order = ["ou", "cubic", "cubic_mult", "cubic_bistable_0", "cubic_bistable_1", "cubic_bistable_2",
             "cubic_bistable_4", "cubic_shift"]
    rows = []
    for proc in order:
        for pipe in ("saai", "direct"):
            rows.append([_proc(proc), PIPE_LABEL[pipe], s1.get((proc, pipe), "--"), s2.get((proc, pipe), "--")])
    spec = r"ll>{\raggedright\arraybackslash}p{0.3\linewidth}>{\raggedright\arraybackslash}p{0.3\linewidth}"
    W.table("tab_k1_s0", spec, ["Process", "Pipeline", "First freeze", "Second freeze"], rows)

    pc1 = W.csv("results/tables/k1_s0_scope_precheck_units.csv").df
    m = pc1.merge(pc2.df, on=["cell_index", "process", "quantity", "period"], suffixes=("_1", "_2"))
    if len(m) != len(pc2.df) or len(m) != len(pc1):
        raise SourceError("precheck units of the two freezes do not match one to one")
    qn = {"xi_trap": "$\\xi^*$", "tau_mfpt": "$\\tau$ ratio", "two_delta_V": "$2\\Delta V$"}
    rows = []
    for _, r in m.iterrows():
        c1 = _n(r["coverage_1"], "dec3") + ("" if bool(r["in_band_1"]) else "$^\\dagger$")
        c2 = _n(r["coverage_2"], "dec3") + ("" if bool(r["in_band_2"]) else "$^\\dagger$")
        st = "gated" if r["status"] == "gated" else "not evaluable (amendment 7)"
        rows.append([fmt(r["cell_index"], "int"), _proc(r["process"]), qn.get(r["quantity"], r["quantity"]),
                     _esc(r["period"]), c1, c2, _esc(r["scope"]), st])
    W.table("tab_k1_precheck", "rlllrrll", ["Cell", "Process", "Quantity", "Period", "First", "Second",
                                             "Scope", "Status (second)"], rows)

    v = W.csv("results/tables/k1_k1c_f2/k1_gates_verdict.csv").df
    v = v[(v["run_label"] == "registered") & (v["primary"].astype(str) == "True")]
    rows = []
    for _, r in v.iterrows():
        verdict = _amendment_label(r["verdict"])
        rows.append([_esc(r["gate"]), _esc(r["rule_set"]), _esc(verdict), fmt(r["n_gating_rows"], "int"),
                     fmt(r["n_pass"], "int"), fmt(r["n_fail"], "int"), fmt(r["n_not_evaluable"], "int"),
                     fmt(r["n_blocked"], "int")])
    W.table("tab_k1_verdicts", "lllrrrrr", ["Criterion", "Rule set", "Verdict", "Rows", "Pass", "Fail",
                                             "Not eval.", "Blocked"], rows)

    gates = W.csv("results/tables/k1_k1c_f2/k1_gates.csv")
    power = W.csv("results/tables/k1_k1c_f2/k1_power.csv")
    rows = []
    for cell, mech in ((23, "a"), (24, "d"), (25, "e")):
        for pipe, role in (("saai", "gated"), ("direct", "report only")):
            base = dict(gate="K1c", rule_set="limit/tol0", cell_index=float(cell), pipeline=pipe)
            rb = gates.row(**base, statistic="V1_relative_bias_xi_shift").get("value")
            cv = gates.row(**base, statistic="V1_ci95_coverage_xi_shift").get("value")
            pw = gates.row(**base, statistic="V1_power_ci_excludes_0").get("value")
            tr = power.row(cell_index=cell, pipeline=pipe, variant="V1", period="contrast",
                           quantity="xi_trap", ci_nan_rule="tol0").get("median_truth")
            rows.append([fmt(cell, "int"), f"({mech})", PIPE_LABEL[pipe], role, _n(tr, "sdec2"), _n(rb, "sig2"),
                         _n(cv, "dec3"), _n(pw, "dec3")])
    W.table("tab_k1c", "rllllrrr", ["Cell", "Exit", "Pipeline", "Role", "True shift", "Rel.\\ bias",
                                     "Coverage", "Power"], rows)

    ex = W.csv("results/tables/k1_exploratory/k1_exploratory_k1a.csv").df
    e = ex[(ex["pipeline"] == "saai") & (ex["null_variant"] == "registered")]

    def oc(s):
        s = str(s).replace("exploratory:", "")
        return {"pass": "pass", "fail": "fail", "indistinguishable_from_null": "null",
                "fail_(non-finite)": "n/f"}.get(s, _esc(s))
    rows = []
    rown = {"projection": "projection", "closed_form": "closed form (OU only)", "zeroth_order": "zeroth order",
            "zeroth_order_mbar": "$\\bar M/|a_1|$ (report only)", "fallback": "fallback design"}
    for cell in (3, 4, 8, 9):
        first = True
        for row in ("projection", "closed_form", "zeroth_order", "zeroth_order_mbar"):
            g = e[(e["cell_index"] == cell) & (e["row"] == row)].set_index("statistic")
            if not len(g):
                continue

            def val(stat, spec):
                k = f"{row}:{stat}"
                if k not in g.index:
                    return "--"
                x = g.loc[k]
                return (_n(x["value"], spec, na="--") + f" ({oc(x['exploratory_outcome'])})")

            def sg(stat):
                k = f"{row}:{stat}"
                return oc(g.loc[k]["exploratory_outcome"]) if k in g.index else "--"
            proc = e[e["cell_index"] == cell]["process"].iloc[0]
            mech = e[e["cell_index"] == cell]["mechanism"].iloc[0]
            lab = f"{cell} {_proc(proc)} ({mech})" if first else ""
            rows.append([lab, rown[row], val("xi_trap_rel", "dec3"), sg("two_delta_V_sign"),
                         sg("tau_mfpt_sign"), val("rho1_rel", "dec2"), sg("rho1_sign")])
            first = False
        rows.append("MID")
    rows = rows[:-1]
    W.table("tab_k1a_expl", "llllllll"[:7], ["Cell", "Row", "$\\xi^*$ rel.\\ err.", "$2\\Delta V$ sign",
                                              "$\\tau$ sign", "$\\rho_1$ rel.\\ err.", "$\\rho_1$ sign"], rows)

    th = set(e[e["statistic"] == "projection:xi_trap_rel"]["threshold"].astype(str))
    if len(th) != 1:
        raise SourceError(f"K1a xi threshold not unique: {th}")
    m_ = re.fullmatch(r"\D*([0-9.]+)", th.pop())
    if not m_:
        raise SourceError("K1a threshold unparsable")
    W.macro("SiKoneaThreshold", fmt(float(m_.group(1)), "dec2"),
            "results/tables/k1_exploratory/k1_exploratory_k1a.csv", "registered K1a xi* relative-error threshold")

    kb = W.csv("results/tables/k1_exploratory/k1_exploratory_k1b.csv").df
    thb = set(kb[kb["statistic"] == "V1_ci95_coverage"]["threshold"].astype(float))
    if len(thb) != 1:
        raise SourceError(f"K1b threshold not unique: {thb}")
    W.macro("SiKonebThreshold", fmt(thb.pop(), "dec2"),
            "results/tables/k1_exploratory/k1_exploratory_k1b.csv", "registered K1b coverage floor")
    k = kb[(kb["pipeline"] == "saai") & (kb["ci_nan_rule"] == "tol0")]
    rows = []
    for cell, g in k.groupby("cell_index"):
        g = g.set_index("quantity")

        def cov(q):
            x = g.loc[q]
            s = _n(x["value"], "dec3")
            o = str(x["exploratory_outcome"])
            if o == "exploratory:fail":
                s += "$^\\dagger$"
            elif "not_evaluable" in o:
                s += "$^\\ast$"
            return s
        rows.append([fmt(cell, "int"), _proc(g["process"].iloc[0]), f"({g['mechanism'].iloc[0]})",
                     cov("xi_trap"), cov("tau_mfpt")])
    W.table("tab_k1b_expl", "rllrr", ["Cell", "Process", "Exit", "$\\xi^*$ shift", "$\\tau$ ratio"], rows)

    ctl = W.csv("results/tables/k1_exploratory/k1_exploratory_controls_k1a.csv").df
    rows = []
    for cell, g in ctl.groupby("cell_index"):
        cnt = g["exploratory_outcome"].value_counts()
        rows.append([fmt(cell, "int"), _proc(g["process"].iloc[0]), f"({g['mechanism'].iloc[0]})",
                     _esc(roles.get(g["gate_use"].iloc[0], g["gate_use"].iloc[0]) if "gate_use" in g else ""),
                     fmt(int(cnt.get("exploratory:pass", 0)), "int"), fmt(int(cnt.get("exploratory:fail", 0)), "int"),
                     fmt(int(cnt.get("exploratory:fail_(non-finite)", 0)), "int"),
                     fmt(int(cnt.get("exploratory:indistinguishable_from_null", 0)), "int")])
    W.table("tab_k1_controls", "rlllrrrr", ["Cell", "Process", "Exit", "Role", "Pass", "Fail", "Non-finite",
                                             "Null"], rows)


def _si_china(W):
    by = W.csv("results/tables/expi_attrition_by_year.csv").df
    by = by[by["saai_spec"] == "multi"].sort_values("year_t")
    rows = []
    for _, r in by.iterrows():
        t = int(r["year_t"])
        rows.append([f"{t}--{(t + 1) % 100:02d}", _esc(r["period"]), fmt(r["n_stay"], "int"),
                     fmt(r["n_leave"], "int"), fmt(r["n_enter"], "int"), _n(r["mean_leave_t"], "sdec2"),
                     _n(r["mean_enter_t1"], "sdec2"),
                     _n((r["n_enter"] + r["n_leave"]) / r["n_stay"], "dec2"),
                     _n(r["mechanical_term"], "sdec3"), _n(abs(r["identity_residual"]), "sci1")])
    W.table("tab_cn_by_year", "llrrrrrrrr", ["Transition", "Window", "$n_S$", "$n_L$", "$n_E$", "$m_L$",
                                              "$m_E$", "Dose", "$M_t$", "$|$Lemma 1 resid.$|$"], rows)

    sbs = W.csv("results/tables/expi_rerun_ci_side_by_side.csv").df
    qn = {"xi_trap (during-pre)": "$\\xi^*$ shift (during$-$pre)",
          "tau_mfpt (during/pre)": "$\\tau$ ratio (during/pre)",
          "gamma (during-pre)": "$\\Gamma$ difference", "mean_diffusion (during/pre)": "$\\bar D$ ratio",
          "B_ratio_w (during/pre)": "$B$ ratio, poor side"}
    rows = []
    for var, vl in (("V0_baseline", "V0"), ("V1_stayer_restandardised", "V1")):
        g = sbs[(sbs["variant"] == var) & (sbs["degenerate"] == "limit")]
        if not len(g):
            raise SourceError(f"side-by-side: variant {var} missing")
        for _, r in g.iterrows():
            spec = "sig3" if "tau" in r["quantity"] else "sdec2" if "-pre" in r["quantity"] else "dec2"
            rows.append([vl, qn.get(r["quantity"], _esc(r["quantity"])), _n(r["point_estimate"], spec),
                         _ci(r["county_block_ci_lo"], r["county_block_ci_hi"], spec),
                         _ci(r["rerun_ci_lo"], r["rerun_ci_hi"], spec),
                         _n(r["width_ratio_rerun_over_county"], "dec1")])
        rows.append("MID")
    W.table("tab_cn_ci", "llrccr", ["", "Quantity", "Point", "County block (superseded)",
                                     "Stratified province block (primary)", "Width ratio"], rows[:-1])


def _si_paired(W):
    """Every row of the China paired addendum (all estimands x conventions; report-only rows marked)."""
    df = W.csv("results/tables/expi_paired_estimands.csv").df
    lab = {"xi_trap_V0_minus_V1_pre": "$\\xi^*_{V0}-\\xi^*_{V1}$, first window",
           "xi_trap_V0_minus_V1_during": "$\\xi^*_{V0}-\\xi^*_{V1}$, second window",
           "xi_shift_V0_minus_V1": "shift difference V0$-$V1 (second$-$first)",
           "tau_ratio_V0_over_V1": "$\\tau$-ratio V0/V1"}
    def name(e):
        if e in lab:
            return lab[e]
        for p, w in (("_pre", "first"), ("_during", "second")):
            if e.endswith(p):
                core = e[: -len(p)]
                break
        else:
            raise SourceError(f"paired estimand {e!r} has no window suffix")
        m = {"rho1_V0": "$\\rho_1$, V0", "rho1_V1": "$\\rho_1$, V1", "a1_V0": "$a_1$, V0", "a1_V1": "$a_1$, V1",
             "amp_V1": "$1/|a_1|$, V1", "amp_V0": "$1/|a_1|$, V0", "dose": "dose $\\bar\\delta$",
             "dose_max": "largest transition dose", "dose_x_amp_V1": "$\\bar\\delta/|a_{1}|$, V1",
             "rho1_V0_minus_V1": "$\\rho_{1,V0}-\\rho_{1,V1}$", "Mbar": "$\\bar M$",
             "zeroth_order_Mbar_over_abs_a1_V1": "$\\bar M/|a_{1}|$, V1"}
        if core not in m:
            raise SourceError(f"paired estimand {e!r} not labelled")
        return f"{m[core]}, {w} window"
    rows = []
    for _, r in df.iterrows():
        role = "report only" if r["role"] == "report" else "descriptive"
        spec = "sig3"
        ci = _ci(r["ci_lo"], r["ci_hi"], spec) if r["ci_status"] == "ok" else "not evaluable (tol0)"
        drop = _ci(r["supplement_rule_lo"], r["supplement_rule_hi"], spec)
        rows.append([name(r["estimand"]), _esc(r["convention"]), role, _n(r["point_estimate"], spec), ci,
                     fmt(r["n_nonfinite"], "int"), drop])
    W.table("tab_cn_paired", "lllrcrc", ["Estimand", "Conv.", "Role", "Point", "tol0 95\\% CI",
                                          "Non-finite", "Supplement rule"], rows)


PIPE_LABEL = {"saai": "regression", "direct": "direct-$x$"}


def _parse_mass(status):
    m = re.search(r"stationary mass ([0-9.eE+-]+)", str(status))
    return float(m.group(1)) if m else float("nan")


def _si_us(W):
    ver = W.csv("results/tables/k2/k2_verdicts.csv").df
    met = ver[ver["scope"] == "metro"]
    pdiff = W.csv("results/tables/k2/k2_paired_differences.csv").df
    pdl = pdiff[(pdiff["scope"] == "metro") & (pdiff["link"] == "overlap") & (pdiff["conv"] == "limit")]
    h3a = W.csv("results/tables/k2/k2_h3a.csv").df
    rows = []
    for out, ol in (("pi", "Personal income"), ("gdp", "GDP")):
        for per, pl in (("main", "main (2003--2024)"), ("secondary", "secondary (1993--2024)")):
            if out == "gdp" and per == "secondary":
                pl = "secondary"
            q = met[(met["outcome"] == out) & (met["period"] == per)]

            def one(h, **f):
                g = q[q["hypothesis"] == h]
                for k, v in f.items():
                    g = g[g[k] == v]
                if len(g) != 1:
                    raise SourceError(f"k2_verdicts: {h} {out} {per} {f}: {len(g)} rows")
                return g.iloc[0]
            h1 = q[(q["hypothesis"] == "H1") & (q["registered"].astype(str) == "True")]["value"].abs().max()
            h2 = one("H2", registered=True)
            h3 = one("H3a", link="overlap")
            hb = one("H3b", link="overlap", registered=True)
            hc = one("H3c", link="overlap")
            h4 = one("H4", link="overlap", registered=True)
            hs = one("H4_state", link="overlap")
            rows.append([ol if per == "main" else "", pl, _n(h1, "sci2"), _n(h2["value"], "int"),
                         f"{fmt(h3['n_transitions'], 'int')}; {_n(h3['value'], 'sci2')}",
                         _esc(str(hb["verdict"]).replace("not_evaluable_prediction_undefined",
                                                         "not evaluable (prediction undefined)")),
                         f"{_n(hc['value'], 'ssig2')} {_ci(hc['ci_lo'], hc['ci_hi'], 'ssig2')}",
                         f"{_n(h4['value'], 'sdec3')} {_ci(h4['ci_lo'], h4['ci_hi'], 'sdec3')}",
                         f"{_n(h4['value_rho1'], 'ssig2')} {_ci(h4['ci_lo_rho1'], h4['ci_hi_rho1'], 'ssig2')}",
                         _esc(h4["verdict"]), _esc(hs["verdict"])])
    spec = r"ll" + r">{\raggedleft\arraybackslash}p{0.06\linewidth}" + "r" * 2 + \
           r">{\raggedright\arraybackslash}p{0.1\linewidth}" + "lllll"
    W.table("tab_us_h", "lllllllllll", ["Outcome", "Period", "H1 max", "H2", "H3a: $n$; min $|M_t|$", "H3b",
                                         "H3c $\\Delta\\rho_1$ [95\\%]", "H4 $\\Delta\\xi^*$ [95\\%]",
                                         "H4 $\\Delta\\rho_1$ [95\\%]", "H4", "H4 state"], rows)

    cn = {"V0-V1": "V0$-$V1 (B)", "V0(B)-A": "V0(B)$-$A", "V1(B)-A": "V1(B)$-$A",
          "V1d(B)-A": "V1d(B)$-$A", "V1-V1d": "V1$-$V1d", "V1d_stay(B)-A": "V1d stayers$-$A"}
    rows = []
    for out, ol in (("pi", "Personal income"), ("gdp", "GDP")):
        for per in ("main", "secondary"):
            first = True
            for c in ("V0-V1", "V0(B)-A", "V1(B)-A", "V1d(B)-A", "V1-V1d"):
                g = pdl[(pdl["outcome"] == out) & (pdl["period"] == per) & (pdl["contrast"] == c)].set_index("quantity")
                x, r_ = g.loc["xi_trap"], g.loc["rho1"]
                rows.append([f"{ol}, {per}" if first else "", cn[c],
                             f"{_n(x['estimate'], 'sdec3')}",
                             _ci(x["lineage_tol0_lo"], x["lineage_tol0_hi"], "sdec3"),
                             _ci(x["state_tol0_lo"], x["state_tol0_hi"], "sdec3"),
                             f"{_n(r_['estimate'], 'ssig2')}",
                             _ci(r_["lineage_tol0_lo"], r_["lineage_tol0_hi"], "ssig2")])
                first = False
            rows.append("MID")
    W.table("tab_us_pairs", "llrccrc", ["Spec", "Contrast", "$\\Delta\\xi^*$", "lineage [95\\%]",
                                        "state [95\\%]", "$\\Delta\\rho_1$", "lineage [95\\%]"], rows[:-1])

    a = h3a[(h3a["scope"] == "metro") & (h3a["link"] == "overlap") & (h3a["panel"] == "B")
            & (h3a["period"] == "secondary")]
    pi = a[a["outcome"] == "pi"].set_index("year_t")
    gd = a[a["outcome"] == "gdp"].set_index("year_t")
    years = sorted(set(pi.index) | set(gd.index))
    rows = []
    for t in years:
        p = pi.loc[t] if t in pi.index else None
        g = gd.loc[t] if t in gd.index else None
        b = p if p is not None else g
        rows.append([f"{int(t)}--{(int(t) + 1) % 100:02d}", f"{_esc(b['vintage_t'])} $\\to$ {_esc(b['vintage_t1'])}",
                     fmt(b["n_S"], "int"), fmt(b["n_E"], "int"), fmt(b["n_L"], "int"), fmt(b["n_changed"], "int"),
                     _n(p["M_t"], "ssig2") if p is not None else "--",
                     _n(p["M_over_bound"], "dec2") if p is not None else "--",
                     _n(g["M_t"], "ssig2") if g is not None else "--",
                     _n(g["M_over_bound"], "dec2") if g is not None else "--"])
    W.table("tab_us_transitions", "llrrrrrrrr", ["Transition", "Delineations", "$n_S$", "$n_E$", "$n_L$",
                                                  "Redefined", "$M_t$ (PI)", "$M_t$/bound", "$M_t$ (GDP)",
                                                  "$M_t$/bound"], rows)


def _si_guard(W):
    h = W.csv("results/tables/k2/k2_h3b.csv").df
    rows = []
    lab = {("pi", "main", "overlap"): ("US PI main", "registered (primary)"),
           ("pi", "main", "code"): ("US PI main, code link", "report only"),
           ("pi", "secondary", "overlap"): ("US PI secondary", "registered"),
           ("gdp", "main", "overlap"): ("US GDP main", "registered"),
           ("gdp", "secondary", "overlap"): ("US GDP secondary", "registered")}
    n_refused = 0
    for key, (name, role) in lab.items():
        g = h[(h["scope"] == "metro") & (h["outcome"] == key[0]) & (h["period"] == key[1]) & (h["link"] == key[2])]
        if len(g) != 1:
            raise SourceError(f"k2_h3b: {key} matched {len(g)}")
        r = g.iloc[0]
        if str(r["status"]) == "ok":
            gp = _n(r["pred_xi_trap"], "ssig3")
        else:
            gp = f"undefined (mass {_n(_parse_mass(r['status']), 'sci2')})"
        if str(r["unguarded_status"]) == "ok":
            up = _n(r["unguarded_pred_xi_trap"], "ssig3")
        else:
            up = f"refused (mass {_n(_parse_mass(r['unguarded_status']), 'sci2')})"
            n_refused += 1
        gd = r["guard"]
        gl = [float(x) for x in re.findall(r"[-0-9.eE]+", str(gd))]
        guard = f"[{fmt(gl[0], 'sdec2')}, {fmt(gl[1], 'sdec2')}]; {fmt(gl[2], 'ssig3')}" if len(gl) == 3 else "--"
        verdict = str(r["verdict"]).replace("not_evaluable_prediction_undefined", "not evaluable")
        rows.append([name, role, _n(r["observed"], "ssig3"), gp, up, _n(r["zeroth_order_xi_trap"], "ssig3"),
                     _n(R.v("h3b.gate_factor") * float(r["half_width_V1"]), "sig3"),
                     guard, _esc(verdict)])
    rows.append("MID")
    pr = W.csv("results/tables/k2br_predictions.csv").df
    bv = W.csv("results/tables/k2br_verdicts.csv").df
    for win, out, name in (("ref2010", "pib", "BR ref2010 GDP"), ("ref2010", "vab_agro", "BR ref2010 agri.\\ VAB"),
                           ("ref2002", "pib", "BR ref2002 GDP"), ("ref2002", "vab_agro", "BR ref2002 agri.\\ VAB")):
        p = pr[(pr["window"] == win) & (pr["outcome"] == out)].iloc[0]
        v = bv[(bv["window"] == win) & (bv["outcome"] == out)].iloc[0]
        gp = _n(p["pred_dxi_star"], "ssig3") if p["status"] == "ok" else "undefined"
        up = _n(p["unguarded_pred_dxi_star"], "ssig3") if p["unguarded_status"] == "ok" else "refused"
        guard = f"[{_n(p['guard_lo'], 'sdec2')}, {_n(p['guard_hi'], 'sdec2')}]; {_n(p['guard_slope_floor'], 'ssig3')}"
        rows.append([name, "registered" + (" (primary)" if win == "ref2010" and out == "pib" else ""),
                     _n(v["H3b_observed"], "ssig3"), gp, up, _n(p["zeroth_order_dxi_star"], "ssig3"),
                     _n(v["H3b_gate_threshold"], "sig3"), guard, _esc(v["H3b"])])
    W.table("tab_guard", "llrrrrrll", ["Specification", "Role", "Observed", "Guarded", "Unguarded",
                                        "$\\bar M/|a_1|$", "Gate", "Guard support; slope", "H3b"], rows)
    W.macro("SiUsUnguardedRefusedAll", fmt(n_refused, "int"), "results/tables/k2/k2_h3b.csv",
            "US metro specs (incl. code link) where the unguarded projection refused")


def _si_identify(W):
    """Post-hoc identifiability diagnostic of the Chinese province-block draws."""
    fr = W.csv("results/tables/expi_identify_fractions.csv").df
    rows = []
    for _, r in fr.iterrows():
        rows.append([_esc(r["version"]), "first" if r["window"] == "pre" else "second",
                     fmt(r["n_draws"], "int"), f"{fmt(r['n_identified'], 'int')} ({_n(r['frac_identified'], 'dec3')})",
                     fmt(r["n_not_identified"], "int"), fmt(r["n_fit_failed"], "int"),
                     fmt(r["n_multiple_stable_zeros"], "int"), fmt(r["n_subgrid_path"], "int"),
                     fmt(r["n_nan_convention_nonfinite"], "int")])
    W.table("tab_cn_identify", "llrrrrrrr",
            ["Version", "Window", "Draws", "Identified (share)", "Not identified", "Fit failed",
             "$\\geq2$ stable zeros", "Short path", "Non-finite (nan)"], rows)

    iv = W.csv("results/tables/expi_identify_intervals.csv").df
    cv = W.csv("results/tables/expi_identify_conventions.csv").df
    lab = {"xi_trap_V0_pre": "$\\xi^*_{V0}$, first window", "xi_trap_V0_during": "$\\xi^*_{V0}$, second window",
           "xi_trap_V1_pre": "$\\xi^*_{V1}$, first window", "xi_trap_V1_during": "$\\xi^*_{V1}$, second window",
           "xi_trap_V0_during_minus_pre": "V0 shift (second$-$first)",
           "xi_trap_V1_during_minus_pre": "V1 shift (second$-$first)",
           "xi_trap_V0_minus_V1_pre": "$\\xi^*_{V0}-\\xi^*_{V1}$, first window",
           "xi_trap_V0_minus_V1_during": "$\\xi^*_{V0}-\\xi^*_{V1}$, second window",
           "xi_shift_V0_minus_V1": "shift difference V0$-$V1",
           "tau_ratio_V0_over_V1": "$\\tau$-ratio V0/V1 (report only)"}
    rows, rows_c = [], []
    for e, name in lab.items():
        spec = "sig3" if e.startswith("tau") else "sdec3"
        a = iv[(iv["estimand"] == e) & (iv["convention"] == "limit") & (iv["subset"] == "all")]
        b = iv[(iv["estimand"] == e) & (iv["convention"] == "limit") & (iv["subset"] == "identified")]
        c = cv[(cv["estimand"] == e) & (cv["subset"] == "all")]
        if len(a) != 1 or len(b) != 1 or len(c) != 1:
            raise SourceError(f"identify tables: estimand {e} rows missing")
        a, b, c = a.iloc[0], b.iloc[0], c.iloc[0]
        flags = iv[(iv["estimand"] == e) & iv["subset"].isin(["all", "identified"])][
            ["tol0_contains_null", "tol5_contains_null", "supp_contains_null"]]
        flags = flags.stack().dropna().astype(str)
        call = "yes" if (flags == "True").all() else "no"
        nan_tol5 = _ci(c["nan_tol5_lo"], c["nan_tol5_hi"], spec)
        rows.append([name, _n(a["point_estimate"], spec), _ci(a["tol0_lo"], a["tol0_hi"], spec),
                     f"{_ci(b['tol0_lo'], b['tol0_hi'], spec)} ({fmt(b['n_subset'], 'int')})", call])
        rows_c.append([name, "none ($>$5\\% non-finite)" if nan_tol5 == "--" else nan_tol5,
                       _ci(c["nan_supp_lo"], c["nan_supp_hi"], spec), fmt(c["n_nan_nonfinite_only"], "int")])
    W.table("tab_cn_identify_ci", "lrccc",
            ["Estimand", "Point", "All draws, tol0 (primary)", "Identified only, tol0 ($n$)",
             "Null inside, every rule"], rows)
    W.table("tab_cn_identify_conv", "lccr",
            ["Estimand", "nan, tol5", "nan, supplement rule", "Extra non-finite (nan)"], rows_c)


def _si_guard_sens(W):
    """Post-hoc guard sensitivity: every row of k1_guard_sensitivity.csv."""
    gs = W.csv("results/tables/k1_guard_sensitivity.csv").df
    names = {"metro|pi|main|B|overlap": "US PI main (primary)", "ref2010|pib": "BR ref2010 GDP",
             "ref2010|vab_agro": "BR ref2010 agri.\\ VAB", "ref2002|pib": "BR ref2002 GDP",
             "ref2002|vab_agro": "BR ref2002 agri.\\ VAB"}
    floors = sorted(gs["floor_factor"].unique())
    rows = []
    for cell, name in names.items():
        g = gs[gs["cell"] == cell]
        if not len(g):
            raise SourceError(f"k1_guard_sensitivity: cell {cell} missing")
        first = True
        for (lo, hi), q in g.groupby(["support_lo_pct", "support_hi_pct"], sort=True):
            cells = []
            for ff in floors:
                r = q[q["floor_factor"] == ff]
                if len(r) != 1:
                    raise SourceError(f"k1_guard_sensitivity: {cell} {lo}/{hi} floor {ff}: {len(r)} rows")
                r = r.iloc[0]
                if str(r["status"]) == "ok":
                    s = f"{_n(r['pred_xi_trap'], 'ssig3')} ({_n(r['pred_over_registered'], 'dec2')})"
                else:
                    s = f"undefined (mass {_n(_parse_mass(r['status']), 'sci2')})"
                if str(r["registered"]) == "True":
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            if cell == "metro|pi|main|B|overlap":
                tol_ = float(W.json("results/tables/k2br_meta.json").get("constants.H3B_REL_TOL"))
                okq = q[q["status"] == "ok"]
                marks = []
                for ff in floors:
                    r = q[q["floor_factor"] == ff].iloc[0]
                    if str(r["status"]) != "ok":
                        marks.append("--")
                        continue
                    err = abs(float(r["observed_V0_minus_V1"]) - float(r["pred_xi_trap"]))
                    marks.append(f"{_n(err, 'dec3')} ({'yes' if err <= tol_ * abs(float(r['pred_xi_trap'])) else 'no'})")
                tolcol = " / ".join(marks)
            else:
                tolcol = "--"
            ung = q["pred_over_unguarded"].dropna()
            ungs = (f"{fmt(float(ung.min()), 'dec1')}--{fmt(float(ung.max()), 'dec1')}" if ung.nunique() > 1
                    else (fmt(float(ung.iloc[0]), "dec1") if len(ung) else "--"))
            rows.append([name if first else "", _n(g.iloc[0]["observed_V0_minus_V1"], "ssig3") if first else "",
                         f"{fmt(lo, 'dec1')}--{fmt(hi, 'dec1')}"] + cells + [ungs, tolcol])
            first = False
        rows.append("MID")
    hdr = ["Cell", "Observed", "Support (percentiles)"] + \
          [f"floor $-{fmt(ff, 'dec1')}|a_1|$" for ff in floors] + ["Guarded/unguarded",
                                                                    "$|$obs.$-$proj.$|$ (within tol.)"]
    W.table("tab_guard_sens", "llc" + "c" * len(floors) + "c" +
            r">{\raggedright\arraybackslash}p{0.2\linewidth}", hdr, rows[:-1])


def _si_brazil(W):
    bv = W.csv("results/tables/k2br_verdicts.csv").df
    ct = W.csv("results/tables/k2br_contrasts.csv").df
    ct = ct[ct["convention"] == "limit"]
    rows = []
    names = {("ref2010", "pib"): "ref2010, GDP (primary)", ("ref2010", "vab_agro"): "ref2010, agri.\\ VAB",
             ("ref2002", "pib"): "ref2002, GDP", ("ref2002", "vab_agro"): "ref2002, agri.\\ VAB"}
    for (win, out), name in names.items():
        v = bv[(bv["window"] == win) & (bv["outcome"] == out)].iloc[0]
        c = ct[(ct["window"] == win) & (ct["outcome"] == out) & (ct["contrast"] == "V1B_minus_A")].set_index("quantity")
        x, r_ = c.loc["xi_trap"], c.loc["rho1"]
        rows.append([name, _n(v["H1_max_residual"], "sci2"), _n(v["H2_pairs_max_diff"], "int"),
                     f"{fmt(v['H3a_n_entry_transitions'], 'int')}; {_n(v['H3a_min_abs_M'], 'sci2')}",
                     _esc(v["H3b"]),
                     f"{_n(v['H3c_estimate'], 'ssig2')} {_ci(v['H3c_ci_lo'], v['H3c_ci_hi'], 'ssig2')}",
                     f"{_n(x['estimate'], 'ssig2')} {_ci(x['amc_ci_lo'], x['amc_ci_hi'], 'ssig2')}",
                     f"{_n(r_['estimate'], 'ssig2')} {_ci(r_['amc_ci_lo'], r_['amc_ci_hi'], 'ssig2')}",
                     f"{_esc(v['H4_amc'])} / {_esc(v['H4_immediate'])} / {_esc(v['H4_uf'])}"])
    W.table("tab_br_h", "lllllllll", ["Window, outcome", "H1 max", "H2", "H3a: $n$; min $|M_t|$", "H3b",
                                       "H3c $\\Delta\\rho_1$ [95\\%]", "H4 $\\Delta\\xi^*$ [95\\%]",
                                       "H4 $\\Delta\\rho_1$ [95\\%]", "H4 AMC / region / state"], rows)

    cn = {"V0B_minus_V1B": "V0(B)$-$V1(B)", "V0B_minus_A": "V0(B)$-$A", "V1B_minus_A": "V1(B)$-$A",
          "V1pB_minus_A": "V1p(B)$-$A", "V1mB_minus_A": "V1m(B)$-$A", "V1B_minus_V1mB": "V1(B)$-$V1m(B)"}
    rows = []
    for (win, out), name in names.items():
        first = True
        for cc, cl in cn.items():
            c = ct[(ct["window"] == win) & (ct["outcome"] == out) & (ct["contrast"] == cc)].set_index("quantity")
            x, r_ = c.loc["xi_trap"], c.loc["rho1"]
            rows.append([name if first else "", cl, _n(x["estimate"], "ssig2"),
                         _ci(x["amc_ci_lo"], x["amc_ci_hi"], "ssig2"), _n(r_["estimate"], "ssig2"),
                         _ci(r_["amc_ci_lo"], r_["amc_ci_hi"], "ssig2")])
            first = False
        rows.append("MID")
    W.table("tab_br_variants", "llrcrc", ["Window, outcome", "Contrast", "$\\Delta\\xi^*$", "AMC block [95\\%]",
                                          "$\\Delta\\rho_1$", "AMC block [95\\%]"], rows[:-1])

    d = W.csv("results/tables/k2br_decomposition.csv").df
    d = d[(d["panel"] == "B") & (d["family"] == "V0") & (d["n_E"] > 0)].sort_values(["window", "outcome", "year_t"],
                                                                                   ascending=[False, True, True])
    rows = []
    for _, r in d.iterrows():
        rows.append([_esc(r["window"]), _esc("GDP" if r["outcome"] == "pib" else "agri. VAB"),
                     f"{int(r['year_t'])}--{(int(r['year_t']) + 1) % 100:02d}", fmt(r["n_S"], "int"),
                     fmt(r["n_E"], "int"), fmt(r["n_P"], "int"), _n(r["m_E"], "sdec2"), _n(r["M"], "ssig2"),
                     _n(r["M_over_bound"], "dec2"), _n(abs(r["identity_residual"]), "sci1")])
    W.table("tab_br_entry", "llrrrrrrrr", ["Window", "Outcome", "Transition", "$n_S$", "$n_E$", "Parents",
                                            "$m_E$", "$M_t$", "$M_t$/bound", "$|$Lemma 1 resid.$|$"], rows)


_MASK_DOSE_TEX = {"us_like": "US-like", "intermediate": "intermediate", "china_like": "China-like"}
_MASK_SYS_TEX = {"US": "US", "BR": "Brazil"}


def _mask_cell_tex(cell):
    lv, pos, org, tim = str(cell).split("|")
    return f"{_MASK_DOSE_TEX[lv]}, {pos}, {org}, {tim}"


def _si_mask(W):
    """Controlled masking: verdicts, report-only strata, per dose, residuals, cells, guards."""
    ev = W.csv(f"{MASK_DIR}/k4_evaluation.csv")
    _, fz, _ = _mask_eval()
    W.reads.add(f"{MASK_DIR}/k4_points.csv")
    _, p = mask_points()
    cells = W.csv(f"{MASK_DIR}/k4_cells.csv").df
    pj = W.csv(f"{MASK_DIR}/k4_projections.csv").df

    def m(S_, scope, name):
        return _mrow(ev, S_, scope, name).s

    def vb(S_, scope, name):
        r = fz[(fz["system"] == S_) & (fz["scope"] == scope) & (fz["name"] == name)]
        if len(r) != 1:
            raise SourceError(f"verdict {S_}/{scope}/{name}: {len(r)} rows")
        val = str(r.iloc[0]["value"])
        return MASK_VERDICT_TEXT.get(val, _esc(val.replace("_", " ")))

    def vci(S_, scope, name, spec):
        r = m(S_, scope, name)
        ci = _ci(r["cluster_ci_lo"], r["cluster_ci_hi"], spec)
        return _n(r["value"], spec) + ("" if ci == "--" else f" {ci}")

    crit = lambda k: float(W.json(f"{MASK_DIR}/k4_meta.json").get(f"criteria.{k}"))
    sb = W.json(f"{MASK_DIR}/k4_meta.json").get("criteria.slope_band")
    spec_rows = [
        ("T1 point", "P0 point availability", "point_availability", "dec3", f"$\\ge {fmt(crit('point_availability_min'), 'dec2')}$", "T1_point_P0_availability"),
        ("", "P1 calibration slope", "calibration_slope", "dec3", f"CI $\\subset[{fmt(sb[0], 'dec2')}, {fmt(sb[1], 'dec2')}]$", "T1_point_P1_slope"),
        ("", "P2 intercept$/s$", "intercept_over_s", "sdec3", f"CI $\\subset[-{fmt(crit('intercept_band_rel'), 'dec2')}, {fmt(crit('intercept_band_rel'), 'dec2')}]$", "T1_point_P2_intercept"),
        ("", "P3 sign agreement (gated)", "sign_agreement_gated", "dec2", f"$\\ge {fmt(crit('sign_agreement_min'), 'dec2')}$", "T1_point_P3_sign"),
        ("", "P4 per-dose median error", None, None, f"$\\le {fmt(crit('level_abs_error_rel'), 'dec2')}\\times$ median $|O|$", "T1_point_P4_levels"),
        ("", "verdict", None, None, "", "T1_point"),
        ("Envelope", "R0 availability", "range_availability", "dec2", f"$\\ge {fmt(crit('range_availability_min'), 'dec2')}$", "T1_range_R0_availability"),
        ("", "R1 coverage", "range_coverage", "dec3", f"$\\ge {fmt(crit('range_coverage_min'), 'dec2')}$", "T1_range_R1_coverage"),
        ("", "R2 width$/|O|$ (gated)", "range_width_rel_median_gated", "dec3", f"$\\le {fmt(crit('range_width_rel_max'), 'dec1')}$", "T1_range_R2_width"),
        ("", "verdict", None, None, "", "T1_range"),
        ("T2 recovery", "V1 share $|T_2|<|T_3|$", "recovery_share_v1_closer", "dec2", f"$\\ge {fmt(crit('recovery_share_min'), 'dec2')}$, CI low $> {fmt(crit('recovery_share_ci_lo_min'), 'dec2')}$", "T2_V1_share"),
        ("", "V2 median $|T_2|/$median $|T_3|$", "recovery_median_ratio", "dec3", f"$\\le {fmt(crit('recovery_median_ratio_max'), 'dec2')}$", "T2_V2_ratio"),
        ("", "V3 share $|T_2|<|T_{\\mathrm{naive}}|$", "recovery_share_v1_beats_naive", "dec3", f"CI low $> {fmt(crit('comparator_share_ci_lo_min'), 'dec2')}$", "T2_V3_beats_naive"),
        ("", "verdict", None, None, "", "T2"),
        ("Gates", "T1 gate availability", "gate_T1_availability", "dec2", f"$\\ge {fmt(crit('gate_availability_min'), 'dec2')}$", "T1_gate_availability"),
        ("", "T3 gate availability", "gate_T3_availability", "dec2", f"$\\ge {fmt(crit('gate_availability_min'), 'dec2')}$", "T2_gate_availability"),
        ("T3 positive control", "low cells: median $T_3$", "pc_low_median_T3", "sdec3", f"$<-{fmt(crit('power_factor'), 'int')}\\times$ median half-width", "T3_powered_low"),
        ("", "high cells: median $T_3$", "pc_high_median_T3", "sdec3", f"$>{fmt(crit('power_factor'), 'int')}\\times$ median half-width", "T3_powered_high"),
        ("", "verdict", None, None, "", "T3_powered"),
        ("Null control", "N1 null median $|O|$", None, None, f"$\\le {fmt(crit('null_rel_max'), 'dec2')}\\times$ active median $|O|$", "N1_null_obs"),
        ("", "N2 null median $|P|$", None, None, f"$\\le {fmt(crit('null_rel_max'), 'dec2')}\\times$ active median $|O|$", "N2_null_pred"),
        ("", "verdict", None, None, "", "N_null_control"),
    ]
    rows = []
    for test, lab, metric, spec, rule, vname in spec_rows:
        cells_ = [test, lab, rule]
        for S_ in ("US", "BR"):
            if vname in ("N1_null_obs", "N2_null_pred"):
                nm = "median_abs_obs" if vname == "N1_null_obs" else "median_abs_pred_point"
                val = f"{_n(m(S_, 'null', nm)['value'], 'dec3')} vs {fmt(crit('null_rel_max') * float(m(S_, 'active', 'median_abs_obs')['value']), 'dec3')}"
            elif metric is None and vname == "T1_point_P4_levels":
                parts = []
                for lv, _L in MASK_DOSES:
                    n_ = float(m(S_, "active", f"level_{lv}_n")["value"])
                    e_ = float(m(S_, "active", f"level_{lv}_median_abs_err")["value"])
                    o_ = float(m(S_, "active", f"level_{lv}_median_abs_obs")["value"])
                    parts.append(f"{fmt(e_ / o_, 'dec2')} ({fmt(e_, 'dec3')} of {fmt(o_, 'dec2')}; $n$={fmt(n_, 'int')})")
                val = "; ".join(parts)
            elif metric is None:
                val = ""
            elif metric in ("pc_low_median_T3", "pc_high_median_T3"):
                hw = "pc_low_median_halfwidth_T3" if "low" in metric else "pc_high_median_halfwidth_T3"
                val = f"{_n(m(S_, 'active', metric)['value'], spec)} (half-width {_n(m(S_, 'active', hw)['value'], 'dec3')})"
            else:
                val = vci(S_, "active", metric, spec)
            v_ = vb(S_, "active", vname)
            if vname in ("T1_point", "T1_range", "T2", "T3_powered", "N_null_control"):
                val = f"\\textbf{{{v_}}}"
                cells_ += [val, ""]
            else:
                cells_ += [val, {"True": "yes", "False": "no"}.get(str(fz[(fz['system'] == S_) & (fz['scope'] == 'active') & (fz['name'] == vname)].iloc[0]['value']), v_)]
        rows.append(cells_)
        if lab == "verdict":
            rows.append("MID")
    W.table("tab_mask_verdicts", r"l>{\raggedright\arraybackslash}p{0.17\linewidth}>{\raggedright\arraybackslash}p{0.15\linewidth}"
            r">{\raggedright\arraybackslash}p{0.2\linewidth}c>{\raggedright\arraybackslash}p{0.2\linewidth}c",
            ["Test", "Criterion", "Frozen rule", "US (pilot-informed)", "met", "Brazil", "met"], rows[:-1])

    rows = []
    for S_ in ("US", "BR"):
        for sc in ("balanced", "null", "pooled"):
            rows.append([_MASK_SYS_TEX[S_], _esc(sc), _n(m(S_, sc, "n_masks")["value"], "int"),
                         _n(m(S_, sc, "point_availability")["value"], "dec2"), vb(S_, sc, "T1_point"),
                         vci(S_, sc, "calibration_slope", "dec2"), vci(S_, sc, "intercept_over_s", "sdec2"),
                         vb(S_, sc, "T1_range"), vci(S_, sc, "range_coverage", "dec2"),
                         _n(m(S_, sc, "range_width_rel_median_gated")["value"], "dec2"), vb(S_, sc, "T2"),
                         vci(S_, sc, "recovery_median_ratio", "dec2"), vci(S_, sc, "recovery_share_v1_beats_naive", "dec2")])
        rows.append("MID")
    W.table("tab_mask_strata", "llrrlllllrlll",
            ["System", "Stratum", "$n$", "Point avail.", "T1 point", "Slope [95\\%]", "Intercept$/s$ [95\\%]",
             "Envelope", "Coverage [95\\%]", "Width$/|O|$", "T2", "V2 [95\\%]", "V3 [95\\%]"], rows[:-1])

    rows = []
    for S_ in ("US", "BR"):
        for lv, _L in MASK_DOSES:
            q = p[(p["system"] == S_) & (p["stratum"] == "active") & (p["fraction_level"] == lv)]
            s = mask_summary(q)
            rows.append([_MASK_SYS_TEX[S_], _MASK_DOSE_TEX[lv], fmt(s["n"], "int"),
                         f"{fmt(s['n_point'], 'int')} ({fmt(s['point_share'], 'dec2')})",
                         _n(s["med_obs_point"], "dec3"), _n(s["med_err_point"], "dec3"), _n(s["med_err_zeroth"], "dec3"),
                         _n(s.get("slope_point"), "dec2"), _n(s.get("slope_zeroth"), "dec2"),
                         _n(s["coverage"], "dec2"), _n(s["width_gated"], "dec2"), _n(s["med_T3"], "dec3"),
                         _n(s["med_T2"], "dec3"), _n(s["ratio"], "dec3"), _n(s["naive_share"], "dec2")])
        rows.append("MID")
    W.table("tab_mask_dose", "llrrrrrrrrrrrrr",
            ["System", "Dose", "$n$", "Point avail.", "med.\\ $|O|$", "med.\\ $|O-P|$", "med.\\ $|O-Z|$",
             "slope $O\\sim P$", "slope $O\\sim Z$", "Coverage", "Width$/|O|$", "med.\\ $|T_3|$", "med.\\ $|T_2|$",
             "V2", "V3"], rows[:-1])

    rows = []
    for S_ in ("US", "BR"):
        for sc in ("active", "balanced", "null"):
            for pos in (("low", "high") if sc != "null" else ("mid",)):
                q = p[(p["system"] == S_) & (p["stratum"] == sc) & (p["position"] == pos)]
                s = mask_summary(q)
                rows.append([_MASK_SYS_TEX[S_], _esc(sc), _esc(pos), fmt(s["n"], "int"), fmt(s["n_gated_T3"], "int"),
                             _n(s["med_T3"], "dec3"), _n(s["med_sel"], "dec3"), _n(s["med_event"], "dec3"),
                             _n(s["med_T2"], "dec3"), _n(s["med_T2FS"], "dec3"), _n(s["med_naive"], "dec3"),
                             _n(s["med_T2_signed"], "sdec3"), _n(s["med_sel_signed"], "sdec3"),
                             _n(s["slope_T2_on_sel"], "sdec2"), _n(s["corr_T2_sel"], "sdec2"),
                             _n(s["v1_share"], "dec2"), _n(s["naive_share"], "dec2")])
        rows.append("MID")
    W.table("tab_mask_resid", "lllrrrrrrrrrrrrrr",
            ["System", "Stratum", "Position", "$n$", "gated", "$|T_3|$", "$|F_S-F|$", "$|V0-F_S|$", "$|T_2|$",
             "$|V1-F_S|$", "$|T_{\\mathrm{naive}}|$", "$T_2$", "$F_S-F$", "slope", "corr.", "V1 share", "V3 share"],
            rows[:-1])

    order = {"active": 0, "balanced": 1, "null": 2}
    for S_, P in MASK_SYS:
        c = cells[cells["system"] == S_].copy()
        if len(c) != 36:
            raise SourceError(f"k4_cells.csv: {S_} has {len(c)} cells")
        c["_o"] = c["stratum"].map(order)
        c = c.sort_values(["_o", "cell"])
        rows = []
        for _, r in c.iterrows():
            rows.append([_esc(r["stratum"]), _mask_cell_tex(r["cell"]), _n(r["mean_delta_event"], "dec2"),
                         _n(r["mean_M_bar"], "ssig2"), _n(r["point_availability"], "dec1"), _n(r["mean_T1_obs"], "sdec3"),
                         _n(r["mean_pred_point"], "sdec3"), _n(r["median_abs_err_point"], "dec3"),
                         _n(r["range_coverage"], "dec1"), _n(r["mean_T3"], "sdec3"), _n(r["mean_T3_selection"], "sdec3"),
                         _n(r["mean_T2"], "sdec3"), _n(r["mean_T_naive"], "sdec3"),
                         _n(r["recovery_share_v1_closer"], "dec1"), _n(r["recovery_share_v1_beats_naive"], "dec1")])
        W.table(f"tab_mask_cells_{S_.lower()}", "llrrrrrrrrrrrrr",
                ["Stratum", "Cell (dose, position, organization, timing)", "$\\delta_{\\mathrm{event}}$", "$\\bar M$",
                 "Point avail.", "$O$", "$P$", "med.\\ $|O-P|$", "Cover.", "$T_3$", "$F_S-F$", "$T_2$",
                 "$T_{\\mathrm{naive}}$", "$|T_2|<|T_3|$", "$|T_2|<|T_{\\mathrm{naive}}|$"], rows)

    act = p[p["stratum"] == "active"][["mask_id", "system", "T1_obs", "fraction_level"]]
    g = pj.merge(act, on="mask_id", how="inner")
    if g["mask_id"].nunique() != 240:
        raise SourceError(f"k4_projections: {g['mask_id'].nunique()} active masks, need 240")
    rows = []
    for (lo, hi, ff), q in g.groupby(["support_lo_pct", "support_hi_pct", "floor_factor"], sort=True):
        reg = bool(q["registered_guard"].astype(str).eq("True").all())
        row = [f"{fmt(lo, 'dec1')}--{fmt(hi, 'dec1')}", f"$-{fmt(ff, 'dec1')}|a_1|$" + (" (point)" if reg else "")]
        for S_ in ("US", "BR"):
            r = q[q["system"] == S_]
            ok = r[r["status"] == "ok"]
            row += [fmt(len(ok) / len(r), "dec2"),
                    _n(float((ok["T1_obs"] - ok["pred"]).abs().median()) if len(ok) else float("nan"), "dec3"),
                    _n(float((ok["pred"].abs() > ok["T1_obs"].abs()).mean()) if len(ok) else float("nan"), "dec2")]
        rows.append(row)
    W.table("tab_mask_guards", "llrrrrrr",
            ["Support (percentiles)", "Slope floor", "US defined", "US med.\\ $|O-\\mathrm{pred}|$",
             "US $|\\mathrm{pred}|>|O|$", "BR defined", "BR med.\\ $|O-\\mathrm{pred}|$", "BR $|\\mathrm{pred}|>|O|$"],
            rows)


def _si_constants(W):
    bm = W.json("results/tables/k2br_meta.json")
    for k, n in (("constants.H1_TOL", "SiHoneTol"), ("constants.H2_TOL", "SiHtwoTol"),
                 ("constants.H3A_TOL", "SiHthreeaTol")):
        W.macro(n, fmt(float(bm.get(k)), "sci1"), bm.path, k)
    fz = W.json("results/tables/k1_theory_constants.json")
    W.macro("SiEdgeSd", fmt(float(fz.get("guard_constants.EDGE_SD")), "int"), fz.path, "EDGE_SD")
    W.macro("SiEdgeMassTol", fmt(float(fz.get("guard_constants.EDGE_MASS_TOL")), "sci1"), fz.path, "EDGE_MASS_TOL")
    W.macro("SiTheoryVersion", _esc(fz.get("theory_version")), fz.path, "theory_version")
    km = W.json("results/tables/k2/k2_meta.json")
    W.macro("SiUsSeed", str(int(km.get("seed"))), km.path, "seed")
    W.macro("SiBrSeed", str(int(bm.get("seed"))), bm.path, "seed")
    W.macro("SiBrB", fmt(float(bm.get("n_boot")), "int"), bm.path, "n_boot")


def si_tables():
    W = _SiWriter()
    _si_k1(W)
    _si_china(W)
    _si_paired(W)
    _si_us(W)
    _si_guard(W)
    _si_identify(W)
    _si_guard_sens(W)
    _si_brazil(W)
    _si_mask(W)
    _si_constants(W)
    os.makedirs(SI_DIR, exist_ok=True)
    keep = set(W.files) | {f"{SI_DIR}/si_values.tex"}
    for f in os.listdir(SI_DIR):
        p = f"{SI_DIR}/{f}"
        if p not in keep and f.endswith(".tex"):
            with open(p, encoding="utf-8") as fh:
                first = fh.readline()
            if GENERATED_TAG in first:
                os.remove(p)
    for p, txt in sorted(W.files.items()):
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt)
    lines = ["% si_values.tex -- GENERATED by scripts/paper_k1_numbers.py (si_tables); do not edit.",
             "% Provenance: the files named after each macro.", ""]
    for name, tex, src, note in W.macros:
        lines.append(f"\\newcommand{{\\{name}}}{{{tex}}}% {note} <- {src}")
    with open(f"{SI_DIR}/si_values.tex", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    shas = {p: _SHA[p] for p in sorted(W.reads) if p in _SHA}
    with open(SI_META, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"script": "scripts/paper_k1_numbers.py:si_tables", "n_tables": len(W.files),
                   "n_macros": len(W.macros), "inputs_sha256": shas},
                  fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"paper_k1_numbers: si_tables: {len(W.files)} tables, {len(W.macros)} macros -> {SI_DIR}/")


def main():
    china()
    k1()
    us()
    brazil()
    diagnostics()
    posthoc()
    ladder()
    masking()
    write()
    si_tables()
    print(f"paper_k1_numbers: {len(R.rows)} values -> {OUT_CSV}, {OUT_TEX}")


if __name__ == "__main__":
    try:
        main()
    except SourceError as e:
        print(f"paper_k1_numbers: FAILED: {e}", file=sys.stderr)
        sys.exit(1)
