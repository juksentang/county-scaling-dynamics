"""
DataV current-vintage county boundaries + panel match diagnostic.

Phase 1 — fetch Aliyun DataV GeoJSON county boundaries (current vintage):
  traverse 100000_full.json -> province _full.json -> prefecture _full.json,
  collect level=='district' features (municipalities expose districts at the
  province level), keep counties with adcode % 100 >= 21 (same rule as
  src/data_cleaner.py filter_county_type), simplify geometry, cache to
  data/boundaries/datav_counties.geojson.

Phase 2 — match diagnostic against the panel:
  match rates by county / county-year / year / u_GDP decile.
  Decision rule: county-year >= 95% AND top-u-decile county match >= 90%
  -> single vintage suffices; otherwise multi-vintage patch required.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import requests
from shapely.geometry import shape, mapping

BASE = "https://geo.datav.aliyun.com/areas_v3/bound/{adcode}_full.json"
OUT_GEO = "data/boundaries/datav_counties.geojson"
OUT_DIAG = "results/tables/boundary_match_diagnostic.csv"
MUNICIPALITIES = {110000, 120000, 310000, 500000}


def fetch(adcode: int, sess: requests.Session, retries: int = 3) -> dict | None:
    for i in range(retries):
        try:
            r = sess.get(BASE.format(adcode=adcode), timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            time.sleep(1 + i)
    return None


def phase1_fetch() -> list[dict]:
    os.makedirs("data/boundaries", exist_ok=True)
    if os.path.exists(OUT_GEO):
        print(f"cache hit: {OUT_GEO}")
        return json.load(open(OUT_GEO))["features"]

    sess = requests.Session()
    sess.headers["User-Agent"] = "EconP-boundary-fetch/1.0"
    china = fetch(100000, sess)
    provinces = [f["properties"]["adcode"] for f in china["features"]]
    print(f"{len(provinces)} provinces")

    feats, n_req = [], 0
    for prov in provinces:
        pj = fetch(prov, sess)
        n_req += 1
        if pj is None:
            continue
        children = pj["features"]
        if prov in MUNICIPALITIES:
            pref_feats = children          # districts directly
            pref_codes = []
        else:
            pref_codes = [f["properties"]["adcode"] for f in children]
            pref_feats = [f for f in children
                          if f["properties"].get("level") == "district"]
        for f in pref_feats:
            if f["properties"].get("level") == "district":
                feats.append(f)
        for pref in pref_codes:
            cj = fetch(pref, sess)
            n_req += 1
            time.sleep(0.05)
            if cj is None:
                continue
            for f in cj["features"]:
                if f["properties"].get("level") == "district":
                    feats.append(f)
        print(f"  prov {prov}: cumulative districts={len(feats)} req={n_req}")

    # dedupe by adcode, filter counties (CC>=21), simplify
    seen, out = set(), []
    for f in feats:
        ad = int(f["properties"]["adcode"])
        if ad in seen or ad % 100 < 21:
            continue
        seen.add(ad)
        try:
            geom = shape(f["geometry"]).simplify(0.002, preserve_topology=True)
            f = {"type": "Feature",
                 "properties": {"adcode": ad,
                                "name": f["properties"].get("name", "")},
                 "geometry": mapping(geom)}
        except Exception:
            f = {"type": "Feature",
                 "properties": {"adcode": ad,
                                "name": f["properties"].get("name", "")},
                 "geometry": f["geometry"]}
        out.append(f)
    json.dump({"type": "FeatureCollection", "features": out},
              open(OUT_GEO, "w"))
    print(f"saved {len(out)} county features -> {OUT_GEO} "
          f"({os.path.getsize(OUT_GEO)/1e6:.1f} MB)")
    return out


def phase2_diagnostic(feats: list[dict]) -> None:
    adcodes = {f["properties"]["adcode"] for f in feats}
    panel = pd.read_parquet("data_clean/panel_full.parquet",
                            columns=["CountyCode", "Year", "NonAgriGDP",
                                     "RegGDP"])
    panel["matched"] = panel["CountyCode"].astype(int).isin(adcodes)
    by_county = panel.groupby("CountyCode")["matched"].first()
    cy_rate = panel["matched"].mean()
    print(f"\n=== match diagnostic (current vintage, {len(adcodes)} DataV counties) ===")
    print(f"county match:      {by_county.mean():.1%}  ({by_county.sum()}/{len(by_county)})")
    print(f"county-year match: {cy_rate:.1%}  ({panel['matched'].sum():,}/{len(panel):,})")

    rows = [{"level": "county", "group": "all", "rate": by_county.mean(),
             "n": len(by_county)},
            {"level": "county-year", "group": "all", "rate": cy_rate,
             "n": len(panel)}]
    print("\nby year:")
    for y, g in panel.groupby("Year"):
        rows.append({"level": "county-year", "group": f"year_{y}",
                     "rate": g["matched"].mean(), "n": len(g)})
        if y % 4 == 0 or y >= 2020:
            print(f"  {y}: {g['matched'].mean():.1%} (n={len(g):,})")

    u = (panel["NonAgriGDP"] / panel["RegGDP"]).clip(0, 1.5)
    panel["u_decile"] = pd.qcut(u, 10, labels=False, duplicates="drop")
    print("\nby u_GDP decile (county-year):")
    for d, g in panel.dropna(subset=["u_decile"]).groupby("u_decile"):
        rows.append({"level": "county-year", "group": f"u_decile_{int(d)}",
                     "rate": g["matched"].mean(), "n": len(g)})
        print(f"  d{int(d)}: {g['matched'].mean():.1%} (n={len(g):,})")

    top = rows[-1]["rate"]
    gate = cy_rate >= 0.95 and top >= 0.90
    print(f"\nGATE (county-year>=95% & top-decile>=90%): "
          f"{'PASS - single vintage suffices' if gate else 'FAIL - multi-vintage patch required'}")
    unmatched = sorted(set(panel.loc[~panel["matched"], "CountyCode"].astype(int)))
    pd.DataFrame(rows).to_csv(OUT_DIAG, index=False)
    pd.Series(unmatched, name="CountyCode").to_csv(
        "results/tables/boundary_unmatched_initial.csv", index=False)
    print(f"unmatched county codes: {len(unmatched)} -> results/tables/boundary_unmatched_initial.csv")


if __name__ == "__main__":
    feats = phase1_fetch()
    phase2_diagnostic(feats)
