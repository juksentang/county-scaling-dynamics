"""
GEE-side county aggregation of nighttime lights + GAIA impervious surface.

Boundaries: data/boundaries/counties_merged.geojson (2,060 counties,
DataV current vintage + VKGEBX 2000-2010 patch).

Submits Drive export tasks:
  - NTL  (projects/sat-io/open-datasets/npp-viirs-ntl, 2000-2024, 500 m):
    per county-year sums of radiance*area, lit area at tau in {0.5,1,2,5},
    county area.  6 chunk tasks.
  - GAIA (Tsinghua/FROM-GLC/GAIA/v10, 1985-2018, 30 m): impervious area by
    year 2000-2018 per county.  6 chunk tasks (--gaia flag).

Usage: python3 scripts/gee_export_urbanization.py [--ntl] [--gaia] [--dry]
Monitor: earthengine task list | head -30
"""
import argparse
import json
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ee

BOUNDS = "data/boundaries/counties_merged.geojson"
FOLDER = "econp_gee_urbanization"
N_CHUNKS = 6
NTL_IC = "projects/sat-io/open-datasets/npp-viirs-ntl"
GAIA_IMG = "Tsinghua/FROM-GLC/GAIA/v10"
NTL_YEARS = (2000, 2024)
GAIA_YEARS = (2000, 2018)
TAUS = {"t05": 0.5, "t1": 1.0, "t2": 2.0, "t5": 5.0}


def load_chunks() -> list[ee.FeatureCollection]:
    feats = json.load(open(BOUNDS))["features"]
    feats.sort(key=lambda f: f["properties"]["adcode"])
    size = -(-len(feats) // N_CHUNKS)
    chunks = []
    for i in range(0, len(feats), size):
        part = [ee.Feature(ee.Geometry(f["geometry"]),
                           {"adcode": f["properties"]["adcode"]})
                for f in feats[i:i + size]]
        chunks.append(ee.FeatureCollection(part))
    print(f"{len(feats)} counties -> {len(chunks)} chunks of <= {size}")
    return chunks


def submit_ntl(chunks, dry=False):
    ic = ee.ImageCollection(NTL_IC)
    area = ee.Image.pixelArea()

    def yearly(fc):
        def one_year(y):
            y = ee.Number(y)
            img = ic.filter(ee.Filter.calendarRange(y, y, "year")).mosaic()
            ntl = img.select(0).rename("ntl")
            stack = ntl.multiply(area).rename("ntl_sum_w") \
                .addBands([area.updateMask(ntl.gt(tau)).rename(f"lit_area_{k}")
                           for k, tau in TAUS.items()]) \
                .addBands(area.rename("county_area"))
            red = stack.reduceRegions(collection=fc, reducer=ee.Reducer.sum(),
                                      scale=500, tileScale=2)
            return red.map(lambda f: f.set("year", y))
        years = ee.List.sequence(NTL_YEARS[0], NTL_YEARS[1])
        return ee.FeatureCollection(years.map(one_year)).flatten()

    for i, fc in enumerate(chunks):
        desc = f"econp_ntl_chunk{i}"
        task = ee.batch.Export.table.toDrive(
            collection=yearly(fc), description=desc, folder=FOLDER,
            fileFormat="CSV",
            selectors=["adcode", "year", "ntl_sum_w", "county_area"]
                      + [f"lit_area_{k}" for k in TAUS])
        if dry:
            print(f"[dry] {desc}")
        else:
            task.start()
            print(f"submitted {desc} -> Drive/{FOLDER}")


def submit_gaia(chunks, dry=False):
    gaia = ee.Image(GAIA_IMG).select(0)
    area = ee.Image.pixelArea()

    def yearly(fc):
        def one_year(y):
            y = ee.Number(y)
            # change_year_index: 34=1985 ... 1=2018; a pixel is impervious by
            # year Y iff index >= (2019 - Y)
            mask = gaia.gte(ee.Number(2019).subtract(y))
            stack = area.updateMask(mask).rename("imp_area") \
                .addBands(area.rename("county_area"))
            red = stack.reduceRegions(collection=fc, reducer=ee.Reducer.sum(),
                                      scale=30, tileScale=4)
            return red.map(lambda f: f.set("year", y))
        years = ee.List.sequence(GAIA_YEARS[0], GAIA_YEARS[1])
        return ee.FeatureCollection(years.map(one_year)).flatten()

    for i, fc in enumerate(chunks):
        desc = f"econp_gaia_chunk{i}"
        task = ee.batch.Export.table.toDrive(
            collection=yearly(fc), description=desc, folder=FOLDER,
            fileFormat="CSV",
            selectors=["adcode", "year", "imp_area", "county_area"])
        if dry:
            print(f"[dry] {desc}")
        else:
            task.start()
            print(f"submitted {desc} -> Drive/{FOLDER}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntl", action="store_true")
    ap.add_argument("--gaia", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    if not (args.ntl or args.gaia):
        args.ntl = True

    ee.Initialize()
    chunks = load_chunks()
    if args.ntl:
        submit_ntl(chunks, dry=args.dry)
    if args.gaia:
        submit_gaia(chunks, dry=args.dry)


if __name__ == "__main__":
    main()
