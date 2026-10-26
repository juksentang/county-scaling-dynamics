"""
Download the public US and Brazilian input files and check them against the
sha256 recorded when the analysis was run.

  python3 scripts/fetch_public_data.py us      [--dest data/us]
  python3 scripts/fetch_public_data.py brazil  [--dest data/brazil]
  add --verify-only to check files already in place without downloading.

The files land under <dest>/raw/ at the relative paths the result tables
record (e.g. raw/omb/2023/list1_2023.xlsx, raw/sidra/t6579_p2001.json).

United States: county personal income (BEA CAINC1) and county GDP (BEA
CAGDP1), public domain, and the OMB metropolitan delineation files published
by the Census Bureau. The expected sha256 are the `data_sha256` entries of
results/tables/k2/k2_meta.json. The BEA files are taken out of the zip
archives BEA publishes; BEA replaces these archives with each annual release,
so a later release has other file names or hashes and is reported as a
mismatch (the hashes identify the release that was used). A BEA zip archive
already placed at <dest>/raw/bea/CAINC1.zip or <dest>/raw/bea/CAGDP1.zip is
unpacked instead of downloaded, also under --verify-only.

Brazil: IBGE population (SIDRA tables 202, 793, 4709, 6579 and the FTP
estimates), the territorial division and genealogy files, and the PIB dos
Municipios. URLs and expected sha256 are the rows of
results/tables/k2br_coverage_sources.csv.

Standard library only. A mismatch or a failed download exits non-zero.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
US_META = os.path.join(ROOT, "results", "tables", "k2", "k2_meta.json")
BR_SOURCES = os.path.join(ROOT, "results", "tables", "k2br_coverage_sources.csv")

BEA = "https://apps.bea.gov/regional/zip"
OMB = "https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files"

US_URLS = {
    "raw/bea/x/CAINC1__ALL_AREAS_1969_2024.csv": (f"{BEA}/CAINC1.zip", "CAINC1__ALL_AREAS_1969_2024.csv"),
    "raw/bea/x/CAGDP1__ALL_AREAS_2001_2024.csv": (f"{BEA}/CAGDP1.zip", "CAGDP1__ALL_AREAS_2001_2024.csv"),
    "raw/omb/1993/93mfips.txt": (f"{OMB}/1993/historical-delineation-files/93mfips.txt", None),
    "raw/omb/1993/93nfips.txt": (f"{OMB}/1993/historical-delineation-files/93nfips.txt", None),
    "raw/omb/1999/99mfips.txt": (f"{OMB}/1999/historical-delineation-files/99mfips.txt", None),
    "raw/omb/1999/99nfips.txt": (f"{OMB}/1999/historical-delineation-files/99nfips.txt", None),
    "raw/omb/2003/030606omb-cbsa-csa.xls": (f"{OMB}/2003/historical-delineation-files/030606omb-cbsa-csa.xls", None),
    "raw/omb/200312/0312cbsas-csas.xls": (f"{OMB}/2003/historical-delineation-files/0312cbsas-csas.xls", None),
    "raw/omb/2004/list3.xls": (f"{OMB}/2004/historical-delineation-files/list3.xls", None),
    "raw/omb/2005/list3.xls": (f"{OMB}/2005/historical-delineation-files/list3.xls", None),
    "raw/omb/2006/list3.xls": (f"{OMB}/2006/historical-delineation-files/list3.xls", None),
    "raw/omb/2007/list3.xls": (f"{OMB}/2007/historical-delineation-files/list3.xls", None),
    "raw/omb/2008/list3.xls": (f"{OMB}/2008/historical-delineation-files/list3.xls", None),
    "raw/omb/2009/list3.xls": (f"{OMB}/2009/historical-delineation-files/list3.xls", None),
    "raw/omb/2013/list1.xls": (f"{OMB}/2013/delineation-files/list1.xls", None),
    "raw/omb/2015/list1.xls": (f"{OMB}/2015/delineation-files/list1.xls", None),
    "raw/omb/2017/list1.xls": (f"{OMB}/2017/delineation-files/list1.xls", None),
    "raw/omb/201804/list1.xls": (f"{OMB}/2018/delineation-files/list1.xls", None),
    "raw/omb/201809/list1_Sep_2018.xls": (f"{OMB}/2018/delineation-files/list1_Sep_2018.xls", None),
    "raw/omb/2020/list1_2020.xls": (f"{OMB}/2020/delineation-files/list1_2020.xls", None),
    "raw/omb/2023/list1_2023.xlsx": (f"{OMB}/2023/delineation-files/list1_2023.xlsx", None),
}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except Exception as e:
            if i == tries - 1:
                raise RuntimeError(f"download failed: {url}: {e}") from e
            time.sleep(5 * (i + 1))


def us_expected():
    with open(US_META, encoding="utf-8") as fh:
        want = json.load(fh)["data_sha256"]
    missing = set(US_URLS) ^ set(want)
    if missing:
        raise SystemExit(f"US source list and {US_META} disagree: {sorted(missing)}")
    return {p: (US_URLS[p][0], US_URLS[p][1], want[p]) for p in sorted(want)}


def br_expected():
    with open(BR_SOURCES, encoding="utf-8", newline="") as fh:
        return {r["file"]: (r["url"], None, r["sha256"]) for r in csv.DictReader(fh)}


def run(entries, dest, verify_only):
    bad = []
    archives = {}
    for rel, (url, member, want) in entries.items():
        out = os.path.join(dest, rel)
        local_zip = os.path.join(dest, "raw", "bea", os.path.basename(url)) if member else None
        use_local = local_zip is not None and os.path.isfile(local_zip)
        if (use_local or not verify_only) and not os.path.isfile(out):
            if member is None:
                data = download(url)
            else:
                if url not in archives:
                    archives[url] = zipfile.ZipFile(local_zip if use_local else io.BytesIO(download(url)))
                z = archives[url]
                if member not in z.namelist():
                    bad.append((rel, f"{member} not in {url} (a later BEA release?)"))
                    continue
                data = z.read(member)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out + ".part", "wb") as fh:
                fh.write(data)
            os.replace(out + ".part", out)
        if not os.path.isfile(out):
            bad.append((rel, "missing"))
            continue
        got = sha256_file(out)
        status = "ok" if got == want else "MISMATCH"
        print(f"{status:8s} {rel}")
        if got != want:
            bad.append((rel, f"sha256 {got} != {want}"))
    if bad:
        for rel, why in bad:
            print(f"FAILED   {rel}: {why}", file=sys.stderr)
        sys.exit(1)
    print(f"{len(entries)} files verified under {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("country", choices=["us", "brazil"])
    ap.add_argument("--dest", default=None, help="default data/us or data/brazil")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    dest = a.dest or os.path.join(ROOT, "data", a.country)
    entries = us_expected() if a.country == "us" else br_expected()
    run(entries, dest, a.verify_only)


if __name__ == "__main__":
    main()
