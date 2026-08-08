# Data sources

This repository ships **code and derived results only**. Primary data are not
redistributed: the county statistical database is licensed commercially, and
several boundary layers are supplied under terms that do not permit
redistribution. Everything needed to rebuild the analysis panel from those
sources is included here.

---

## 1. County economic panel (primary input)

**China County Statistical Database (CSMAR).** Commercial subscription;
**not redistributable**. Twenty-two thematic workbooks (population, gross
regional product by sector, agricultural output value, farm product output,
crop sown area, cultivated and administrative area, agricultural production
conditions, rural incomes, fiscal and financial indicators, employment, and
county geographic linkage) covering 2000–2023.

To rebuild the panel:

1. Obtain the workbooks through an institutional CSMAR subscription
   (<https://data.csmar.com>) and place them in `data/`.
2. Run `notebooks/01_panel_construction.ipynb`, which calls `src.data_loader`,
   `src.data_cleaner`, `src.deflator` and `src.data_merger`.

The file registry (`THEMED_FILES`, `LONG_FORMAT_FILES`, `MAIN_DATABASE` in
`src/config.py`) lists the exact workbook names, sheets and column mappings, so
the required fields can be identified in any equivalent source.

The build produces two panels:

| File | Rows | Counties | Role |
|---|---|---|---|
| `data_clean/panel_full.parquet` | 38,308 | 2,297 | Coverage diagnostics and audits |
| `data_clean/panel_scaling.parquet` | 19,392 | 1,882 | **Analysis panel** — all reported results |

**Poverty designation list.** The 832 nationally designated poverty counties
and their delisting years (2016–2020) are public government records, compiled
into `data/全国832个国家级贫困县名单、摘帽整理数据.xlsx`. 755 of these counties
are present in the analysis panel.

---

## 2. County administrative boundaries

Boundaries are used only to aggregate satellite rasters to counties; the merged
layer is keyed by the six-digit GB administrative division code, and each county
carries a `boundary_source` tag.

| Vintage | Source |
|---|---|
| 2010, 2015 | Resource and Environment Science and Data Center, Chinese Academy of Sciences (RESDC), <https://www.resdc.cn> |
| 2020 | Ministry of Natural Resources of China, map approval number **GS(2020)4619** |
| 2024 | Ministry of Natural Resources of China, map approval number **GS(2024)0650** |
| 2000, 2005 | Obtained online; attribute schema matches CTAmap (see below) |

The 2000 and 2005 vintages carry the CTAmap schema and should be cited as:

> Rui C., Zhang H., Chen B. China temporal administrative map: a multitemporal
> database for Chinese historical administrative divisions (2009–2023).
> *Proc. SPIE* **13565**, ERSGIT 2024. <https://doi.org/10.1117/12.3059430>
> (dataset: CTAmap v1.30, <http://www.shengshixian.com>, MIT licence)

Their geometries were cross-validated against the official RESDC 2010 layer:
median intersection-over-union 0.994, with 93% of counties above 0.90. The few
low-overlap cases are genuine boundary reallocations at county-to-district
conversions, which is the intended behaviour for pre-conversion years.

Two openly licensed layers complete the coverage of counties absent from the
above vintages:

- **Aliyun DataV GeoAtlas** (current vintage), <https://datav.aliyun.com/tools/atlas/>
  — used for geometry only; not redistributed.
- **China County Map with 2000–2010 Population Census Data**, Spatial Data Lab /
  China Data Institute, Harvard Dataverse, CC0 1.0,
  <https://doi.org/10.7910/DVN/VKGEBX>

Twenty-three county codes in the panel have no matching geometry in any vintage
and are excluded from the satellite measures (0.3% of county-years).

---

## 3. Satellite urbanisation measures

**Nighttime lights (primary independent measure).** Harmonised NPP-VIIRS-like
annual composites, 2000–2024, 500 m, accessed through Google Earth Engine as
`projects/sat-io/open-datasets/npp-viirs-ntl`.

> Chen Z. et al. An extended time series (2000–2018) of global NPP-VIIRS-like
> nighttime light data. *Earth Syst. Sci. Data* **13**, 889–906 (2021).
> <https://doi.org/10.5194/essd-13-889-2021> (CC0 1.0)

A level shift is present at the 2012/2013 sensor-era transition inside the
harmonised product; within-year quantile comparisons are unaffected.

**Impervious surface (cross-validation measure).** Global Artificial Impervious
Area, 1985–2018, 30 m, Google Earth Engine asset `Tsinghua/FROM-GLC/GAIA/v10`.

> Gong P. et al. Annual maps of global artificial impervious area (GAIA) between
> 1985 and 2018. *Remote Sens. Environ.* **236**, 111510 (2020).
> <https://doi.org/10.1016/j.rse.2019.111510>

County aggregation is performed on Earth Engine
(`scripts/gee_export_urbanization.py`) and the resulting county-year tables are
downloaded with `scripts/download_gee_urbanization.py`. Reproducing this step
requires an Earth Engine account:

> Gorelick N. et al. Google Earth Engine: planetary-scale geospatial analysis for
> everyone. *Remote Sens. Environ.* **202**, 18–27 (2017).
> <https://doi.org/10.1016/j.rse.2017.06.031>

---

## 4. Input–output tables

National input–output tables published by the **National Bureau of Statistics of
China**, competitive-type sheets for the benchmark years 2002, 2007, 2012, 2017
and the extension years 2005, 2010, 2015, 2018, 2020 (42–153 sector
resolutions). Place the workbooks under `data/io_tables/` and run
`scripts/run_io_linkage.py`; the derived sector-linkage indicators are included
in `results/tables/`.

---

## 5. What this repository contains

| Path | Contents | Licence |
|---|---|---|
| `src/`, `scripts/`, `notebooks/`, `Makefile` | Analysis code | MIT (see `LICENSE`) |
| `results/tables/` | Derived statistics (coefficients, confidence intervals, test results) | CC BY 4.0 |
| `results/figures/` | Generated figures | CC BY 4.0 |

Derived tables contain estimated statistics only, never county-level values from
the licensed source database.
