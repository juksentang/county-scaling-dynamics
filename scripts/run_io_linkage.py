#!/usr/bin/env python3
"""Driver for the IO-table sectoral linkage indicators + ordering test.

Usage:
    python3 scripts/run_io_linkage.py            # all available vintages
    python3 scripts/run_io_linkage.py 2002 2012  # subset

Outputs to results/tables/: io_linkage_indicators.csv, io_ordering_test.csv,
io_table_validation.csv, io_sector_mapping_{year}.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.io_linkage import run_all

if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or None
    run_all(years)
