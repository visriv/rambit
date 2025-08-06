#!/usr/bin/env python3
import gzip
import shutil
from pathlib import Path

def extract_all(input_dir: Path, output_dir: Path):
    """
    Decompresses every .csv.gz in input_dir into output_dir,
    preserving the filename but dropping the .gz extension.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for gz in sorted(input_dir.glob("*.csv.gz")):
        out_csv = output_dir / gz.with_suffix("").name  # e.g. ADMISSIONS.csv
        print(f"Extracting {gz.name} → {out_csv.name}")
        with gzip.open(gz, "rt") as f_in, open(out_csv, "wt") as f_out:
            shutil.copyfileobj(f_in, f_out)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Un-gzip all MIMIC CSVs into a clean folder"
    )
    p.add_argument("input_dir",  type=Path, help="Folder containing *.csv.gz")
    p.add_argument("output_dir", type=Path, help="Where to dump *.csv files")
    args = p.parse_args()

    extract_all(args.input_dir, args.output_dir)
