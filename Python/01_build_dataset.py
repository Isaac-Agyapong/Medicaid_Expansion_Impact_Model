"""Export the model dataset from the analytics database to Data/county_panel.csv.gz.

The database is built by the analytics project (Medicaid_Expansion_Coverage_Analysis, `python run_all.py`).
The exported file is small (3,035 counties x 16 years) and committed, so everything after this step, including the
web app, runs without a database.

Connection settings come from the standard PG* environment variables (default postgres@localhost:5432,
database medicaid_coverage); the password is read from pgpass.conf.
"""
import os
from pathlib import Path

import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Data" / "county_panel.csv.gz"
CONNINFO = " ".join([f"host={os.getenv('PGHOST', 'localhost')}", f"port={os.getenv('PGPORT', '5432')}",
                     f"user={os.getenv('PGUSER', 'postgres')}", f"dbname={os.getenv('PGDATABASE', 'medicaid_coverage')}"])


def main():
    sql = (ROOT / "SQL" / "01_model_panel.sql").read_text(encoding="utf-8-sig")
    with psycopg.connect(CONNINFO) as conn:
        cur = conn.execute(sql)
        df = pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])
    num = ["pct_uninsured", "pct_uninsured_moe", "pct_uninsured_138_400", "pct_uninsured_children", "pct_nh_white_2013",
           "pct_nh_black_2013", "pct_hispanic_2013", "pct_age_65plus_2013", "poverty_pct_2013"]
    df[num] = df[num].astype(float)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df):,} rows, {df.county_fips.nunique():,} counties, "
          f"{df.year.min()}-{df.year.max()}, {OUT.stat().st_size / 1e6:.1f} MB")


def load_panel():
    """Read the committed dataset with the same types the database returns."""
    return pd.read_csv(OUT, dtype={"county_fips": str, "state_fips": str})


if __name__ == "__main__":
    main()
