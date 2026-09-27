"""Rebuild the model project in order.

    python run_all.py              full rebuild (~5 min)
    python run_all.py --skip-data  reuse the committed Data/county_panel.csv.gz (no database needed)

Step 1 reads the PostgreSQL database built by the analytics project (Medicaid_Expansion_Coverage_Analysis).
"""
import subprocess
import sys
import time
from pathlib import Path

PY = Path(__file__).resolve().parent / "Python"
STEPS = [
    ("01_build_dataset.py", "export the county panel from the analytics database"),
    ("02_causal_effects.py", "difference-in-differences effect of expansion, placebo tests, bootstrap"),
    ("03_train_causal_forest.py", "causal forest: validation, county effects, predictions, app files"),
    ("04_build_notebook.py", "build and execute Python/04_model_results.ipynb"),
]


def main():
    start = time.time()
    for script, what in STEPS:
        if script.startswith("01_") and "--skip-data" in sys.argv:
            continue
        print(f"\n== {script}: {what}")
        t = time.time()
        subprocess.run([sys.executable, script], cwd=PY, check=True)
        print(f"   done in {time.time() - t:.0f}s")
    print(f"\nall steps finished in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
