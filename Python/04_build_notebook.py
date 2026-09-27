"""Build Python/04_model_results.ipynb with nbformat and execute it, so the outputs show on GitHub.

    python Python/04_build_notebook.py
"""
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
NB = HERE / "04_model_results.ipynb"
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Machine Learning Model for Medicaid Expansion Impact: results

**In short:** this notebook shows how much Medicaid expansion itself reduced the share of low-income adults without
health insurance, whether that result holds up, and what a machine learning model predicts for the 10 states that
have not expanded. Data: Census estimates for 3,035 US counties, 2008-2023, prepared in PostgreSQL by the companion
analytics project.

1. **Dataset**: the county panel exported by `01_build_dataset.py`.
2. **Effect of expansion**: staggered difference-in-differences (`02_causal_effects.py`).
3. **Machine learning**: causal forest, validation on held-out states, predictions (`03_train_causal_forest.py`).
""")
code("""
import json
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd().parent
pd.set_option("display.float_format", "{:,.2f}".format)
panel = pd.read_csv(ROOT / "Data" / "county_panel.csv.gz", dtype={"county_fips": str})
cr = json.loads((ROOT / "models" / "causal_results.json").read_text())
mr = json.loads((ROOT / "models" / "ml_results.json").read_text())
print(f"{len(panel):,} county-years, {panel.county_fips.nunique():,} counties, {panel.state_abbrev.nunique()} states, "
      f"{panel.year.min()}-{panel.year.max()}")
panel.groupby("analysis_group").county_fips.nunique().rename("counties").to_frame()
""")

md("""
## 1. How much did expansion itself change the uninsured rate?

Each expansion county's change is compared with the change in similar counties in states that did not expand over
the same years (Callaway & Sant'Anna 2021), adjusted for 2013 county traits, weighted by population, with 95%
intervals from 499 bootstrap resamples of states.
""")
code("""
display(Image(ROOT / "Image" / "01_event_study.png"))
pd.DataFrame({
    "estimate (pts)": [cr["adjusted"]["att_years_0_2"], cr["unadjusted"]["att_years_0_2"], cr["placebo_138_400"]["att_years_0_2"],
                       cr["placebo_fake_2011"]["att"], cr["adjusted"]["att_all_post_years"], cr["twfe"]],
    "95% CI": [f"{a:.1f} to {b:.1f}" for a, b in [cr["adjusted"]["ci"], cr["unadjusted"]["ci"], cr["placebo_138_400"]["ci"],
                                                  cr["placebo_fake_2011"]["ci"]]] + ["", ""],
}, index=["Main: covariate-adjusted, years 0-2", "Unadjusted, years 0-2", "Adults 138-400% FPL (not made eligible)",
          "Placebo: fake 2011 expansion", "Adjusted, all post years", "Two-way fixed effects (traditional)"])
""")
code("""
display(Image(ROOT / "Image" / "02_robustness_checks.png"))
print(f"Low-income adults insured in 2023 because of expansion (33 expansion states studied): "
      f"{cr['people_covered_2023']['estimate']:,.0f} (95% CI {cr['people_covered_2023']['ci'][0]:,.0f} to {cr['people_covered_2023']['ci'][1]:,.0f})")
pd.read_csv(ROOT / "Data" / "results" / "causal_by_cohort.csv")
""")

md("""
## 2. Machine learning: which counties gain the most?

A causal forest (EconML `CausalForestDML`) estimates a separate effect for every county from its pre-expansion traits.
It must agree with the difference-in-differences estimate on average, and on states held out of training, the counties
it predicts will gain more must really have gained more.
""")
code("""
print(f"Causal forest average effect: {mr['forest_ate_expansion_counties']:.2f} pts "
      f"(difference-in-differences: {mr['did_estimate_years_0_2']:.2f} pts)")
display(Image(ROOT / "Image" / "03_model_validation.png"))
pd.DataFrame(mr["validation_splits"])[["split", "gap_top_vs_bottom_predicted", "gap_top_vs_bottom_actual"]]
""")
code("""
display(Image(ROOT / "Image" / "04_effect_drivers.png"))
pd.DataFrame(mr["effect_by"]).T
""")

md("## 3. What if the 10 remaining states expanded?")
code("""
display(Image(ROOT / "Image" / "05_nonexpansion_predictions.png"))
top = pd.read_csv(ROOT / "Data" / "results" / "ml_nonexpansion_county_predictions.csv")
top = top[~top.state.isin(["NC", "SD"])]
top.head(15)[["county_name", "state", "rurality", "base_rate", "predicted_effect_pts", "effect_lo", "effect_hi", "adults_gaining_coverage"]]
""")
md("""
## Limitations

* SAHIE figures are model-based estimates with margins of error; every model weights counties by population.
* States chose whether to expand; the design removes fixed differences and shared yearly changes, and pre-trends
  match, but a state-specific shock at the same time as expansion would bias the estimate.
* Predictions assume expansion would work as it did in similar counties, under 2023 conditions; they estimate the
  drop in the uninsured rate, not Medicaid enrollment.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, NB)
subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(NB),
                "--ExecutePreprocessor.timeout=600"], check=True, cwd=HERE)
print("built and executed", NB.name)
