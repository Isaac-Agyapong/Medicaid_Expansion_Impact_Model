"""
Medicaid Expansion Impact: machine learning model web app (Streamlit).

    streamlit run app/app.py

Tabs: Overview, What if the rest expanded, Try the model (live causal forest on any county), How it works.
Runs from the files committed in models/, Data/results/ and Image/, so it deploys to Streamlit Community Cloud
without a database. All data are public county-level Census estimates.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
BLUE, BLUE_L, TANGERINE, TANGERINE_L = "#2F5BEA", "#A9BBF5", "#F08A24", "#F9C796"
INK, INK_2, GREY, GREY_L, LINE = "#111827", "#4B5563", "#9CA3AF", "#E5E7EB", "#ECEEF1"
FEATURES = ["base_rate", "pre_trend", "base_rate_138_400", "poverty_pct_2013", "log_income_2013", "pct_nh_black_2013",
            "pct_hispanic_2013", "pct_age_65plus_2013", "log_pop_2013", "rucc_2013"]
GEOJSON = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
STATE_NAMES = {"TX": "Texas", "FL": "Florida", "GA": "Georgia", "TN": "Tennessee", "SC": "South Carolina", "AL": "Alabama",
               "MS": "Mississippi", "KS": "Kansas", "WI": "Wisconsin", "WY": "Wyoming"}

st.set_page_config(page_title="Medicaid Expansion Impact | ML model", page_icon="▣", layout="wide")


# --------------------------------------------------------------------------------------------- data
@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "causal_forest.joblib")


@st.cache_data
def load_data():
    counties = pd.read_csv(ROOT / "models" / "app_counties.csv", dtype={"county_fips": str})
    states = pd.read_csv(ROOT / "Data" / "results" / "ml_nonexpansion_state_predictions.csv")
    causal = json.loads((ROOT / "models" / "causal_results.json").read_text())
    ml = json.loads((ROOT / "models" / "ml_results.json").read_text())
    return counties, states, causal, ml


@st.cache_data(show_spinner=False)
def load_geojson():
    import urllib.request
    with urllib.request.urlopen(GEOJSON, timeout=20) as r:
        return json.loads(r.read())


model = load_model()
COUNTIES, STATES, CAUSAL, ML = load_data()
EFFECT = abs(CAUSAL["adjusted"]["att_years_0_2"])
EFFECT_LO, EFFECT_HI = abs(CAUSAL["adjusted"]["ci"][1]), abs(CAUSAL["adjusted"]["ci"][0])
COVERED = CAUSAL["people_covered_2023"]["estimate"]
GAIN = ML["nonexpansion_total_adults_gaining"]
TEN = STATES[~STATES.expanded_late_2023]
RATE_NOW = 100 * TEN.uninsured_2023.sum() / TEN.low_income_adults.sum()
RATE_AFTER = 100 * (TEN.uninsured_2023.sum() - TEN.adults_gaining_coverage.sum()) / TEN.low_income_adults.sum()

# --------------------------------------------------------------------------------------------- style
st.markdown(f"""
<style>
  html, body, [class*="css"], .stMarkdown, .stTabs {{ font-family: Arial, Helvetica, sans-serif; }}
  .stApp {{ background: #FFFFFF; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: 1.2rem; max-width: 1180px; }}
  .topbar {{ background: {INK}; color: #fff; padding: 18px 26px; display: flex; align-items: center;
            justify-content: space-between; border-radius: 0; margin-bottom: 6px; }}
  .topbar .t {{ font-size: 26px; font-weight: 800; letter-spacing: -0.5px; }}
  .topbar .s {{ font-size: 13px; color: #C7CBD3; margin-top: 2px; }}
  .topbar .tag {{ background: {BLUE}; color: #fff; font-weight: 700; font-size: 12px; padding: 6px 12px; }}
  .rule {{ border-top: 5px solid {INK}; margin: 18px 0 8px 0; }}
  .kicker {{ color: {BLUE}; font-weight: 800; font-size: 12px; letter-spacing: 1.5px; text-transform: uppercase; }}
  .headline {{ font-size: 30px; font-weight: 800; color: {INK}; line-height: 1.15; margin: 4px 0 6px 0; letter-spacing: -0.5px; }}
  .lede {{ font-size: 16px; color: {INK_2}; max-width: 900px; }}
  .big {{ font-size: 54px; font-weight: 800; line-height: 1; letter-spacing: -1.5px; }}
  .bigunit {{ font-size: 22px; font-weight: 700; }}
  .label {{ font-size: 14px; color: {INK}; font-weight: 700; margin-top: 8px; }}
  .note {{ font-size: 13px; color: {INK_2}; }}
  .stat {{ border-top: 3px solid {INK}; padding-top: 12px; }}
  .check {{ font-size: 15px; color: {INK}; margin: 8px 0; }}
  .check b {{ color: {BLUE}; }}
  .pill {{ display: inline-block; border: 2px solid {INK}; padding: 3px 10px; font-weight: 700; font-size: 12px; margin-right: 6px; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 2px solid {INK}; }}
  .stTabs [data-baseweb="tab"] {{ font-weight: 700; font-size: 15px; padding: 10px 22px; color: {INK_2}; }}
  .stTabs [aria-selected="true"] {{ color: #fff !important; background: {INK}; }}
  .stTabs [data-baseweb="tab-highlight"] {{ display: none; }}
  footer {{ visibility: hidden; }}
</style>
<div class="topbar">
  <div><div class="t">Medicaid Expansion Impact</div>
       <div class="s">Machine learning model · 3,035 US counties · Census data 2008-2023</div></div>
  <div class="tag">Built by Isaac Agyapong</div>
</div>
""", unsafe_allow_html=True)


def stat(col, value, unit, label, note, colour=INK):
    col.markdown(f'<div class="stat"><span class="big" style="color:{colour}">{value}</span>'
                 f'<span class="bigunit" style="color:{colour}"> {unit}</span>'
                 f'<div class="label">{label}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)


def section(kicker, headline, lede=None):
    st.markdown(f'<div class="rule"></div><div class="kicker">{kicker}</div><div class="headline">{headline}</div>'
                + (f'<div class="lede">{lede}</div>' if lede else ""), unsafe_allow_html=True)


def base_layout(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=20, t=10, b=10), paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      font=dict(family="Arial", color=INK, size=14), showlegend=False)
    return fig


tab1, tab2, tab3, tab4 = st.tabs(["Overview", "What if the rest expanded", "Try the model", "How it works"])

# --------------------------------------------------------------------------------------------- 1. overview
with tab1:
    section("The question", "Did Medicaid expansion work, and what if the 10 remaining states expanded?",
            "Medicaid is free or low-cost government health insurance for people with low incomes. Since 2014, 40 states "
            "and DC have let more low-income adults qualify (\"Medicaid expansion\"). This model measures what expansion "
            "itself did, separate from everything else that changed after 2014, and predicts what it would do elsewhere. "
            "Low-income adults = people aged 18-64 earning about $20,000 a year or less.")
    c1, c2, c3 = st.columns(3)
    stat(c1, f"{EFFECT:.1f}", "in 100", "fewer uninsured adults, thanks to expansion",
         f"in its first three years · likely between {EFFECT_LO:.0f} and {EFFECT_HI:.0f} in 100", BLUE)
    stat(c2, f"{COVERED / 1000:,.0f}K", "", "more adults insured in 2023 because of expansion",
         "in the 33 expansion states studied", BLUE)
    stat(c3, f"{round(GAIN, -4) / 1000:,.0f}K", "", "more adults could be insured if the 10 remaining states expanded",
         f"uninsured would fall from {RATE_NOW:.0f}% to {RATE_AFTER:.0f}% · {TEN.set_index('state').adults_gaining_coverage['TX'] / GAIN:.0%} in Texas",
         TANGERINE)

    left, right = st.columns([1.35, 1])
    with left:
        section("What happened vs what would have happened", "In 2023, expansion still meant about 5 fewer uninsured adults in every 100")
        es = pd.read_csv(ROOT / "Data" / "results" / "causal_att_gt_adjusted.csv").query("g == 2014 and t == 2023")
        counties = COUNTIES.query("expansion_year == 2014")
        panel = pd.read_csv(ROOT / "Data" / "county_panel.csv.gz", dtype={"county_fips": str}).query("expansion_year == 2014 and year == 2023")
        actual = 100 * panel.uninsured.sum() / panel.population.sum()
        without = actual - float(es.att.iloc[0])
        fig = go.Figure(go.Bar(y=["Without expansion (estimated)", "With expansion (what happened)"], x=[without, actual],
                               orientation="h", marker_color=[GREY_L, BLUE], text=[f"{without:.0f}%", f"{actual:.0f}%"],
                               textposition="outside", textfont=dict(size=22, color=INK), width=0.55))
        fig.update_xaxes(visible=False, range=[0, without * 1.25])
        fig.update_yaxes(tickfont=dict(size=15))
        st.plotly_chart(base_layout(fig, 250), use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="note">States that expanded in 2014: share of low-income adults with no health insurance in 2023. '
                    'The grey bar is the model\'s estimate of the share if they had not expanded.</div>', unsafe_allow_html=True)
    with right:
        section("Checks", "Can this result be trusted?")
        spill = abs(CAUSAL["placebo_138_400"]["att_years_0_2"])
        for c in ["Before 2014, counties in both groups were on the same path, so the comparison is fair.",
                  "Pretending expansion happened in 2011, when it did not, shows no effect, as it should.",
                  f"Adults who earn too much to qualify changed much less ({spill:.0f} in 100, not {EFFECT:.0f}).",
                  "A second, separate method (the machine learning model) gives the same answer."]:
            st.markdown(f'<div class="check"><b>✓</b>&nbsp; {c}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------- 2. what if
with tab2:
    section("Prediction", f"About {round(GAIN, -4):,.0f} more adults would have health insurance if the 10 remaining states expanded",
            "Estimated by the machine learning model from what happened in similar counties that expanded between 2014 and 2021, "
            "applied to each county's situation in 2023.")
    ten = TEN.sort_values("adults_gaining_coverage")
    fig = go.Figure(go.Bar(y=[STATE_NAMES.get(s, s) for s in ten.state], x=ten.adults_gaining_coverage, orientation="h",
                           marker_color=[INK if s == "TX" else TANGERINE for s in ten.state],
                           text=[f"{v:,.0f}  ({a:.0f}% → {b:.0f}% uninsured)" for v, a, b in
                                 zip(ten.adults_gaining_coverage, ten.rate_2023, ten.predicted_rate_after)],
                           textposition="outside", textfont=dict(size=13)))
    fig.update_xaxes(visible=False, range=[0, ten.adults_gaining_coverage.max() * 1.6])
    st.plotly_chart(base_layout(fig, 420), use_container_width=True, config={"displayModeBar": False})

    section("By county", "Where the gains would be largest",
            "Map: darker = bigger drop in the share of uninsured adults. Table: counties where the most people would gain coverage.")
    pick = st.selectbox("State", [STATE_NAMES[s] for s in STATES.sort_values("adults_gaining_coverage", ascending=False).state
                                  if s in STATE_NAMES], index=0)
    code = {v: k for k, v in STATE_NAMES.items()}[pick]
    sc = COUNTIES[(COUNTIES.state == code)].copy()
    sc["rate_after"] = sc.base_rate + sc.effect_pts
    m1, m2 = st.columns([1.3, 1])
    with m1:
        try:
            geo = load_geojson()
            fmap = go.Figure(go.Choropleth(geojson=geo, locations=sc.county_fips, z=-sc.effect_pts, featureidkey="id",
                                           colorscale=[[0, "#FFF4E8"], [1, "#B45309"]], marker_line_color="#FFFFFF",
                                           marker_line_width=0.5, colorbar=dict(title="Fewer<br>uninsured<br>per 100", thickness=12),
                                           customdata=np.stack([sc.adults_gaining], axis=-1), text=sc.county_name,
                                           hovertemplate="%{text}<br>%{z:.1f} fewer uninsured in every 100"
                                                         "<br>%{customdata[0]:,.0f} adults gaining coverage<extra></extra>"))
            fmap.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(base_layout(fmap, 430), use_container_width=True, config={"displayModeBar": False})
        except Exception:
            st.info("The county map needs an internet connection to load county shapes.")
    with m2:
        top = sc.sort_values("adults_gaining", ascending=False).head(12)
        st.dataframe(pd.DataFrame({"County": top.county_name, "Uninsured today": top.base_rate.map("{:.0f}%".format),
                                   "If expanded": top.rate_after.map("{:.0f}%".format),
                                   "Adults gaining": top.adults_gaining.map("{:,.0f}".format)}),
                     hide_index=True, use_container_width=True, height=430)

# --------------------------------------------------------------------------------------------- 3. try the model
with tab3:
    section("Live model", "Pick any county and see what the model estimates",
            "For counties in states that expanded, the model estimates how much their expansion helped. For counties in states "
            "that have not expanded, it estimates what expanding now would do. Then change the county's situation to see how "
            "the answer moves.")
    labels = (COUNTIES.county_name + ", " + COUNTIES.state).tolist()
    default = labels.index("Harris County, TX") if "Harris County, TX" in labels else 0
    choice = st.selectbox("County", labels, index=default)
    row = COUNTIES.iloc[labels.index(choice)]
    expanded = pd.notna(row.expansion_year)
    status = (f"Expanded Medicaid in {int(row.expansion_year)}" if expanded else "Has not expanded Medicaid")
    st.markdown(f'<span class="pill">{status}</span><span class="pill">{row.rurality}</span>'
                f'<span class="pill">described as of {int(row.base_year)}</span>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.1])
    with c1:
        st.markdown('<div class="rule"></div><div class="kicker">Change the county</div>', unsafe_allow_html=True)
        base = st.slider("Share of low-income adults uninsured (%)", 2.0, 75.0, float(round(row.base_rate, 1)), 0.5)
        pov = st.slider("Poverty rate (%)", 2.0, 55.0, float(round(row.poverty_pct_2013, 1)), 0.5)
        hisp = st.slider("Hispanic share of residents (%)", 0.0, 99.0, float(round(row.pct_hispanic_2013, 1)), 0.5)
        income = st.slider("Median household income ($)", 20000, 130000, int(row.median_income_2013), 1000)
        x = row[FEATURES].astype(float).copy()
        # use the county's exact values unless a slider was moved (sliders round to their step)
        if base != float(round(row.base_rate, 1)):
            x["base_rate"] = base
        if pov != float(round(row.poverty_pct_2013, 1)):
            x["poverty_pct_2013"] = pov
        if hisp != float(round(row.pct_hispanic_2013, 1)):
            x["pct_hispanic_2013"] = hisp
        if income != int(row.median_income_2013):
            x["log_income_2013"] = np.log(income)
        base = x["base_rate"]
        X = x.values.reshape(1, -1)
        eff = float(model.effect(X)[0])
        lo, hi = model.effect_interval(X, alpha=0.05)
        adults = -eff / 100 * row.low_income_adults_base
    with c2:
        st.markdown('<div class="rule"></div><div class="kicker">Model estimate</div>', unsafe_allow_html=True)
        verb = "Expansion cut the share uninsured by" if expanded else "Expanding would cut the share uninsured by"
        st.markdown(f'<div class="big" style="color:{BLUE if expanded else TANGERINE}">{abs(eff):.1f}<span class="bigunit"> in 100</span></div>'
                    f'<div class="label">{verb} about {abs(eff):.1f} in every 100 low-income adults</div>'
                    f'<div class="note">likely between {abs(hi[0]):.1f} and {abs(lo[0]):.1f} in 100</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat" style="margin-top:22px"><span class="big" style="color:{INK}">{adults:,.0f}</span>'
                    f'<div class="label">{"adults insured because of expansion" if expanded else "more adults would have health insurance"}</div>'
                    f'<div class="note">out of {row.low_income_adults_base:,.0f} low-income adults · uninsured {base:.0f}% → {base + eff:.0f}%</div></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="note" style="margin-top:16px">The model learned from 1,707 county expansions (2014-2021) '
                    'and 1,136 counties in states that did not expand. Places with more uninsured people to begin with, '
                    'and more poverty, tend to gain the most.</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------- 4. how it works
with tab4:
    section("How it works", "Two methods, one answer",
            "Step 1 measures the effect of expansion by comparing counties in states that expanded with similar counties in "
            "states that did not, before and after. Step 2 trains a machine learning model to estimate the effect for each "
            "county separately, checks it against step 1, and tests it on states it never saw.")
    a, b = st.columns(2)
    a.image(str(ROOT / "Image" / "01_event_study.png"), use_container_width=True)
    b.image(str(ROOT / "Image" / "03_model_validation.png"), use_container_width=True)
    with st.expander("Technical details"):
        st.markdown(f"""
* **Data:** US Census Small Area Health Insurance Estimates (SAHIE) 2008-2023 for 3,035 counties in 45 states; KFF
  expansion dates; 2013 county traits. Built in PostgreSQL by the companion analytics project.
* **Step 1, difference-in-differences:** Callaway & Sant'Anna (2021) staggered design, adjusted for county traits,
  weighted by population, 95% intervals from 499 bootstrap resamples of states. Effect over years 0-2:
  **{CAUSAL['adjusted']['att_years_0_2']:.2f}** percentage points (95% CI {CAUSAL['adjusted']['ci'][0]:.2f} to {CAUSAL['adjusted']['ci'][1]:.2f}).
* **Step 2, causal forest:** EconML `CausalForestDML` (double machine learning, gradient-boosted nuisance models,
  2,000 honest trees, folds grouped by state). Average effect **{ML['forest_ate_expansion_counties']:.2f}** points.
  On held-out states, counties ranked in the top quarter dropped 9.3 points vs 4.5 in the bottom quarter.
* **Limitations:** estimates carry Census margins of error; states chose whether to expand; predictions assume
  expansion would work as it did in similar counties and are not enrollment forecasts.
""")
    st.markdown('<div class="note">Code and full write-up: github.com/Isaac-Agyapong/Medicaid_Expansion_Impact_Model · '
                'Analytics and Power BI dashboard: github.com/Isaac-Agyapong/Health_Insurance_Coverage_Gap_Analysis</div>',
                unsafe_allow_html=True)
