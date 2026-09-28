"""
Medicaid Expansion Impact: machine learning model web app (Streamlit).

    streamlit run app/app.py

Tabs: Overview, What if the rest expanded, Try the model (live causal forest on any county), How it works.
Runs from the files committed in models/, Data/, Image/ and app/assets/, so it deploys to Streamlit Community Cloud
without a database. All data are public county-level Census estimates.
"""
import base64
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "assets"
NAVY, BLUE, BLUE_SOFT = "#0F1E52", "#2F5BEA", "#E8EEFF"
ORANGE, ORANGE_SOFT, GREEN, GREEN_SOFT = "#F08A24", "#FFF1E3", "#12A06E", "#E3F6EE"
INK, INK_2, GREY, GREY_L, BG = "#101828", "#475467", "#98A2B3", "#E4E7EC", "#F3F5FA"
FEATURES = ["base_rate", "pre_trend", "base_rate_138_400", "poverty_pct_2013", "log_income_2013", "pct_nh_black_2013",
            "pct_hispanic_2013", "pct_age_65plus_2013", "log_pop_2013", "rucc_2013"]
GEOJSON = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
STATE_NAMES = {"TX": "Texas", "FL": "Florida", "GA": "Georgia", "TN": "Tennessee", "SC": "South Carolina", "AL": "Alabama",
               "MS": "Mississippi", "KS": "Kansas", "WI": "Wisconsin", "WY": "Wyoming"}
ICON = {  # inline SVG icons (stroke = currentColor)
    "down": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M5 12l7 7 7-7"/></svg>',
    "people": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="8" r="3.5"/><path d="M2.5 20c0-3.6 2.9-6 6.5-6s6.5 2.4 6.5 6"/><circle cx="17" cy="9" r="2.8"/><path d="M16 14.2c3 .3 5.5 2.4 5.5 5.8"/></svg>',
    "target": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.3" fill="currentColor"/></svg>',
    "check": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
}

st.set_page_config(page_title="Medicaid Expansion Impact | ML model", page_icon="🩺", layout="wide")


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
    panel = pd.read_csv(ROOT / "Data" / "county_panel.csv.gz", dtype={"county_fips": str})
    att = pd.read_csv(ROOT / "Data" / "results" / "causal_att_gt_adjusted.csv")
    return counties, states, causal, ml, panel, att


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


@st.cache_data(show_spinner=False)
def load_geojson():
    import urllib.request
    with urllib.request.urlopen(GEOJSON, timeout=20) as r:
        return json.loads(r.read())


model = load_model()
COUNTIES, STATES, CAUSAL, ML, PANEL, ATT = load_data()
EFFECT = abs(CAUSAL["adjusted"]["att_years_0_2"])
EFFECT_LO, EFFECT_HI = abs(CAUSAL["adjusted"]["ci"][1]), abs(CAUSAL["adjusted"]["ci"][0])
COVERED = CAUSAL["people_covered_2023"]["estimate"]
GAIN = ML["nonexpansion_total_adults_gaining"]
TEN = STATES[~STATES.expanded_late_2023]
RATE_NOW = 100 * TEN.uninsured_2023.sum() / TEN.low_income_adults.sum()
RATE_AFTER = 100 * (TEN.uninsured_2023.sum() - TEN.adults_gaining_coverage.sum()) / TEN.low_income_adults.sum()
TX_SHARE = TEN.set_index("state").adults_gaining_coverage["TX"] / GAIN

# --------------------------------------------------------------------------------------------- style
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  .stApp *:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded) {{
      font-family: 'Inter', system-ui, sans-serif !important; }}
  .stApp {{ background: {BG}; }}
  header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
  .block-container {{ padding-top: 1rem; padding-bottom: 3rem; max-width: 1220px; }}
  footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none !important; }}
  /* hero */
  .hero {{ border-radius: 24px; padding: 30px 38px 86px 38px; color: #fff; position: relative; overflow: hidden;
           background: {NAVY} url('data:image/png;base64,{b64(ASSETS / "hero.png")}') right center / cover no-repeat;
           box-shadow: 0 20px 45px -20px rgba(15,30,82,.55); }}
  .hero .top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 34px; }}
  .brand {{ display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 15px; letter-spacing: .2px; }}
  .brand .logo {{ width: 34px; height: 34px; border-radius: 10px; background: rgba(255,255,255,.14);
                  border: 1px solid rgba(255,255,255,.25); display: grid; place-items: center; font-size: 17px; }}
  .badge {{ display: flex; align-items: center; gap: 10px; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.25);
            padding: 6px 14px 6px 6px; border-radius: 999px; font-size: 13px; font-weight: 600; backdrop-filter: blur(6px); }}
  .badge .av {{ width: 28px; height: 28px; border-radius: 50%; background: {ORANGE}; color: #fff; display: grid; place-items: center;
                font-weight: 800; font-size: 12px; }}
  .kicker {{ display: inline-block; font-size: 12px; font-weight: 700; letter-spacing: 1.6px; text-transform: uppercase;
             color: #C7D4FF; background: rgba(255,255,255,.10); padding: 6px 12px; border-radius: 999px; }}
  .hero h1 {{ font-size: 44px; line-height: 1.1; font-weight: 800; letter-spacing: -1.2px; margin: 14px 0 12px 0; color: #fff;
              max-width: 540px; }}
  .hero p {{ font-size: 16.5px; line-height: 1.6; color: #D5DEFF; max-width: 560px; margin: 0; }}
  .legend {{ margin-top: 18px; font-size: 13px; color: #C7D4FF; display: flex; gap: 18px; }}
  .legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
  /* stat cards overlapping the hero */
  .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin: -62px 22px 8px 22px; position: relative; }}
  .stat {{ background: #fff; border-radius: 18px; padding: 20px 22px; box-shadow: 0 12px 30px -12px rgba(16,24,40,.22);
           display: flex; gap: 16px; align-items: flex-start; }}
  .ic {{ width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center; flex: none; }}
  .num {{ font-size: 34px; font-weight: 800; letter-spacing: -1px; line-height: 1; color: {INK}; }}
  .num small {{ font-size: 16px; font-weight: 700; color: {INK_2}; letter-spacing: 0; }}
  .lbl {{ font-size: 14px; font-weight: 600; color: {INK}; margin-top: 8px; line-height: 1.35; }}
  .sub {{ font-size: 12.5px; color: {INK_2}; margin-top: 4px; }}
  /* tabs as a pill bar */
  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; background: #fff; padding: 6px; border-radius: 999px; width: fit-content;
                                       box-shadow: 0 4px 14px -6px rgba(16,24,40,.18); margin: 18px 0 6px 0; border: 0; }}
  .stTabs [data-baseweb="tab"] {{ height: 40px; padding: 0 20px; border-radius: 999px; font-weight: 600; font-size: 14px; color: {INK_2}; }}
  .stTabs [aria-selected="true"] {{ background: {NAVY}; color: #fff !important; }}
  .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
  /* cards */
  [class*="st-key-card-"] {{
      background: #fff; border-radius: 20px !important; border: 1px solid #EDF0F5 !important; padding: 22px 24px;
      box-shadow: 0 10px 28px -16px rgba(16,24,40,.25); }}
  .card-k {{ font-size: 12px; font-weight: 700; letter-spacing: 1.4px; text-transform: uppercase; color: {BLUE}; }}
  .card-h {{ font-size: 22px; font-weight: 800; letter-spacing: -.4px; color: {INK}; line-height: 1.25; margin: 6px 0 4px 0; }}
  .card-p {{ font-size: 14.5px; color: {INK_2}; line-height: 1.55; margin-bottom: 6px; }}
  .checks {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }}
  .chk {{ background: #F8FAFC; border: 1px solid #EEF2F6; border-radius: 14px; padding: 14px; font-size: 13.5px; color: {INK}; line-height: 1.45; }}
  .chk .dot {{ width: 26px; height: 26px; border-radius: 50%; background: {GREEN_SOFT}; color: {GREEN}; display: grid; place-items: center;
               margin-bottom: 8px; }}
  .pill {{ display: inline-block; padding: 5px 12px; border-radius: 999px; font-size: 12.5px; font-weight: 600; margin: 0 6px 6px 0; }}
  .big {{ font-size: 58px; font-weight: 800; letter-spacing: -2px; line-height: 1; }}
  .big small {{ font-size: 22px; letter-spacing: 0; }}
  .foot {{ text-align: center; color: {GREY}; font-size: 13px; margin-top: 26px; }}
  .foot a {{ color: {BLUE}; text-decoration: none; font-weight: 600; }}
  [data-testid="stSlider"] [role="slider"] {{ background: {BLUE}; }}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------- hero + stats
st.markdown(f"""
<div class="hero">
  <div class="top">
    <div class="brand"><div class="logo">✚</div>Medicaid Expansion Impact</div>
    <div class="badge"><div class="av">IA</div>Built by Isaac Agyapong</div>
  </div>
  <span class="kicker">Machine learning · 3,035 US counties · 2008-2023</span>
  <h1>Did Medicaid expansion work, and what if every state expanded?</h1>
  <p>Medicaid is free or low-cost health insurance for people with low incomes. This model measures how much letting more
     low-income adults sign up actually helped, and predicts what would happen in the 10 states that have not done it.</p>
  <div class="legend"><span><i style="background:#BED2FF"></i>States that expanded</span>
                      <span><i style="background:#FFAA5A"></i>10 states that have not</span></div>
</div>
<div class="stats">
  <div class="stat"><div class="ic" style="background:{BLUE_SOFT};color:{BLUE}">{ICON["down"]}</div>
    <div><div class="num">{EFFECT:.1f} <small>in 100</small></div>
    <div class="lbl">fewer uninsured low-income adults because of expansion</div>
    <div class="sub">first three years · likely {EFFECT_LO:.0f} to {EFFECT_HI:.0f} in 100</div></div></div>
  <div class="stat"><div class="ic" style="background:{GREEN_SOFT};color:{GREEN}">{ICON["people"]}</div>
    <div><div class="num">{COVERED / 1000:,.0f}K</div>
    <div class="lbl">more adults had health insurance in 2023 because of it</div>
    <div class="sub">in the 33 expansion states studied</div></div></div>
  <div class="stat"><div class="ic" style="background:{ORANGE_SOFT};color:{ORANGE}">{ICON["target"]}</div>
    <div><div class="num">{round(GAIN, -4) / 1000:,.0f}K</div>
    <div class="lbl">more adults insured if the last 10 states expand</div>
    <div class="sub">{TX_SHARE:.0%} in Texas · uninsured {RATE_NOW:.0f}% → {RATE_AFTER:.0f}%</div></div></div>
</div>
""", unsafe_allow_html=True)


_CARDS = [0]


def card():
    """White rounded card: a bordered container with a key, so CSS can target it (class st-key-card-N)."""
    _CARDS[0] += 1
    return st.container(border=True, key=f"card-{_CARDS[0]}")


def head(kicker, title, text=None):
    st.markdown(f'<div class="card-k">{kicker}</div><div class="card-h">{title}</div>'
                + (f'<div class="card-p">{text}</div>' if text else ""), unsafe_allow_html=True)


def base_layout(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=6, r=16, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color=INK, size=14),
                      showlegend=False, hoverlabel=dict(font_family="Inter"))
    return fig


CFG = {"displayModeBar": False}
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "What if the rest expanded", "Try the model", "How it works"])

# --------------------------------------------------------------------------------------------- 1. overview
with tab1:
    left, right = st.columns([1.25, 1], gap="medium")
    with left, card():
        es = ATT.query("g == 2014 and t == 2023")
        p23 = PANEL.query("expansion_year == 2014 and year == 2023")
        actual = 100 * p23.uninsured.sum() / p23.population.sum()
        without = actual - float(es.att.iloc[0])
        head("What happened vs what would have happened",
             f"In 2023, expansion still meant about {round(without) - round(actual)} fewer uninsured adults in every 100",
             "States that expanded in 2014: share of low-income adults with no health insurance. The grey bar is the "
             "model's estimate if they had not expanded.")
        fig = go.Figure(go.Bar(y=["Without expansion<br>(estimated)", "With expansion<br>(what happened)"], x=[without, actual],
                               orientation="h", marker=dict(color=["#CBD2DC", BLUE], cornerradius=10),
                               text=[f"{without:.0f}%", f"{actual:.0f}%"], textposition="outside",
                               textfont=dict(size=24, color=INK, family="Inter"), width=0.62))
        fig.update_xaxes(visible=False, range=[0, without * 1.25])
        fig.update_yaxes(tickfont=dict(size=14, color=INK_2))
        st.plotly_chart(base_layout(fig, 250), use_container_width=True, config=CFG)
    with right, card():
        spill = abs(CAUSAL["placebo_138_400"]["att_years_0_2"])
        head("Checks", "Can this result be trusted?", "Four tests, all passed.")
        checks = ["Before 2014, both groups of counties were on the same path, so the comparison is fair.",
                  "Pretending expansion happened in 2011, when it did not, shows no effect.",
                  f"People who earn too much to qualify changed much less ({spill:.0f} in 100, not {EFFECT:.0f}).",
                  "A separate machine learning model gives the same answer."]
        st.markdown('<div class="checks">' + "".join(f'<div class="chk"><div class="dot">{ICON["check"]}</div>{c}</div>'
                                                     for c in checks) + "</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------- 2. what if
with tab2:
    with card():
        head("Prediction by state", f"About {round(GAIN, -4):,.0f} more adults would have health insurance",
             "If the 10 remaining states expanded Medicaid. Estimated from what happened in similar counties that expanded "
             "between 2014 and 2021, applied to each county's situation in 2023.")
        ten = TEN.sort_values("adults_gaining_coverage")
        fig = go.Figure(go.Bar(y=[STATE_NAMES.get(s, s) for s in ten.state], x=ten.adults_gaining_coverage, orientation="h",
                               marker=dict(color=[NAVY if s == "TX" else ORANGE for s in ten.state], cornerradius=8),
                               text=[f"<b>{v:,.0f}</b>   {a:.0f}% → {b:.0f}% uninsured" for v, a, b in
                                     zip(ten.adults_gaining_coverage, ten.rate_2023, ten.predicted_rate_after)],
                               textposition="outside", textfont=dict(size=13, color=INK_2)))
        fig.update_xaxes(visible=False, range=[0, ten.adults_gaining_coverage.max() * 1.55])
        fig.update_yaxes(tickfont=dict(size=14, color=INK))
        st.plotly_chart(base_layout(fig, 430), use_container_width=True, config=CFG)

    with card():
        head("By county", "Where the gains would be largest",
             "Map: darker orange = bigger drop in the share of uninsured adults. Table: where the most people would gain.")
        pick = st.selectbox("State", [STATE_NAMES[s] for s in STATES.sort_values("adults_gaining_coverage", ascending=False).state
                                      if s in STATE_NAMES], index=0)
        code = {v: k for k, v in STATE_NAMES.items()}[pick]
        sc = COUNTIES[COUNTIES.state == code].copy()
        sc["rate_after"] = sc.base_rate + sc.effect_pts
        m1, m2 = st.columns([1.25, 1], gap="medium")
        with m1:
            try:
                fmap = go.Figure(go.Choropleth(
                    geojson=load_geojson(), locations=sc.county_fips, z=-sc.effect_pts, featureidkey="id",
                    colorscale=[[0, "#FFF4E8"], [1, "#C2410C"]], marker_line_color="#FFFFFF", marker_line_width=0.6,
                    colorbar=dict(title="Fewer<br>uninsured<br>per 100", thickness=12, outlinewidth=0),
                    customdata=np.stack([sc.adults_gaining], axis=-1), text=sc.county_name,
                    hovertemplate="<b>%{text}</b><br>%{z:.1f} fewer uninsured in every 100"
                                  "<br>%{customdata[0]:,.0f} adults gaining coverage<extra></extra>"))
                fmap.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(base_layout(fmap, 420), use_container_width=True, config=CFG)
            except Exception:
                st.info("The county map needs an internet connection to load county shapes.")
        with m2:
            top = sc.sort_values("adults_gaining", ascending=False).head(12)
            st.dataframe(pd.DataFrame({"County": top.county_name, "Uninsured today": top.base_rate.map("{:.0f}%".format),
                                       "If expanded": top.rate_after.map("{:.0f}%".format),
                                       "Adults gaining": top.adults_gaining.map("{:,.0f}".format)}),
                         hide_index=True, use_container_width=True, height=420)

# --------------------------------------------------------------------------------------------- 3. try the model
with tab3:
    labels = (COUNTIES.county_name + ", " + COUNTIES.state).tolist()
    with card():
        head("Live model", "Pick any county and see what the model estimates",
             "For counties in states that expanded, it estimates how much their expansion helped. For the others, it "
             "estimates what expanding now would do. Then change the county's situation and watch the answer move.")
        choice = st.selectbox("County", labels, index=labels.index("Harris County, TX") if "Harris County, TX" in labels else 0)
    row = COUNTIES.iloc[labels.index(choice)]
    expanded = pd.notna(row.expansion_year)
    colour, soft = (BLUE, BLUE_SOFT) if expanded else (ORANGE, ORANGE_SOFT)
    c1, c2 = st.columns([1, 1.1], gap="medium")
    with c1, card():
        status = f"Expanded Medicaid in {int(row.expansion_year)}" if expanded else "Has not expanded Medicaid"
        st.markdown(f'<span class="pill" style="background:{soft};color:{colour}">{status}</span>'
                    f'<span class="pill" style="background:#F2F4F7;color:{INK_2}">{row.rurality}</span>'
                    f'<span class="pill" style="background:#F2F4F7;color:{INK_2}">as of {int(row.base_year)}</span>',
                    unsafe_allow_html=True)
        head("Change the county", "What if this county were different?")
        base = st.slider("Share of low-income adults uninsured (%)", 2.0, 75.0, float(round(row.base_rate, 1)), 0.5)
        pov = st.slider("Poverty rate (%)", 2.0, 55.0, float(round(row.poverty_pct_2013, 1)), 0.5)
        hisp = st.slider("Hispanic share of residents (%)", 0.0, 99.0, float(round(row.pct_hispanic_2013, 1)), 0.5)
        income = st.slider("Median household income ($)", 20000, 130000, int(row.median_income_2013), 1000, format="$%d")
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
    with c2, card():
        verb = "Expansion cut the share uninsured here by about" if expanded else "Expanding would cut the share uninsured here by about"
        head("Model estimate", choice)
        st.markdown(f'<div class="big" style="color:{colour}">{abs(eff):.1f}<small> in 100</small></div>'
                    f'<div class="lbl">{verb} {abs(eff):.1f} in every 100 low-income adults</div>'
                    f'<div class="sub">likely between {abs(hi[0]):.1f} and {abs(lo[0]):.1f} in 100</div>',
                    unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=[base, base + eff], y=["Uninsured today" if not expanded else "Before expansion",
                                                         "If expanded" if not expanded else "After expansion"],
                               orientation="h", marker=dict(color=["#CBD2DC", colour], cornerradius=8),
                               text=[f"{base:.0f}%", f"{base + eff:.0f}%"], textposition="outside",
                               textfont=dict(size=18, color=INK), width=0.6))
        fig.update_xaxes(visible=False, range=[0, max(base, 5) * 1.3])
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=13, color=INK_2))
        st.plotly_chart(base_layout(fig, 170), use_container_width=True, config=CFG)
        st.markdown(f'<div class="stat" style="box-shadow:none;background:{soft};padding:16px 18px">'
                    f'<div class="ic" style="background:#fff;color:{colour}">{ICON["people"]}</div><div>'
                    f'<div class="num">{adults:,.0f}</div>'
                    f'<div class="lbl">{"adults insured because of expansion" if expanded else "more adults would have health insurance"}</div>'
                    f'<div class="sub">out of {row.low_income_adults_base:,.0f} low-income adults in this county</div></div></div>',
                    unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------- 4. how it works
with tab4:
    a, b = st.columns(2, gap="medium")
    with a, card():
        head("Step 1", "Compare like with like",
             "For each county in a state that expanded, I compared how much its share of uninsured people changed with "
             "similar counties in states that did not, over the same years. The extra drop is the effect of expansion.")
        st.image(str(ROOT / "Image" / "01_event_study.png"), use_container_width=True)
    with b, card():
        head("Step 2", "Train a machine learning model",
             "A causal forest learned from 1,707 county expansions how the effect depends on things like poverty and how "
             "many people were uninsured. Tested on states it never saw, the counties it ranked highest really gained most.")
        st.image(str(ROOT / "Image" / "03_model_validation.png"), use_container_width=True)
    with st.expander("Technical details"):
        st.markdown(f"""
* **Data:** US Census Small Area Health Insurance Estimates 2008-2023, 3,035 counties in 45 states; KFF expansion dates;
  2013 county traits. Prepared in PostgreSQL by the companion analytics project.
* **Step 1:** Callaway & Sant'Anna (2021) staggered difference-in-differences, adjusted for county traits, population
  weighted, 95% intervals from 499 bootstrap resamples of states. Effect over years 0-2:
  **{CAUSAL['adjusted']['att_years_0_2']:.2f}** percentage points (95% CI {CAUSAL['adjusted']['ci'][0]:.2f} to {CAUSAL['adjusted']['ci'][1]:.2f}).
* **Step 2:** EconML `CausalForestDML` (double machine learning, gradient-boosted nuisance models, 2,000 honest trees,
  folds grouped by state). Average effect **{ML['forest_ate_expansion_counties']:.2f}** points.
* **Limitations:** Census estimates have margins of error; states chose whether to expand; predictions assume
  expansion would work as it did in similar counties and are not enrollment forecasts.
""")

st.markdown('<div class="foot">Built by Isaac Agyapong · '
            '<a href="https://github.com/Isaac-Agyapong/Medicaid_Expansion_Impact_Model">Code and write-up</a> · '
            '<a href="https://github.com/Isaac-Agyapong/Health_Insurance_Coverage_Gap_Analysis">Analytics and Power BI dashboard</a>'
            '</div>', unsafe_allow_html=True)
