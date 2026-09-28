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
ALL_STATES = dict(AL="Alabama", AK="Alaska", AZ="Arizona", AR="Arkansas", CA="California", CO="Colorado", CT="Connecticut",
                  DE="Delaware", DC="District of Columbia", FL="Florida", GA="Georgia", HI="Hawaii", ID="Idaho", IL="Illinois",
                  IN="Indiana", IA="Iowa", KS="Kansas", KY="Kentucky", LA="Louisiana", ME="Maine", MD="Maryland",
                  MA="Massachusetts", MI="Michigan", MN="Minnesota", MS="Mississippi", MO="Missouri", MT="Montana",
                  NE="Nebraska", NV="Nevada", NH="New Hampshire", NJ="New Jersey", NM="New Mexico", NY="New York",
                  NC="North Carolina", ND="North Dakota", OH="Ohio", OK="Oklahoma", OR="Oregon", PA="Pennsylvania",
                  RI="Rhode Island", SC="South Carolina", SD="South Dakota", TN="Tennessee", TX="Texas", UT="Utah",
                  VT="Vermont", VA="Virginia", WA="Washington", WV="West Virginia", WI="Wisconsin", WY="Wyoming")
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
  .stButton button {{ border-radius: 999px; border: 1px solid #E4E7EC; background: #fff; color: {INK}; font-weight: 600;
                      font-size: 13px; padding: 6px 12px; min-height: 38px; }}
  .stButton button:hover {{ border-color: {BLUE}; color: {BLUE}; background: {BLUE_SOFT}; }}
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
PERSON = ('<svg viewBox="0 0 24 24" width="100%" height="100%"><circle cx="12" cy="6.2" r="4.2" fill="{c}"/>'
          '<path d="M4 22c0-5 3.6-8.6 8-8.6s8 3.6 8 8.6z" fill="{c}"/></svg>')
QUICK = ["Harris County, TX", "Miami-Dade County, FL", "Fulton County, GA", "Hidalgo County, TX", "Cook County, IL",
         "Los Angeles County, CA"]
SLIDERS = [("base_rate", "Share of low-income adults uninsured", 2.0, 75.0, 0.5, "%.1f%%"),
           ("poverty_pct_2013", "Poverty rate", 2.0, 55.0, 0.5, "%.1f%%"),
           ("pct_hispanic_2013", "Hispanic share of residents", 0.0, 99.0, 0.5, "%.1f%%"),
           ("median_income_2013", "Median household income", 20000.0, 130000.0, 1000.0, "$%d")]


def people_grid(still, gain, colour_gain, colour_still):
    """100 person icons, filled row by row: still uninsured, then gaining coverage, then already insured."""
    cells = []
    for i in range(100):
        c = colour_still if i < still else colour_gain if i < still + gain else "#DDE2EA"
        cells.append(f'<div style="width:100%;aspect-ratio:1">{PERSON.format(c=c)}</div>')
    return ('<div style="display:grid;grid-template-columns:repeat(20,minmax(0,1fr));gap:5px 5px;margin:10px 0 8px 0;max-width:100%">'
            + "".join(cells) + "</div>")


def county_locator(row):
    try:
        st_counties = COUNTIES[COUNTIES.state == row.state]
        z = (st_counties.county_fips == row.county_fips).astype(int)
        fig = go.Figure(go.Choropleth(geojson=load_geojson(), locations=st_counties.county_fips, z=z, featureidkey="id",
                                      colorscale=[[0, "#E4E9F2"], [1, ORANGE if pd.isna(row.expansion_year) else BLUE]],
                                      showscale=False, marker_line_color="#FFFFFF", marker_line_width=0.5,
                                      text=st_counties.county_name, hovertemplate="%{text}<extra></extra>"))
        fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(base_layout(fig, 190), use_container_width=True, config=CFG)
    except Exception:
        pass


def pick_county(label):
    st.session_state.county = label


with tab3:
    labels = (COUNTIES.county_name + ", " + COUNTIES.state).tolist()
    st.session_state.setdefault("county", "Harris County, TX")
    with card():
        k1, k2 = st.columns([1, 1.25], gap="large")
        with k1:
            head("Live model", "Pick any county and see what the model estimates",
                 "The machine learning model runs live on the county you choose.")
            st.selectbox("Search for a county", labels, key="county")
        with k2:
            st.markdown('<div style="height:34px"></div><div class="card-k" style="color:#98A2B3">Or try one of these</div>',
                        unsafe_allow_html=True)
            cols = st.columns(3)
            for i, q in enumerate(QUICK):
                cols[i % 3].button(q.replace(" County", ""), key=f"quick{i}", on_click=pick_county, args=(q,),
                                   use_container_width=True)

    choice = st.session_state.county
    row = COUNTIES.iloc[labels.index(choice)]
    fips = row.county_fips
    expanded = pd.notna(row.expansion_year)
    colour, soft = (BLUE, BLUE_SOFT) if expanded else (ORANGE, ORANGE_SOFT)

    # sliders live in the right column but their values drive the estimate on the left, so read them first
    defaults = {"base_rate": float(round(row.base_rate, 1)), "poverty_pct_2013": float(round(row.poverty_pct_2013, 1)),
                "pct_hispanic_2013": float(round(row.pct_hispanic_2013, 1)), "median_income_2013": float(row.median_income_2013)}
    for key, *_ in SLIDERS:
        st.session_state.setdefault(f"{key}_{fips}", defaults[key])

    x = row[FEATURES].astype(float).copy()
    for key, *_ in SLIDERS:
        v = st.session_state[f"{key}_{fips}"]
        if v != defaults[key]:                       # keep the county's exact value unless a slider was moved
            if key == "median_income_2013":
                x["log_income_2013"] = np.log(v)
            else:
                x[key] = v
    moved = any(st.session_state[f"{k}_{fips}"] != defaults[k] for k, *_ in SLIDERS)
    base = float(x["base_rate"])
    X = x.values.reshape(1, -1)
    eff = float(model.effect(X)[0])
    lo, hi = model.effect_interval(X, alpha=0.05)
    after = max(base + eff, 0.0)
    adults = -eff / 100 * row.low_income_adults_base
    same_state = COUNTIES[COUNTIES.state == row.state]
    rank = int((same_state.adults_gaining > row.adults_gaining).sum()) + 1

    left, right = st.columns([1.35, 1], gap="medium")
    with left, card():
        status = f"Expanded Medicaid in {int(row.expansion_year)}" if expanded else "Has not expanded Medicaid"
        st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px">'
                    f'<div><div class="card-k">Model estimate{" · what-if" if moved else ""}</div>'
                    f'<div class="card-h" style="font-size:28px">{choice}</div></div></div>'
                    f'<span class="pill" style="background:{soft};color:{colour}">{status}</span>'
                    f'<span class="pill" style="background:#F2F4F7;color:{INK_2}">{row.rurality}</span>'
                    f'<span class="pill" style="background:#F2F4F7;color:{INK_2}">'
                    f'#{rank} of {len(same_state)} {ALL_STATES.get(row.state, row.state)} counties for people gaining</span>',
                    unsafe_allow_html=True)
        before_lbl, after_lbl = ("Before expansion", "After expansion") if expanded else ("Uninsured today", "If the state expanded")
        gain_lbl = "insured because of expansion" if expanded else "would gain health insurance"
        tiles = [(before_lbl, f"{base:.0f}%", "of low-income adults uninsured", INK, "#F8FAFC"),
                 (after_lbl, f"{after:.0f}%", f"likely {base + lo[0]:.0f}% to {base + hi[0]:.0f}%", colour, soft),
                 ("People", f"{adults:,.0f}", gain_lbl, GREEN, GREEN_SOFT)]
        st.markdown('<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:14px 0 4px 0">' + "".join(
            f'<div style="background:{bg};border-radius:14px;padding:14px 16px"><div class="sub" style="margin:0">{t}</div>'
            f'<div class="num" style="color:{c};font-size:30px;margin-top:4px">{v}</div><div class="sub">{s}</div></div>'
            for t, v, s, c, bg in tiles) + "</div>", unsafe_allow_html=True)
        still, gain = round(after), round(base) - round(after)
        st.markdown(f'<div class="lbl" style="margin-top:16px">Out of every 100 low-income adults in {row.county_name}</div>'
                    + people_grid(still, gain, GREEN, colour)
                    + f'<div style="display:flex;gap:18px;font-size:13px;color:{INK_2};flex-wrap:wrap">'
                      f'<span><b style="color:{colour}">●</b> {still} still uninsured</span>'
                      f'<span><b style="color:{GREEN}">●</b> {gain} {"gained" if expanded else "would gain"} coverage</span>'
                      f'<span><b style="color:#C4CBD6">●</b> {100 - still - gain} already insured</span></div>',
                    unsafe_allow_html=True)

    with right:
        with card():
            head("Where it is", f"{row.county_name}, {ALL_STATES.get(row.state, row.state)}")
            county_locator(row)
        with card():
            head("What if", "Change the county and watch the estimate move")
            for key, label, lo_, hi_, step, fmt in SLIDERS:
                st.slider(label, lo_, hi_, step=step, format=fmt, key=f"{key}_{fips}")

            def reset():
                for k, *_ in SLIDERS:
                    st.session_state[f"{k}_{fips}"] = defaults[k]
            st.button("Reset to the real county", on_click=reset, use_container_width=True, disabled=not moved)

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
