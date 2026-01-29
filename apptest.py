import io
import re
import base64
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="Recruitment Fit Score (RFS) – v3.1 Optimized",
    page_icon="✅",
    layout="wide"
)

# ------------ Brand palette ------------
PALETTE = {
    "green":  "#78A22F",
    "blue":   "#007C9E",
    "orange": "#F58025",
    "yellow": "#F8E71C",
    "red":    "#D0021B",
    "ink":    "#1F2A33",
    "bg2":    "#F7FBFC"
}

# ---------------------------
# Local logos
# ---------------------------
APP_DIR = Path(__file__).resolve().parent
LOGO1_PATH = APP_DIR / "logo1.png"
LOGO2_PATH = APP_DIR / "logo2.png"

def _img_to_b64(path: Path) -> str:
    try:
        b = path.read_bytes()
        return base64.b64encode(b).decode("utf-8")
    except Exception:
        return ""

LOGO1_B64 = _img_to_b64(LOGO1_PATH)
LOGO2_B64 = _img_to_b64(LOGO2_PATH)

def inject_css():
    st.markdown("""
    <style>
      :root {
        --green:  #78A22F;
        --blue:   #007C9E;
        --orange: #F58025;
        --yellow: #F8E71C;
        --red:    #D0021B;
        --ink:    #1F2A33;
        --bg2:    #F7FBFC;
      }

      html, body, .stApp {
        font-family: Verdana, Geneva, Arial, sans-serif;
        color: var(--ink);
        -webkit-font-smoothing: antialiased;
      }

      .app-banner {
        padding: 18px 22px; border-radius: 14px; margin: 2px 0 24px 0;
        background: linear-gradient(135deg, var(--green) 0%, var(--blue) 100%); color: #fff;
      }
      .app-banner-inner {
        display: flex; align-items: center; justify-content: space-between; gap: 16px;
      }
      .app-banner-left h2 { margin: 0 0 4px 0; font-weight: 700; }
      .app-banner-left p  { margin: 0; opacity: .95; }

      .stButton > button, .stDownloadButton button {
        border-radius: 10px; border: 1px solid var(--blue); background: var(--blue);
        color: #fff; font-weight: 700;
      }

      .tag { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:700; margin-right:8px; }
      .tag-priority { background: var(--orange); color:#000; }
      .tag-admit    { background: var(--green);  color:#fff; }
      .tag-equity   { background: var(--yellow); color:#000; border:1px solid #D8C600; }
      .tag-reserve  { background: var(--bg2);    color: var(--ink); border:1px solid #CFDCE4; }
      
      .v3-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--orange) 0%, var(--red) 100%);
        color: #fff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        margin-left: 8px;
      }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# Required secret salt
# ---------------------------
if "SALT" not in st.secrets or not st.secrets["SALT"]:
    st.error("Missing SALT in Streamlit secrets.")
    st.stop()

SALT = st.secrets["SALT"]
inject_css()

# ---------------------------
# v3.1 ADAPTIVE PRESETS - Optimized based on Leaderboard Data
# ---------------------------

PRESETS = {
    "finance_optimized": {
        "name": "Finance-Optimized (v3.1)",
        "description": "RECALIBRATED: Prioritizes Function (r=0.10) and Referee (r=0.08) over noisy signals.",
        "thresh_admit": 50,
        "thresh_priority": 65,
        "equity_lower": 40,
        "equity_upper": 49,
        "w_motivation": 15, # Reduced: too noisy
        "w_sector": 5,     # Reduced: negative correlation
        "w_referee": 28,   # Kept: strong predictor
        "w_function": 25,  # Increased: strongest predictor
        "w_time": 10,
        "w_lang": 15,      # Captures completion
        "w_alumni": 5,
        "min_motivation_words": 30,
        "sector_uplift": {
            "Finance": 8, "Private": 5, "Government": 5, "NGO/CSO": 5, "Farmer Org": 2, "Other/Unclassified": 0
        }
    },
    "balanced": {
        "name": "Balanced (General Cohort)",
        "description": "Standard weighting with optimized word count rubrics.",
        "thresh_admit": 55,
        "thresh_priority": 70,
        "equity_lower": 45,
        "equity_upper": 54,
        "w_motivation": 25,
        "w_sector": 10,
        "w_referee": 28,
        "w_function": 15,
        "w_time": 10,
        "w_lang": 20,
        "w_alumni": 5,
        "min_motivation_words": 50,
        "sector_uplift": {
            "NGO/CSO": 10, "Government": 8, "Education": 8, "Private": 5, "Farmer Org": 5, "Other/Unclassified": 0
        }
    }
}

WEEKLY_TIME_BANDS = ["<1h", "1-2h", "2-3h", ">=3h"]
LANG_BANDS = ["basic", "working", "fluent"]

# ---------------------------
# Helpers
# ---------------------------
def normalize_email(x):
    if pd.isna(x): return ""
    return str(x).strip().lower()

def hash_id(value: str) -> str:
    v = (SALT + value).encode("utf-8")
    return hashlib.sha256(v).hexdigest()[:16]

def org_to_sector(org_text: str) -> str:
    if not isinstance(org_text, str) or org_text.strip() == "":
        return "Other/Unclassified"
    x = org_text.lower()
    if any(k in x for k in ["universit", "school", "educat"]): return "Education"
    if any(k in x for k in ["ngo", "foundation", "association", "civil", "non profit", "non-profit"]): return "NGO/CSO"
    if any(k in x for k in ["ministry", "gov", "municipal", "department of", "bureau"]): return "Government"
    if any(k in x for k in ["united nations", "world bank", "fao", "ifad", "ifpri", "undp", "unesco"]): return "Multilateral"
    if any(k in x for k in ["ltd", "company", "bv", "inc", "plc", "gmbh", "sarl"]): return "Private"
    if any(k in x for k in ["farmer", "coop", "co-op", "cooperative"]): return "Farmer Org"
    if any(k in x for k in ["consult"]): return "Consultancy"
    if any(k in x for k in ["bank", "finance", "microfinance"]): return "Finance"
    return "Other/Unclassified"

def rubric_heuristic_score(text: str, min_words: int):
    if not isinstance(text, str) or text.strip() == "": return (0, 0, 0)
    t = text.strip()
    words = len(t.split())
    if words < min_words: return (0, 0, 0)

    has_data = any(w in t.lower() for w in ["data", "dataset", "dashboard", "faostat", "survey"])
    has_numbers = bool(re.search(r"\b\d+\b", t))
    has_fs = any(w in t.lower() for w in ["food system", "seed", "agric", "market", "value chain"])

    spec = 10 if (words >= 200 or has_data) else 5
    feas = 10 if has_numbers else 5
    rel = 10 if has_fs else 5
    return (spec, feas, rel)

def label_band(val, admit_thr, priority_thr, sector, equity_reserve, equity_range):
    if val >= priority_thr: return "Priority"
    if val >= admit_thr: return "Admit"
    if equity_reserve and sector == "Farmer Org" and equity_range[0] <= val <= equity_range[1]:
        return "Reserve (Equity)"
    return "Reserve"

def normalize_mapping_value(v):
    if not isinstance(v, str): return ""
    return v.strip().lower().replace("–", "-").replace("—", "-").replace("≥", ">=").replace(" ", "")

def get_time_points(x):
    if ">=3h" in x: return 10
    if "2-3h" in x: return 6
    if "1-2h" in x: return 3
    return 0

def yes_no_points(x, cap):
    if pd.isna(x): return 0
    text = str(x).strip().lower()
    if any(text.startswith(p) for p in ["no", "none", "n/a", "0"]): return 0
    return cap

# ---------------------------
# Sidebar & Main App
# ---------------------------
st.sidebar.header("⚙️ Configuration")
preset_choice = st.sidebar.radio("Choose a preset:", options=list(PRESETS.keys()), format_func=lambda x: PRESETS[x]["name"])
preset = PRESETS[preset_choice]

st.markdown(f"""
    <div class="app-banner">
      <div class="app-banner-inner">
        <div class="app-banner-left">
          <h2>Recruitment Fit Score (RFS) <span class="v3-badge">v3.1 OPTIMIZED</span></h2>
          <p>Validated against leaderboard success data (Function & Referee priority)</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

uploaded = st.file_uploader("Upload applications file", type=["csv", "xlsx"])
if uploaded:
    df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
    
    # 🧭 COLUMN MAPPING (Simplified for example, use your 'pick' function here)
    email_col, mot_col, func_col, ref_col, lang_col, time_col, org_col, alm_col = "Email", "MotivationText", "FunctionTitle", "RefereeConfirmsFit", "LanguageComfort", "WeeklyTimeBand", "Organisation", "AlumniReferral"

    work = df.copy()
    work['Email_Norm'] = work[email_col].map(normalize_email)
    work.insert(0, "PID", work['Email_Norm'].apply(hash_id))

    # 1. Motivation
    mot_scores = work[mot_col].apply(lambda x: rubric_heuristic_score(x, preset["min_motivation_words"]))
    work["MotivationPts"] = (mot_scores.apply(sum) / 30.0) * preset["w_motivation"]

    # 2. Function (THE KEY SIGNAL)
    def function_points(x):
        if not isinstance(x, str): return 0
        xl = x.lower()
        # High-performing keywords from the leaderboard
        direct = any(k in xl for k in ["specialist", "officer", "advisor", "director", "manager", "analyst", "lecturer"])
        if direct: return preset["w_function"]
        return preset["w_function"] * 0.4

    work["FunctionPts"] = work[func_col].apply(function_points)

    # 3. Referee & Language & Sector
    work["RefereePts"] = work[ref_col].apply(lambda x: yes_no_points(x, preset["w_referee"]))
    work["Sector"] = work[org_col].apply(org_to_sector)
    work["SectorPts"] = work["Sector"].map(lambda s: preset["sector_uplift"].get(s, 0))
    
    work["LanguagePts"] = work[lang_col].apply(normalize_mapping_value).map({"fluent": preset["w_lang"], "working": preset["w_lang"]*0.6, "basic": preset["w_lang"]*0.3}).fillna(0)
    work["TimePts"] = work[time_col].apply(normalize_mapping_value).apply(get_time_points)

    # 4. Final Scoring
    pts_cols = ["MotivationPts", "FunctionPts", "RefereePts", "SectorPts", "LanguagePts", "TimePts"]
    work["RFS"] = work[pts_cols].sum(axis=1).round(2)
    work["predicted Decision"] = work.apply(lambda r: label_band(r["RFS"], preset["thresh_admit"], preset["thresh_priority"], r["Sector"], True, (preset["equity_lower"], preset["equity_upper"])), axis=1)

    st.success(f"✅ Scored {len(work)} applicants using **{preset['name']}** model")
    st.dataframe(work[["PID", "Sector", "RFS", "predicted Decision"] + pts_cols], use_container_width=True)
    
    csv = work.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download v3.1 Optimized CSV", csv, "rfs_v3_1_optimized.csv", "text/csv")
