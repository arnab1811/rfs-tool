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
    page_title="Recruitment Fit Score (RFS) – v3 Adaptive",
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
        font-family: Verdana, Geneva, Arial, "Segoe UI Emoji", "Noto Color Emoji", "Apple Color Emoji", sans-serif;
        color: var(--ink);
        -webkit-font-smoothing: antialiased;
      }

      .material-icons,
      .material-icons-outlined,
      .material-icons-round,
      .material-icons-sharp,
      .material-icons-two-tone {
        font-family: "Material Icons", "Material Icons Outlined", "Material Icons Round",
                     "Material Icons Sharp", "Material Icons Two Tone" !important;
        font-weight: normal; font-style: normal; line-height: 1; letter-spacing: normal;
        text-transform: none; display: inline-block; white-space: nowrap; word-wrap: normal;
        direction: ltr; -webkit-font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased;
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

      .app-logos {
        display: flex; align-items: center; gap: 10px;
      }
      .app-logos img {
        height: 40px; width: auto;
        background: rgba(255,255,255,0.95);
        border-radius: 10px;
        padding: 6px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.12);
      }

      .stButton > button, .stDownloadButton button {
        border-radius: 10px; border: 1px solid var(--blue); background: var(--blue);
        color: #fff; font-weight: 700;
      }
      .stButton > button:hover, .stDownloadButton button:hover { filter: brightness(0.92); }

      .tag { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:700; margin-right:8px; }
      .tag-priority { background: var(--orange); color:#000; }
      .tag-admit    { background: var(--green);  color:#fff; }
      .tag-equity   { background: var(--yellow); color:#000; border:1px solid #D8C600; }
      .tag-reserve  { background: var(--bg2);    color: var(--ink); border:1px solid #CFDCE4; }

      .stDataFrame tbody td, .stDataFrame thead th { font-size: 13px; }
      h1,h2,h3 { letter-spacing:.2px; }
      
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
      
      .preset-card {
        border: 2px solid var(--blue);
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
        background: var(--bg2);
      }
      .preset-card h4 { margin: 0 0 8px 0; color: var(--blue); }
      .preset-card p { margin: 0; font-size: 13px; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# Required secret salt
# ---------------------------
if "SALT" not in st.secrets or not st.secrets["SALT"]:
    st.error("Missing SALT in Streamlit secrets. Set SALT locally in .streamlit/secrets.toml and in Streamlit Cloud app settings.")
    st.stop()

SALT = st.secrets["SALT"]
inject_css()

# ---------------------------
# v3 ADAPTIVE PRESETS - Based on validation data
# ---------------------------

PRESETS = {
    "balanced": {
        "name": "Balanced (Main Cohort)",
        "description": "Best for general food systems course (r=0.13 validation)",
        "thresh_admit": 55,
        "thresh_priority": 70,
        "equity_lower": 45,
        "equity_upper": 54,
        "w_motivation": 30,
        "w_sector": 12,
        "w_referee": 28,
        "w_function": 15,
        "w_time": 10,
        "w_lang": 20,
        "w_alumni": 10,
        "min_motivation_words": 30,  # More lenient
        "sector_uplift": {
            "Education": 12, "NGO/CSO": 10, "Government": 8, "Multilateral": 8,
            "Private": 8, "Farmer Org": 0, "Consultancy": 8, "Finance": 8,
            "Other/Unclassified": 0
        }
    },
    
    "lenient": {
        "name": "Lenient (High Acceptance)",
        "description": "Lower bar, maximize admits (use when capacity available)",
        "thresh_admit": 40,
        "thresh_priority": 60,
        "equity_lower": 30,
        "equity_upper": 39,
        "w_motivation": 20,
        "w_sector": 8,
        "w_referee": 20,
        "w_function": 12,
        "w_time": 10,
        "w_lang": 15,
        "w_alumni": 10,
        "min_motivation_words": 20,  # Very lenient
        "sector_uplift": {
            "Education": 8, "NGO/CSO": 8, "Government": 6, "Multilateral": 6,
            "Private": 6, "Farmer Org": 0, "Consultancy": 6, "Finance": 6,
            "Other/Unclassified": 0
        }
    },
    
    "strict": {
        "name": "Strict (Quality Focus)",
        "description": "Higher bar, focus on completion (use when limited capacity)",
        "thresh_admit": 65,
        "thresh_priority": 80,
        "equity_lower": 55,
        "equity_upper": 64,
        "w_motivation": 35,
        "w_sector": 10,
        "w_referee": 30,
        "w_function": 12,
        "w_time": 12,
        "w_lang": 25,
        "w_alumni": 10,
        "min_motivation_words": 75,  # Strict
        "sector_uplift": {
            "Education": 10, "NGO/CSO": 8, "Government": 6, "Multilateral": 6,
            "Private": 6, "Farmer Org": 0, "Consultancy": 6, "Finance": 6,
            "Other/Unclassified": 0
        }
    },
    
    "finance_optimized": {
        "name": "Finance-Optimized",
        "description": "Specialized for finance cohort (FunctionPts weighted higher)",
        "thresh_admit": 45,
        "thresh_priority": 65,
        "equity_lower": 35,
        "equity_upper": 44,
        "w_motivation": 25,
        "w_sector": 8,
        "w_referee": 25,
        "w_function": 20,  # Higher for finance
        "w_time": 12,
        "w_lang": 15,
        "w_alumni": 10,
        "min_motivation_words": 30,
        "sector_uplift": {
            "Education": 8, "NGO/CSO": 8, "Government": 8, "Multilateral": 10,
            "Private": 12, "Farmer Org": 0, "Consultancy": 10, "Finance": 15,  # Finance boosted
            "Other/Unclassified": 0
        }
    }
}

WEEKLY_TIME_BANDS = ["<1h", "1–2h", "2–3h", "≥3h", "1-2h", "2-3h", ">=3h"]
LANG_BANDS = ["Basic/With support", "Working", "Fluent"]

# ---------------------------
# Helpers
# ---------------------------
def normalize_email(x):
    if pd.isna(x):
        return ""
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

def rubric_heuristic_score(text: str, min_words: int, length_targets=(200, 300)):
    """
    ADAPTIVE: Minimum word count now configurable based on preset
    """
    if not isinstance(text, str) or text.strip() == "":
        return (0, 0, 0)

    t = text.strip()
    words = len(t.split())

    # Configurable minimum
    if words < min_words:
        return (0, 0, 0)

    has_numbers = bool(re.search(r"\b\d+\b", t))
    has_when = any(w in t.lower() for w in ["week", "month", "timeline", "plan", "schedule"])
    has_where = any(w in t.lower() for w in ["district", "province", "country", "region", "university", "ministry"])
    has_data = any(w in t.lower() for w in ["data", "dataset", "dashboard", "faostat", "survey", "indicator"])
    has_role = any(w in t.lower() for w in ["lecturer", "extension", "officer", "analyst", "programme", "policy"])

    # Specificity
    spec = 0
    if words >= length_targets[0]: spec += 4
    if words >= length_targets[1]: spec += 2
    if has_where or has_data: spec += 3
    if has_role: spec += 1
    spec = min(spec, 10)

    # Feasibility
    feas = 0
    if has_numbers: feas += 3
    if has_when: feas += 4
    if "pilot" in t.lower() or "test" in t.lower(): feas += 2
    if "5–6 weeks" in t.lower() or "5-6 weeks" in t.lower(): feas += 1
    feas = min(feas, 10)

    # Relevance
    rel = 0
    fs_keywords = ["food system", "seed", "agric", "market", "value chain", "policy", "extension", "nutrition", "farm"]
    keyword_count = sum(1 for k in fs_keywords if k in t.lower())
    if keyword_count >= 2: rel += 4
    elif keyword_count >= 1: rel += 2
    if has_where: rel += 3
    if has_data: rel += 2
    if "student" in t.lower() or "farmer" in t.lower(): rel += 1
    rel = min(rel, 10)

    return (spec, feas, rel)

def label_band(val, admit_thr, priority_thr, sector, equity_reserve, equity_range):
    if val >= priority_thr:
        return "Priority"
    if val >= admit_thr:
        return "Admit"
    if equity_reserve and sector == "Farmer Org" and equity_range[0] <= val <= equity_range[1]:
        return "Reserve (Equity)"
    return "Reserve"

def normalize_time_value(v):
    if not isinstance(v, str):
        return v
    x = v.strip()
    x = x.replace("–", "-").replace("—", "-")
    x = x.replace("≥", ">=")
    x_compact = re.sub(r"\s+", "", x.lower())

    if x_compact in ["<1h", "<1hour", "<1hrs", "<1hr"]:
        return "<1h"
    if x_compact in ["1-2h", "1-2hours", "1-2hrs", "1-2hr"]:
        return "1-2h"
    if x_compact in ["2-3h", "2-3hours", "2-3hrs", "2-3hr"]:
        return "2-3h"
    if x_compact in [">=3h", ">=3hours", ">=3hrs", ">=3hr"]:
        return ">=3h"
    return x

def map_band(val, valid):
    if not isinstance(val, str):
        return None
    v = val.strip()
    return v if v in valid else None

def get_time_points(x):
    if x in ["≥3h", ">=3h"]: return 10
    if x in ["2–3h", "2-3h"]: return 6
    if x in ["1–2h", "1-2h"]: return 3
    if x == "<1h": return 0
    return 0

def yes_no_points(x, cap):
    if pd.isna(x) or x is None:
        return 0
    text = str(x).strip().lower()
    if not text:
        return 0

    no_prefixes = ["no", "none", "n/a", "na", "not applicable", "0"]
    if any(text == p or text.startswith(p + " ") or text.startswith(p + ",") for p in no_prefixes):
        return 0

    yes_prefixes = ["yes", "y", "true", "1", "ja", "oui", "si"]
    if any(text == p or text.startswith(p + " ") or text.startswith(p + ",") for p in yes_prefixes):
        return cap

    return cap

# ---------------------------
# Sidebar – Configuration
# ---------------------------
st.sidebar.header("⚙️ Configuration")

st.sidebar.info("🎯 **v3 Adaptive** - Choose a preset or customize weights")

# PRESET SELECTION
st.sidebar.subheader("1️⃣ Select Preset")
preset_choice = st.sidebar.radio(
    "Choose a scoring preset:",
    options=list(PRESETS.keys()),
    format_func=lambda x: PRESETS[x]["name"],
    help="Each preset is optimized for different use cases based on validation data"
)

preset = PRESETS[preset_choice]

with st.sidebar.expander("ℹ️ About this preset"):
    st.markdown(f"**{preset['name']}**")
    st.markdown(preset['description'])

st.sidebar.markdown("---")
st.sidebar.subheader("2️⃣ Adjust Settings (Optional)")

# Allow overrides
custom_mode = st.sidebar.checkbox("Customize weights", value=False, 
    help="Override preset values with your own")

if custom_mode:
    st.sidebar.caption("⚠️ Custom mode - preset values overridden")
    thr_admit = st.sidebar.number_input("Admit threshold", 0, 100, preset["thresh_admit"], 1)
    thr_priority = st.sidebar.number_input("Priority threshold", 0, 100, preset["thresh_priority"], 1)
    
    w_motivation = st.sidebar.slider("Motivation", 0, 40, preset["w_motivation"])
    w_sector = st.sidebar.slider("Sector", 0, 30, preset["w_sector"])
    w_referee = st.sidebar.slider("Referee", 0, 30, preset["w_referee"])
    w_function = st.sidebar.slider("Function", 0, 25, preset["w_function"])
    w_time = st.sidebar.slider("Time", 0, 20, preset["w_time"])
    w_lang = st.sidebar.slider("Language", 0, 25, preset["w_lang"])
    w_alumni = st.sidebar.slider("Alumni", 0, 15, preset["w_alumni"])
    
    min_mot_words = st.sidebar.slider("Min motivation words", 10, 100, preset["min_motivation_words"])
    
    sector_uplift = preset["sector_uplift"].copy()
else:
    # Use preset values
    thr_admit = preset["thresh_admit"]
    thr_priority = preset["thresh_priority"]
    w_motivation = preset["w_motivation"]
    w_sector = preset["w_sector"]
    w_referee = preset["w_referee"]
    w_function = preset["w_function"]
    w_time = preset["w_time"]
    w_lang = preset["w_lang"]
    w_alumni = preset["w_alumni"]
    min_mot_words = preset["min_motivation_words"]
    sector_uplift = preset["sector_uplift"]

equity_on = st.sidebar.checkbox("Enable Equity Reserve for Farmer Orgs", value=True)
equity_lower = preset["equity_lower"]
equity_upper = preset["equity_upper"]

st.sidebar.markdown("---")
st.sidebar.caption(f"💡 Using **{preset['name']}** preset")

# ---------------------------
# Banner
# ---------------------------
logos_html = ""
if LOGO1_B64:
    logos_html += f'<img src="data:image/png;base64,{LOGO1_B64}" alt="Logo 1">'
if LOGO2_B64:
    logos_html += f'<img src="data:image/png;base64,{LOGO2_B64}" alt="Logo 2">'

st.markdown(
    f"""
    <div class="app-banner">
      <div class="app-banner-inner">
        <div class="app-banner-left">
          <h2>Recruitment Fit Score (RFS) <span class="v3-badge">v3 ADAPTIVE</span></h2>
          <p>Smart presets for different cohorts - choose the right model for your needs</p>
        </div>
        <div class="app-logos">
          {logos_html}
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Main
# ---------------------------
st.title("Recruitment Fit Score (RFS) – v3 Adaptive")
st.write("Upload an **applications CSV** (UTF-8) or **XLSX**. Emails are immediately replaced with `PID` (hashed).")

with st.expander("🆕 What's New in v3 - Adaptive Presets"):
    st.markdown("""
**v3 introduces smart presets based on validation data:**

### 📊 **Available Presets:**

<div class="preset-card">
<h4>🎯 Balanced (Main Cohort)</h4>
<p>Best for general food systems course. Validated r=0.13 correlation. Good balance between precision and recall.</p>
</div>

<div class="preset-card">
<h4>🟢 Lenient (High Acceptance)</h4>
<p>Lower bar, maximize admits. Use when you have capacity and want to be inclusive. Lower thresholds (40/60).</p>
</div>

<div class="preset-card">
<h4>🔴 Strict (Quality Focus)</h4>
<p>Higher bar, focus on completion. Use when capacity is limited. Requires 75+ word motivations, higher thresholds (65/80).</p>
</div>

<div class="preset-card">
<h4>💼 Finance-Optimized</h4>
<p>Specialized for finance cohort. FunctionPts weighted higher (r=0.103). Finance sector boosted to 15 points.</p>
</div>

### **Key Improvements over v2:**
1. ✅ **Adaptive minimum word counts** (20-75 based on preset)
2. ✅ **Cohort-specific weights** (Finance gets different weights)
3. ✅ **Flexible thresholds** (40-80 range)
4. ✅ **Validated against real data** (all presets tested)
5. ✅ **Easy to switch** between strict/lenient modes

### **When to Use Each Preset:**
- **Balanced**: Default choice, works for most cohorts
- **Lenient**: High capacity, want to give more people a chance
- **Strict**: Limited capacity, prioritize completion rates
- **Finance-Optimized**: Specifically for finance/economics courses

**Still getting low correlations?** That means the **components themselves** need to change, not just the weights. Consider collecting: prior experience, organization reputation, time availability evidence, etc.
""", unsafe_allow_html=True)

with st.expander("📄 Expected columns"):
    st.markdown("""
**All fields optional. Map what you have:**

**Core:**
- **Email** (unique ID; hashed to PID)
- **Organisation / SectorText** (text for sector inference)
- **MotivationText** (free text)

**Optional:**
- **Sector** (structured)
- **FunctionTitle**
- **WeeklyTimeBand** (`<1h`, `1-2h`, `2-3h`, `>=3h`)
- **LanguageComfort** (`Basic/With support`, `Working`, `Fluent`)
- **RefereeConfirmsFit** (`yes`/`no` or free text)
- **AlumniReferral** (`yes`/`no` or free text)
- **ApplicationDate** (for dedup)
""")

def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    name = (uploaded_file.name or "").lower()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file, engine="openpyxl")

    raw_bytes = uploaded_file.getvalue()
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, sep=None, engine="python")
        except UnicodeDecodeError:
            continue
        except Exception:
            for sep in [",", ";", "\t", "|"]:
                try:
                    return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, sep=sep)
                except Exception:
                    pass

    text = raw_bytes.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")

uploaded = st.file_uploader("Upload applications file", type=["csv", "xlsx"])
if uploaded is None:
    st.info("💡 Select a preset in the sidebar, then upload your file.")
    st.stop()

df = read_uploaded_table(uploaded)
cols = df.columns.tolist()

st.subheader("🧭 Map your columns")

def pick(label, guess_key):
    guesses = {
        "Email": ["Email", "email", "E-mail", "EMAIL"],
        "Organisation": ["Organisation", "Organization", "organisation", "organization", "Organisation / SectorText"],
        "Sector": ["Sector", "sector", "SECTOR"],
        "MotivationText": ["MotivationText", "Motivation", "motivation"],
        "FunctionTitle": ["FunctionTitle", "Function", "function", "Position", "Job Title"],
        "WeeklyTimeBand": ["WeeklyTimeBand", "TimeBand", "Time", "Weekly Time"],
        "LanguageComfort": ["LanguageComfort", "Language", "language"],
        "RefereeConfirmsFit": ["RefereeConfirmsFit", "Referee", "referee", "Recommendation", "Recommender"],
        "AlumniReferral": ["AlumniReferral", "Alumni", "alumni", "Referral", "AlumniRef"],
        "ApplicationDate": ["ApplicationDate", "Date", "date", "Timestamp"],
    }

    guess_list = guesses.get(guess_key, [guess_key])
    default_idx = 0
    for g in guess_list:
        if g in cols:
            default_idx = cols.index(g) + 1
            break
    return st.selectbox(label, ["— none —"] + cols, index=default_idx)

email_col = pick("Email (optional; for PID generation)", "Email")
org_col   = pick("Organisation / SectorText (optional)", "Organisation")
sector_col= pick("Sector (structured; optional)", "Sector")
mot_col   = pick(f"MotivationText (min {min_mot_words} words)", "MotivationText")
func_col  = pick("FunctionTitle (optional)", "FunctionTitle")
time_col  = pick("WeeklyTimeBand (optional)", "WeeklyTimeBand")
lang_col  = pick("LanguageComfort (optional)", "LanguageComfort")
ref_col   = pick("RefereeConfirmsFit (optional)", "RefereeConfirmsFit")
alm_col   = pick("AlumniReferral (optional)", "AlumniReferral")
date_col  = pick("ApplicationDate (optional; for dedup)", "ApplicationDate")

if email_col == "— none —":
    st.warning("⚠️ No Email column mapped. Row-based PIDs will be generated.")
if ref_col == "— none —" and w_referee > 0:
    st.warning("⚠️ Referee column not mapped but weighted in your preset. Consider mapping it.")
if lang_col == "— none —" and w_lang > 15:
    st.warning("⚠️ Language column not mapped but heavily weighted in your preset.")

# ---------------------------
# Processing
# ---------------------------
work = df.copy()

# Remove PID if exists
if "PID" in work.columns:
    work.drop(columns=["PID"], inplace=True)

if email_col != "— none —" and email_col in work.columns:
    work["_original_email"] = work[email_col].copy()
    emails_norm = work[email_col].map(normalize_email)
    work.insert(0, "PID", emails_norm.apply(hash_id))
    work.drop(columns=[email_col], inplace=True)
else:
    work["_original_email"] = None
    work.insert(0, "PID", [hash_id(f"row_{i}") for i in range(len(work))])

# Deduplicate
if date_col != "— none —" and date_col in work.columns:
    work["_app_date"] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.sort_values("_app_date").drop_duplicates(subset=["PID"], keep="last")
else:
    work = work.drop_duplicates(subset=["PID"], keep="first")

# Resolve sector
if sector_col != "— none —" and sector_col in work.columns:
    work["_sector"] = work[sector_col].fillna("Other/Unclassified")
elif org_col != "— none —" and org_col in work.columns:
    work["_sector"] = work[org_col].apply(org_to_sector)
else:
    work["_sector"] = "Other/Unclassified"

# Motivation scores (adaptive minimum)
if mot_col != "— none —" and mot_col in work.columns:
    mot_scores = work[mot_col].apply(lambda x: rubric_heuristic_score(x, min_mot_words))
    work["_mot_specificity"] = mot_scores.apply(lambda t: t[0])
    work["_mot_feasibility"] = mot_scores.apply(lambda t: t[1])
    work["_mot_relevance"]   = mot_scores.apply(lambda t: t[2])
    work["_mot_total"] = work[["_mot_specificity", "_mot_feasibility", "_mot_relevance"]].sum(axis=1)
else:
    work["_mot_specificity"] = 0
    work["_mot_feasibility"] = 0
    work["_mot_relevance"] = 0
    work["_mot_total"] = 0

work["_mot_scaled"] = (work["_mot_total"] / 30.0) * w_motivation
work["_mot_scaled"] = work["_mot_scaled"].clip(lower=0, upper=w_motivation)

# Sector uplift
def sector_points(s):
    s_clean = s if s in sector_uplift else "Other/Unclassified"
    return sector_uplift.get(s_clean, 0)

work["_sector_points"] = work["_sector"].map(sector_points).clip(0, w_sector)

# Referee + Alumni
work["_ref_points"] = (
    work[ref_col].apply(lambda x: yes_no_points(x, w_referee))
    if ref_col != "— none —" and ref_col in work.columns else 0
)
work["_alm_points"] = (
    work[alm_col].apply(lambda x: yes_no_points(x, w_alumni))
    if alm_col != "— none —" and alm_col in work.columns else 0
)

# Function relevance
def function_points(x):
    if not isinstance(x, str):
        return 0
    xl = x.lower()
    # Finance-specific keywords
    finance_keywords = ["finance", "financial", "accountant", "economist", "banking", "treasury", "audit"]
    is_finance = any(k in xl for k in finance_keywords)
    
    direct = any(k in xl for k in ["lecturer", "extension", "analyst", "programme", "program officer", "policy", "teacher", "advisor", "manager", "director"])
    indirect = any(k in xl for k in ["assistant", "admin", "coordinator", "student", "intern"])
    
    if is_finance and preset_choice == "finance_optimized":
        return w_function  # Full points for finance roles in finance preset
    elif direct:
        return w_function
    elif indirect:
        return w_function * 0.5
    return 0

work["_func_points"] = (
    work[func_col].apply(function_points)
    if func_col != "— none —" and func_col in work.columns else 0
)

# Time & language
work["_time_band"] = (
    work[time_col].apply(lambda x: map_band(normalize_time_value(x), WEEKLY_TIME_BANDS))
    if time_col != "— none —" and time_col in work.columns else None
)
work["_lang_band"] = (
    work[lang_col].apply(lambda x: map_band(x, LANG_BANDS))
    if lang_col != "— none —" and lang_col in work.columns else None
)

work["_time_points"] = work["_time_band"].apply(get_time_points) if isinstance(work["_time_band"], pd.Series) else 0
work["_lang_points"] = work["_lang_band"].apply(lambda x: {"Fluent": w_lang, "Working": w_lang*0.6, "Basic/With support": w_lang*0.3}.get(x, 0)) if isinstance(work["_lang_band"], pd.Series) else 0

work["_time_points"] = np.minimum(work["_time_points"], w_time) if isinstance(work["_time_points"], pd.Series) else 0
work["_lang_points"] = np.minimum(work["_lang_points"], w_lang) if isinstance(work["_lang_points"], pd.Series) else 0

# Final RFS
rfs_cols = ["_mot_scaled", "_sector_points", "_ref_points", "_func_points", "_time_points", "_lang_points", "_alm_points"]
work["_RFS"] = work[rfs_cols].sum(axis=1).round(2)

work["_label"] = work.apply(
    lambda r: label_band(
        r["_RFS"],
        thr_admit,
        thr_priority,
        r["_sector"],
        equity_on,
        (equity_lower, equity_upper),
    ),
    axis=1
)

# Prepare output
out_cols = ["PID", "_sector", "_RFS", "_label", "_mot_scaled", "_sector_points", "_ref_points", "_func_points", "_time_points", "_lang_points", "_alm_points"]
pretty = work[out_cols].rename(columns={
    "_sector": "Sector",
    "_RFS": "RFS",
    "_label": "predicted Decision",
    "_mot_scaled": "MotivationPts",
    "_sector_points": "SectorPts",
    "_ref_points": "RefereePts",
    "_func_points": "FunctionPts",
    "_time_points": "TimePts",
    "_lang_points": "LanguagePts",
    "_alm_points": "AlumniPts"
})

# Score distribution stats
mean_rfs = pretty["RFS"].mean()
median_rfs = pretty["RFS"].median()
q25 = pretty["RFS"].quantile(0.25)
q75 = pretty["RFS"].quantile(0.75)

st.success(f"✅ Scored {len(pretty)} applicants using **{preset['name']}** preset | Mean RFS: {mean_rfs:.1f} | Median: {median_rfs:.1f}")

if mean_rfs < 30:
    st.warning(f"⚠️ Low average RFS ({mean_rfs:.1f}). Consider switching to **Lenient** preset for more admits.")
elif mean_rfs > 70:
    st.info(f"ℹ️ High average RFS ({mean_rfs:.1f}). Consider switching to **Strict** preset if you want higher bar.")

tab_score, tab_summary, tab_presets, tab_about = st.tabs(["📊 Scores", "📈 Summary", "🎯 Presets", "ℹ️ About"])

with tab_score:
    st.dataframe(pretty, use_container_width=True)
    
    # Score distribution
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean RFS", f"{mean_rfs:.1f}")
    with col2:
        st.metric("Median RFS", f"{median_rfs:.1f}")
    with col3:
        st.metric("25th percentile", f"{q25:.1f}")
    with col4:
        st.metric("75th percentile", f"{q75:.1f}")

    download_df = pretty.copy()
    if "_original_email" in work.columns:
        download_df.insert(1, "Email", work.loc[pretty.index, "_original_email"].values)

    csv_buf = io.StringIO()
    download_df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download scored CSV (with emails)",
        data=csv_buf.getvalue(),
        file_name=f"rfs_scored_v3_{preset_choice}.csv",
        mime="text/csv"
    )

with tab_summary:
    dec_counts = pretty["predicted Decision"].value_counts().to_dict()

    def pill(lbl, css_class):
        n = dec_counts.get(lbl, 0)
        pct = (n / len(pretty) * 100) if len(pretty) > 0 else 0
        st.markdown(f'<span class="tag {css_class}">{lbl}: {n} ({pct:.1f}%)</span>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: pill("Priority", "tag-priority")
    with c2: pill("Admit", "tag-admit")
    with c3: pill("Reserve (Equity)", "tag-equity")
    with c4: pill("Reserve", "tag-reserve")

    st.divider()
    st.subheader("By sector")
    by_sector = pretty.groupby(["Sector", "predicted Decision"]).size().to_frame("N").reset_index()
    pivot = by_sector.pivot(index="Sector", columns="predicted Decision", values="N").fillna(0).astype(int)
    st.dataframe(pivot, use_container_width=True)
    
    st.divider()
    st.subheader("Component breakdown")
    comp_means = {
        "Motivation": pretty["MotivationPts"].mean(),
        "Sector": pretty["SectorPts"].mean(),
        "Referee": pretty["RefereePts"].mean(),
        "Function": pretty["FunctionPts"].mean(),
        "Time": pretty["TimePts"].mean(),
        "Language": pretty["LanguagePts"].mean(),
        "Alumni": pretty["AlumniPts"].mean()
    }
    
    comp_df = pd.DataFrame(list(comp_means.items()), columns=["Component", "Mean Points"])
    comp_df["% of Max"] = [
        comp_means["Motivation"] / w_motivation * 100 if w_motivation > 0 else 0,
        comp_means["Sector"] / w_sector * 100 if w_sector > 0 else 0,
        comp_means["Referee"] / w_referee * 100 if w_referee > 0 else 0,
        comp_means["Function"] / w_function * 100 if w_function > 0 else 0,
        comp_means["Time"] / w_time * 100 if w_time > 0 else 0,
        comp_means["Language"] / w_lang * 100 if w_lang > 0 else 0,
        comp_means["Alumni"] / w_alumni * 100 if w_alumni > 0 else 0
    ]
    comp_df["% of Max"] = comp_df["% of Max"].round(1)
    st.dataframe(comp_df, use_container_width=True)

with tab_presets:
    st.subheader("🎯 Available Presets")
    
    for key, p in PRESETS.items():
        with st.expander(f"{p['name']}" + (" ← Current" if key == preset_choice else "")):
            st.markdown(f"**{p['description']}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Thresholds:**")
                st.markdown(f"- Admit: {p['thresh_admit']}")
                st.markdown(f"- Priority: {p['thresh_priority']}")
                st.markdown(f"- Min words: {p['min_motivation_words']}")
            
            with col2:
                st.markdown("**Weights:**")
                st.markdown(f"- Motivation: {p['w_motivation']}")
                st.markdown(f"- Referee: {p['w_referee']}")
                st.markdown(f"- Language: {p['w_lang']}")
                st.markdown(f"- Function: {p['w_function']}")

with tab_about:
    st.markdown(f"""
    **RFS v3 Adaptive - Smart Presets for Different Cohorts**
    
    **Current Preset:** {preset['name']}
    
    **What v3 Does Differently:**
    1. **Adaptive scoring** - Minimum word counts and weights adjust by preset
    2. **Cohort-specific** - Finance gets different rules than Main cohort
    3. **Evidence-based** - All presets validated against real data
    4. **Flexible** - Easy to switch between strict/lenient modes
    
    **About Your Selected Preset:**
    - {preset['description']}
    - Admit threshold: {thr_admit}
    - Priority threshold: {thr_priority}
    - Minimum motivation words: {min_mot_words}
    
    **Expected Results:**
    - **Balanced**: ~13% correlation (r=0.13), moderate precision
    - **Lenient**: More admits, lower bar, inclusive
    - **Strict**: Fewer admits, higher completion rates
    - **Finance-Optimized**: Best for finance/economics cohorts
    
    **Still Low Correlations?**
    If you're still getting r<0.15 after trying different presets, the issue is **what you're measuring**, not how you're weighing it. Consider adding:
    - Prior experience/education in the field
    - Organization reputation/type
    - Evidence of time availability (not just commitment)
    - Technical skills assessment
    - Employer support letter
    
    **Privacy:**
    - Screen: PIDs only (salted hashes)
    - Download: Includes emails for your records
    """)

st.caption(f"🎯 RFS v3 Adaptive - Using **{preset['name']}** preset | Privacy: PIDs on screen, emails in download")
