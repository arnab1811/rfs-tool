import io
import re
import base64
import hashlib
import urllib.request
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

# ---------------------------
# Solid palette (from your image)
# ---------------------------
PALETTE = {
    "accent": "#8B6C23",   # gold/brown
    "primary":  "#C02080",   # magenta
    "white":   "#FFFFFF",
    "ink":     "#1F2A33",
    "bg":      "#FFFFFF",
    "bg2":     "#F6F3EA",   # warm off-white (optional)
    "border":  "#E6E0D2"
}

# ---------------------------
# Logos (local + remote fallback)
# ---------------------------
APP_DIR = Path(__file__).resolve().parent
LOGO1_PATH = APP_DIR / "logo1.png"
LOGO2_PATH = APP_DIR / "logo2.png"

LOGO1_URL = "https://raw.githubusercontent.com/arnab1811/rfs-tool/main/logo1.png"
LOGO2_URL = "https://raw.githubusercontent.com/arnab1811/rfs-tool/main/logo2.png"

def ensure_logo(path: Path, url: str):
    """Download logo from GitHub raw if it doesn't exist locally."""
    if path.exists():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(path))
    except Exception:
        # If download fails, UI will still render without that logo.
        pass

def _img_to_b64(path: Path) -> str:
    try:
        b = path.read_bytes()
        return base64.b64encode(b).decode("utf-8")
    except Exception:
        return ""

ensure_logo(LOGO1_PATH, LOGO1_URL)
ensure_logo(LOGO2_PATH, LOGO2_URL)

LOGO1_B64 = _img_to_b64(LOGO1_PATH)
LOGO2_B64 = _img_to_b64(LOGO2_PATH)

# ---------------------------
# CSS injection (solid styling)
# ---------------------------
def inject_css():
    st.markdown(f"""
    <style>
      :root {{
        --primary: {PALETTE["primary"]};
        --accent:  {PALETTE["accent"]};
        --white:   {PALETTE["white"]};
        --ink:     {PALETTE["ink"]};
        --bg:      {PALETTE["bg"]};
        --bg2:     {PALETTE["bg2"]};
        --border:  {PALETTE["border"]};
      }}

      html, body, .stApp {{
        font-family: Verdana, Geneva, Arial, sans-serif;
        color: var(--ink);
        background: var(--bg);
        -webkit-font-smoothing: antialiased;
      }}

      /* Top banner */
      .app-banner {{
        padding: 16px 18px;
        border-radius: 14px;
        margin: 2px 0 22px 0;
        background: var(--primary);
        color: var(--white);
        border: 1px solid rgba(255,255,255,.15);
      }}
      .app-banner-inner {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
      }}
      .brand-left {{
        display: flex;
        align-items: center;
        gap: 14px;
        min-width: 0;
      }}
      .brand-title {{
        min-width: 0;
      }}
      .brand-title h2 {{
        margin: 0 0 4px 0;
        font-weight: 700;
        line-height: 1.15;
      }}
      .brand-title p {{
        margin: 0;
        opacity: .95;
      }}
      .logo-img {{
        height: 44px;
        width: auto;
        display: block;
        background: rgba(255,255,255,.06);
        padding: 6px 10px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,.18);
      }}
      .logo-right {{
        height: 44px;
        width: auto;
        display: block;
        background: rgba(255,255,255,.06);
        padding: 6px 10px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,.18);
      }}

      /* Version badge */
      .v3-badge {{
        display: inline-block;
        background: var(--accent);
        color: #fff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        margin-left: 8px;
        vertical-align: middle;
      }}

      /* Buttons */
      .stButton > button, .stDownloadButton button {{
        border-radius: 10px !important;
        border: 1px solid var(--primary) !important;
        background: var(--primary) !important;
        color: #fff !important;
        font-weight: 700 !important;
      }}
      .stButton > button:hover, .stDownloadButton button:hover {{
        border-color: var(--accent) !important;
        background: var(--accent) !important;
      }}

      /* Tags */
      .tag {{
        display:inline-block;
        padding:3px 10px;
        border-radius:999px;
        font-size:12px;
        font-weight:700;
        margin-right:8px;
        border: 1px solid transparent;
      }}
      .tag-priority {{ background: var(--accent); color:#fff; }}
      .tag-admit    {{ background: var(--primary); color:#fff; }}
      .tag-equity   {{ background: var(--bg2); color: var(--ink); border:1px solid var(--border); }}
      .tag-reserve  {{ background: #fff; color: var(--ink); border:1px solid var(--border); }}

      /* Make the sidebar header slightly cleaner */
      section[data-testid="stSidebar"] > div {{
        border-right: 1px solid var(--border);
      }}
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
        "w_motivation": 15,  # Reduced: too noisy
        "w_sector": 5,       # Reduced: negative correlation
        "w_referee": 28,     # Kept: strong predictor
        "w_function": 25,    # Increased: strongest predictor
        "w_time": 10,
        "w_lang": 15,        # Captures completion
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
    if any(k in x for k in ["universit", "school", "educat"]):
        return "Education"
    if any(k in x for k in ["ngo", "foundation", "association", "civil", "non profit", "non-profit"]):
        return "NGO/CSO"
    if any(k in x for k in ["ministry", "gov", "municipal", "department of", "bureau"]):
        return "Government"
    if any(k in x for k in ["united nations", "world bank", "fao", "ifad", "ifpri", "undp", "unesco"]):
        return "Multilateral"
    if any(k in x for k in ["ltd", "company", "bv", "inc", "plc", "gmbh", "sarl"]):
        return "Private"
    if any(k in x for k in ["farmer", "coop", "co-op", "cooperative"]):
        return "Farmer Org"
    if any(k in x for k in ["consult"]):
        return "Consultancy"
    if any(k in x for k in ["bank", "finance", "microfinance"]):
        return "Finance"
    return "Other/Unclassified"

def rubric_heuristic_score(text: str, min_words: int):
    if not isinstance(text, str) or text.strip() == "":
        return (0, 0, 0)
    t = text.strip()
    words = len(t.split())
    if words < min_words:
        return (0, 0, 0)

    has_data = any(w in t.lower() for w in ["data", "dataset", "dashboard", "faostat", "survey"])
    has_numbers = bool(re.search(r"\b\d+\b", t))
    has_fs = any(w in t.lower() for w in ["food system", "seed", "agric", "market", "value chain"])

    spec = 10 if (words >= 200 or has_data) else 5
    feas = 10 if has_numbers else 5
    rel = 10 if has_fs else 5
    return (spec, feas, rel)

def label_band(val, admit_thr, priority_thr, sector, equity_reserve, equity_range):
    if val >= priority_thr:
        return "Priority"
    if val >= admit_thr:
        return "Admit"
    if equity_reserve and sector == "Farmer Org" and equity_range[0] <= val <= equity_range[1]:
        return "Reserve (Equity)"
    return "Reserve"

def normalize_mapping_value(v):
    if not isinstance(v, str):
        return ""
    return v.strip().lower().replace("–", "-").replace("—", "-").replace("≥", ">=").replace(" ", "")

def get_time_points(x):
    if ">=3h" in x:
        return 10
    if "2-3h" in x:
        return 6
    if "1-2h" in x:
        return 3
    return 0

def yes_no_points(x, cap):
    if pd.isna(x):
        return 0
    text = str(x).strip().lower()
    if any(text.startswith(p) for p in ["no", "none", "n/a", "0"]):
        return 0
    return cap

# ---------------------------
# Sidebar & Main App
# ---------------------------
st.sidebar.header("⚙️ Configuration")
preset_choice = st.sidebar.radio(
    "Choose a preset:",
    options=list(PRESETS.keys()),
    format_func=lambda x: PRESETS[x]["name"]
)
preset = PRESETS[preset_choice]

# ---------------------------
# Banner with two logos (left + right)
# ---------------------------
logo1_html = ""
logo2_html = ""

if LOGO1_B64:
    logo1_html = f'<img class="logo-img" src="data:image/png;base64,{LOGO1_B64}" alt="Logo 1"/>'
if LOGO2_B64:
    logo2_html = f'<img class="logo-right" src="data:image/png;base64,{LOGO2_B64}" alt="Logo 2"/>'

st.markdown(f"""
<div class="app-banner">
  <div class="app-banner-inner">
    <div class="brand-left">
      {logo1_html}
      <div class="brand-title">
        <h2>Recruitment Fit Score (RFS) <span class="v3-badge">v3.1 OPTIMIZED</span></h2>
        <p>Validated against leaderboard success data (Function &amp; Referee priority)</p>
      </div>
    </div>
    <div>
      {logo2_html}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload applications file", type=["csv", "xlsx"])
if uploaded:
    # ---- Robust read ----
    name = uploaded.name.lower()
    if name.endswith(".xlsx"):
        df = pd.read_excel(uploaded, engine="openpyxl")
    else:
        df = pd.read_csv(uploaded, encoding_errors="ignore")

    # ---- Clean column names (prevents silly KeyErrors) ----
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]  # keep first if duplicate headers

    # ---- Column detection + mapping UI ----
    def _find_cols(patterns):
        hits = []
        for c in df.columns:
            cl = str(c).lower()
            if any(re.search(p, cl) for p in patterns):
                hits.append(c)
        return hits

    def pick_column(label, patterns, required=True):
        hits = _find_cols(patterns)
        options = ["— Select —"] + list(df.columns)
        default = hits[0] if hits else "— Select —"
        idx = options.index(default) if default in options else 0
        chosen = st.selectbox(label, options, index=idx)
        if chosen == "— Select —":
            if required:
                st.error(f"Missing required mapping for: {label}")
                st.stop()
            return None
        return chosen

    st.markdown("### Column mapping (auto-detected; change only if needed)")
    c1, c2 = st.columns(2)

    with c1:
        email_col = pick_column("Email", [r"\bemail\b", r"e-mail"])
        mot_col   = pick_column("Motivation text", [r"motivation", r"why.*apply", r"statement", r"interest", r"reason"])
        func_col  = pick_column("Function / job title", [r"\bfunction\b", r"job", r"title", r"position", r"role"])
        org_col   = pick_column("Organisation", [r"organi[sz]ation", r"employer", r"company", r"institution", r"affiliation"], required=False)

    with c2:
        ref_col   = pick_column("Referee / recommendation", [r"refere", r"reference", r"recommend", r"endorse"], required=False)
        lang_col  = pick_column("Language level", [r"language", r"english", r"fluency", r"proficien", r"comfort"], required=False)
        time_col  = pick_column("Weekly time commitment", [r"time", r"hours", r"weekly", r"commit"], required=False)
        alm_col   = pick_column("Alumni referral", [r"alumni", r"referral", r"referred", r"how.*hear"], required=False)

    # ---- More tolerant parsers ----
    def language_points(x):
        if pd.isna(x):
            return 0.0
        t = str(x).strip().lower()
        if any(k in t for k in ["fluent", "native", "advanced", "excellent"]):
            return float(preset["w_lang"])
        if any(k in t for k in ["working", "intermediate", "good", "professional"]):
            return float(preset["w_lang"]) * 0.6
        if any(k in t for k in ["basic", "limited", "beginner"]):
            return float(preset["w_lang"]) * 0.3
        return 0.0

    def time_points(x):
        if pd.isna(x):
            return 0.0
        t = str(x).strip().lower().replace("–", "-").replace("—", "-")
        if any(k in t for k in [">=3", "3+", "3 h", "3h", "more than 3", "at least 3"]):
            return 10.0
        if any(k in t for k in ["2-3", "2 to 3", "2.5", "2 h", "2h"]):
            return 6.0
        if any(k in t for k in ["1-2", "1 to 2", "1.5", "1 h", "1h"]):
            return 3.0
        if any(k in t for k in ["<1", "less than 1", "0.5", "30 min"]):
            return 0.0
        return 0.0

    def safe_text(x):
        return "" if pd.isna(x) else str(x)

    # ---- Build working frame ----
    work = df.copy()
    work["Email_Norm"] = work[email_col].apply(normalize_email)

    # Drop empty emails (otherwise everyone hashes to same PID)
    work = work[work["Email_Norm"] != ""].copy()
    if work.empty:
        st.error("No valid email values found after normalization. Check your Email column mapping.")
        st.stop()

    work.insert(0, "PID", work["Email_Norm"].apply(hash_id))

    # ---- Optional: de-duplicate repeated applicants by email ----
    ts_hits = _find_cols([r"timestamp", r"submitted", r"submission", r"date"])
    if ts_hits:
        ts_col = ts_hits[0]
        work["_ts"] = pd.to_datetime(work[ts_col], errors="coerce")
        work = work.sort_values("_ts").drop_duplicates("Email_Norm", keep="last").drop(columns=["_ts"])
    else:
        before = len(work)
        work = work.drop_duplicates("Email_Norm", keep="last")
        after = len(work)
        if after < before:
            st.info(f"De-duplicated {before-after} repeated emails (kept last entry per email).")

    # ---- 1) Motivation ----
    if mot_col:
        mot_scores = work[mot_col].apply(lambda x: rubric_heuristic_score(safe_text(x), preset["min_motivation_words"]))
        work["MotivationPts"] = (mot_scores.apply(sum) / 30.0) * preset["w_motivation"]
    else:
        work["MotivationPts"] = 0.0

    # ---- 2) Function ----
    def function_points(x):
        xl = safe_text(x).lower()
        if xl == "":
            return 0.0
        direct = any(k in xl for k in ["specialist", "officer", "advisor", "director", "manager", "analyst", "lecturer"])
        return float(preset["w_function"]) if direct else float(preset["w_function"]) * 0.4

    work["FunctionPts"] = work[func_col].apply(function_points)

    # ---- 3) Referee / Sector / Language / Time ----
    work["RefereePts"] = work[ref_col].apply(lambda x: yes_no_points(x, preset["w_referee"])) if ref_col else 0.0

    if org_col:
        work["Sector"] = work[org_col].apply(org_to_sector)
    else:
        work["Sector"] = "Other/Unclassified"

    work["SectorPts"] = work["Sector"].map(lambda s: preset["sector_uplift"].get(s, 0))

    work["LanguagePts"] = work[lang_col].apply(language_points) if lang_col else 0.0
    work["TimePts"]     = work[time_col].apply(time_points) if time_col else 0.0

    # ---- 4) Final scoring ----
    pts_cols = ["MotivationPts", "FunctionPts", "RefereePts", "SectorPts", "LanguagePts", "TimePts"]
    work["RFS"] = work[pts_cols].sum(axis=1).round(2)
    work["predicted Decision"] = work.apply(
        lambda r: label_band(
            r["RFS"],
            preset["thresh_admit"],
            preset["thresh_priority"],
            r["Sector"],
            True,
            (preset["equity_lower"], preset["equity_upper"])
        ),
        axis=1
    )

    st.success(f"✅ Scored {len(work)} applicants using **{preset['name']}** model")
    st.dataframe(work[["PID", "Email_Norm", "Sector", "RFS", "predicted Decision"] + pts_cols], use_container_width=True)

    csv = work.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download v3.1 Optimized CSV", csv, "rfs_v3_1_optimized.csv", "text/csv")
