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
    page_title="Recruitment Fit Score (RFS) – Pseudonymized",
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
# Local logos (you added these to repo root)
# ---------------------------
APP_DIR = Path(__file__).resolve().parent
LOGO1_PATH = APP_DIR / "logo1.png"
LOGO2_PATH = APP_DIR / "logo2.png"

def _img_to_b64(path: Path) -> str:
    """
    Read a local image and return base64 string.
    Using base64 avoids hotlinking issues and works on Streamlit Cloud.
    """
    try:
        b = path.read_bytes()
        return base64.b64encode(b).decode("utf-8")
    except Exception:
        return ""

LOGO1_B64 = _img_to_b64(LOGO1_PATH)
LOGO2_B64 = _img_to_b64(LOGO2_PATH)

def inject_css():
    # NOTE: no f-string here -> normal CSS braces are safe.
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

      /* Restore Material Icons ligature fonts so arrows/chevrons render as icons */
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

      /* Banner */
      .app-banner {
        padding: 18px 22px; border-radius: 14px; margin: 2px 0 24px 0;
        background: var(--green); color: #fff;
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
# Defaults / constants
# ---------------------------
DEFAULT_THRESH_ADMIT = 60
DEFAULT_THRESH_PRIORITY = 70
DEFAULT_EQUITY_LOWER = 50
DEFAULT_EQUITY_UPPER = 59

SECTOR_UPLIFT_DEFAULT = {
    "Education": 20,
    "NGO/CSO": 15,
    "Government": 10,
    "Multilateral": 10,
    "Private": 10,
    "Farmer Org": 0,
    "Consultancy": 10,
    "Finance": 10,
    "Other/Unclassified": 0,
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

def rubric_heuristic_score(text: str, length_targets=(200, 300)):
    if not isinstance(text, str) or text.strip() == "":
        return (0, 0, 0)

    t = text.strip()
    words = len(t.split())

    # tighten: very short motivations get zero (cuts noise/gaming)
    if words < 30:
        return (0, 0, 0)

    has_numbers = bool(re.search(r"\b\d+\b", t))
    has_when = any(w in t.lower() for w in ["week", "month", "timeline", "plan", "schedule"])
    has_where = any(w in t.lower() for w in ["district", "province", "country", "region", "university", "ministry"])
    has_data = any(w in t.lower() for w in ["data", "dataset", "dashboard", "faostat", "survey", "indicator"])
    has_role = any(w in t.lower() for w in ["lecturer", "extension", "officer", "analyst", "programme", "policy"])

    spec = 0
    if words >= length_targets[0]: spec += 5
    if words >= length_targets[1]: spec += 2
    if has_where or has_data: spec += 2
    if has_role: spec += 1
    spec = min(spec, 10)

    feas = 0
    if has_numbers: feas += 3
    if has_when: feas += 4
    if "pilot" in t.lower() or "test" in t.lower(): feas += 2
    if "5–6 weeks" in t.lower() or "5-6 weeks" in t.lower(): feas += 1
    feas = min(feas, 10)

    rel = 0
    if any(k in t.lower() for k in ["food system", "seed", "agric", "market", "value chain", "policy"]): rel += 4
    if has_where: rel += 3
    if has_data: rel += 2
    if "student" in t.lower() or "farmer" in t.lower(): rel += 1
    rel = min(rel, 10)

    return (spec, feas, rel)

def label_band(val, admit_thr, priority_thr, sector, equity_reserve=False, equity_range=(50, 59)):
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
    """
    Treat any non-empty free text as YES unless it clearly indicates NO.
    Prevents silent zeros when fields contain a recommender/referral name/email.
    """
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

st.sidebar.subheader("Thresholds")
thr_admit = st.sidebar.number_input("Admit threshold", min_value=0, max_value=100, value=DEFAULT_THRESH_ADMIT, step=1)
thr_priority = st.sidebar.number_input("Priority threshold", min_value=0, max_value=100, value=DEFAULT_THRESH_PRIORITY, step=1)
equity_on = st.sidebar.checkbox("Enable Equity Reserve for Farmer Orgs (50–59)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Weights")
w_motivation = st.sidebar.slider("Motivation rubric (max)", 0, 40, 30)
w_sector = st.sidebar.slider("Sector uplift (max)", 0, 30, 20)
w_referee = st.sidebar.slider("Referee / recommendation (max)", 0, 20, 10)
w_function = st.sidebar.slider("Function relevance (max)", 0, 20, 10)
w_time = st.sidebar.slider("Weekly time (max)", 0, 20, 10)
w_lang = st.sidebar.slider("Language comfort (max)", 0, 10, 5)
w_alumni = st.sidebar.slider("Alumni/referral bonus (max)", 0, 10, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("Sector uplift values")
sector_uplift = {}
for k, v in SECTOR_UPLIFT_DEFAULT.items():
    sector_uplift[k] = st.sidebar.number_input(f"{k}", value=v, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Privacy: PIDs shown on screen. Downloaded CSV includes emails for local use.")

# ---------------------------
# Banner with local logos
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
          <h2>Recruitment Fit Score (RFS)</h2>
          <p>Pseudonymized shortlisting for fair, transparent intake – Verdana UI & NFP palette.</p>
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
# Main – Upload & mapping
# ---------------------------
st.title("Recruitment Fit Score (RFS) – Pseudonymized Reviewer Tool")
st.write("Upload an **applications CSV** (UTF-8) or **XLSX**. Emails are immediately replaced with `PID` (hashed).")

with st.expander("📄 Expected columns (you can map after upload)"):
    st.markdown("""
**All fields are optional.** Map what you have:

**Recommended:**
- **Email** (unique applicant ID; will be hashed to PID)
- **Organisation / SectorText** (text; used to infer sector)
- **MotivationText** (free text)

**Optional:**
- **Sector** (structured dropdown if available)
- **FunctionTitle**
- **WeeklyTimeBand** (`<1h`, `1-2h`, `2-3h`, `>=3h`)
- **LanguageComfort** (`Basic/With support`, `Working`, `Fluent`)
- **RefereeConfirmsFit** (`yes`/`no` OR free text) — free text counts as "yes" unless explicitly "no"
- **AlumniReferral** (`yes`/`no` OR free text) — free text counts as "yes" unless explicitly "no"
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
    st.info("Tip: test with the sample file in the repo.")
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
mot_col   = pick("MotivationText (optional)", "MotivationText")
func_col  = pick("FunctionTitle (optional)", "FunctionTitle")
time_col  = pick("WeeklyTimeBand (optional)", "WeeklyTimeBand")
lang_col  = pick("LanguageComfort (optional)", "LanguageComfort")
ref_col   = pick("RefereeConfirmsFit (yes/no or free text; optional)", "RefereeConfirmsFit")
alm_col   = pick("AlumniReferral (yes/no or free text; optional)", "AlumniReferral")
date_col  = pick("ApplicationDate (optional; for dedup)", "ApplicationDate")

if email_col == "— none —":
    st.warning("⚠️ No Email column mapped. Row-based PIDs will be generated (less reliable for deduplication).")
if mot_col == "— none —":
    st.warning("⚠️ No MotivationText column mapped. Motivation scores will be 0 for all applicants.")
if org_col == "— none —":
    st.warning("⚠️ No Organisation/SectorText column mapped. Sector will default to 'Other/Unclassified' for all applicants.")

# ---------------------------
# Force pseudonymization - keep original for download
# ---------------------------
work = df.copy()

if email_col != "— none —" and email_col in work.columns:
    work["_original_email"] = work[email_col].copy()
    emails_norm = work[email_col].map(normalize_email)
    work.insert(0, "PID", emails_norm.apply(hash_id))
    work.drop(columns=[email_col], inplace=True)
else:
    work["_original_email"] = None
    work.insert(0, "PID", [hash_id(f"row_{i}") for i in range(len(work))])

with st.expander("🔐 Optional: hash additional identifier columns"):
    available_cols = [c for c in cols if c != email_col and c in work.columns and not c.startswith("_")]
    ident_cols = st.multiselect("Select any additional columns to hash (will be replaced by HASH_<col>)", available_cols)
    for c in ident_cols:
        work[f"HASH_{c}"] = work[c].astype(str).apply(lambda v: hash_id(v))
        drop_original = st.checkbox(f"Drop original '{c}' after hashing", value=True, key=f"drop_{c}")
        if drop_original and c in work.columns:
            work.drop(columns=[c], inplace=True)

# ---------------------------
# Deduplicate by PID
# ---------------------------
if date_col != "— none —" and date_col in work.columns:
    work["_app_date"] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.sort_values("_app_date").drop_duplicates(subset=["PID"], keep="last")
else:
    work = work.drop_duplicates(subset=["PID"], keep="first")

# ---------------------------
# Resolve sector
# ---------------------------
if sector_col != "— none —" and sector_col in work.columns:
    work["_sector"] = work[sector_col].fillna("Other/Unclassified")
elif org_col != "— none —" and org_col in work.columns:
    work["_sector"] = work[org_col].apply(org_to_sector)
else:
    work["_sector"] = "Other/Unclassified"

# ---------------------------
# Motivation scores
# ---------------------------
if mot_col != "— none —" and mot_col in work.columns:
    mot_scores = work[mot_col].apply(rubric_heuristic_score)
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

# Referee + Alumni (robust)
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
    direct = any(k in xl for k in ["lecturer", "extension", "analyst", "programme", "program officer", "policy", "teacher", "advisor"])
    indirect = any(k in xl for k in ["assistant", "admin", "coordinator", "student", "intern"])
    if direct:
        return w_function
    if indirect:
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
work["_lang_points"] = work["_lang_band"].apply(lambda x: {"Fluent": 5, "Working": 3}.get(x, 0)) if isinstance(work["_lang_band"], pd.Series) else 0

work["_time_points"] = np.minimum(work["_time_points"], w_time) if isinstance(work["_time_points"], pd.Series) else 0
work["_lang_points"] = np.minimum(work["_lang_points"], w_lang) if isinstance(work["_lang_points"], pd.Series) else 0

# Final RFS
rfs_cols = ["_mot_scaled", "_sector_points", "_ref_points", "_func_points", "_time_points", "_lang_points", "_alm_points"]
work["_RFS"] = work[rfs_cols].sum(axis=1).round(2)

work["_label"] = work.apply(
    lambda r: label_band(
        r["_RFS"],
        DEFAULT_THRESH_ADMIT if pd.isna(thr_admit) else thr_admit,
        DEFAULT_THRESH_PRIORITY if pd.isna(thr_priority) else thr_priority,
        r["_sector"],
        equity_reserve=equity_on,
        equity_range=(DEFAULT_EQUITY_LOWER, DEFAULT_EQUITY_UPPER),
    ),
    axis=1
)

# Prepare output
out_cols = ["PID", "_sector", "_RFS", "_label", "_mot_scaled", "_sector_points", "_ref_points", "_func_points", "_time_points", "_lang_points", "_alm_points"]
pretty = work[out_cols].rename(columns={
    "_sector": "Sector",
    "_RFS": "RFS",
    "_label": "Decision",
    "_mot_scaled": "MotivationPts",
    "_sector_points": "SectorPts",
    "_ref_points": "RefereePts",
    "_func_points": "FunctionPts",
    "_time_points": "TimePts",
    "_lang_points": "LanguagePts",
    "_alm_points": "AlumniPts"
})

hash_cols = [c for c in work.columns if c.startswith("HASH_")]
pretty = pd.concat([pretty, work[hash_cols]], axis=1)

st.success(f"Scored {len(pretty)} applicants. Screen shows PIDs; download includes emails.")
tab_score, tab_summary, tab_about = st.tabs(["📊 Score", "📈 Summary", "ℹ️ About"])

with tab_score:
    st.dataframe(pretty, use_container_width=True)

    with st.expander("🧪 Diagnostics (value counts)"):
        for c in [ref_col, alm_col, time_col, lang_col]:
            if c != "— none —" and c in work.columns:
                st.write(f"**{c}**")
                st.write(work[c].astype(str).str.strip().str.lower().value_counts(dropna=False))

    download_df = pretty.copy()
    if "_original_email" in work.columns:
        download_df.insert(1, "Email", work.loc[pretty.index, "_original_email"].values)

    csv_buf = io.StringIO()
    download_df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download scored CSV (with emails)",
        data=csv_buf.getvalue(),
        file_name="rfs_scored.csv",
        mime="text/csv"
    )

with tab_summary:
    dec_counts = pretty["Decision"].value_counts().to_dict()

    def pill(lbl, css_class):
        n = dec_counts.get(lbl, 0)
        st.markdown(f'<span class="tag {css_class}">{lbl}: {n}</span>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: pill("Priority", "tag-priority")
    with c2: pill("Admit", "tag-admit")
    with c3: pill("Reserve (Equity)", "tag-equity")
    with c4: pill("Reserve", "tag-reserve")

    st.divider()
    st.subheader("By sector")
    by_sector = pretty.groupby(["Sector", "Decision"]).size().to_frame("N").reset_index()
    st.dataframe(by_sector, use_container_width=True)

with tab_about:
    st.markdown("""
    **What this tool does**  
    - Scores applicants **only** on application-time fields.  
    - Displays PIDs on screen (salted hash) for privacy during review.
    - Downloaded CSV includes original emails for local record-keeping.
    - Uses robust handling for referee/referral fields: free text counts as "yes" unless explicitly "no".

    **Privacy**  
    - Screen display: PIDs only (salted hashes)
    - Downloaded CSV: Includes original emails
    - Configure SALT in `.streamlit/secrets.toml` (local) or Streamlit Cloud settings.
    """)

st.caption("Privacy: PIDs shown on screen (salted hashes). Downloaded CSV includes original emails for local use.")
