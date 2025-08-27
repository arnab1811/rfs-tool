\# Recruitment Fit Score (RFS) – Streamlit App (Pseudonymized)



A transparent shortlisting helper for the NFP e-course. \*\*Forced pseudonymization\*\*: emails are immediately hashed with a secret salt and never shown or saved in outputs.



\## Features

\- RFS scoring using only application-time info

\- Sector uplift configurable

\- Heuristic motivation scorer (optional)

\- Equity reserve flag for Farmer Orgs

\- \*\*Emails replaced by `PID` (hashed) immediately\*\*

\- Optional hashing of additional identifier columns

\- Download of scored CSV (only pseudonyms + scores)



\## Run locally

```bash

\# PowerShell in rfs-tool/

python -m venv .venv

. .venv\\\\Scripts\\\\Activate.ps1

pip install -r requirements.txt

streamlit run app.py



