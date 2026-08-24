"""
AureXus — AI-Powered Resume Parser
Streamlit port. Deploy target: GitHub -> share.streamlit.io (free tier).

Config resolution order for every secret/setting: Streamlit Cloud's
st.secrets (Settings -> Secrets, TOML format) first, then a local .env
file (python-dotenv, for local dev), then an explicit default.
"""

import os
import json
import base64
from io import BytesIO
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv, set_key
from pypdf import PdfReader

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import gspread
from google.oauth2.service_account import Credentials


# ============================================================
# 1. Configuration
# ============================================================
load_dotenv()


def cfg(key: str, default: str = "") -> str:
    """Resolve a setting from st.secrets first (Streamlit Cloud), then
    the environment / local .env (local dev), then a default."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


OPENAI_API_KEY = cfg("OPENAI_API_KEY")
GOOGLE_SHEET_ID = cfg("GOOGLE_SHEET_ID", "1Ol24AVe2mmnokJWwDHSGmTf34iG82kxN56ZMl9HAxkQ")
GOOGLE_SERVICE_ACCOUNT_JSON = cfg("GOOGLE_SERVICE_ACCOUNT_JSON")

ADMIN_USERNAME = cfg("ADMIN_USERNAME")
ADMIN_PASSWORD = cfg("ADMIN_PASSWORD")
USER_USERNAME = cfg("USER_USERNAME")
USER_PASSWORD = cfg("USER_PASSWORD")

COMPANY_NAME = "AureXus"
COMPANY_LOGO_URL = "https://www.aurexus.com/wp-content/uploads/2021/10/1_80x50mm_adobe_illustratorwhite.png"
# Streamlit's HTML sanitizer strips inline onerror handlers, so a hotlinked
# image with a JS fallback isn't reliable here. Default to a monogram — swap
# COMPANY_LOGO_URL for a local asset (e.g. "logo.png" checked into the repo)
# if you want the real logo without depending on an external host.
USE_HOTLINKED_LOGO = False
MODEL_DISPLAY_NAME = "Vertical SLM Engine"

PRICE_INPUT_PER_1M = 0.15
PRICE_OUTPUT_PER_1M = 0.60

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SHEET_TABS = {"EN": "english", "FR": "french"}

USERS = {}
if ADMIN_USERNAME and ADMIN_PASSWORD:
    USERS[ADMIN_USERNAME] = {"password": ADMIN_PASSWORD, "role": "admin"}
if USER_USERNAME and USER_PASSWORD:
    USERS[USER_USERNAME] = {"password": USER_PASSWORD, "role": "user"}


st.set_page_config(
    page_title=f"{COMPANY_NAME} — AI Resume Parser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. Session State
# ============================================================
_DEFAULTS = {
    "authenticated": False,
    "username": None,
    "role": None,
    "parse_counter": 0,
    "sync_counter": 0,
    "session_cost_usd": 0.0,
    "recent_logs": [],
    "current_parsed": None,
    "current_filename": None,
    "current_file_bytes": None,
    "current_language": "EN",
    "parse_status": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Runtime-editable copies of secrets (admin can update these live from the
# Admin Settings tab without redeploying). Kept separate from the cfg()
# values above so a save doesn't require a rerun-order dance.
if "runtime_openai_key" not in st.session_state:
    st.session_state.runtime_openai_key = OPENAI_API_KEY
if "runtime_sheet_id" not in st.session_state:
    st.session_state.runtime_sheet_id = GOOGLE_SHEET_ID
if "runtime_service_json" not in st.session_state:
    st.session_state.runtime_service_json = GOOGLE_SERVICE_ACCOUNT_JSON


# ============================================================
# 3. Backend helpers (ported 1:1 from the Gradio version)
# ============================================================
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=st.session_state.runtime_openai_key)


def compute_call_cost(response) -> tuple:
    in_tok = out_tok = 0
    usage = getattr(response, "usage_metadata", None)
    if usage:
        in_tok = usage.get("input_tokens", 0) or 0
        out_tok = usage.get("output_tokens", 0) or 0
    else:
        meta = getattr(response, "response_metadata", {}) or {}
        tu = meta.get("token_usage", {}) or {}
        in_tok = tu.get("prompt_tokens", 0) or 0
        out_tok = tu.get("completion_tokens", 0) or 0
    cost = (in_tok / 1_000_000) * PRICE_INPUT_PER_1M + (out_tok / 1_000_000) * PRICE_OUTPUT_PER_1M
    return cost, in_tok, out_tok


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def get_sheets_client():
    raw = st.session_state.runtime_service_json
    if not raw or not raw.strip():
        raise FileNotFoundError(
            "❌ No Google credentials configured. Set GOOGLE_SERVICE_ACCOUNT_JSON in "
            "Streamlit secrets, or paste it in Admin Settings."
        )
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Google service-account JSON is not valid: {e}")
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def build_english_prompt(text):
    return f"""
You are an expert ATS (Applicant Tracking System) resume parser.

Extract the following fields from the resume text below and return ONLY a valid JSON object.

⚠️ Strict Rules:
- If any field is missing or cannot be determined, set its value as null
- Return ONLY valid JSON — no markdown, no code fences, no explanations
- Use camelCase field names exactly as listed below
- keywords and languages must be JSON arrays of strings
- educationDegrees and experience must be plain text summaries (not arrays)
- Infer seniorityLevel from total years of experience + role seniority in titles:
    • junior  = 0–2 years
    • mid     = 3–6 years
    • senior  = 7+ years
- Separate skills into domain tiers:
    • domain1PrimarySkill   = core professional identity / main expertise (e.g. "Data Engineering")
    • domain2SecondarySkill = next strongest complementary skill or experience
    • domain3OtherSkills    = brief summary of remaining / supporting skills
- keywords: extract specific skills, tools, certifications, frameworks, and industry terms as a flat array
- contractType: map to one of [Fixed-term contract, Permanent contract, Self-employed, Freelancer / Consultant] or null
- yearsOfExperience: numeric total; compute from date ranges if not explicitly stated
- age: numeric only if explicitly mentioned in the resume
- salaryExpectations: extract verbatim if present, else null
- teams: Microsoft Teams handle or any collaboration handle if present, else null

Fields to extract:
{{
  "firstName": string or null,
  "lastName": string or null,
  "targetPosition": string or null,
  "contractType": string or null,
  "domain1PrimarySkill": string or null,
  "domain2SecondarySkill": string or null,
  "domain3OtherSkills": string or null,
  "age": number or null,
  "seniorityLevel": "junior" | "mid" | "senior" | null,
  "yearsOfExperience": number or null,
  "keywords": [ array of strings ],
  "languages": [ array of strings ],
  "linkedin": string or null,
  "educationDegrees": string or null,
  "experience": string or null,
  "salaryExpectations": string or null,
  "email": string or null,
  "phoneNumber": string or null,
  "teams": string or null,
  "city": string or null,
  "regionCountry": string or null
}}

Resume Text:
{text}
"""


def build_french_prompt(text):
    return f"""
Tu es un expert en parsing de CV pour un Système de Suivi des Candidatures (ATS).

Extrais les champs suivants du texte de CV ci-dessous et retourne UNIQUEMENT un objet JSON valide.

⚠️ Règles strictes :
- Si un champ est manquant ou ne peut pas être déterminé, définis sa valeur comme null
- Retourne UNIQUEMENT du JSON valide — pas de markdown, pas de balises de code, pas d'explications
- Utilise exactement les noms de champs en camelCase listés ci-dessous
- keywords et languages doivent être des tableaux JSON de chaînes de caractères
- formation, diplomes et experience doivent être des résumés en texte brut (pas des tableaux)
- Déduis senioriteNiveau à partir des années d'expérience totales + niveau des titres de poste :
    • Junior    = 0–2 ans
    • Confirmé  = 3–6 ans
    • Senior    = 7+ ans
- Sépare les compétences en niveaux :
    • competence1Principale = identité professionnelle principale
    • competence2Secondaire = compétence complémentaire forte suivante
    • competence3Autres     = résumé des compétences secondaires/additionnelles
- keywords : extrais les compétences spécifiques, outils, certifications, frameworks, termes métier
- typeContrat : l'un de [CDD, CDI, Micro-entreprise, Freelance / Consultant] ou null
- anneesExperience : nombre total d'années; calcule à partir des plages de dates si non mentionné
- age : numérique uniquement si explicitement mentionné dans le CV
- pretentionsSalariales : extrais tel quel si présent, sinon null
- mobiliteGeo : mobilité géographique si mentionnée, sinon null
- departement : numéro ou nom du département français si mentionné, sinon null
- teams : handle Microsoft Teams ou outil de communication si présent, sinon null

Champs à extraire :
{{
  "prenom": string or null,
  "nom": string or null,
  "posteCible": string or null,
  "typeContrat": string or null,
  "competence1Principale": string or null,
  "competence2Secondaire": string or null,
  "competence3Autres": string or null,
  "senioriteNiveau": "Junior" | "Confirmé" | "Senior" | null,
  "anneesExperience": number or null,
  "keywords": [ tableau de chaînes ],
  "languages": [ tableau de chaînes ],
  "linkedin": string or null,
  "formation": string or null,
  "diplomes": string or null,
  "experience": string or null,
  "pretentionsSalariales": string or null,
  "email": string or null,
  "telephone": string or null,
  "teams": string or null,
  "ville": string or null,
  "departement": string or null,
  "mobiliteGeo": string or null,
  "age": number or null
}}

Texte du CV :
{text}
"""


def arr_to_text(v):
    if isinstance(v, list):
        return "\n".join(str(x) for x in v)
    return v or ""


def build_english_row(parsed: dict, cv_filename: str = "") -> list:
    today = datetime.today().strftime("%Y-%m-%d")
    row = [""] * 84
    row[1] = today
    row[3] = parsed.get("firstName") or ""
    row[4] = parsed.get("lastName") or ""
    row[5] = parsed.get("targetPosition") or ""
    row[6] = parsed.get("contractType") or ""
    row[7] = parsed.get("domain1PrimarySkill") or ""
    row[8] = parsed.get("domain2SecondarySkill") or ""
    row[9] = parsed.get("domain3OtherSkills") or ""
    row[10] = parsed.get("age") or ""
    row[11] = parsed.get("seniorityLevel") or ""
    row[12] = parsed.get("yearsOfExperience") or ""
    row[13] = arr_to_text(parsed.get("keywords"))
    row[14] = arr_to_text(parsed.get("languages"))
    row[15] = cv_filename
    row[16] = parsed.get("linkedin") or ""
    row[19] = parsed.get("educationDegrees") or ""
    row[20] = parsed.get("experience") or ""
    row[22] = parsed.get("salaryExpectations") or ""
    row[23] = parsed.get("email") or ""
    row[24] = parsed.get("phoneNumber") or ""
    row[25] = parsed.get("teams") or ""
    row[26] = parsed.get("city") or ""
    row[27] = parsed.get("regionCountry") or ""
    return row


def build_french_row(parsed: dict, cv_filename: str = "") -> list:
    today = datetime.today().strftime("%Y-%m-%d")
    row = [""] * 86
    row[1] = today
    row[3] = parsed.get("nom") or ""
    row[4] = parsed.get("prenom") or ""
    row[5] = parsed.get("posteCible") or ""
    row[6] = parsed.get("typeContrat") or ""
    row[7] = parsed.get("competence1Principale") or ""
    row[8] = parsed.get("competence2Secondaire") or ""
    row[9] = parsed.get("competence3Autres") or ""
    row[10] = parsed.get("senioriteNiveau") or ""
    row[11] = parsed.get("anneesExperience") or ""
    row[12] = arr_to_text(parsed.get("keywords"))
    row[13] = arr_to_text(parsed.get("languages"))
    row[14] = cv_filename
    row[15] = parsed.get("linkedin") or ""
    row[18] = parsed.get("formation") or ""
    row[19] = parsed.get("diplomes") or ""
    row[20] = parsed.get("experience") or ""
    row[22] = parsed.get("pretentionsSalariales") or ""
    row[23] = parsed.get("email") or ""
    row[24] = parsed.get("telephone") or ""
    row[25] = parsed.get("teams") or ""
    row[26] = parsed.get("ville") or ""
    row[27] = parsed.get("departement") or ""
    row[28] = parsed.get("mobiliteGeo") or ""
    row[29] = parsed.get("age") or ""
    return row


def process_resume(file_bytes: bytes, filename: str, language: str):
    """Runs extraction, updates session_state in place. Returns nothing —
    read results back from st.session_state after calling."""
    try:
        resume_text = extract_text_from_pdf(file_bytes)
        prompt = build_french_prompt(resume_text) if language == "FR" else build_english_prompt(resume_text)

        response = get_llm().invoke([HumanMessage(content=prompt)])
        raw_output = response.content.strip()

        call_cost, _in, _out = compute_call_cost(response)
        st.session_state.session_cost_usd += call_cost

        if raw_output.startswith("```"):
            parts = raw_output.split("```")
            cleaned = parts[1] if len(parts) > 1 else raw_output
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        else:
            cleaned = raw_output

        parsed_json = json.loads(cleaned)
        st.session_state.current_parsed = parsed_json
        st.session_state.current_filename = filename
        st.session_state.current_file_bytes = file_bytes
        st.session_state.current_language = language
        st.session_state.parse_counter += 1

        filled = sum(1 for v in parsed_json.values() if v not in (None, "", [], {}))
        total = len(parsed_json)
        tab_name = SHEET_TABS.get(language, "english")

        st.session_state.parse_status = (
            "ok",
            f"Resume Extracted Successfully via **AI-Powered** Analysis ({language})",
            f"Populated **{filled}/{total}** fields • Target Google Sheet Tab: **'{tab_name}'**",
        )

    except json.JSONDecodeError:
        st.session_state.current_parsed = None
        st.session_state.parse_status = ("warn", f"{MODEL_DISPLAY_NAME} returned non-JSON output.", raw_output[:300])
    except Exception as e:
        st.session_state.current_parsed = None
        st.session_state.parse_status = ("err", "Extraction Error", str(e))


def push_to_sheet():
    parsed = st.session_state.current_parsed
    language = st.session_state.current_language
    filename = st.session_state.current_filename or ""

    if not parsed:
        return ("err", "No valid extraction to push. Extract a resume first.", "")

    try:
        tab_name = SHEET_TABS.get(language, "english")
        if language == "FR":
            row = build_french_row(parsed, filename)
            candidate = f"{parsed.get('prenom', '')} {parsed.get('nom', '')}".strip() or "Inconnu"
        else:
            row = build_english_row(parsed, filename)
            candidate = f"{parsed.get('firstName', '')} {parsed.get('lastName', '')}".strip() or "Unknown"

        gc = get_sheets_client()
        sh = gc.open_by_key(st.session_state.runtime_sheet_id)

        try:
            ws = sh.worksheet(tab_name)
        except gspread.exceptions.WorksheetNotFound:
            return ("err", f"Tab '{tab_name}' not found in Google Sheet.", "")

        ws.append_row(row, value_input_option="USER_ENTERED")

        st.session_state.sync_counter += 1
        populated = len([x for x in row if x != ""])
        st.session_state.recent_logs.append({
            "candidate": candidate,
            "tab": ws.title,
            "fields": f"{populated}/{len(row)}",
            "time": datetime.today().strftime("%H:%M:%S"),
        })

        return ("ok", f"Successfully Synced Row to Google Sheet! ({language})",
                f"Candidate: **{candidate}** | Tab: **'{ws.title}'** | Fields: **{populated}/{len(row)}**")

    except FileNotFoundError as e:
        return ("err", str(e), "")
    except Exception as e:
        return ("err", f"Google Sheet Write Error: {e}", "")


def mask_secret(value: str, keep: int = 4) -> str:
    if not value:
        return "Not configured"
    value = value.strip()
    if len(value) <= keep:
        return "•" * len(value)
    return value[:keep] + "•" * max(4, len(value) - keep)


# ============================================================
# 4. Styling (config.toml handles native widget theming; this adds
#    polish for our own custom HTML blocks — cards, chips, badges)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.5rem; max-width: 1400px; }

.brand-row { display:flex; justify-content:space-between; align-items:center;
             padding-bottom:16px; margin-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.07); }
.brand-left { display:flex; align-items:center; gap:12px; }
.brand-left img { height:32px; }
.brand-left h1 { font-size:1.2rem; font-weight:700; margin:0; color:#f2f3f6; }
.brand-badge { font-size:0.65rem; font-weight:600; letter-spacing:0.05em; text-transform:uppercase;
                color:#9aa1b2; border:1px solid rgba(255,255,255,0.1); border-radius:5px; padding:2px 8px; margin-left:8px; }
.user-pill { display:flex; align-items:center; gap:8px; background:#12141c; border:1px solid rgba(255,255,255,0.07);
             padding:6px 14px; border-radius:9999px; font-size:0.82rem; color:#9aa1b2; }
.user-avatar { width:24px; height:24px; border-radius:50%; background:#5b4fd1; color:#fff; font-size:0.72rem;
               font-weight:700; display:flex; align-items:center; justify-content:center; }

.card { background:#12141c; border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:16px 20px; }
.card-flat { background:#0d0f15; border:1px solid rgba(255,255,255,0.07); border-radius:8px; padding:12px 14px; }
.box-label { font-size:0.68rem; color:#6b7280; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; display:block; margin-bottom:4px; }
.box-value { font-size:0.9rem; color:#f2f3f6; margin:0; }

.status-banner { padding:12px 16px; border-radius:8px; border-left:3px solid; background:#12141c; margin:10px 0; }
.status-ok { border-left-color:#22c55e; }
.status-err { border-left-color:#f0546c; }
.status-warn { border-left-color:#f0a93c; }
.status-title { font-weight:600; font-size:0.9rem; color:#f2f3f6; }
.status-desc { font-size:0.82rem; color:#9aa1b2; margin-top:3px; }

.tag-chip { display:inline-block; background:rgba(124,108,240,0.1); border:1px solid rgba(124,108,240,0.3);
            color:#c7d2fe; padding:3px 10px; border-radius:6px; font-size:0.78rem; margin:2px; }
.lang-chip { background:rgba(34,197,94,0.1); border-color:rgba(34,197,94,0.3); color:#bbf7d0; }

[data-testid="stMetric"] { background:#12141c; border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:14px 18px; }
.stButton>button { border-radius:8px; font-weight:600; white-space:nowrap; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 5. Auth gate
# ============================================================
def login_screen():
    logo_html = (f'<img src="{COMPANY_LOGO_URL}" style="height:40px; margin-bottom:12px;" />'
                 if USE_HOTLINKED_LOGO else
                 '<div style="width:48px;height:48px;border-radius:10px;background:#171a24;border:1px solid rgba(255,255,255,0.14);'
                 'display:flex;align-items:center;justify-content:center;font-weight:700;color:#9b7cf0;'
                 'font-size:1.1rem;margin:0 auto 12px auto;">AU</div>')
    st.markdown(f"""
    <div style="max-width:420px; margin:80px auto 0 auto; text-align:center;">
        {logo_html}
        <h2 style="color:#f2f3f6; margin-bottom:4px;">{COMPANY_NAME}</h2>
        <p style="color:#9aa1b2; font-size:0.9rem; margin-bottom:24px;"><strong>AI-Powered</strong> Resume Parser</p>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", width="stretch", type="primary")
            if submitted:
                account = USERS.get(username)
                if account and account["password"] == password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.role = account["role"]
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")


if not USERS:
    st.error("❌ No accounts configured. Set ADMIN_USERNAME / ADMIN_PASSWORD in secrets or .env.")
    st.stop()

if not OPENAI_API_KEY and not st.session_state.runtime_openai_key:
    st.warning("⚠️ OPENAI_API_KEY is not set. Extraction will fail until it's configured (Admin Settings, once logged in, or via secrets).")

if not st.session_state.authenticated:
    login_screen()
    st.stop()


# ============================================================
# 6. Top bar
# ============================================================
role = st.session_state.role
username = st.session_state.username

top_l, top_r = st.columns([3, 1])
with top_l:
    logo_html = (f'<img src="{COMPANY_LOGO_URL}" />' if USE_HOTLINKED_LOGO else
                 '<div style="width:32px;height:32px;border-radius:7px;background:#171a24;border:1px solid rgba(255,255,255,0.14);'
                 'display:flex;align-items:center;justify-content:center;font-weight:700;color:#9b7cf0;font-size:0.75rem;">AU</div>')
    st.markdown(f"""
    <div class="brand-row" style="border-bottom:none; margin-bottom:0; padding-bottom:0;">
        <div class="brand-left">
            {logo_html}
            <h1>{COMPANY_NAME} <span class="brand-badge">AI-Powered Resume Parser</span></h1>
        </div>
    </div>
    """, unsafe_allow_html=True)
with top_r:
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"""
        <div class="user-pill" style="justify-content:flex-end;">
            <span>{username} <span style="opacity:.6; text-transform:uppercase; font-size:0.68rem;">({role})</span></span>
            <div class="user-avatar">{username[0].upper()}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        if st.button("Logout", width="stretch"):
            for k in ("authenticated", "username", "role"):
                st.session_state[k] = _DEFAULTS[k]
            st.rerun()

st.markdown("<hr style='margin-top:10px; margin-bottom:20px; border-color:rgba(255,255,255,0.07);'>", unsafe_allow_html=True)


# ============================================================
# 7. Metrics row
# ============================================================
m1, m2, m3, m4 = st.columns(4)
avg_cost = (st.session_state.session_cost_usd / st.session_state.parse_counter) if st.session_state.parse_counter else 0.0
m1.metric("📄 Resumes Parsed", st.session_state.parse_counter)
m2.metric(f"💰 Session API Cost ({MODEL_DISPLAY_NAME})", f"${st.session_state.session_cost_usd:,.4f}", f"avg ${avg_cost:.4f}/resume")
m3.metric("📤 Synced Rows", st.session_state.sync_counter)
m4.metric("🌐 Supported Schemas", "2 (EN & FR)")

st.write("")


# ============================================================
# 8. Tabs (Admin Settings only exists for admins)
# ============================================================
tab_names = ["⚡ Resume Extractor", "📋 Sheet Sync & History", "🌐 Schema Guide", "🚪 Session"]
if role == "admin":
    tab_names.append("⚙️ Admin Settings")

tabs = st.tabs(tab_names)

# ---------- TAB: Resume Extractor ----------
with tabs[0]:
    left, right = st.columns([2, 1])

    with left:
        st.markdown("#### 📥 Resume Upload & **AI-Powered** Extraction")
        language_label = st.radio(
            "Target ATS Language & Google Sheet Tab",
            options=["EN — English", "FR — French"],
            horizontal=True,
        )
        language = "FR" if language_label.startswith("FR") else "EN"

        uploaded = st.file_uploader("Upload Candidate Resume (PDF)", type=["pdf"])

        if st.button("🚀 Extract & Parse Resume", type="primary", width="stretch", disabled=(uploaded is None)):
            with st.spinner(f"Running **AI-Powered** extraction via {MODEL_DISPLAY_NAME}..."):
                process_resume(uploaded.getvalue(), uploaded.name, language)

        if st.session_state.parse_status:
            kind, title, desc = st.session_state.parse_status
            st.markdown(f"""
            <div class="status-banner status-{kind}">
                <div class="status-title">{'✅' if kind=='ok' else '⚠️' if kind=='warn' else '❌'} {title}</div>
                <div class="status-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        sub_tabs = st.tabs(["📄 Preview", "📊 Profile", "📋 Data Grid", "💻 Raw JSON"])

        with sub_tabs[0]:
            if uploaded is not None:
                b64 = base64.b64encode(uploaded.getvalue()).decode()
                st.markdown(
                    f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="520" '
                    f'style="border:1px solid rgba(255,255,255,0.07); border-radius:8px;"></iframe>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Upload a PDF to preview it here.")

        with sub_tabs[1]:
            parsed = st.session_state.current_parsed
            if not parsed:
                st.info("Extract a resume to view the candidate profile.")
            else:
                is_fr = st.session_state.current_language == "FR"
                name = (f"{parsed.get('prenom','')} {parsed.get('nom','')}" if is_fr
                        else f"{parsed.get('firstName','')} {parsed.get('lastName','')}").strip() or "Unknown Candidate"
                title = (parsed.get("posteCible") if is_fr else parsed.get("targetPosition")) or "N/A"
                sen = (parsed.get("senioriteNiveau") if is_fr else parsed.get("seniorityLevel")) or "N/A"
                yoe = parsed.get("anneesExperience" if is_fr else "yearsOfExperience") or "N/A"
                email = parsed.get("email") or "N/A"
                phone = parsed.get("telephone" if is_fr else "phoneNumber") or "N/A"
                keywords = parsed.get("keywords") or []
                langs = parsed.get("languages") or []

                st.markdown(f"### {name}")
                st.caption(f"{title} • {sen} • {yoe} years experience")
                c1, c2 = st.columns(2)
                c1.markdown(f"<div class='card-flat'><span class='box-label'>Email</span><p class='box-value'>{email}</p></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='card-flat'><span class='box-label'>Phone</span><p class='box-value'>{phone}</p></div>", unsafe_allow_html=True)
                st.write("")
                st.markdown("**Keywords & Competencies**")
                st.markdown("".join(f"<span class='tag-chip'>{k}</span>" for k in keywords) or "_None extracted_", unsafe_allow_html=True)
                st.write("")
                st.markdown("**Languages**")
                st.markdown("".join(f"<span class='tag-chip lang-chip'>{l}</span>" for l in langs) or "_None extracted_", unsafe_allow_html=True)

        with sub_tabs[2]:
            parsed = st.session_state.current_parsed
            if not parsed:
                st.info("Extract a resume to view the ATS data grid.")
            else:
                st.dataframe(
                    [{"Field": k, "Value": (", ".join(v) if isinstance(v, list) else (v if v is not None else "—"))}
                     for k, v in parsed.items()],
                    width="stretch", hide_index=True,
                )

        with sub_tabs[3]:
            parsed = st.session_state.current_parsed
            if not parsed:
                st.info("Extract a resume to view raw JSON.")
            else:
                st.json(parsed)

    with right:
        st.markdown("#### 📤 Google Sheet Integration")
        if st.button("📤 Push Row to Google Sheet", width="stretch", disabled=(st.session_state.current_parsed is None)):
            kind, title, desc = push_to_sheet()
            st.session_state["_push_result"] = (kind, title, desc)

        if "_push_result" in st.session_state:
            kind, title, desc = st.session_state["_push_result"]
            st.markdown(f"""
            <div class="status-banner status-{kind}">
                <div class="status-title">{'✅' if kind=='ok' else '❌'} {title}</div>
                <div class="status-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.caption("Ready to sync candidate row to ATS Google Sheet.")

        st.markdown("#### ⏱️ Recent Session Sync Log")
        if not st.session_state.recent_logs:
            st.caption("No sync operations yet. This fills in after you extract a resume and push it.")
        else:
            for entry in reversed(st.session_state.recent_logs[-6:]):
                st.markdown(f"""
                <div class="card-flat" style="margin-bottom:8px;">
                    <strong>{entry['candidate']}</strong> <span style="float:right; color:#6b7280; font-size:0.75rem;">{entry['time']}</span><br/>
                    <span style="font-size:0.8rem; color:#9aa1b2;">Tab: '{entry['tab']}' • Fields: {entry['fields']}</span>
                </div>
                """, unsafe_allow_html=True)


# ---------- TAB: Sheet Sync & History ----------
with tabs[1]:
    st.markdown("#### 📊 Google Sheets Connection & Target Info")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='card-flat'><span class='box-label'>Spreadsheet ID</span><p class='box-value'>{st.session_state.runtime_sheet_id}</p></div>", unsafe_allow_html=True)
    c2.markdown("<div class='card-flat'><span class='box-label'>Active Sheet Tabs</span><p class='box-value'>'english' (84 cols) & 'french' (86 cols)</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card-flat'><span class='box-label'>Credentials</span><p class='box-value'>{'Configured' if st.session_state.runtime_service_json else 'Not configured'}</p></div>", unsafe_allow_html=True)


# ---------- TAB: Schema Guide ----------
with tabs[2]:
    schema_tabs = st.tabs(["EN — English Mapping (84 cols)", "FR — French Mapping (86 cols)"])
    with schema_tabs[0]:
        st.markdown("""
| # | ATS Field | Description | Source |
|---|-----------|-------------|--------|
| 1 | `candidateId` | Candidate ID | Manual |
| 2 | `date` | Date of entry | Auto (today) |
| 4 | `firstName` | First Name | Extracted |
| 5 | `lastName` | Last Name | Extracted |
| 6 | `targetPosition` | Target Position / Role | Extracted |
| 7 | `contractType` | Contract Type | Extracted |
| 8 | `domain1PrimarySkill` | Primary Expertise | Extracted |
| 9 | `domain2SecondarySkill` | Secondary Expertise | Extracted |
| 10 | `domain3OtherSkills` | Other Skills summary | Extracted |
| 11 | `age` | Age | Extracted |
| 12 | `seniorityLevel` | Seniority Level | Inferred |
| 13 | `yearsOfExperience` | Total Years of Exp | Computed |
| 14 | `keywords` | Skill Keywords | Extracted |
| 15 | `languages` | Languages Spoken | Extracted |
| 16 | `cv` | PDF Filename | Filename |
| 17 | `linkedin` | LinkedIn URL | Extracted |
| 20 | `educationDegrees` | Education summary | Extracted |
| 21 | `experience` | Work Experience summary | Extracted |
| 24 | `email` | Email Address | Extracted |
| 25 | `phoneNumber` | Mobile Number | Extracted |
| 27 | `city` | City | Extracted |
| 28 | `regionCountry` | Region / Country | Extracted |
        """)
    with schema_tabs[1]:
        st.markdown("""
| # | Colonne ATS | Description | Source |
|---|------------|-------------|--------|
| 1 | `ID_Candidat` | ID unique | Manuel |
| 2 | `Date` | Date d'entrée | Auto |
| 4 | `nom` | Nom de famille | Extrait |
| 5 | `prenom` | Prénom | Extrait |
| 6 | `posteCible` | Poste recherché | Extrait |
| 7 | `typeContrat` | Type Contrat | Extrait |
| 8 | `competence1Principale` | Compétence principale | Extrait |
| 9 | `competence2Secondaire` | Compétence secondaire | Extrait |
| 10 | `competence3Autres` | Autres compétences | Extrait |
| 11 | `senioriteNiveau` | Niveau de séniorité | Inféré |
| 12 | `anneesExperience` | Années d'expérience | Calculé |
| 13 | `keywords` | Mots clés | Extrait |
| 14 | `languages` | Langues parlées | Extrait |
| 19 | `formation` | Parcours formation | Extrait |
| 20 | `diplomes` | Diplômes obtenus | Extrait |
| 21 | `experience` | Expériences pro | Extrait |
| 24 | `email` | Adresse email | Extrait |
| 25 | `telephone` | Numéro téléphone | Extrait |
| 27 | `ville` | Ville | Extrait |
| 28 | `departement` | Département français | Extrait |
| 29 | `mobiliteGeo` | Mobilité géographique | Extrait |
| 30 | `age` | Âge | Extrait |
        """)


# ---------- TAB: Session ----------
with tabs[3]:
    st.markdown("#### 👤 Your Session")
    st.markdown(f"""
    <div class="card">
        <p style="color:#9aa1b2; font-size:0.88rem; margin:0;">
            You're signed in as <strong>{username}</strong> ({role}) to the <strong>AI-Powered</strong> resume parser.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    if st.button("Sign Out / Logout Session"):
        for k in ("authenticated", "username", "role"):
            st.session_state[k] = _DEFAULTS[k]
        st.rerun()


# ---------- TAB: Admin Settings (admin only — tab doesn't exist for others) ----------
if role == "admin":
    with tabs[4]:
        st.markdown("#### 🔑 Live Configuration")
        st.caption(
            "Changes apply immediately for this session. Best-effort written to a local .env "
            "(works for local dev). On Streamlit Community Cloud, use **Settings → Secrets** for "
            "anything that must survive a redeploy — this form updates the running app only."
        )

        with st.form("admin_settings_form"):
            new_api_key = st.text_input(
                "AI Engine API Key", type="password",
                placeholder=f"Currently: {mask_secret(st.session_state.runtime_openai_key)} — leave blank to keep",
            )
            new_sheet_id = st.text_input(
                "Google Sheet ID",
                placeholder=f"Currently: {st.session_state.runtime_sheet_id}",
            )
            new_service_json = st.text_area(
                "Google Service-Account Credentials (paste full JSON)",
                placeholder="Currently: " + ("configured" if st.session_state.runtime_service_json else "not configured") + " — leave blank to keep",
                height=160,
            )
            save = st.form_submit_button("💾 Save Configuration", type="primary")

        if save:
            updated = []
            error = None
            if new_service_json.strip():
                try:
                    json.loads(new_service_json)
                    st.session_state.runtime_service_json = new_service_json.strip()
                    updated.append("Google service-account credentials")
                except json.JSONDecodeError:
                    error = "❌ Service-account credentials are not valid JSON."
            if not error and new_sheet_id.strip():
                st.session_state.runtime_sheet_id = new_sheet_id.strip()
                updated.append("Google Sheet ID")
            if not error and new_api_key.strip():
                st.session_state.runtime_openai_key = new_api_key.strip()
                updated.append("AI engine API key")

            if error:
                st.error(error)
            elif not updated:
                st.warning("⚠️ No changes submitted — all fields were left blank.")
            else:
                try:
                    for k, v in [("OPENAI_API_KEY", st.session_state.runtime_openai_key),
                                 ("GOOGLE_SHEET_ID", st.session_state.runtime_sheet_id),
                                 ("GOOGLE_SERVICE_ACCOUNT_JSON", st.session_state.runtime_service_json)]:
                        if v:
                            set_key(".env", k, v)
                except Exception:
                    pass
                st.success(f"✅ Updated: {', '.join(updated)}")

        st.markdown("---")
        st.markdown("#### 👥 Registered Accounts")
        c1, c2 = st.columns(2)
        c1.markdown(f"<div class='card-flat'><span class='box-label'>Admin Account</span><p class='box-value'>Username: '{ADMIN_USERNAME}'</p></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card-flat'><span class='box-label'>General User Account</span><p class='box-value'>{('Username: ' + USER_USERNAME) if USER_USERNAME else 'Not configured'}</p></div>", unsafe_allow_html=True)
