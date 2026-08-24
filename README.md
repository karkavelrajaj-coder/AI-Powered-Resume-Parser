# AureXus — AI-Powered Resume Parser (Streamlit)

A Streamlit port of the AureXus resume-parsing tool: upload a PDF resume,
extract structured candidate data via an AI engine, review it, and sync it
into a Google Sheet (English or French schema). Two account roles: **admin**
(full access + live configuration panel) and **general user** (extraction +
sync only).

---

## 1. Push this to GitHub

```bash
git init
git add .
git commit -m "AureXus resume parser (Streamlit)"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

`.gitignore` already excludes `.env`, `.streamlit/secrets.toml`, and
`service_account.json` — **never commit real credentials**. Only
`.streamlit/secrets.toml.example` (a template, no real values) should be
in the repo.

---

## 2. Deploy on Streamlit Community Cloud (free)

1. Go to **share.streamlit.io** and sign in with GitHub.
2. Click **New app**, pick your repo/branch, and set the main file path to
   `app.py`.
3. Before (or right after) the first deploy, open **Settings → Secrets** on
   the app and paste in TOML like this:

   ```toml
   OPENAI_API_KEY = "sk-your-new-key"

   GOOGLE_SHEET_ID = "1Ol24AVe2mmnokJWwDHSGmTf34iG82kxN56ZMl9HAxkQ"

   GOOGLE_SERVICE_ACCOUNT_JSON = '''
   {"type": "service_account", "project_id": "...", "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n", "client_email": "...", ...}
   '''

   ADMIN_USERNAME = "admin"
   ADMIN_PASSWORD = "choose-a-strong-password"

   USER_USERNAME = "staffuser"
   USER_PASSWORD = "choose-a-different-strong-password"
   ```

   Paste your service-account JSON exactly as-is (the `\n` sequences inside
   `private_key` are already correctly escaped from the original file — don't
   reformat them).

4. Save. The app redeploys automatically and picks up the secrets.

That's it — no `service_account.json` file, no `.env` file, nothing sensitive
ever touches the git repo.

---

## 3. Local development

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml with real values (it's gitignored)

streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## 4. What's different from the original Gradio version

- **Theming**: a native `.streamlit/config.toml` dark-purple theme replaces
  all the custom CSS overrides the Gradio version needed — every widget
  (inputs, buttons, tabs, the Raw JSON viewer) is themed consistently and
  automatically.
- **Auth**: simple session-state-based login (no external auth package).
  Admin and general-user accounts are both driven by Secrets — add or
  remove the `USER_USERNAME` / `USER_PASSWORD` pair to enable/disable the
  general account.
- **Admin Settings tab**: only rendered at all for the admin account (not
  just hidden) — the tab object doesn't exist in a general user's session.
  Lets the admin update the AI engine key, Sheet ID, and service-account
  JSON live, without a redeploy. This is a runtime-only change by default;
  it's also best-effort written to a local `.env`, which helps for local
  dev but **won't persist across a Streamlit Cloud redeploy** — use the
  Secrets panel for anything that must survive that.
- **PDF preview**: rendered inline via a base64 data-URI iframe (no file
  server / temp-path plumbing needed, unlike the Gradio version).
- **Raw JSON**: uses `st.json()`, which just works — no CodeMirror
  dark-mode theming issues to fight.

---

## 5. Known limitations

- Session state (parse counter, cost tracker, sync log) is per-browser-tab
  and resets on refresh — this is normal Streamlit behavior, not a bug.
- The AureXus logo defaults to a text monogram ("AU") rather than the
  hotlinked company logo, since relying on an externally-hosted image plus
  a JS fallback isn't reliable inside Streamlit's sanitized HTML rendering.
  To use the real logo: download it, commit it to the repo (e.g. as
  `assets/logo.png`), set `USE_HOTLINKED_LOGO = True` near the top of
  `app.py`, and point `COMPANY_LOGO_URL` at the local path.
- Streamlit Community Cloud's free tier sleeps after inactivity and has an
  ephemeral filesystem — anything written by the Admin Settings "save"
  (the best-effort `.env` write) is lost on redeploy/restart. Use Secrets
  for persistence.
