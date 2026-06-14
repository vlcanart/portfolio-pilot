# Deploying to Streamlit Community Cloud (password-protected)

Your holdings and API key live in **Streamlit secrets**, not in the repo — so even a
public GitHub repo never contains your financial data. The dashboard itself is gated
behind a password.

## 1. Push the code to GitHub

A private repo is recommended (belt-and-suspenders, though no secrets are committed).

```powershell
cd portfolio-pilot
git init
git add .
git commit -m "Portfolio Pilot dashboard"
gh repo create portfolio-pilot --private --source . --push
```

`.gitignore` already excludes `.env`, `data/holdings.csv`, and `.streamlit/secrets.toml`,
so none of your real data is pushed.

## 2. Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. **New app** → pick the `portfolio-pilot` repo, branch `main`.
3. **Main file path:** `src/app.py`
4. Click **Advanced settings → Secrets** and paste the contents of
   `.streamlit/secrets.toml.example`, filling in:
   - `app_password` — your dashboard password
   - `ANTHROPIC_API_KEY` — only if you want the AI brief (optional)
   - `holdings_csv` — your portfolio (already pre-filled with your current positions)
5. **Deploy.**

First build takes a few minutes (it installs pandas, PyPortfolioOpt, etc.).

## 3. Open it

You'll get a URL like `https://<your-app>.streamlit.app`. It will ask for the password
before showing anything.

## Updating your holdings later
Edit the `holdings_csv` value in **App → Settings → Secrets** and save — the app reloads.
No code change or redeploy needed.

## Notes on privacy
- The app URL is reachable by anyone who has the link; the **password gate** is what
  protects the data. Use a strong password.
- Streamlit Community Cloud also offers viewer restriction by email under the app's
  settings if you want a second layer (requires viewers to have Streamlit accounts).
- To run locally with the gate, save a real `.streamlit/secrets.toml` (gitignored) from
  the example file.
