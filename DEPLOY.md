# Putting the system online

**Vercel cannot run this app.** Vercel hosts websites and short serverless functions; Streamlit needs a
server that keeps running. Use the free combination below:

* **Neon** (neon.tech) — the online PostgreSQL database that stores every institution's data.
* **Streamlit Community Cloud** (share.streamlit.io) — runs the app from your GitHub repo and gives you
  a public link such as `https://drspecimen-analytics.streamlit.app`.

Menu names below may differ slightly as these services update their sites.

## 1. Create the database (Neon)

1. Sign up at https://neon.tech (GitHub sign-in works).
2. Create a project, e.g. `learning-analytics`. Pick the region closest to your users.
3. Open **Connect** / **Connection details** and copy the connection string. It looks like
   `postgresql://user:password@ep-xxxx.region.aws.neon.tech/neondb?sslmode=require`
4. Keep it secret — anyone with it can read every institution's data.

The app creates all tables itself the first time it connects.

## 2. Push the code to GitHub

```bash
git add -A
git commit -m "Multi-institution learning analytics platform"
git push
```
Check that `app.db` and `.streamlit/secrets.toml` are NOT in the repo (`.gitignore` excludes them).

## 3. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Repository: `drspecimen/learning-analytics-system`, branch `main`, main file `app.py`.
4. Choose a custom subdomain if offered (e.g. `drspecimen-analytics`).
5. Open **Advanced settings**:
   * Python version: **3.12**
   * Secrets — paste (with your real values):
     ```toml
     DATABASE_URL = "postgresql://...your Neon string...?sslmode=require"
     PLATFORM_ADMIN_EMAIL = "you@drspecimen.online"
     ```
6. **Deploy**. The first build takes a few minutes.

## 4. First use

1. Open your link → **Register your institution**. Register DIG itself first, using the same email you
   put in `PLATFORM_ADMIN_EMAIL` — that account gets the **Platform** page.
2. Share the link with institutions. Each one registers itself and gets its own private workspace.

## Updating the app

Push to GitHub (`git push`). Streamlit Cloud redeploys automatically. Data in Neon is untouched.

## Things to know

* **Sleeping:** free Streamlit apps sleep after a period without visitors; the next visitor wakes it
  (takes a minute). Data is safe in Neon.
* **Install on devices:** in Chrome/Edge open the link → menu → *Install app* / *Add to Home screen*.
  It is a shortcut to the web app and needs internet.
* **Limits:** check Neon's current free-tier storage. A demo workspace is roughly 70,000 rows; a real
  institution grows with its data. Upgrade the Neon plan when needed — nothing in the code changes.
* **Backups:** Neon keeps point-in-time history on its plans; you can also run `pg_dump` with the same
  connection string.
* **Data protection:** institutions upload personal student data. Before taking real customers, publish a
  privacy notice and data processing agreement that meet the **Nigeria Data Protection Act 2023**, choose a
  database region you can justify, and keep the connection string private.
* **Email:** add `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM` to Secrets to send
  student messages as real email.

## Alternative hosts

Render, Railway or a VPS can also run it: start command
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`, and set the same environment
variables. These allow custom domains such as `analytics.drspecimen.online`.
