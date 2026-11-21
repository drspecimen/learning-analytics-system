# Learning Analytics — multi-institution edition

AI-driven student performance prediction, early warning and visualisation that any institution
can register for and use with its own LMS data. Python · Streamlit · PostgreSQL/SQLite ·
Scikit-learn · Plotly. Four layers: Presentation → Business → Machine Learning → Database.

## How institutions use it

1. Open the app link and choose **Register your institution** (institution name, short code,
   master administrator's name, email and password).
2. The new workspace starts with **demo data** (optional) so they can explore every dashboard.
3. **What data you need** (on the landing page and under Data Management) lists every column.
4. **Data Management → Load your institution's data**: upload the CSV → confirm the reset
   (type the institution code) → *Yes, load my data*. The system wipes the demo data, imports
   theirs, trains Logistic Regression / Decision Tree / Random Forest on their past students and
   predicts every current student. All dashboards now show their institution.
5. **Users**: the master admin adds lecturers and assigns courses. Lecturers sign in with their own
   email and see only their courses and students.

## Institution formats

At registration (and later under **Settings**) the administrator sets:

* **Matric number format** — `#` digit, `?` letter, `*` anything, other characters fixed.
  `NOU#########` accepts `NOU212289345`; `???/???/##/####` accepts `DIG/CSC/24/1045`.
  Give just an example and the format is built from it (digits become `#`).
* **Student email format** — e.g. `{matric}@noun.edu.ng` or `{first}.{last}@student.unilag.edu.ng`.
  Blank emails in uploads are filled in from it.
* **Staff email domain** — e.g. `noun.edu.ng`. Lecturer and admin accounts must use it.

Uploads with matric numbers that don't fit the format are rejected with examples of the bad rows.

## Managing staff

* Add, edit, deactivate (keeps history, blocks sign-in) or **permanently delete** accounts
  (for staff who have left — their courses become unassigned; messages they recorded are kept).
* Assign and **unassign** courses; a lecturer instantly stops seeing unassigned courses.
* The master administrator can transfer the role to another administrator.
* 5 wrong passwords lock an account for 15 minutes.

## Isolation

* Every row of academic data carries `institution_id`; every query in `database/db.py` filters on
  the institution of the **signed-in session** (never on anything typed by a user).
* Lecturers are further limited to courses assigned to them.
* Each institution's trained model is stored separately in the database.
* Emails are unique across the platform, so one sign-in always maps to one institution.
* Accounts or institutions that are deactivated are signed out on their next click.
* The platform owner (`PLATFORM_ADMIN_EMAIL`) gets a **Platform** page with counts per institution
  and can suspend access — it never shows student records.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py            # then register an institution in the browser
```
Optional: `python setup_system.py --demo` creates a ready demo institution
(sign in with `admin@demo.las` / `Demo12345`).

Locally data is stored in `app.db` (SQLite). Online, set `DATABASE_URL` to a PostgreSQL database —
see **DEPLOY.md**.

## Tests

```bash
pytest -q                                   # SQLite
DATABASE_URL=postgresql://... pytest -q     # PostgreSQL (use an EMPTY test database: tests wipe it)
```

## Layout

```
app.py                  landing (sign in / register / data requirements), navigation
presentation/           theme, charts, shared pages, admin pages, platform page
business/               auth & registration, onboarding (demo/reset/load), importer, analytics, notifications
ml/                     feature engineering, training, prediction service (models stored in the DB)
database/               schema.sql, data-access layer (SQLite or PostgreSQL)
scripts/                demo data generator
tests/                  unit, end-to-end and isolation tests
DEPLOY.md               putting it online
```
