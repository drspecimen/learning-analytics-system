"""
business/importer.py
CSV import of an institution's own LMS data (one row = one student in one course).

Rows WITH final_result  -> past students: the model learns from them.
Rows WITHOUT final_result -> current students: the model predicts for them.
All inserts are batched, so large files load quickly even on an online database.
"""

import io

import numpy as np
import pandas as pd

from business import formats
from database import db
from scripts.generate_institution_data import ASSESSMENTS

# name, required, description, example
COLUMNS = [
    ("matric_no", True, "Unique student ID / matric number", "ABC/2023/1045"),
    ("first_name", True, "Student's first name", "Chioma"),
    ("last_name", True, "Student's surname", "Okafor"),
    ("course_code", True, "Course code", "CSC201"),
    ("session", True, "Academic session", "2025/2026"),
    ("total_logins", True, "LMS logins in this course so far", "34"),
    ("total_clicks", True, "LMS clicks / page views so far", "580"),
    ("active_weeks", True, "Weeks with at least one login (0-14)", "7"),
    ("avg_score", True, "Average assessment score so far, 0-100", "62.5"),
    ("final_result", False, "PAST students only: Distinction, Pass, Fail or Withdrawn. Leave blank for current students",
     "Pass"),
    ("assessments_missed", False, "Assessments not submitted", "1"),
    ("late_submissions", False, "Assessments submitted late", "0"),
    ("minutes_active", False, "Total minutes on the LMS", "640"),
    ("resources_viewed", False, "Course materials opened", "40"),
    ("forum_posts", False, "Discussion forum posts", "5"),
    ("videos_watched", False, "Lecture videos watched", "18"),
    ("prev_attempts", False, "Times the student took this course before", "0"),
    ("gender", False, "M or F", "F"),
    ("age_band", False, "16-24, 25-34, 35-44 or 45+", "25-34"),
    ("highest_education", False, "SSCE/WAEC, OND, NCE, HND or First Degree", "OND"),
    ("disability", False, "Y or N", "N"),
    ("course_title", False, "Course title", "Data Structures"),
    ("lecturer_email", False, "Email of the lecturer account to assign the course to", "lecturer@yourschool.edu"),
    ("email", False, "Student's email (used for messages)", "student@yourschool.edu"),
    ("phone", False, "Student's phone", "08031234567"),
    ("programme", False, "Programme of study", "B.Sc. Computer Science"),
    ("faculty", False, "Faculty", "Faculty of Computing"),
    ("region", False, "State / region", "Lagos"),
    ("level", False, "Level, e.g. 100-500", "200"),
]
REQUIRED = [c[0] for c in COLUMNS if c[1]]
TEMPLATE_COLUMNS = [c[0] for c in COLUMNS]
DEFAULTS = {"email": "", "phone": "", "gender": "M", "age_band": "25-34", "region": "",
            "highest_education": "SSCE/WAEC", "programme": "", "faculty": "", "level": 200,
            "prev_attempts": 0, "disability": "N", "course_title": "", "minutes_active": 0,
            "resources_viewed": 0, "forum_posts": 0, "videos_watched": 0, "assessments_missed": 0,
            "late_submissions": 0, "final_result": "", "lecturer_email": ""}
VALID_RESULTS = {"", "Distinction", "Pass", "Fail", "Withdrawn"}
NUMERIC = ["total_logins", "total_clicks", "active_weeks", "minutes_active", "resources_viewed",
           "forum_posts", "videos_watched", "avg_score", "assessments_missed", "late_submissions",
           "prev_attempts", "level"]
RESULT_ALIASES = {"distinction": "Distinction", "pass": "Pass", "passed": "Pass", "fail": "Fail",
                  "failed": "Fail", "withdrawn": "Withdrawn", "withdraw": "Withdrawn", "dropped": "Withdrawn"}


def columns_table():
    return pd.DataFrame([{"Column": n, "Required": "Yes" if r else "Optional", "What it means": d,
                          "Example": e} for n, r, d, e in COLUMNS])


def validate(df, inst=None):
    """Returns (errors, cleaned dataframe). Non-blocking warnings are in df.attrs["warnings"].
    `inst` (the institution row) enables its matric number and student email formats."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"Missing required columns: {', '.join(missing)}"], df
    if df.empty:
        return ["The file has no rows."], df
    errors = []
    for c, default in DEFAULTS.items():
        if c not in df.columns:
            df[c] = default
    df = df.fillna({c: v for c, v in DEFAULTS.items()})
    for c in ["matric_no", "first_name", "last_name", "course_code", "session"]:
        df[c] = df[c].astype(str).str.strip()
        blank = (df[c] == "") | (df[c].str.lower() == "nan")
        if blank.any():
            errors.append(f"Column '{c}' is empty in {int(blank.sum())} row(s).")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        bad = int(df[c].isna().sum())
        if bad:
            errors.append(f"Column '{c}' has {bad} empty or non-numeric value(s).")
    res = df["final_result"].astype(str).str.strip()
    res = res.where(~res.str.lower().isin(["nan", "none"]), "")
    df["final_result"] = res.map(lambda v: RESULT_ALIASES.get(v.lower(), v) if v else "")
    bad_res = set(df["final_result"]) - VALID_RESULTS
    if bad_res:
        errors.append(f"final_result must be Distinction, Pass, Fail, Withdrawn or blank. Found: "
                      f"{', '.join(sorted(bad_res)[:6])}")
    if (df["avg_score"] > 100).any() or (df["avg_score"] < 0).any():
        errors.append("avg_score must be between 0 and 100.")
    if (df["active_weeks"] > 14).any() or (df["active_weeks"] < 0).any():
        errors.append("active_weeks must be between 0 and 14.")
    dup = int(df.duplicated(["matric_no", "course_code", "session"]).sum())
    if dup:
        errors.append(f"{dup} duplicate rows (same matric_no, course_code and session).")

    warnings = []
    if inst and inst.get("matric_format"):
        bad = df.loc[~df["matric_no"].map(lambda m: formats.matric_matches(inst["matric_format"], m)), "matric_no"]
        if len(bad):
            errors.append(f"{bad.nunique()} matric number(s) do not match your institution's format "
                          f"'{inst['matric_format']}', e.g. {', '.join(bad.unique()[:4])}. Fix the file or update "
                          "the format under Settings.")
    if inst and inst.get("student_email_format"):
        tpl = inst["student_email_format"]
        df["email"] = df["email"].astype(str).str.strip().replace({"nan": ""})
        blank = df["email"] == ""
        df.loc[blank, "email"] = [formats.student_email(tpl, r.matric_no, r.first_name, r.last_name)
                                  for r in df[blank].itertuples(index=False)]
        if blank.any():
            warnings.append(f"{int(blank.sum())} blank student emails were filled in from your format ({tpl}).")
        domain = tpl.split("@")[-1]
        off = df[~df["email"].str.lower().str.endswith("@" + domain)]
        if len(off):
            warnings.append(f"{len(off)} student email(s) do not use your domain @{domain}, "
                            f"e.g. {off['email'].iloc[0]}. They will be imported as given.")
    df.attrs["warnings"] = warnings
    return errors, df


def summarize(df):
    past = df[df["final_result"] != ""]
    return {"rows": len(df), "past": len(past), "current": len(df) - len(past),
            "students": df["matric_no"].nunique(), "courses": df["course_code"].nunique(),
            "sessions": sorted(df["session"].unique().tolist()),
            "results": past["final_result"].value_counts().to_dict()}


def import_dataframe(iid, df, user_id, file_name, cutoff_week=8):
    """Insert a validated dataframe into institution iid. Returns counts."""
    dataset_id = db.insert_dataset(iid, file_name, "CSV upload", user_id, len(df))

    # ---- students (new ones only)
    existing = dict(db.query_df("SELECT matric_no, student_id FROM Students WHERE institution_id = ?",
                                (iid,)).itertuples(index=False, name=None))
    firsts = df.drop_duplicates("matric_no")
    new = firsts[~firsts["matric_no"].isin(existing)]
    db.executemany(
        "INSERT INTO Students (institution_id, matric_no, first_name, last_name, email, phone, gender, age_band, "
        "region, highest_education, programme, faculty, level, prev_attempts, disability, dataset_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(iid, r.matric_no, r.first_name, r.last_name, str(r.email), str(r.phone), str(r.gender).upper()[:1],
          str(r.age_band), str(r.region), str(r.highest_education), str(r.programme), str(r.faculty),
          int(r.level), int(r.prev_attempts), str(r.disability).upper()[:1], dataset_id)
         for r in new.itertuples(index=False)])
    sid_of = dict(db.query_df("SELECT matric_no, student_id FROM Students WHERE institution_id = ?",
                              (iid,)).itertuples(index=False, name=None))

    # ---- courses (create missing, with a standard assessment plan) and lecturer assignment
    course_of = dict(db.query_df("SELECT code, course_id FROM Courses WHERE institution_id = ?",
                                 (iid,)).itertuples(index=False, name=None))
    lecturers = dict(db.query_df("SELECT username, user_id FROM Users WHERE institution_id = ? AND role = 'Lecturer'",
                                 (iid,)).itertuples(index=False, name=None))
    for code, g in df.groupby("course_code"):
        lec = lecturers.get(str(g["lecturer_email"].iloc[0]).strip().lower())
        if code not in course_of:
            title = str(g["course_title"].iloc[0]) or code
            fac = str(g["faculty"].iloc[0]) or "Not specified"
            cid = db.insert("Courses", {"institution_id": iid, "code": code, "title": title, "faculty": fac,
                                        "department": fac, "level": int(g["level"].iloc[0]), "semester": "",
                                        "credits": 3, "lecturer_id": lec}, "course_id")
            db.executemany("INSERT INTO Assessments (course_id, title, type, due_week, weight, max_score) "
                           "VALUES (?, ?, ?, ?, ?, 100)", [(cid,) + a for a in ASSESSMENTS])
            course_of[code] = cid
        elif lec:
            db.execute("UPDATE Courses SET lecturer_id = ? WHERE course_id = ? AND institution_id = ?",
                       (lec, course_of[code], iid))

    # ---- enrollments (skip duplicates already in the database)
    have = set(db.query_df("SELECT student_id, course_id, session FROM Enrollments WHERE institution_id = ?",
                           (iid,)).itertuples(index=False, name=None))
    rows, keep = [], []
    for r in df.itertuples(index=False):
        key = (sid_of[r.matric_no], course_of[r.course_code], r.session)
        if key in have:
            continue
        have.add(key)
        done = bool(r.final_result)
        rows.append((iid,) + key + ("Completed" if done else "Active", r.final_result or None))
        keep.append(r)
    skipped = len(df) - len(rows)
    db.executemany("INSERT INTO Enrollments (institution_id, student_id, course_id, session, status, final_result) "
                   "VALUES (?, ?, ?, ?, ?, ?)", rows)
    eid_of = {(s, c, ss): e for e, s, c, ss in db.query_df(
        "SELECT enrollment_id, student_id, course_id, session FROM Enrollments WHERE institution_id = ?",
        (iid,)).itertuples(index=False, name=None)}
    assess = db.get_assessments(iid)
    assess = assess[assess["due_week"] <= cutoff_week]
    assess_of = {cid: g.sort_values("due_week")["assessment_id"].tolist() for cid, g in assess.groupby("course_id")}

    # ---- spread each row's totals across its active weeks, and rebuild its grades
    act, grades = [], []
    for r in keep:
        cid = course_of[r.course_code]
        eid = eid_of[(sid_of[r.matric_no], cid, r.session)]
        weeks = int(max(0, min(r.active_weeks, cutoff_week)))
        div = max(weeks, 1)
        for w in range(1, cutoff_week + 1):
            if w <= weeks:
                act.append((eid, w, int(round(r.total_logins / div)), int(round(r.total_clicks / div)),
                            float(r.minutes_active) / div, int(round(r.resources_viewed / div)),
                            int(round(r.forum_posts / div)), int(round(r.videos_watched / div))))
            else:
                act.append((eid, w, 0, 0, 0.0, 0, 0, 0))
        ids = assess_of.get(cid, [])
        missed, late = min(int(r.assessments_missed), len(ids)), int(r.late_submissions)
        submitted = len(ids) - missed
        # keep the student's overall average equal to avg_score even with missed (zero) work
        per = float(np.clip(r.avg_score * len(ids) / submitted, 0, 100)) if submitted else 0.0
        for k, aid in enumerate(ids):
            if k >= submitted:
                grades.append((eid, aid, 0.0, 0, 0, "Not submitted"))
            else:
                grades.append((eid, aid, round(per, 1), 1, 2 if k < late else 0, "Imported score"))
    for i in range(0, len(act), 20000):
        db.executemany("INSERT INTO Weekly_Activity (enrollment_id, week, logins, clicks, minutes_active, "
                       "resources_viewed, forum_posts, videos_watched) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       act[i:i + 20000])
    for i in range(0, len(grades), 20000):
        db.executemany("INSERT INTO Grades (enrollment_id, assessment_id, score, submitted, days_late, feedback) "
                       "VALUES (?, ?, ?, ?, ?, ?)", grades[i:i + 20000])
    db.insert_log(iid, user_id, "DATASET_IMPORTED",
                  f"{file_name}: {len(new)} new students, {len(rows)} enrollments, {skipped} skipped")
    return {"dataset_id": dataset_id, "students": len(new), "enrollments": len(rows), "skipped": skipped}


def template_csv():
    example = {n: e for n, _, _, e in COLUMNS}
    return pd.DataFrame([example], columns=TEMPLATE_COLUMNS).to_csv(index=False).encode("utf-8")


def sample_csv(n_past=400, n_current=200, seed=7, prefix="SMP", inst=None):
    """A realistic test file with past AND current students (enough history to train).
    With `inst`, matric numbers and emails follow that institution's formats."""
    rng = np.random.default_rng(seed)
    fmt = (inst or {}).get("matric_format")
    tpl = (inst or {}).get("student_email_format")
    used = set()

    def matric(i):
        if not fmt:
            return f"{prefix}/{20000 + i}"
        for _ in range(50):
            m = formats.example_from_pattern(fmt, rng)
            if m not in used:
                used.add(m)
                return m
        raise ValueError("The matric format allows too few unique numbers for a sample file.")
    courses = [("CSC201", "Data Structures"), ("CSC203", "Database Systems"), ("MTH201", "Linear Algebra"),
               ("ACC201", "Financial Accounting"), ("ECO201", "Microeconomics"), ("GST201", "Use of English")]
    first = ["Ada", "Bola", "Chidi", "Dayo", "Efe", "Femi", "Gbenga", "Hadiza", "Ife", "Jide", "Kemi", "Lola",
             "Musa", "Nneka", "Ope", "Rasheed", "Sade", "Tolu", "Uju", "Wale"]
    last = ["Ade", "Bello", "Chukwu", "Danladi", "Eze", "Fashola", "Garba", "Ihekwe", "Johnson", "Kalu"]
    rows = []
    for i in range(n_past + n_current):
        past = i < n_past
        engage = float(np.clip(rng.normal(0.55, 0.25), 0.02, 1))
        ability = float(np.clip(rng.normal(0.55, 0.2), 0.05, 1))
        code, title = courses[int(rng.integers(0, len(courses)))]
        weeks = int(np.clip(8 - rng.binomial(6, max(0.0, 0.6 - engage)), 0, 8))
        per = lambda lam: int(rng.poisson(lam, weeks).sum()) if weeks else 0
        missed = int(rng.binomial(4, max(0.05, 0.6 - engage)))
        avg = round(float(np.clip(rng.normal(25 + 65 * ability, 10), 0, 100)) * (1 - missed / 5), 1)
        result = ""
        if past:
            final = avg + rng.normal(0, 8)
            result = ("Withdrawn" if (weeks <= 4 or engage < 0.25) and rng.random() < 0.6 else
                      "Distinction" if final >= 68 else "Pass" if final >= 42 else "Fail")
        m, fn, ln = matric(i), str(rng.choice(first)), str(rng.choice(last))
        rows.append({
            "matric_no": m, "first_name": fn, "last_name": ln,
            "course_code": code, "session": "2024/2025" if past else "2025/2026",
            "total_logins": per(1 + 6 * engage), "total_clicks": per(15 + 90 * engage * (0.6 + ability)),
            "active_weeks": weeks, "avg_score": avg, "final_result": result, "assessments_missed": missed,
            "late_submissions": int(rng.binomial(3, max(0.05, 0.45 - 0.35 * engage))),
            "minutes_active": round(float(rng.gamma(2, 20 + 80 * engage, weeks).sum()), 1) if weeks else 0.0,
            "resources_viewed": per(1 + 8 * engage), "forum_posts": per(0.2 + 2.5 * engage * ability),
            "videos_watched": per(0.5 + 4 * engage), "prev_attempts": int(rng.choice([0, 0, 0, 1])),
            "gender": rng.choice(["M", "F"]), "age_band": rng.choice(["16-24", "25-34", "35-44"]),
            "highest_education": rng.choice(["SSCE/WAEC", "OND", "HND"]), "disability": "N",
            "course_title": title, "lecturer_email": "",
            "email": formats.student_email(tpl, m, fn, ln) if tpl else f"student{20000 + i}@example.edu.ng",
            "phone": "0803" + str(rng.integers(1000000, 9999999)), "programme": "B.Sc. (Sample)",
            "faculty": "Faculty of Sciences", "region": rng.choice(["Lagos", "Oyo", "Kano", "Abuja (FCT)"]),
            "level": 200,
        })
    buf = io.StringIO()
    pd.DataFrame(rows, columns=TEMPLATE_COLUMNS).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
