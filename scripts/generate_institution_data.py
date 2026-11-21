"""
scripts/generate_institution_data.py
Loads realistic demo LMS data into ONE institution, so a newly registered
institution can explore a complete system before uploading its own data.

  - 3 faculties, 6 departments, 26 courses, demo lecturer accounts
  - students with demographics
  - 2024/2025 completed (used to TRAIN), 2025/2026 in progress at week 8 (PREDICTED)
  - weekly LMS activity, assessments, grades with feedback

Each student has a hidden ability and motivation that drive their engagement,
scores and outcome, so the models have real patterns to learn.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database import db                      # noqa: E402
from business.auth import hash_password      # noqa: E402
from business import formats                 # noqa: E402

TOTAL_WEEKS = 14
CURRENT_WEEK = 8
PAST_SESSION = "2024/2025"
CURRENT_SESSION = "2025/2026"
DEMO_LECTURER_PASSWORD = "Lecturer123"

FACULTIES = {
    "Faculty of Computing": {
        "Computer Science": ["CSC201 Data Structures", "CSC203 Database Systems",
                             "CSC205 Operating Systems", "CSC207 Web Programming"],
        "Information Technology": ["IFT201 Networking Fundamentals", "IFT203 Systems Analysis",
                                   "IFT205 Cloud Computing", "IFT207 Cybersecurity Basics"],
    },
    "Faculty of Management Sciences": {
        "Accounting": ["ACC201 Financial Accounting", "ACC203 Cost Accounting",
                       "ACC205 Taxation", "ACC207 Auditing Principles"],
        "Business Administration": ["BUS201 Principles of Management", "BUS203 Marketing",
                                    "BUS205 Business Statistics", "BUS207 Entrepreneurship"],
    },
    "Faculty of Sciences": {
        "Mathematics": ["MTH201 Linear Algebra", "MTH203 Real Analysis",
                        "MTH205 Numerical Methods", "MTH207 Probability Theory"],
        "Economics": ["ECO201 Microeconomics", "ECO203 Macroeconomics",
                      "ECO205 Econometrics", "ECO207 Development Economics"],
    },
}
GST_COURSES = ["GST201 Nigerian Peoples and Culture", "GST203 Communication in English"]
LECTURERS = [
    ("Dr. Adebayo Okonkwo", "Computer Science"), ("Dr. Chiamaka Eze", "Information Technology"),
    ("Prof. Ibrahim Musa", "Accounting"), ("Dr. Funke Adeyemi", "Business Administration"),
    ("Dr. Emeka Nwosu", "Mathematics"), ("Dr. Halima Bello", "Economics"),
    ("Mrs. Ngozi Okafor", "General Studies"), ("Mr. Tunde Bakare", "Computer Science"),
]
FIRST_M = ["Chinedu", "Oluwaseun", "Ibrahim", "Emeka", "Tunde", "Musa", "Segun", "Obinna", "Kelechi",
           "Yusuf", "Femi", "Ikenna", "Abdul", "Damilola", "Uche", "Babajide", "Tobi", "Sani", "Chukwudi", "Kunle"]
FIRST_F = ["Adaeze", "Funmilayo", "Aisha", "Ngozi", "Chioma", "Zainab", "Bisola", "Amaka", "Fatima", "Yetunde",
           "Ifeoma", "Hauwa", "Temitope", "Nkechi", "Folake", "Blessing", "Kemi", "Halima", "Ebere", "Ronke"]
LAST = ["Adeyemi", "Okafor", "Bello", "Eze", "Ogunleye", "Nwankwo", "Abubakar", "Olawale", "Okonkwo", "Ibrahim",
        "Adebayo", "Chukwu", "Usman", "Afolabi", "Obi", "Lawal", "Okoro", "Balogun", "Danjuma", "Akinola",
        "Nnamdi", "Oyelaran", "Garba", "Ezeh"]
REGIONS = ["Lagos", "Abuja (FCT)", "Oyo", "Kano", "Rivers", "Enugu", "Kaduna", "Ogun", "Anambra", "Delta",
           "Benue", "Kwara"]
EDUCATION = ["SSCE/WAEC", "OND", "NCE", "HND", "First Degree"]
AGE_BANDS = ["16-24", "25-34", "35-44", "45+"]
FEEDBACK = {
    "good": ["Excellent work, well structured.", "Very good understanding shown.", "Strong answer, keep it up."],
    "mid": ["Good attempt, revise the key concepts.", "Fair work; add more examples.",
            "Satisfactory, but more depth needed."],
    "low": ["Below expectation, please see me.", "Incomplete answer; review the lecture notes.",
            "Weak attempt; attend the tutorial sessions."],
}
ASSESSMENTS = [("Assignment 1", "Assignment", 3, 10), ("Quiz 1", "Quiz", 5, 10),
               ("Assignment 2", "Assignment", 7, 10), ("Mid-Semester Test", "Test", 8, 20),
               ("Assignment 3", "Assignment", 11, 10), ("Final Exam", "Exam", 14, 40)]


def demo_lecturer_email(code, i):
    return f"lecturer{i}.{code.lower()}@demo.las"


def generate(iid, n_students=400, seed=42, verbose=True, progress=None):
    """Load demo data into institution `iid`. `progress(fraction, text)` is optional."""
    rng = np.random.default_rng(seed)
    log = print if verbose else (lambda *a, **k: None)
    step = progress or (lambda f, t: None)
    inst = db.get_institution(iid)
    code = inst["code"]
    fmt, tpl, used = inst.get("matric_format"), inst.get("student_email_format"), set()
    id_rng = np.random.default_rng(seed + 1)   # separate stream: ID formats never change the simulated data

    step(0.05, "Creating demo lecturers and courses")
    lecturer_ids = {}
    for i, (name, dept) in enumerate(LECTURERS, start=1):
        email = demo_lecturer_email(code, i)
        existing = db.get_user_by_username(email)
        uid = existing["user_id"] if existing else db.insert_user(
            iid, email, hash_password(DEMO_LECTURER_PASSWORD), "Lecturer", name, email, dept, is_demo=1)
        lecturer_ids.setdefault(dept, []).append(uid)

    course_ids, dept_courses = {}, {}
    for faculty, depts in list(FACULTIES.items()) + [("General Studies", {"General Studies": GST_COURSES})]:
        for dept, courses in depts.items():
            for c in courses:
                ccode, title = c.split(" ", 1)
                pool = lecturer_ids.get(dept) or lecturer_ids["General Studies"]
                cid = db.insert("Courses", {"institution_id": iid, "code": ccode, "title": title,
                                            "faculty": faculty, "department": dept, "level": 200,
                                            "semester": "First", "credits": int(rng.choice([2, 3, 3, 4])),
                                            "lecturer_id": pool[len(course_ids) % len(pool)]}, "course_id")
                course_ids[ccode] = cid
                dept_courses.setdefault(dept, []).append(ccode)
    db.executemany("INSERT INTO Assessments (course_id, title, type, due_week, weight, max_score) "
                   "VALUES (?, ?, ?, ?, ?, 100)",
                   [(cid,) + a for cid in course_ids.values() for a in ASSESSMENTS])
    assess = db.get_assessments(iid)
    assess_by_course = {cid: g.to_dict("records") for cid, g in assess.groupby("course_id")}

    step(0.15, f"Creating {n_students} students")
    dataset_id = db.insert_dataset(iid, "Demo institution LMS feed", "Demo data", None, 0)
    depts = [d for d in dept_courses if d != "General Studies"]
    dept_faculty = {d: f for f, ds in FACULTIES.items() for d in ds}
    student_rows, profiles = [], []
    for i in range(n_students):
        gender = str(rng.choice(["M", "F"]))
        first, last = str(rng.choice(FIRST_M if gender == "M" else FIRST_F)), str(rng.choice(LAST))
        dept = str(rng.choice(depts))
        edu = str(rng.choice(EDUCATION, p=[0.40, 0.20, 0.12, 0.15, 0.13]))
        prev = int(rng.choice([0, 0, 0, 0, 1, 1, 2]))
        ability = float(np.clip(rng.normal(0.55, 0.18) + 0.04 * EDUCATION.index(edu) - 0.06 * prev, 0.05, 0.98))
        motivation = float(np.clip(rng.normal(0.6, 0.2), 0.05, 1.0))
        matric = f"{code}/{dept[:3].upper()}/{24 if i % 2 else 25}/{1000 + i}"
        if fmt:   # follow the institution's own matric format
            for _ in range(50):
                matric = formats.example_from_pattern(fmt, id_rng)
                if matric not in used:
                    break
        used.add(matric)
        email = formats.student_email(tpl, matric, first, last) if tpl else \
            f"{first.lower()}.{last.lower()}{i}@student.demo"
        student_rows.append((iid, matric, first, last, email,
                             "080" + "".join(str(d) for d in rng.integers(0, 10, 8)), gender,
                             str(rng.choice(AGE_BANDS, p=[0.35, 0.40, 0.18, 0.07])), str(rng.choice(REGIONS)),
                             edu, f"B.Sc. {dept}", dept_faculty[dept], 200, prev,
                             "Y" if rng.random() < 0.06 else "N", dataset_id))
        profiles.append((matric, dept, ability, motivation))
    db.executemany("INSERT INTO Students (institution_id, matric_no, first_name, last_name, email, phone, gender, "
                   "age_band, region, highest_education, programme, faculty, level, prev_attempts, disability, "
                   "dataset_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", student_rows)
    sid_of = dict(db.query_df("SELECT matric_no, student_id FROM Students WHERE institution_id = ?",
                              (iid,)).itertuples(index=False, name=None))

    # ---- decide every enrollment's story first, insert enrollments in one batch
    step(0.3, "Simulating two sessions of LMS activity")
    plans = []
    for matric, dept, ability, motivation in profiles:
        sid = sid_of[matric]
        my_courses = dept_courses[dept] + [str(rng.choice(GST_COURSES)).split(" ", 1)[0]]
        for session in (PAST_SESSION, CURRENT_SESSION):
            for ccode in my_courses:
                a = float(np.clip(ability - rng.normal(0, 0.06) + rng.normal(0, 0.05), 0.02, 0.99))
                m = float(np.clip(motivation + rng.normal(0, 0.08), 0.02, 1.0))
                p_withdraw = 0.03 + max(0.0, 0.55 - m) * 0.9 + (0.05 if a < 0.3 else 0)
                dropout = int(rng.integers(3, 11)) if rng.random() < p_withdraw else None
                fade = rng.random() < (0.35 if m < 0.45 else 0.08)
                plans.append((sid, course_ids[ccode], session, a, m, dropout, fade))
    db.executemany("INSERT INTO Enrollments (institution_id, student_id, course_id, session, status) "
                   "VALUES (?, ?, ?, ?, 'Active')", [(iid, p[0], p[1], p[2]) for p in plans])
    eid_of = {(s, c, ss): e for e, s, c, ss in db.query_df(
        "SELECT enrollment_id, student_id, course_id, session FROM Enrollments WHERE institution_id = ?",
        (iid,)).itertuples(index=False, name=None)}

    act_rows, grade_rows, updates = [], [], []
    for sid, cid, session, a, m, dropout, fade in plans:
        eid = eid_of[(sid, cid, session)]
        weeks_run = TOTAL_WEEKS if session == PAST_SESSION else CURRENT_WEEK
        for w in range(1, weeks_run + 1):
            if dropout and w >= dropout:
                break
            level = max(m * (1 - (0.06 * w if fade else 0)), 0.03)
            logins = int(rng.poisson(1 + 6 * level))
            if logins == 0:
                act_rows.append((eid, w, 0, 0, 0.0, 0, 0, 0))
                continue
            act_rows.append((eid, w, logins, int(rng.poisson(15 + 90 * level * (0.6 + a))),
                             round(float(rng.gamma(2, 20 + 80 * level)), 1), int(rng.poisson(1 + 8 * level)),
                             int(rng.poisson(0.2 + 2.5 * level * a)), int(rng.poisson(0.5 + 4 * level))))
        total = weight_sum = 0.0
        for asmt in assess_by_course[cid]:
            if asmt["due_week"] > weeks_run:
                continue
            aid = int(asmt["assessment_id"])
            if dropout and asmt["due_week"] >= dropout:
                grade_rows.append((eid, aid, None, 0, 0, "Not submitted"))
                continue
            p_submit = 0.55 + 0.45 * m if asmt["type"] != "Exam" else 0.92
            if fade and asmt["due_week"] > 7:
                p_submit -= 0.25
            weight_sum += asmt["weight"]
            if rng.random() > p_submit:
                grade_rows.append((eid, aid, 0.0, 0, 0, "Not submitted"))
                continue
            score = float(np.clip(rng.normal(25 + 70 * a + 8 * (m - 0.5), 9), 0, 100))
            late = int(rng.integers(1, 6)) if rng.random() < (0.45 - 0.35 * m) else 0
            score = max(0.0, score - 2 * late)
            band = "good" if score >= 70 else "mid" if score >= 45 else "low"
            grade_rows.append((eid, aid, round(score, 1), 1, late, str(rng.choice(FEEDBACK[band]))))
            total += score * asmt["weight"]
        if session == PAST_SESSION:
            if dropout:
                updates.append(("Withdrawn", "Withdrawn", None, eid))
            else:
                final = round(total / weight_sum, 1) if weight_sum else 0.0
                result = "Distinction" if final >= 70 else "Pass" if final >= 45 else "Fail"
                updates.append(("Completed", result, final, eid))
        elif dropout:
            updates.append(("Inactive", None, None, eid))

    step(0.6, f"Saving {len(act_rows):,} activity records and {len(grade_rows):,} grades")
    chunk = 20000
    for i in range(0, len(act_rows), chunk):
        db.executemany("INSERT INTO Weekly_Activity (enrollment_id, week, logins, clicks, minutes_active, "
                       "resources_viewed, forum_posts, videos_watched) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       act_rows[i:i + chunk])
    for i in range(0, len(grade_rows), chunk):
        db.executemany("INSERT INTO Grades (enrollment_id, assessment_id, score, submitted, days_late, feedback) "
                       "VALUES (?, ?, ?, ?, ?, ?)", grade_rows[i:i + chunk])
    db.executemany("UPDATE Enrollments SET status = ?, final_result = ?, final_score = ? WHERE enrollment_id = ?",
                   updates)
    db.execute("UPDATE Datasets SET row_count = ? WHERE dataset_id = ?", (len(plans), dataset_id))
    db.set_institution_mode(iid, "demo")
    counts = db.table_counts(iid)
    log("Demo data loaded: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    return counts
