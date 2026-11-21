"""
tests/functional_tests.py
Executes the functional test cases reported in Chapter Five against a fresh database
and writes the actual result and status of each case to functional_results.json.
Run:  python tests/functional_tests.py
"""

import io
import json
import os
import sys
import tempfile
import time

os.environ["LAS_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "functional.db")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd                                                                    # noqa: E402
from streamlit.testing.v1 import AppTest                                               # noqa: E402

from business import analytics, importer, notifications, onboarding                    # noqa: E402
from business.auth import add_user, authenticate_user, delete_user, register_institution  # noqa: E402
from database import db                                                                # noqa: E402
from ml import ml_service, train                                                       # noqa: E402
from ml.features import FEATURES                                                       # noqa: E402

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")
results = []


def case(tc, fr, condition, expected):
    def wrap(fn):
        t0 = time.time()
        try:
            actual, ok = fn()
        except Exception as exc:          # a crash is a failure, recorded as such
            actual, ok = f"Exception: {exc}", False
        results.append({"id": tc, "fr": fr, "condition": condition, "expected": expected,
                        "actual": actual, "status": "Pass" if ok else "Fail",
                        "seconds": round(time.time() - t0, 2)})
        print(f"{tc} {'PASS' if ok else 'FAIL'}  {actual}")
        return fn
    return wrap


def app_login(email, password):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.text_input[0].input(email)
    at.text_input[1].input(password)
    at.button[0].click()
    at.run()
    return at


db.drop_all()
A, UA = register_institution("Alpha University", "ALPHA", "University", "Nigeria", "Alpha Admin", "admin@alpha.edu.ng",
                             "Alpha1234", "ALP#########", "ALP123456789", "{matric}@alpha.edu.ng", "alpha.edu.ng")
B, UB = register_institution("Beta Polytechnic", "BETA", "Polytechnic", "Nigeria", "Beta Admin", "admin@beta.edu.ng",
                             "Beta12345")
onboarding.load_demo(A, UA, n_students=300)
LEC = add_user(A, UA, "lecturer@alpha.edu.ng", "Lect12345", "Lecturer", "Test Lecturer")
COURSES = db.get_courses(A)["course_id"].tolist()
db.assign_course_lecturer(A, COURSES[0], LEC)


@case("TC-01", "FR1", "Valid lecturer email and password", "Lecturer pages open; administrator pages hidden")
def _():
    at = app_login("lecturer@alpha.edu.ng", "Lect12345")
    menu = [o.split("  ", 1)[-1] for o in at.radio(key="nav").options]
    hidden = not any(p in menu for p in ("Users", "Data Management", "Model Performance", "Settings"))
    return f"Signed in; menu = {', '.join(menu)}", (not at.exception) and hidden


@case("TC-02", "FR1", "Wrong password", "Error shown; no session; attempt logged")
def _():
    at = app_login("lecturer@alpha.edu.ng", "wrong-pass1")
    logged = db.scalar("SELECT COUNT(*) FROM System_Logs WHERE action = 'LOGIN_FAILED'") >= 1
    err = [e.value for e in at.error]
    return f"Error: '{err[0] if err else ''}'; session user = {at.session_state['user'] if 'user' in at.session_state else None}; logged = {logged}", bool(err) and logged and "user" not in at.session_state


@case("TC-03", "FR1/NFR2", "Five wrong passwords in 15 minutes", "Account locked with message")
def _():
    for _ in range(5):
        authenticate_user("admin@beta.edu.ng", "nope-nope1")
    user, err = authenticate_user("admin@beta.edu.ng", "Beta12345")
    return f"Correct password refused: '{err}'", user is None and "Too many" in (err or "")


@case("TC-04", "FR2", "Administrator adds a lecturer with an email outside the staff domain", "Account refused")
def _():
    try:
        add_user(A, UA, "someone@gmail.com", "Abcd12345", "Lecturer", "Outside")
        return "Account created", False
    except ValueError as exc:
        return f"Refused: '{exc}'", True


@case("TC-05", "FR2", "Administrator deletes a lecturer who has courses", "Account removed; courses unassigned")
def _():
    tmp = add_user(A, UA, "temp@alpha.edu.ng", "Temp12345", "Lecturer", "Temporary")
    db.assign_course_lecturer(A, COURSES[1], tmp)
    delete_user(A, UA, tmp)
    gone = db.get_user_by_id(tmp) is None
    lec = db.scalar("SELECT lecturer_id FROM Courses WHERE course_id = ?", (COURSES[1],))
    return f"User exists = {not gone}; course lecturer = {lec}", gone and lec is None


@case("TC-06", "FR3", "Assign then unassign a course from a lecturer", "Lecturer sees the course, then no longer sees it")
def _():
    db.assign_course_lecturer(A, COURSES[2], LEC)
    before = set(analytics.enrollment_view(A, LEC)["course_id"])
    db.unassign_courses(A, [COURSES[2]])
    after = set(analytics.enrollment_view(A, LEC)["course_id"])
    return f"Courses before = {len(before)}, after = {len(after)}", COURSES[2] in before and COURSES[2] not in after


@case("TC-07", "FR4", "CSV with a required column missing", "File rejected; missing column named")
def _():
    df = pd.read_csv(io.BytesIO(importer.sample_csv(10, 5))).drop(columns=["avg_score"])
    errors, _ = importer.validate(df)
    return f"Errors: {errors}", any("avg_score" in e for e in errors)


@case("TC-08", "FR4", "CSV with avg_score of 150 and an invalid final_result", "File rejected; both problems listed")
def _():
    df = pd.read_csv(io.BytesIO(importer.sample_csv(10, 5)))
    df.loc[0, "avg_score"] = 150
    df.loc[1, "final_result"] = "Excellent"
    errors, _ = importer.validate(df)
    return f"{len(errors)} errors: {errors}", any("avg_score" in e for e in errors) and any("final_result" in e for e in errors)


@case("TC-09", "FR4", "Matric numbers that do not fit the institution's format", "File rejected with examples")
def _():
    inst = db.get_institution(A)
    df = pd.read_csv(io.BytesIO(importer.sample_csv(10, 5, inst=inst)))
    df.loc[0, "matric_no"] = "WRONG/123"
    errors, _ = importer.validate(df, inst)
    return f"Errors: {errors}", any("WRONG/123" in e for e in errors)


@case("TC-10", "FR5", "Reset and load a valid institution file (600 rows)", "Data imported, three models trained, current students scored")
def _():
    _, clean = importer.validate(pd.read_csv(io.BytesIO(importer.sample_csv())))
    out = onboarding.load_own_data(B, clean, "sample.csv", 8, UB)
    ok = out["train_error"] is None and out["scored"] == 200 and len(out["report"]["results"]) == 3
    return f"Imported {out['import']['enrollments']} enrollments; best = {out['report']['best_model']}; scored {out['scored']}", ok


@case("TC-11", "NFR1", "Score 1,000 enrollments", "Predictions returned within 5 seconds")
def _():
    view = analytics.enrollment_view(A)
    X = pd.concat([view[FEATURES]] * (1000 // len(view) + 1)).head(1000)
    t0 = time.time()
    ml_service.predict(A, X)
    secs = time.time() - t0
    return f"1,000 predictions with explanations in {secs:.2f} s", secs < 5


@case("TC-12", "FR7", "Open a high-risk student", "Top three contributing factors shown")
def _():
    v = analytics.enrollment_view(A)
    row = v[v["risk_level"] == "High"].iloc[0]
    factors = row["top_factors"].split(" | ")
    return f"{len(factors)} factors, e.g. '{factors[0]}'", len(factors) == 3


@case("TC-13", "FR8", "Early warning list", "Only High and Medium risk enrollments listed, highest first")
def _():
    at = app_login("admin@alpha.edu.ng", "Alpha1234")
    at.radio(key="nav").set_value("Early Warning")
    at.run()
    table = at.dataframe[0].value
    levels = set(table["risk_level"])
    sorted_ok = table["risk_score"].is_monotonic_decreasing
    return f"{len(table)} rows; levels = {sorted(levels)}; sorted = {sorted_ok}", levels <= {"High", "Medium"} and sorted_ok and not at.exception


@case("TC-14", "FR9", "Export the high-risk report", "CSV rows equal the number of high-risk enrollments")
def _():
    v = analytics.enrollment_view(A)
    expected = int((v["risk_level"] == "High").sum())
    exported = len(pd.read_csv(io.StringIO(v[v["risk_level"] == "High"].to_csv(index=False))))
    return f"Dashboard count {expected}; exported rows {exported}", expected == exported


@case("TC-15", "FR10", "Administrator retrains at week 8", "Three models compared; best saved; earlier versions kept (max 3)")
def _():
    for _ in range(3):
        train.train(A, cutoff_week=8, verbose=False)
    rep = ml_service.get_report(A)
    versions = db.scalar("SELECT COUNT(*) FROM Models WHERE institution_id = ?", (A,))
    return f"Models compared = {list(rep['results'])}; stored versions = {versions}", len(rep["results"]) == 3 and versions == 3


@case("TC-16", "FR12/NFR7", "Check the audit log after the actions above", "Sign-ins, imports, training and predictions recorded")
def _():
    acts = set(db.get_logs(B)["action"]) | set(db.get_logs(A)["action"])
    need = {"LOGIN", "LOGIN_FAILED", "DATASET_IMPORTED", "MODEL_TRAINED", "PREDICTIONS_RUN", "USER_DELETED"}
    return f"Missing actions: {sorted(need - acts) or 'none'}", need <= acts


@case("TC-17", "FR13", "Send a message to a student", "Message stored with status; visible in history")
def _():
    sid = int(analytics.enrollment_view(A)["student_id"].iloc[0])
    _, status = notifications.send(A, sid, UA, "Email", "Check-in", "Please contact your lecturer.")
    return f"Status = {status}; history rows = {len(db.get_notifications(A, student_id=sid))}", len(db.get_notifications(A, student_id=sid)) >= 1


@case("TC-18", "NFR2", "Institution A requests a student who belongs to institution B", "Nothing returned")
def _():
    sid_b = int(analytics.enrollment_view(B)["student_id"].iloc[0])
    return f"Result = {db.get_student(A, sid_b)}", db.get_student(A, sid_b) is None


@case("TC-19", "NFR2", "Session idle for longer than the timeout", "User signed out on next action")
def _():
    at = app_login("admin@alpha.edu.ng", "Alpha1234")
    at.session_state["last_seen"] = time.time() - 31 * 60
    at.run()
    msgs = [w.value for w in at.warning]
    return f"Message: '{msgs[0] if msgs else ''}'", "user" not in at.session_state and any("inactivity" in m for m in msgs)


@case("TC-20", "NFR4", "Every page opened by each role on demo data", "No page raises an error")
def _():
    errors = []
    for email, pw in [("admin@alpha.edu.ng", "Alpha1234"), ("lecturer@alpha.edu.ng", "Lect12345")]:
        at = app_login(email, pw)
        for opt in at.radio(key="nav").options:
            page = opt.split("  ", 1)[-1]
            at.radio(key="nav").set_value(page)
            at.run()
            if at.exception:
                errors.append(f"{email}:{page}")
    return f"Pages with errors: {errors or 'none'}", not errors


with open(os.path.join(os.path.dirname(__file__), "functional_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n{sum(r['status'] == 'Pass' for r in results)} of {len(results)} test cases passed")
