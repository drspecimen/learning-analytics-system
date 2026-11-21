"""
tests/test_system.py
Unit and end-to-end tests, including institution isolation.
Run:  pytest -q          (SQLite)   ·   DATABASE_URL=postgresql://... pytest -q   (PostgreSQL)
"""

import io
import os
import tempfile

import pandas as pd
import pytest

if not os.environ.get("DATABASE_URL"):
    os.environ["LAS_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")

from business import analytics, importer, notifications, onboarding   # noqa: E402
from business.auth import (add_user, authenticate_user, change_password, hash_password,  # noqa: E402
                           register_institution, verify_password)
from database import db                                                 # noqa: E402
from ml import ml_service                                               # noqa: E402
from ml.features import FEATURES                                        # noqa: E402


@pytest.fixture(scope="module")
def two_institutions():
    db.drop_all()
    a, ua = register_institution("Alpha University", "ALPHA", "University", "Nigeria", "Ada", "ada@alpha.edu", "Alpha1234")
    b, ub = register_institution("Beta Polytechnic", "BETA", "Polytechnic", "Nigeria", "Bola", "bola@beta.edu", "Beta12345")
    onboarding.load_demo(a, ua, n_students=150)
    _, clean = importer.validate(pd.read_csv(io.BytesIO(importer.sample_csv())))
    out = onboarding.load_own_data(b, clean, "sample.csv", 8, ub)
    return a, ua, b, ub, out


def test_password_hashing():
    h = hash_password("secret123")
    assert verify_password("secret123", h) and not verify_password("wrong", h)
    assert h != hash_password("secret123")


def test_registration_rules(two_institutions):
    with pytest.raises(ValueError):
        register_institution("Alpha Again", "ALPHA", "University", "NG", "X", "x@x.edu", "Abcdef123")
    with pytest.raises(ValueError):
        register_institution("Gamma", "GAMMA", "University", "NG", "X", "ada@alpha.edu", "Abcdef123")
    with pytest.raises(ValueError):
        register_institution("Gamma", "GAMMA", "University", "NG", "X", "g@g.edu", "short")


def test_sign_in(two_institutions):
    a, *_ = two_institutions
    user, err = authenticate_user("ADA@alpha.edu", "Alpha1234")
    assert err is None and user["institution_id"] == a
    assert authenticate_user("ada@alpha.edu", "nope")[0] is None


def test_institutions_never_share_data(two_institutions):
    a, _, b, _, _ = two_institutions
    va, vb = analytics.enrollment_view(a), analytics.enrollment_view(b)
    assert len(va) and len(vb)
    assert not set(va["student_id"]) & set(vb["student_id"])
    assert not set(va["course_id"]) & set(vb["course_id"])
    sid_b = int(vb["student_id"].iloc[0])
    assert db.get_student(a, sid_b) is None          # A cannot open B's student
    with pytest.raises(ValueError):
        notifications.send(a, sid_b, None, "SMS", "Hi", "Hello there")


def test_own_data_trained_and_scored(two_institutions):
    *_, out = two_institutions
    assert out["train_error"] is None and out["scored"] == 200
    assert set(out["report"]["results"]) == {"Logistic Regression", "Decision Tree", "Random Forest"}


def test_predictions_valid(two_institutions):
    a, *_ = two_institutions
    v = analytics.enrollment_view(a)
    assert v["risk_score"].between(0, 1).all()
    assert set(v["risk_level"]) <= {"High", "Medium", "Low"}
    assert not v[FEATURES].isna().any().any()


def test_lecturer_scope(two_institutions):
    a, ua, *_ = two_institutions
    lec = add_user(a, ua, "lec@alpha.edu", "Lecturer1", "Lecturer", "Lec One")
    assert len(analytics.enrollment_view(a, lec)) == 0          # nothing assigned yet
    cid = int(db.get_courses(a)["course_id"].iloc[0])
    db.assign_course_lecturer(a, cid, lec)
    v = analytics.enrollment_view(a, lec)
    assert len(v) and (v["course_id"] == cid).all()


def test_cannot_assign_foreign_lecturer(two_institutions):
    a, _, b, ub, _ = two_institutions
    lec_b = add_user(b, ub, "lec@beta.edu", "Lecturer2", "Lecturer", "Lec Two")
    cid_a = int(db.get_courses(a)["course_id"].iloc[0])
    with pytest.raises(ValueError):
        db.assign_course_lecturer(a, cid_a, lec_b)


def test_reset_only_touches_one_institution(two_institutions):
    a, ua, b, *_ = two_institutions
    before_b = db.table_counts(b)
    onboarding.reset(a, ua)
    assert db.table_counts(a)["Students"] == 0 and not ml_service.model_available(a)
    assert db.table_counts(b) == before_b


def test_bad_files_rejected():
    errors, _ = importer.validate(pd.DataFrame({"matric_no": ["x"]}))
    assert "Missing required columns" in errors[0]
    df = pd.read_csv(io.BytesIO(importer.sample_csv(5, 0)))
    df.loc[0, "avg_score"] = 150
    df.loc[1, "final_result"] = "Excellent"
    errors, _ = importer.validate(df)
    assert any("avg_score" in e for e in errors) and any("final_result" in e for e in errors)


def test_change_password(two_institutions):
    uid = db.get_user_by_username("lec@alpha.edu")["user_id"]
    change_password(uid, "Lecturer1", "NewPass99")
    assert authenticate_user("lec@alpha.edu", "NewPass99")[0]


# ---- institution formats, deletion, unassignment, lockout
from business import formats                      # noqa: E402
from business.auth import delete_user, transfer_owner, update_settings   # noqa: E402


def test_format_patterns():
    assert formats.matric_matches("NOU#########", "NOU212289345")
    assert not formats.matric_matches("NOU#########", "NOU2122")
    assert formats.matric_matches("???/???/##/####", "DIG/CSC/24/1045")
    assert formats.derive_pattern("DIG/CSC/24/1045") == "DIG/CSC/##/####"
    assert formats.student_email("{matric}@noun.edu.ng", "NOU212289345") == "nou212289345@noun.edu.ng"
    assert formats.staff_email_ok("noun.edu.ng", "Sishola@noun.edu.ng")
    assert not formats.staff_email_ok("noun.edu.ng", "someone@gmail.com")


def test_formatted_institution():
    iid, uid = register_institution("Open Univ", "OPEN", "University", "NG", "Sam", "sam@open.edu.ng",
                                    "Sam123456", "NOU#########", "NOU212289345", "{matric}@open.edu.ng",
                                    "open.edu.ng")
    with pytest.raises(ValueError):            # staff must use the domain
        add_user(iid, uid, "x@gmail.com", "Lecturer1", "Lecturer", "X")
    add_user(iid, uid, "lec@open.edu.ng", "Lecturer1", "Lecturer", "Lec")
    inst = db.get_institution(iid)
    good = pd.read_csv(io.BytesIO(importer.sample_csv(60, 20, inst=inst)))
    good["email"] = ""
    errors, clean = importer.validate(good, inst)
    assert errors == [] and clean["email"].str.endswith("@open.edu.ng").all()
    bad = good.copy()
    bad.loc[0, "matric_no"] = "WRONG/1"
    errors, _ = importer.validate(bad, inst)
    assert any("do not match" in e for e in errors)
    onboarding.load_demo(iid, uid, n_students=60)
    m = db.query_df("SELECT matric_no FROM Students WHERE institution_id = ?", (iid,))["matric_no"]
    assert m.map(lambda x: formats.matric_matches("NOU#########", x)).all()
    with pytest.raises(ValueError):
        update_settings(iid, uid, matric_format="NOU#########", matric_example="ABC")


def test_delete_and_unassign():
    iid, uid = db.get_user_by_username("sam@open.edu.ng")["institution_id"], db.get_user_by_username("sam@open.edu.ng")["user_id"]
    lec = db.get_user_by_username("lec@open.edu.ng")["user_id"]
    cids = db.get_courses(iid)["course_id"].head(2).tolist()
    for c in cids:
        db.assign_course_lecturer(iid, int(c), lec)
    db.unassign_courses(iid, [cids[0]])
    assert len(analytics.enrollment_view(iid, lec)["course_id"].unique()) == 1
    with pytest.raises(ValueError):
        delete_user(iid, lec, uid)             # master admin cannot be deleted
    delete_user(iid, uid, lec)
    assert db.get_user_by_username("lec@open.edu.ng") is None
    assert db.get_courses(iid)["lecturer_id"].isna().all() or lec not in db.get_courses(iid)["lecturer_id"].tolist()


def test_transfer_owner():
    iid = db.get_user_by_username("sam@open.edu.ng")["institution_id"]
    owner = db.get_user_by_username("sam@open.edu.ng")["user_id"]
    new = add_user(iid, owner, "admin2@open.edu.ng", "Admin2345", "Administrator", "Second Admin")
    transfer_owner(iid, owner, new)
    assert db.get_user_by_id(new)["is_owner"] == 1 and db.get_user_by_id(owner)["is_owner"] == 0


def test_lockout_after_failed_logins():
    for _ in range(5):
        authenticate_user("bola@beta.edu", "wrong-password")
    user, err = authenticate_user("bola@beta.edu", "Beta12345")
    assert user is None and "Too many" in err
