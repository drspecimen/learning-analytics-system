"""
database/db.py
Database Layer — every SQL statement in the system lives here.

Two backends, same code:
  * PostgreSQL when the DATABASE_URL environment variable (or Streamlit secret) is set —
    used online, one central database for every institution.
  * SQLite file app.db otherwise — used for local development.

Isolation rule: every function that touches institution data takes `iid`
(the institution_id of the signed-in user) and filters on it. The id always comes
from the server-side session, never from anything the user types.
"""

import os
import re
import sqlite3
import threading

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

_local = threading.local()
_ready = set()


def database_url():
    return os.environ.get("DATABASE_URL", "").strip()


def backend():
    return "postgres" if database_url() else "sqlite"


def sqlite_path():
    return os.environ.get("LAS_DB_PATH", os.path.join(BASE_DIR, "app.db"))


# ---------------------------------------------------------------- connection

def _connect():
    if backend() == "postgres":
        import psycopg
        return psycopg.connect(database_url(), autocommit=True)
    conn = sqlite3.connect(sqlite_path(), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _pg_schema(sql):
    sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
    sql = sql.replace("BLOB", "BYTEA").replace(" REAL", " DOUBLE PRECISION")
    sql = sql.replace("TEXT DEFAULT CURRENT_TIMESTAMP",
                      "TEXT DEFAULT to_char(now(), 'YYYY-MM-DD HH24:MI:SS')")
    return sql


def _conn():
    """One persistent connection per thread; reconnects automatically if it dropped."""
    key = (backend(), database_url() or sqlite_path())
    conn = getattr(_local, "conn", None)
    if not (conn is not None and getattr(_local, "key", None) == key and not getattr(conn, "closed", False)):
        conn = _connect()
        _local.conn, _local.key = conn, key
    if key not in _ready:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = f.read()
        if backend() == "postgres":
            clean = "\n".join(l for l in schema.splitlines() if not l.strip().startswith("--"))
            for stmt in _pg_schema(clean).split(";"):
                if stmt.strip():
                    conn.execute(stmt.strip())
        else:
            conn.executescript(schema)
        _migrate(conn)
        _ready.add(key)
    return conn


def _retry(fn):
    """Run fn(conn); if the connection was dropped (common with hosted databases), reconnect once."""
    try:
        return fn(_conn())
    except Exception as exc:
        name = exc.__class__.__name__
        if name not in ("OperationalError", "InterfaceError", "ProgrammingError") or \
                (name == "ProgrammingError" and "closed" not in str(exc).lower()):
            raise
        close()
        return fn(_conn())


# columns added after the first release: (table, column, type). Added automatically to older databases.
MIGRATIONS = [
    ("Institutions", "matric_format", "TEXT"),
    ("Institutions", "matric_example", "TEXT"),
    ("Institutions", "student_email_format", "TEXT"),
    ("Institutions", "staff_email_domain", "TEXT"),
]


def _migrate(conn):
    for table, col, ctype in MIGRATIONS:
        if backend() == "postgres":
            conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {ctype}")
        else:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if col not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ctype}")
                conn.commit()


def _sql(sql):
    if backend() == "postgres":
        return sql.replace("%", "%%").replace("?", "%s")
    return sql


def init_db():
    _conn()


def close():
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _local.conn = None


def drop_all():
    """Delete EVERY table (all institutions). Used only by local setup --reset and tests."""
    if backend() == "sqlite":
        close()
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(sqlite_path() + suffix):
                os.remove(sqlite_path() + suffix)
    else:
        conn = _conn()
        for t in ["System_Logs", "Interventions", "Notifications", "Models", "Model_Results", "Predictions",
                  "Grades", "Assessments", "Weekly_Activity", "Enrollments", "Students", "Datasets",
                  "Courses", "Users", "Institutions"]:
            conn.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
    _ready.clear()
    init_db()


# ---------------------------------------------------------------- primitives

def query_df(sql, params=()):
    def run(conn):
        cur = conn.execute(_sql(sql), tuple(params))
        cols = [d[0] for d in cur.description] if cur.description else []
        return pd.DataFrame([tuple(r) for r in cur.fetchall()], columns=cols)
    return _retry(run)


def query_one(sql, params=()):
    def run(conn):
        cur = conn.execute(_sql(sql), tuple(params))
        row = cur.fetchone()
        return None if row is None else dict(zip([d[0] for d in cur.description], row))
    return _retry(run)


def scalar(sql, params=()):
    def run(conn):
        row = conn.execute(_sql(sql), tuple(params)).fetchone()
        return row[0] if row else None
    return _retry(run)


def execute(sql, params=()):
    conn = _conn()
    conn.execute(_sql(sql), tuple(params))
    if backend() == "sqlite":
        conn.commit()


def executemany(sql, rows):
    rows = list(rows)
    if not rows:
        return
    conn = _conn()
    if backend() == "postgres":
        with conn.cursor() as cur:
            cur.executemany(_sql(sql), rows)
    else:
        conn.executemany(sql, rows)
        conn.commit()


def insert(table, values, pk):
    """Insert one row from a dict and return its new primary key."""
    cols = list(values)
    sql = (f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
           f"RETURNING {pk}")
    conn = _conn()
    new_id = conn.execute(_sql(sql), tuple(values[c] for c in cols)).fetchone()[0]
    if backend() == "sqlite":
        conn.commit()
    return new_id


def _in(ids):
    ids = [int(i) for i in ids] or [-1]
    return ",".join(str(i) for i in ids)          # integers only, safe to inline


# ---------------------------------------------------------------- institutions

def create_institution(name, code, itype, country, contact_email, matric_format=None, matric_example=None,
                       student_email_format=None, staff_email_domain=None):
    return insert("Institutions", {"name": name, "code": code, "type": itype, "country": country,
                                   "contact_email": contact_email, "matric_format": matric_format or None,
                                   "matric_example": matric_example or None,
                                   "student_email_format": student_email_format or None,
                                   "staff_email_domain": staff_email_domain or None}, "institution_id")


def update_institution(iid, **fields):
    allowed = {"name", "type", "country", "contact_email", "matric_format", "matric_example",
               "student_email_format", "staff_email_domain"}
    cols = [k for k in fields if k in allowed]
    if cols:
        execute(f"UPDATE Institutions SET {', '.join(c + ' = ?' for c in cols)} WHERE institution_id = ?",
                tuple(fields[c] or None for c in cols) + (iid,))


def get_institution(iid):
    return query_one("SELECT * FROM Institutions WHERE institution_id = ?", (iid,))


def institution_code_exists(code):
    return scalar("SELECT COUNT(*) FROM Institutions WHERE LOWER(code) = LOWER(?)", (code,)) > 0


def set_institution_mode(iid, mode):
    execute("UPDATE Institutions SET data_mode = ? WHERE institution_id = ?", (mode, iid))


def set_institution_active(iid, active):
    execute("UPDATE Institutions SET is_active = ? WHERE institution_id = ?", (1 if active else 0, iid))


def platform_overview():
    """Platform owner view: counts only, never any student details."""
    return query_df("""
        SELECT i.institution_id, i.name, i.code, i.type, i.country, i.contact_email, i.data_mode,
               i.is_active, i.created_at,
               (SELECT COUNT(*) FROM Users u WHERE u.institution_id = i.institution_id) AS users,
               (SELECT COUNT(*) FROM Students s WHERE s.institution_id = i.institution_id) AS students,
               (SELECT COUNT(*) FROM Enrollments e WHERE e.institution_id = i.institution_id) AS enrollments,
               (SELECT MAX(logged_at) FROM System_Logs l WHERE l.institution_id = i.institution_id) AS last_activity
        FROM Institutions i ORDER BY i.institution_id
    """)


def wipe_institution_data(iid, keep_demo_users=False):
    """Delete all academic data of ONE institution. Real user accounts are kept."""
    enr = f"SELECT enrollment_id FROM Enrollments WHERE institution_id = {int(iid)}"
    crs = f"SELECT course_id FROM Courses WHERE institution_id = {int(iid)}"
    execute(f"DELETE FROM Weekly_Activity WHERE enrollment_id IN ({enr})")
    execute(f"DELETE FROM Grades WHERE enrollment_id IN ({enr})")
    for t in ["Predictions", "Notifications", "Interventions", "Enrollments", "Students",
              "Model_Results", "Models", "Datasets"]:
        execute(f"DELETE FROM {t} WHERE institution_id = ?", (iid,))
    execute(f"DELETE FROM Assessments WHERE course_id IN ({crs})")
    execute("DELETE FROM Courses WHERE institution_id = ?", (iid,))
    if not keep_demo_users:
        execute("DELETE FROM Users WHERE institution_id = ? AND is_demo = 1", (iid,))
    set_institution_mode(iid, "empty")


# ---------------------------------------------------------------- users

def insert_user(iid, username, password_hash, role, name=None, email=None, department=None,
                is_owner=0, is_demo=0):
    return insert("Users", {"institution_id": iid, "username": username.lower(),
                            "password_hash": password_hash, "role": role, "name": name,
                            "email": email or username.lower(), "department": department,
                            "is_owner": is_owner, "is_demo": is_demo}, "user_id")


def get_user_by_username(username):
    return query_one("SELECT * FROM Users WHERE username = ?", ((username or "").strip().lower(),))


def get_user_by_id(user_id):
    return query_one("SELECT * FROM Users WHERE user_id = ?", (user_id,))


def get_all_users(iid):
    return query_df(
        "SELECT user_id, username AS email, name, role, department, is_active, is_owner, is_demo, "
        "created_at, last_login FROM Users WHERE institution_id = ? ORDER BY is_owner DESC, role, name", (iid,))


def get_lecturers(iid):
    return query_df(
        "SELECT user_id, name, username, department FROM Users "
        "WHERE institution_id = ? AND role = 'Lecturer' AND is_active = 1 ORDER BY name", (iid,))


def set_user_active(iid, user_id, active):
    execute("UPDATE Users SET is_active = ? WHERE user_id = ? AND institution_id = ? AND is_owner = 0",
            (1 if active else 0, user_id, iid))


def delete_user(iid, user_id):
    """Permanently remove a staff account. History (messages, interventions, logs) is kept
    but no longer linked to the person; their courses become unassigned."""
    target = query_one("SELECT * FROM Users WHERE user_id = ? AND institution_id = ?", (user_id, iid))
    if not target:
        raise ValueError("User not found in your institution.")
    if target["is_owner"]:
        raise ValueError("The master administrator cannot be deleted. Transfer the role first.")
    execute("UPDATE Courses SET lecturer_id = NULL WHERE lecturer_id = ? AND institution_id = ?", (user_id, iid))
    execute("UPDATE Notifications SET sender_id = NULL WHERE sender_id = ? AND institution_id = ?", (user_id, iid))
    execute("UPDATE Interventions SET created_by = NULL WHERE created_by = ? AND institution_id = ?", (user_id, iid))
    execute("UPDATE Datasets SET uploaded_by = NULL WHERE uploaded_by = ? AND institution_id = ?", (user_id, iid))
    execute("UPDATE System_Logs SET user_id = NULL WHERE user_id = ? AND institution_id = ?", (user_id, iid))
    execute("DELETE FROM Users WHERE user_id = ? AND institution_id = ?", (user_id, iid))
    return target


def transfer_owner(iid, from_user_id, to_user_id):
    target = query_one("SELECT * FROM Users WHERE user_id = ? AND institution_id = ?", (to_user_id, iid))
    if not target or target["role"] != "Administrator" or not target["is_active"]:
        raise ValueError("The new master administrator must be an active Administrator of your institution.")
    execute("UPDATE Users SET is_owner = 0 WHERE user_id = ? AND institution_id = ?", (from_user_id, iid))
    execute("UPDATE Users SET is_owner = 1 WHERE user_id = ? AND institution_id = ?", (to_user_id, iid))


def update_user_details(iid, user_id, name, department, role=None):
    if role:
        execute("UPDATE Users SET name = ?, department = ?, role = ? WHERE user_id = ? AND institution_id = ? "
                "AND is_owner = 0", (name, department, role, user_id, iid))
    else:
        execute("UPDATE Users SET name = ?, department = ? WHERE user_id = ? AND institution_id = ?",
                (name, department, user_id, iid))


def recent_failed_logins(user_id, minutes=15):
    since = (pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")
    last_ok = scalar("SELECT MAX(logged_at) FROM System_Logs WHERE user_id = ? AND action = 'LOGIN'", (user_id,))
    since = max(since, last_ok or "")
    return scalar("SELECT COUNT(*) FROM System_Logs WHERE user_id = ? AND action = 'LOGIN_FAILED' "
                  "AND logged_at > ?", (user_id, since))


def update_password_hash(user_id, password_hash):
    execute("UPDATE Users SET password_hash = ? WHERE user_id = ?", (password_hash, user_id))


def touch_last_login(user_id):
    execute("UPDATE Users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?", (user_id,))


# ---------------------------------------------------------------- courses

def get_courses(iid, lecturer_id=None):
    sql = """
        SELECT c.*, u.name AS lecturer_name,
               (SELECT COUNT(*) FROM Enrollments e WHERE e.course_id = c.course_id) AS enrolled
        FROM Courses c LEFT JOIN Users u ON c.lecturer_id = u.user_id
        WHERE c.institution_id = ?
    """
    if lecturer_id:
        return query_df(sql + " AND c.lecturer_id = ? ORDER BY c.code", (iid, lecturer_id))
    return query_df(sql + " ORDER BY c.code", (iid,))


def assign_course_lecturer(iid, course_id, lecturer_id):
    if scalar("SELECT COUNT(*) FROM Users WHERE user_id = ? AND institution_id = ?", (lecturer_id, iid)) == 0:
        raise ValueError("Lecturer does not belong to this institution.")
    execute("UPDATE Courses SET lecturer_id = ? WHERE course_id = ? AND institution_id = ?",
            (lecturer_id, course_id, iid))


def unassign_courses(iid, course_ids):
    execute(f"UPDATE Courses SET lecturer_id = NULL WHERE institution_id = ? AND course_id IN ({_in(course_ids)})",
            (iid,))


def get_assessments(iid, course_id=None):
    sql = ("SELECT a.* FROM Assessments a JOIN Courses c ON a.course_id = c.course_id "
           "WHERE c.institution_id = ?")
    if course_id:
        return query_df(sql + " AND a.course_id = ? ORDER BY a.due_week", (iid, course_id))
    return query_df(sql + " ORDER BY a.course_id, a.due_week", (iid,))


# ---------------------------------------------------------------- students

def get_student(iid, student_id):
    return query_one("SELECT * FROM Students WHERE student_id = ? AND institution_id = ?", (student_id, iid))


def update_student(iid, student_id, **fields):
    allowed = {"first_name", "last_name", "email", "phone", "programme", "faculty", "level", "region"}
    cols = [k for k in fields if k in allowed]
    if cols:
        execute(f"UPDATE Students SET {', '.join(c + ' = ?' for c in cols)} "
                "WHERE student_id = ? AND institution_id = ?",
                tuple(fields[c] for c in cols) + (student_id, iid))


def get_student_history(iid, student_id):
    return query_df("""SELECT e.session, c.code, c.title, e.final_result, e.final_score
                       FROM Enrollments e JOIN Courses c ON e.course_id = c.course_id
                       WHERE e.student_id = ? AND e.institution_id = ? AND e.final_result IS NOT NULL
                       ORDER BY e.session DESC, c.code""", (student_id, iid))


# ---------------------------------------------------------------- analytics tables
# enrollment_ids passed here always come from get_enrollment_base_df(iid, ...),
# so they already belong to the institution; the join re-checks it anyway.

def get_activity_df(iid, enrollment_ids):
    return query_df(f"""SELECT a.* FROM Weekly_Activity a
                        JOIN Enrollments e ON a.enrollment_id = e.enrollment_id
                        WHERE e.institution_id = ? AND a.enrollment_id IN ({_in(enrollment_ids)})""", (iid,))


def get_grades_df(iid, enrollment_ids):
    return query_df(f"""
        SELECT g.*, a.title, a.type, a.due_week, a.weight, a.max_score, a.course_id
        FROM Grades g JOIN Assessments a ON g.assessment_id = a.assessment_id
        JOIN Enrollments e ON g.enrollment_id = e.enrollment_id
        WHERE e.institution_id = ? AND g.enrollment_id IN ({_in(enrollment_ids)})""", (iid,))


def get_enrollment_base_df(iid, lecturer_id=None):
    """One row per enrollment with student and course details plus the latest prediction."""
    sql = """
        SELECT e.enrollment_id, e.student_id, e.course_id, e.session, e.status,
               e.final_result, e.final_score,
               s.matric_no, s.first_name, s.last_name, s.email, s.phone, s.gender, s.age_band,
               s.region, s.highest_education, s.programme, s.faculty, s.level,
               s.prev_attempts, s.disability,
               c.code AS course_code, c.title AS course_title, c.department, c.credits,
               c.lecturer_id, u.name AS lecturer_name,
               p.predicted_result, p.risk_score, p.risk_level, p.confidence_score,
               p.top_factors, p.model_used, p.predicted_at
        FROM Enrollments e
        JOIN Students s ON e.student_id = s.student_id
        JOIN Courses c ON e.course_id = c.course_id
        LEFT JOIN Users u ON c.lecturer_id = u.user_id
        LEFT JOIN (
            SELECT p1.* FROM Predictions p1
            JOIN (SELECT enrollment_id, MAX(prediction_id) AS mid FROM Predictions
                  WHERE institution_id = ? GROUP BY enrollment_id) last
              ON p1.prediction_id = last.mid
        ) p ON p.enrollment_id = e.enrollment_id
        WHERE e.institution_id = ?
    """
    if lecturer_id:
        return query_df(sql + " AND c.lecturer_id = ?", (iid, iid, lecturer_id))
    return query_df(sql, (iid, iid))


def current_session(iid):
    s = scalar("SELECT MAX(session) FROM Enrollments WHERE institution_id = ? AND final_result IS NULL", (iid,))
    return s or scalar("SELECT MAX(session) FROM Enrollments WHERE institution_id = ?", (iid,))


def current_week(iid, session):
    return scalar("""SELECT MAX(a.week) FROM Weekly_Activity a
                     JOIN Enrollments e ON a.enrollment_id = e.enrollment_id
                     WHERE e.institution_id = ? AND e.session = ?""", (iid, session))


def historical_outcomes(iid):
    return query_df("""
        SELECT e.session, c.code AS course_code, c.department, c.faculty, e.final_result, e.final_score
        FROM Enrollments e JOIN Courses c ON e.course_id = c.course_id
        WHERE e.institution_id = ? AND e.final_result IS NOT NULL""", (iid,))


# ---------------------------------------------------------------- predictions & models

def insert_predictions(iid, rows):
    """rows: (enrollment_id, predicted_result, risk_score, risk_level, confidence, top_factors, model_used)"""
    executemany(
        "INSERT INTO Predictions (institution_id, enrollment_id, predicted_result, risk_score, risk_level, "
        "confidence_score, top_factors, model_used) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(iid,) + tuple(r) for r in rows])


def insert_model_result(iid, model_name, accuracy, precision, recall, f1, auc, is_best, params, rows):
    insert("Model_Results", {"institution_id": iid, "model_name": model_name, "accuracy": accuracy,
                             "precision_score": precision, "recall_score": recall, "f1_score": f1,
                             "auc_roc": auc, "is_best": int(is_best), "params": params,
                             "trained_rows": rows}, "result_id")


def get_model_history(iid):
    return query_df("SELECT model_name, accuracy, precision_score, recall_score, f1_score, auc_roc, is_best, "
                    "params, trained_rows, evaluated_at FROM Model_Results WHERE institution_id = ? "
                    "ORDER BY result_id DESC", (iid,))


def save_model(iid, bundle_bytes, report_json):
    return insert("Models", {"institution_id": iid, "bundle": bundle_bytes, "report": report_json}, "model_id")


def latest_model_meta(iid):
    return query_one("SELECT model_id, report, trained_at FROM Models WHERE institution_id = ? "
                     "ORDER BY model_id DESC LIMIT 1", (iid,))


def load_model_bytes(iid, model_id):
    b = scalar("SELECT bundle FROM Models WHERE institution_id = ? AND model_id = ?", (iid, model_id))
    return bytes(b) if b is not None else None


def prune_models(iid, keep=3):
    execute("""DELETE FROM Models WHERE institution_id = ? AND model_id NOT IN
               (SELECT model_id FROM Models WHERE institution_id = ? ORDER BY model_id DESC LIMIT ?)""",
            (iid, iid, keep))


# ---------------------------------------------------------------- datasets

def insert_dataset(iid, file_name, source, uploaded_by, row_count):
    return insert("Datasets", {"institution_id": iid, "file_name": file_name, "source": source,
                               "uploaded_by": uploaded_by, "row_count": row_count}, "dataset_id")


def get_datasets(iid):
    return query_df("""SELECT d.dataset_id, d.file_name, d.source, d.row_count, d.uploaded_at,
                              u.name AS uploaded_by FROM Datasets d
                       LEFT JOIN Users u ON d.uploaded_by = u.user_id
                       WHERE d.institution_id = ? ORDER BY d.dataset_id DESC""", (iid,))


# ---------------------------------------------------------------- notifications & interventions

def insert_notification(iid, student_id, sender_id, channel, subject, message, status="Recorded"):
    return insert("Notifications", {"institution_id": iid, "student_id": student_id, "sender_id": sender_id,
                                    "channel": channel, "subject": subject, "message": message,
                                    "status": status}, "notification_id")


def get_notifications(iid, student_id=None, sender_id=None, limit=500):
    sql = """
        SELECT n.*, s.matric_no, s.first_name || ' ' || s.last_name AS student_name, u.name AS sender_name
        FROM Notifications n
        JOIN Students s ON n.student_id = s.student_id
        LEFT JOIN Users u ON n.sender_id = u.user_id
        WHERE n.institution_id = ?
    """
    params = [iid]
    if student_id:
        sql += " AND n.student_id = ?"; params.append(student_id)
    if sender_id:
        sql += " AND n.sender_id = ?"; params.append(sender_id)
    sql += " ORDER BY n.notification_id DESC LIMIT ?"
    params.append(limit)
    return query_df(sql, params)


def insert_intervention(iid, student_id, created_by, itype, note):
    return insert("Interventions", {"institution_id": iid, "student_id": student_id, "created_by": created_by,
                                    "type": itype, "note": note}, "intervention_id")


def update_intervention_status(iid, intervention_id, status):
    execute("UPDATE Interventions SET status = ? WHERE intervention_id = ? AND institution_id = ?",
            (status, intervention_id, iid))


def get_interventions(iid, student_id=None, created_by=None):
    sql = """
        SELECT i.*, s.matric_no, s.first_name || ' ' || s.last_name AS student_name, u.name AS created_by_name
        FROM Interventions i
        JOIN Students s ON i.student_id = s.student_id
        LEFT JOIN Users u ON i.created_by = u.user_id
        WHERE i.institution_id = ?
    """
    params = [iid]
    if student_id:
        sql += " AND i.student_id = ?"; params.append(student_id)
    if created_by:
        sql += " AND i.created_by = ?"; params.append(created_by)
    return query_df(sql + " ORDER BY i.intervention_id DESC", params)


# ---------------------------------------------------------------- logs

def insert_log(iid, user_id, action, details=None):
    try:
        execute("INSERT INTO System_Logs (institution_id, user_id, action, details) VALUES (?, ?, ?, ?)",
                (iid, user_id, action, details))
    except Exception:
        pass  # logging must never break the request


def get_logs(iid, limit=1000):
    return query_df("""
        SELECT l.log_id, l.logged_at, u.username AS user_email, l.action, l.details
        FROM System_Logs l LEFT JOIN Users u ON l.user_id = u.user_id
        WHERE l.institution_id = ? ORDER BY l.log_id DESC LIMIT ?""", (iid, limit))


# ---------------------------------------------------------------- counts

def table_counts(iid):
    i = int(iid)
    q = {
        "Students": f"SELECT COUNT(*) FROM Students WHERE institution_id = {i}",
        "Courses": f"SELECT COUNT(*) FROM Courses WHERE institution_id = {i}",
        "Enrollments": f"SELECT COUNT(*) FROM Enrollments WHERE institution_id = {i}",
        "Weekly_Activity": f"SELECT COUNT(*) FROM Weekly_Activity a JOIN Enrollments e "
                           f"ON a.enrollment_id = e.enrollment_id WHERE e.institution_id = {i}",
        "Grades": f"SELECT COUNT(*) FROM Grades g JOIN Enrollments e ON g.enrollment_id = e.enrollment_id "
                  f"WHERE e.institution_id = {i}",
        "Predictions": f"SELECT COUNT(*) FROM Predictions WHERE institution_id = {i}",
        "Users": f"SELECT COUNT(*) FROM Users WHERE institution_id = {i}",
    }
    return {k: scalar(v) for k, v in q.items()}


def data_version(iid):
    """Changes whenever the institution's data changes — used to refresh cached analytics."""
    i = int(iid)
    return (scalar(f"SELECT MAX(prediction_id) FROM Predictions WHERE institution_id = {i}"),
            scalar(f"SELECT COUNT(*) FROM Enrollments WHERE institution_id = {i}"),
            scalar(f"SELECT COALESCE(SUM(COALESCE(lecturer_id, 0) * course_id), 0) FROM Courses "
                   f"WHERE institution_id = {i}"),
            scalar(f"SELECT MAX(dataset_id) FROM Datasets WHERE institution_id = {i}"))


if __name__ == "__main__":
    init_db()
    print(f"Database ready ({backend()})")
