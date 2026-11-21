-- database/schema.sql
-- Multi-institution schema. Written in SQLite syntax; db.py converts it for PostgreSQL.
-- Every institution's data is tagged with institution_id, and every query in db.py
-- filters on it, so no institution can ever see another's data.
-- Chapter 3 tables (Figure 3.3): Users, Students, Datasets, Predictions, Model_Results, System_Logs.

CREATE TABLE IF NOT EXISTS Institutions (
    institution_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    code TEXT NOT NULL UNIQUE,
    type TEXT,
    country TEXT,
    contact_email TEXT,
    is_active INTEGER DEFAULT 1,
    data_mode TEXT DEFAULT 'empty',
    matric_format TEXT,
    matric_example TEXT,
    student_email_format TEXT,
    staff_email_domain TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS Users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    institution_id INTEGER NOT NULL,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('Administrator', 'Lecturer')),
    name TEXT,
    email TEXT,
    department TEXT,
    is_active INTEGER DEFAULT 1,
    is_owner INTEGER DEFAULT 0,
    is_demo INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    last_login TEXT,
    FOREIGN KEY (institution_id) REFERENCES Institutions(institution_id)
);

CREATE TABLE IF NOT EXISTS Courses (
    course_id INTEGER PRIMARY KEY AUTOINCREMENT,
    institution_id INTEGER NOT NULL,
    code TEXT NOT NULL,
    title TEXT NOT NULL,
    faculty TEXT,
    department TEXT,
    level INTEGER,
    semester TEXT,
    credits INTEGER,
    lecturer_id INTEGER,
    UNIQUE (institution_id, code),
    FOREIGN KEY (institution_id) REFERENCES Institutions(institution_id),
    FOREIGN KEY (lecturer_id) REFERENCES Users(user_id)
);

CREATE TABLE IF NOT EXISTS Datasets (
    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    institution_id INTEGER NOT NULL,
    file_name TEXT NOT NULL,
    source TEXT,
    uploaded_by INTEGER,
    row_count INTEGER,
    uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (institution_id) REFERENCES Institutions(institution_id)
);

CREATE TABLE IF NOT EXISTS Students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    institution_id INTEGER NOT NULL,
    matric_no TEXT NOT NULL,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone TEXT,
    gender TEXT,
    age_band TEXT,
    region TEXT,
    highest_education TEXT,
    programme TEXT,
    faculty TEXT,
    level INTEGER,
    prev_attempts INTEGER DEFAULT 0,
    disability TEXT DEFAULT 'N',
    dataset_id INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (institution_id, matric_no),
    FOREIGN KEY (institution_id) REFERENCES Institutions(institution_id)
);

CREATE TABLE IF NOT EXISTS Enrollments (
    enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    institution_id INTEGER NOT NULL,
    student_id INTEGER NOT NULL,
    course_id INTEGER NOT NULL,
    session TEXT,
    status TEXT DEFAULT 'Active',
    final_result TEXT,
    final_score REAL,
    UNIQUE (student_id, course_id, session),
    FOREIGN KEY (student_id) REFERENCES Students(student_id),
    FOREIGN KEY (course_id) REFERENCES Courses(course_id)
);

CREATE TABLE IF NOT EXISTS Weekly_Activity (
    activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    enrollment_id INTEGER NOT NULL,
    week INTEGER NOT NULL,
    logins INTEGER DEFAULT 0,
    clicks INTEGER DEFAULT 0,
    minutes_active REAL DEFAULT 0,
    resources_viewed INTEGER DEFAULT 0,
    forum_posts INTEGER DEFAULT 0,
    videos_watched INTEGER DEFAULT 0,
    FOREIGN KEY (enrollment_id) REFERENCES Enrollments(enrollment_id)
);

CREATE TABLE IF NOT EXISTS Assessments (
    assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    type TEXT,
    due_week INTEGER,
    weight REAL,
    max_score REAL DEFAULT 100,
    FOREIGN KEY (course_id) REFERENCES Courses(course_id)
);

CREATE TABLE IF NOT EXISTS Grades (
    grade_id INTEGER PRIMARY KEY AUTOINCREMENT,
    enrollment_id INTEGER NOT NULL,
    assessment_id INTEGER NOT NULL,
    score REAL,
    submitted INTEGER DEFAULT 1,
    days_late INTEGER DEFAULT 0,
    feedback TEXT,
    FOREIGN KEY (enrollment_id) REFERENCES Enrollments(enrollment_id),
    FOREIGN KEY (assessment_id) REFERENCES Assessments(assessment_id)
);

CREATE TABLE IF NOT EXISTS Predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    institution_id INTEGER NOT NULL,
    enrollment_id INTEGER NOT NULL,
    predicted_result TEXT,
    risk_score REAL,
    risk_level TEXT,
    confidence_score REAL,
    top_factors TEXT,
    model_used TEXT,
    predicted_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (enrollment_id) REFERENCES Enrollments(enrollment_id)
);

CREATE TABLE IF NOT EXISTS Model_Results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    institution_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    accuracy REAL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    auc_roc REAL,
    is_best INTEGER DEFAULT 0,
    params TEXT,
    trained_rows INTEGER,
    evaluated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS Models (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    institution_id INTEGER NOT NULL,
    bundle BLOB NOT NULL,
    report TEXT NOT NULL,
    trained_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS Notifications (
    notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
    institution_id INTEGER NOT NULL,
    student_id INTEGER NOT NULL,
    sender_id INTEGER,
    channel TEXT,
    subject TEXT,
    message TEXT,
    status TEXT DEFAULT 'Recorded',
    sent_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS Interventions (
    intervention_id INTEGER PRIMARY KEY AUTOINCREMENT,
    institution_id INTEGER NOT NULL,
    student_id INTEGER NOT NULL,
    created_by INTEGER,
    type TEXT,
    note TEXT,
    status TEXT DEFAULT 'Open',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS System_Logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    institution_id INTEGER,
    user_id INTEGER,
    action TEXT NOT NULL,
    details TEXT,
    logged_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_inst ON Users(institution_id);
CREATE INDEX IF NOT EXISTS idx_courses_inst ON Courses(institution_id);
CREATE INDEX IF NOT EXISTS idx_students_inst ON Students(institution_id);
CREATE INDEX IF NOT EXISTS idx_enr_inst ON Enrollments(institution_id);
CREATE INDEX IF NOT EXISTS idx_enr_student ON Enrollments(student_id);
CREATE INDEX IF NOT EXISTS idx_enr_course ON Enrollments(course_id);
CREATE INDEX IF NOT EXISTS idx_act_enr ON Weekly_Activity(enrollment_id);
CREATE INDEX IF NOT EXISTS idx_grade_enr ON Grades(enrollment_id);
CREATE INDEX IF NOT EXISTS idx_assess_course ON Assessments(course_id);
CREATE INDEX IF NOT EXISTS idx_pred_enr ON Predictions(enrollment_id);
CREATE INDEX IF NOT EXISTS idx_pred_inst ON Predictions(institution_id);
CREATE INDEX IF NOT EXISTS idx_models_inst ON Models(institution_id);
CREATE INDEX IF NOT EXISTS idx_notif_inst ON Notifications(institution_id);
CREATE INDEX IF NOT EXISTS idx_logs_inst ON System_Logs(institution_id);
