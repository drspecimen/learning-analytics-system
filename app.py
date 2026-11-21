"""
app.py
Presentation Layer entry point: landing page (sign in, register an institution,
dataset requirements), role-based navigation and page routing.
Run:  streamlit run app.py
"""

import os
import time

import streamlit as st

st.set_page_config(page_title="Learning Analytics", page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

# Online settings (Streamlit Cloud "Secrets") become environment variables before the DB loads.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float)):
            os.environ.setdefault(_k, str(_v))
except Exception:
    pass

from business import onboarding                                                     # noqa: E402
from business.auth import (authenticate_user, is_platform_owner, password_problems,  # noqa: E402
                           refresh_session_user, register_institution)
from business.formats import from_samples                                           # noqa: E402
from database import db                                                             # noqa: E402
from presentation import admin_pages, pages, theme                                  # noqa: E402

theme.apply()
db.init_db()
IDLE_TIMEOUT_MIN = int(os.environ.get("IDLE_TIMEOUT_MINUTES", "60"))

ADMIN_PAGES = {
    "Overview": ("📊", pages.overview),
    "Courses": ("📚", pages.courses),
    "Students": ("👥", pages.students),
    "Early Warning": ("🚨", pages.early_warning),
    "Student Profile": ("🧑‍🎓", pages.student_profile),
    "Model Performance": ("🤖", admin_pages.model_performance),
    "Data Management": ("🗂️", admin_pages.data_management),
    "Users": ("🔐", admin_pages.users),
    "Communications": ("✉️", pages.communications),
    "Reports": ("📥", pages.reports),
    "Audit Log": ("📜", admin_pages.audit_logs),
    "Settings": ("🏛️", admin_pages.settings),
    "My Account": ("⚙️", pages.account),
}
LECTURER_PAGES = {
    "Overview": ("📊", pages.overview),
    "My Courses": ("📚", pages.courses),
    "My Students": ("👥", pages.students),
    "Early Warning": ("🚨", pages.early_warning),
    "Student Profile": ("🧑‍🎓", pages.student_profile),
    "Communications": ("✉️", pages.communications),
    "Reports": ("📥", pages.reports),
    "My Account": ("⚙️", pages.account),
}


def sign_in(user):
    st.session_state.clear()
    st.session_state.user = user
    st.session_state.last_seen = time.time()


# ---------------------------------------------------------------- landing

def landing():
    st.markdown(
        "<div class='las-header' style='text-align:center;padding:34px 24px'>"
        "<h2 style='font-size:2rem'>🎓 Learning Analytics</h2>"
        "<p style='font-size:1.02rem'>AI-driven student performance prediction, early warning and "
        "visualisation for your institution's LMS data</p></div>", unsafe_allow_html=True)
    tab_in, tab_reg, tab_data, tab_how = st.tabs(["Sign in", "Register your institution",
                                                  "What data you need", "How it works"])

    with tab_in:
        _, mid, _ = st.columns([1, 1.3, 1])
        with mid:
            with st.form("login"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                go = st.form_submit_button("Sign in", type="primary", width="stretch")
            if go:
                if not email or not password:
                    st.error("Enter your email and password.")
                else:
                    user, err = authenticate_user(email, password)
                    if user:
                        sign_in(user)
                        st.rerun()
                    st.error(err)
            st.caption("Lecturers: use the email and password your institution's administrator created for you. "
                       "New institution? Use **Register your institution**.")

    with tab_reg:
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.markdown("Create your institution's private workspace. You become its **master administrator** "
                        "and can then add lecturers. No other institution can see your data.")
            with st.form("register"):
                st.markdown("**Institution**")
                c1, c2 = st.columns([2, 1])
                name = c1.text_input("Institution name")
                code = c2.text_input("Short code", max_chars=15,
                                     help="A short unique code for your institution (2-15 letters or numbers).")
                c3, c4 = st.columns(2)
                itype = c3.selectbox("Type", ["University", "Polytechnic", "College of Education",
                                              "Secondary School", "Training Institute", "Other"])
                country = c4.text_input("Country", value="Nigeria")
                st.markdown("**Your institution's formats** — enter one real sample of each. They are used "
                            "to check uploads and new staff accounts. See *How it works* for details.")
                c7, c8, c9 = st.columns(3)
                m_sample = c7.text_input("Sample student matric number")
                s_sample = c8.text_input("Sample student email")
                staff_sample = c9.text_input("Sample staff email")
                st.markdown("**Master administrator**")
                admin_name = st.text_input("Full name")
                admin_email = st.text_input("Work email (you will sign in with this)",
                                            help="Must use the same domain as your sample staff email.")
                c5, c6 = st.columns(2)
                pw = c5.text_input("Password", type="password", help="8+ characters with letters and numbers")
                pw2 = c6.text_input("Confirm password", type="password")
                demo = st.checkbox("Start with demo data so I can explore the system first "
                                   "(replace it with our own data any time)", value=True)
                agree = st.checkbox("I am authorised to manage student data for this institution")
                submit = st.form_submit_button("Create institution", type="primary", width="stretch")
            if submit:
                problems = password_problems(pw)
                if pw != pw2:
                    st.error("The passwords do not match.")
                elif problems:
                    st.error("Password must have " + " and ".join(problems) + ".")
                elif not agree:
                    st.error("Please confirm you are authorised to manage this data.")
                else:
                    try:
                        m_format, s_format, staff_dom, notes = from_samples(m_sample, s_sample, staff_sample)
                        bad = [n for n in notes if "not a valid" in n]
                        if bad:
                            raise ValueError(" ".join(bad))
                        iid, uid = register_institution(name, code, itype, country, admin_name, admin_email, pw,
                                                        m_format, m_sample.strip(), s_format, staff_dom)
                        for n in notes:
                            st.session_state.setdefault("flash", []).append(n)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        if demo:
                            bar = st.progress(0.0, text="Preparing your demo workspace...")
                            onboarding.load_demo(iid, uid, progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
                        flash = st.session_state.get("flash", [])
                        user, _ = authenticate_user(admin_email, pw)
                        sign_in(user)
                        st.session_state.flash = flash
                        st.session_state.nav_target = "Overview" if demo else "Data Management"
                        st.rerun()

    with tab_data:
        admin_pages.dataset_requirements()

    with tab_how:
        st.markdown(
            "1. **Register** your institution — you become its master administrator.\n"
            "2. **Explore** with demo data, or go straight to **Data Management**.\n"
            "3. **Upload** a CSV exported from your LMS with past students (with their final results) "
            "and current students. The system resets, loads your data, trains Logistic Regression, "
            "Decision Tree and Random Forest on your history, keeps the best, and predicts every current "
            "student.\n"
            "4. **Add lecturers** under Users and assign their courses. Each lecturer sees only their own "
            "courses and students.\n"
            "5. **Act**: open any student's profile to see why they are at risk, message them and log "
            "interventions.\n\n"
            "**Your institution's formats (asked at registration)**\n"
            "- *Sample student matric number*: type one real matric number exactly as your institution writes "
            "it, including any letters, slashes or dashes. Every digit position becomes a rule, so all uploaded "
            "matric numbers must have the same shape.\n"
            "- *Sample student email*: one real student email. If it is built from the matric number, the system "
            "fills in missing student emails the same way.\n"
            "- *Sample staff email*: one real staff email. Its domain (the part after @) becomes the rule for "
            "every lecturer and administrator account, including yours.\n"
            "- Leave any of them blank if your institution has no fixed format. All three can be changed later "
            "under **Settings**, where you can also test a matric number or email.\n\n"
            "Each institution's data is stored separately and every screen and query is restricted to the "
            "signed-in person's institution (and, for lecturers, to their own courses).")
        st.caption("Install on a phone, tablet or computer: open this page in Chrome or Edge and choose "
                   "**Install app** / **Add to Home screen** from the browser menu.")


# ---------------------------------------------------------------- signed-in shell

def sidebar(user, menu):
    if "nav_target" in st.session_state:
        st.session_state.nav = st.session_state.pop("nav_target")
    if st.session_state.get("nav") not in menu:
        st.session_state.nav = next(iter(menu))
    with st.sidebar:
        st.markdown(f"## 🎓 {user['institution_code']}")
        st.caption(user["institution_name"])
        st.markdown(f"**{user.get('name') or user['username']}**  \n{user['role']}"
                    + (" · master admin" if user.get("is_owner") else ""))
        if user.get("data_mode") == "demo":
            st.markdown("🧪 *Demo data loaded*")
        st.divider()
        choice = st.radio("Navigate", list(menu), key="nav", label_visibility="collapsed",
                          format_func=lambda p: f"{menu[p][0]}  {p}")
        st.divider()
        if st.button("Sign out", width="stretch"):
            db.insert_log(user["institution_id"], user["user_id"], "LOGOUT", "Signed out")
            st.session_state.clear()
            st.rerun()
    return choice


def main():
    user = st.session_state.get("user")
    if not user:
        landing()
        return
    if time.time() - st.session_state.get("last_seen", 0) > IDLE_TIMEOUT_MIN * 60:
        st.session_state.clear()
        st.warning("You were signed out after a period of inactivity.")
        landing()
        return
    st.session_state.last_seen = time.time()
    user = refresh_session_user(user)
    if not user:
        st.session_state.clear()
        st.error("Your account or institution is no longer active.")
        landing()
        return
    st.session_state.user = user
    menu = dict(ADMIN_PAGES if user["role"] == "Administrator" else LECTURER_PAGES)
    if is_platform_owner(user):
        menu["Platform"] = ("🌍", admin_pages.platform)
    page = sidebar(user, menu)
    for note in st.session_state.pop("flash", []):
        st.info(note)
    menu[page][1](user)


main()
