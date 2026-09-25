"""
app.py
Presentation Layer — Streamlit UI.
Handles login and routes Administrators and Lecturers to their views.
"""

import streamlit as st
from business.auth import authenticate_user
from business.validation import validate_login_input

st.set_page_config(page_title="Learning Analytics System", layout="wide")

if "user" not in st.session_state:
    st.session_state.user = None


def login_page():
    st.title("Learning Analytics System")
    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Log In"):
        errors = validate_login_input(username, password)
        if errors:
            for e in errors:
                st.error(e)
            return

        user = authenticate_user(username, password)
        if user:
            st.session_state.user = user
            st.rerun()
        else:
            st.error("Invalid username or password.")


def admin_dashboard():
    st.title(f"Administrator Dashboard — Welcome, {st.session_state.user['name'] or st.session_state.user['username']}")
    st.info("Full dashboard features (user management, datasets, model comparison) come in Sprint 5.")
    if st.button("Log Out"):
        st.session_state.user = None
        st.rerun()


def lecturer_dashboard():
    st.title(f"Lecturer Dashboard — Welcome, {st.session_state.user['name'] or st.session_state.user['username']}")
    st.info("Full dashboard features (student data, predictions, reports) come in Sprint 5.")
    if st.button("Log Out"):
        st.session_state.user = None
        st.rerun()


def main():
    if st.session_state.user is None:
        login_page()
    elif st.session_state.user["role"] == "Administrator":
        admin_dashboard()
    else:
        lecturer_dashboard()


if __name__ == "__main__":
    main()