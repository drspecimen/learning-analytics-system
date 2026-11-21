"""
business/validation.py
Business Layer — input validation, so bad data never reaches the DB or ML layers.
"""

import re

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_login_input(username, password):
    errors = []
    if not (username or "").strip():
        errors.append("Username is required.")
    if not password:
        errors.append("Password is required.")
    return errors


def validate_registration_input(username, password, role, email=None):
    errors = []
    if not (username or "").strip():
        errors.append("Username is required.")
    if len(password or "") < 6:
        errors.append("Password must be at least 6 characters.")
    if role not in ("Administrator", "Lecturer"):
        errors.append("Role must be Administrator or Lecturer.")
    if email and not EMAIL_RE.match(email):
        errors.append("Email address looks invalid.")
    return errors


def validate_message(subject, message):
    errors = []
    if not (subject or "").strip():
        errors.append("Subject is required.")
    if len((message or "").strip()) < 5:
        errors.append("Message is too short.")
    return errors
