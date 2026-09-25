"""
business/validation.py
Business Layer — input validation helpers.
Keeps bad data from reaching the Database or ML layers.
"""

def validate_login_input(username, password):
    errors = []
    if not username or not username.strip():
        errors.append("Username is required.")
    if not password or len(password) < 4:
        errors.append("Password must be at least 4 characters.")
    return errors


def validate_registration_input(username, password, role, email=None):
    errors = validate_login_input(username, password)
    if role not in ("Administrator", "Lecturer"):
        errors.append("Role must be 'Administrator' or 'Lecturer'.")
    if email and "@" not in email:
        errors.append("Email address looks invalid.")
    return errors


def validate_uploaded_file(uploaded_file):
    errors = []
    if uploaded_file is None:
        errors.append("No file was uploaded.")
        return errors
    if not uploaded_file.name.lower().endswith(".csv"):
        errors.append("Only CSV files are supported.")
    return errors