"""
business/auth.py
Business Layer — institution registration, sign-in and account management.

* Every account belongs to exactly one institution.
* Sign-in is by email (unique across the whole platform) + password.
* Passwords are hashed with PBKDF2-SHA256 and a per-user salt; never stored in plain text.
* The institution's first account is its master administrator (is_owner).
"""

import hashlib
import hmac
import os
import re
import secrets

from business import formats
from database import db

ROLES = ("Administrator", "Lecturer")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
CODE_RE = re.compile(r"^[A-Za-z0-9-]{2,15}$")
MIN_PASSWORD = 8
MAX_FAILED_LOGINS = 5
LOCKOUT_MINUTES = 15


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
    return f"{salt}${digest.hex()}"


def verify_password(password, stored_hash):
    try:
        salt, hex_digest = stored_hash.split("$")
    except (ValueError, AttributeError):
        return False
    for rounds in (200_000, 100_000):  # 100k = accounts created by earlier versions
        check = hashlib.pbkdf2_hmac("sha256", (password or "").encode(), salt.encode(), rounds).hex()
        if hmac.compare_digest(check, hex_digest):
            return True
    return False


def password_problems(password):
    p = password or ""
    issues = []
    if len(p) < MIN_PASSWORD:
        issues.append(f"at least {MIN_PASSWORD} characters")
    if not re.search(r"[A-Za-z]", p) or not re.search(r"\d", p):
        issues.append("both letters and numbers")
    return issues


def _check_password(password):
    issues = password_problems(password)
    if issues:
        raise ValueError("Password must have " + " and ".join(issues) + ".")


def _check_email(email):
    if not EMAIL_RE.match((email or "").strip()):
        raise ValueError("Enter a valid email address.")
    if db.get_user_by_username(email):
        raise ValueError("An account with this email already exists.")


def register_institution(name, code, itype, country, admin_name, admin_email, password,
                         matric_format="", matric_example="", student_email_format="", staff_email_domain=""):
    """Create a new institution and its master administrator. Returns (institution_id, user_id)."""
    matric_format, matric_example = (matric_format or "").strip(), (matric_example or "").strip()
    student_email_format = (student_email_format or "").strip().lower()
    staff_email_domain = (staff_email_domain or "").strip().lower()
    problems = formats.check_formats(matric_format, matric_example, student_email_format, staff_email_domain)
    if problems:
        raise ValueError(" ".join(problems))
    name, code, admin_email = (name or "").strip(), (code or "").strip().upper(), (admin_email or "").strip().lower()
    if len(name) < 3:
        raise ValueError("Enter the institution's full name.")
    if not CODE_RE.match(code):
        raise ValueError("Short code must be 2-15 letters, numbers or dashes.")
    if db.institution_code_exists(code):
        raise ValueError(f"The short code '{code}' is already registered. Choose another.")
    if not (admin_name or "").strip():
        raise ValueError("Enter the administrator's full name.")
    _check_email(admin_email)
    if not formats.staff_email_ok(staff_email_domain, admin_email):
        raise ValueError(f"The administrator's email must use your staff domain ({staff_email_domain}).")
    _check_password(password)
    iid = db.create_institution(name, code, itype, country, admin_email, matric_format, matric_example,
                                student_email_format, staff_email_domain)
    uid = db.insert_user(iid, admin_email, hash_password(password), "Administrator", admin_name.strip(),
                         admin_email, "Administration", is_owner=1)
    db.insert_log(iid, uid, "INSTITUTION_REGISTERED", f"{name} ({code})")
    return iid, uid


def add_user(iid, created_by, email, password, role, name, department=None):
    """Master admin / admins add lecturers or other admins to THEIR institution only."""
    if role not in ROLES:
        raise ValueError("Role must be Administrator or Lecturer.")
    if not (name or "").strip():
        raise ValueError("Enter the person's full name.")
    _check_email(email)
    domain = db.get_institution(iid).get("staff_email_domain")
    if not formats.staff_email_ok(domain, email):
        raise ValueError(f"Staff emails for your institution must end with: {domain}")
    _check_password(password)
    uid = db.insert_user(iid, email.strip().lower(), hash_password(password), role, name.strip(),
                         email.strip().lower(), department)
    db.insert_log(iid, created_by, "USER_CREATED", f"{role} {email}")
    return uid


def authenticate_user(email, password):
    user = db.get_user_by_username(email)
    if not user:
        return None, "Incorrect email or password."
    inst = db.get_institution(user["institution_id"])
    if not inst or not inst["is_active"]:
        db.insert_log(user["institution_id"], user["user_id"], "LOGIN_BLOCKED", "Institution suspended")
        return None, "This institution's access is suspended. Contact the platform administrator."
    if not user["is_active"]:
        db.insert_log(user["institution_id"], user["user_id"], "LOGIN_BLOCKED", "Inactive account")
        return None, "This account has been deactivated by your administrator."
    if db.recent_failed_logins(user["user_id"], LOCKOUT_MINUTES) >= MAX_FAILED_LOGINS:
        db.insert_log(user["institution_id"], user["user_id"], "LOGIN_LOCKED", "Too many failed attempts")
        return None, (f"Too many failed attempts. Try again in {LOCKOUT_MINUTES} minutes, or ask your "
                      "administrator to reset your password.")
    if not verify_password(password, user["password_hash"]):
        db.insert_log(user["institution_id"], user["user_id"], "LOGIN_FAILED", "Incorrect password")
        return None, "Incorrect email or password."
    db.touch_last_login(user["user_id"])
    db.insert_log(user["institution_id"], user["user_id"], "LOGIN", "Signed in")
    return user, None


def refresh_session_user(user):
    """Re-read the account on every page load so deactivation takes effect immediately."""
    fresh = db.get_user_by_id(user["user_id"])
    if not fresh or not fresh["is_active"]:
        return None
    inst = db.get_institution(fresh["institution_id"])
    if not inst or not inst["is_active"]:
        return None
    fresh["institution_name"], fresh["institution_code"] = inst["name"], inst["code"]
    fresh["data_mode"] = inst["data_mode"]
    return fresh


def change_password(user_id, old_password, new_password):
    user = db.get_user_by_id(user_id)
    if not user or not verify_password(old_password, user["password_hash"]):
        raise ValueError("Current password is incorrect.")
    _check_password(new_password)
    db.update_password_hash(user_id, hash_password(new_password))
    db.insert_log(user["institution_id"], user_id, "PASSWORD_CHANGED", "User changed password")


def reset_password(iid, admin_id, user_id, new_password):
    target = db.get_user_by_id(user_id)
    if not target or target["institution_id"] != iid:
        raise ValueError("User not found in your institution.")
    _check_password(new_password)
    db.update_password_hash(user_id, hash_password(new_password))
    db.insert_log(iid, admin_id, "PASSWORD_RESET", f"Reset password for {target['username']}")


def set_active(iid, admin_id, user_id, active):
    db.set_user_active(iid, user_id, active)
    db.insert_log(iid, admin_id, "USER_ACTIVATED" if active else "USER_DEACTIVATED", f"user_id {user_id}")


def delete_user(iid, admin_id, user_id):
    if user_id == admin_id:
        raise ValueError("You cannot delete your own account.")
    target = db.delete_user(iid, user_id)
    db.insert_log(iid, admin_id, "USER_DELETED", f"{target['role']} {target['name']} ({target['username']})")
    return target


def transfer_owner(iid, owner_id, new_owner_id):
    owner = db.get_user_by_id(owner_id)
    if not owner or not owner["is_owner"] or owner["institution_id"] != iid:
        raise ValueError("Only the master administrator can transfer this role.")
    db.transfer_owner(iid, owner_id, new_owner_id)
    db.insert_log(iid, owner_id, "OWNER_TRANSFERRED", f"to user_id {new_owner_id}")


def update_settings(iid, admin_id, **fields):
    fields = {k: (v or "").strip() for k, v in fields.items()}
    for k in ("student_email_format", "staff_email_domain"):
        if k in fields:
            fields[k] = fields[k].lower()
    problems = formats.check_formats(fields.get("matric_format"), fields.get("matric_example"),
                                     fields.get("student_email_format"), fields.get("staff_email_domain"))
    if fields.get("name") is not None and len(fields["name"]) < 3:
        problems.append("Enter the institution's full name.")
    if problems:
        raise ValueError(" ".join(problems))
    db.update_institution(iid, **fields)
    db.insert_log(iid, admin_id, "SETTINGS_UPDATED", ", ".join(sorted(fields)))


def is_platform_owner(user):
    """The platform owner (you, the system's builder) is set with PLATFORM_ADMIN_EMAIL."""
    owner = os.environ.get("PLATFORM_ADMIN_EMAIL", "").strip().lower()
    return bool(owner) and user.get("username", "").lower() == owner
