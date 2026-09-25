"""
business/auth.py
Business Layer — authentication logic.
Handles password hashing and login/registration, calling the Database Layer underneath.
"""

import hashlib
import secrets
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from database import db


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}${hashed.hex()}"


def verify_password(password, stored_hash):
    try:
        salt, hashed_hex = stored_hash.split("$")
    except ValueError:
        return False
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return check.hex() == hashed_hex


def register_user(username, password, role, name=None, email=None):
    if role not in ("Administrator", "Lecturer"):
        raise ValueError("Role must be 'Administrator' or 'Lecturer'")

    existing = db.get_user_by_username(username)
    if existing:
        raise ValueError(f"Username '{username}' already exists")

    password_hash = hash_password(password)
    user_id = db.insert_user(username, password_hash, role, name, email)
    db.insert_log(user_id, "REGISTER", f"New {role} account created")
    return user_id


def authenticate_user(username, password):
    user = db.get_user_by_username(username)
    if not user:
        return None

    if verify_password(password, user["password_hash"]):
        db.insert_log(user["user_id"], "LOGIN", "Successful login")
        return user

    db.insert_log(user["user_id"], "LOGIN_FAILED", "Incorrect password")
    return None