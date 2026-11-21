"""
business/notifications.py
Sends messages to students and records every message in the Notifications table.

Email goes out through SMTP when these environment variables are set:
  SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASSWORD, SMTP_FROM
Without them, messages are recorded with status "Recorded" (ready for a gateway).
SMS and In-App are recorded for the institution's SMS gateway / LMS inbox to deliver.
"""

import os
import smtplib
from email.message import EmailMessage

from business.validation import validate_message
from database import db

CHANNELS = ["Email", "SMS", "In-App"]

TEMPLATES = {
    "Check-in (friendly)": (
        "Checking in on {course}",
        "Dear {first_name},\n\nWe noticed you have been less active in {course} recently. "
        "Is everything okay? Your lecturer is happy to help you catch up — reply to this message "
        "or visit during office hours.\n\nBest regards,\n{sender}",
    ),
    "Missed assessments": (
        "Outstanding assessments in {course}",
        "Dear {first_name},\n\nOur records show some assessments in {course} have not been "
        "submitted. Please log in to the LMS, review the pending tasks and reach out if you "
        "need an extension or support.\n\nRegards,\n{sender}",
    ),
    "Academic support invitation": (
        "Invitation: academic support session",
        "Dear {first_name},\n\nYou are invited to an academic support session for {course}. "
        "These sessions help students strengthen key topics before the examinations. "
        "Please confirm your attendance.\n\nRegards,\n{sender}",
    ),
    "Commendation": (
        "Great work in {course}",
        "Dear {first_name},\n\nWell done on your excellent engagement and performance in "
        "{course}. Keep it up!\n\nWarm regards,\n{sender}",
    ),
}


def smtp_configured():
    return bool(os.environ.get("SMTP_HOST"))


def _send_email(to_addr, subject, body):
    msg = EmailMessage()
    msg["From"] = os.environ.get("SMTP_FROM", os.environ.get("SMTP_USER", ""))
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", 587)), timeout=20) as s:
        s.starttls()
        if os.environ.get("SMTP_USER"):
            s.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASSWORD", ""))
        s.send_message(msg)


def render_template(name, student, course, sender):
    subject, body = TEMPLATES[name]
    ctx = {"first_name": student.get("first_name", "Student"), "course": course or "your course",
           "sender": sender}
    return subject.format(**ctx), body.format(**ctx)


def send(iid, student_id, sender_id, channel, subject, message):
    errors = validate_message(subject, message)
    if errors:
        raise ValueError(" ".join(errors))
    student = db.get_student(iid, student_id)
    if not student:
        raise ValueError("Student not found.")
    status = "Recorded"
    if channel == "Email" and smtp_configured() and student.get("email"):
        try:
            _send_email(student["email"], subject, message)
            status = "Delivered"
        except Exception as exc:  # keep the record even if delivery fails
            status = f"Failed: {exc.__class__.__name__}"
    nid = db.insert_notification(iid, student_id, sender_id, channel, subject, message, status)
    db.insert_log(iid, sender_id, "NOTIFICATION_SENT", f"{channel} to student_id {student_id} ({status})")
    return nid, status


def send_bulk(iid, student_ids, sender_id, channel, template_name, course, sender_name):
    sent = 0
    for sid in student_ids:
        student = db.get_student(iid, int(sid))
        if not student:
            continue
        subject, body = render_template(template_name, student, course, sender_name)
        send(iid, int(sid), sender_id, channel, subject, body)
        sent += 1
    return sent
