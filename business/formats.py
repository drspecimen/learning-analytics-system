"""
business/formats.py
Each institution's own identifier formats, set at registration and editable in Settings.

Matric number format — a pattern where
    #  = one digit        ?  = one letter        *  = any characters (one or more)
    anything else is literal (letters, slashes, dashes, dots...)
  Examples:   NOU#########         -> NOU212289345
              ???/???/##/####      -> DIG/CSC/24/1045
              U##/##/*             -> U20/55/ACC1234

Student email format — a template with placeholders:
    {matric}  matric number with symbols removed, lower case (NOU212289345 -> nou212289345)
    {first} {last}  first name / surname, lower case
  Examples:   {matric}@noun.edu.ng          {first}.{last}@student.unilag.edu.ng

Staff email domain(s) — e.g. noun.edu.ng   (several allowed, comma separated)
"""

import re

import numpy as np


def pattern_to_regex(pattern):
    out = []
    for ch in pattern or "":
        if ch == "#":
            out.append(r"\d")
        elif ch == "?":
            out.append("[A-Za-z]")
        elif ch == "*":
            out.append(".+")
        else:
            out.append(re.escape(ch))
    return "^" + "".join(out) + "$"


def matric_matches(pattern, value):
    if not pattern:
        return True
    return re.match(pattern_to_regex(pattern), str(value).strip(), flags=re.IGNORECASE) is not None


def example_from_pattern(pattern, rng=None, n=None):
    """A matric number that fits the pattern (used for demo data and previews)."""
    rng = rng or np.random.default_rng(0 if n is None else n)
    letters = "ABCDEFGHJKLMNPRSTUVWXYZ"
    out = []
    for ch in pattern:
        if ch == "#":
            out.append(str(rng.integers(0, 10)))
        elif ch == "?":
            out.append(letters[int(rng.integers(0, len(letters)))])
        elif ch == "*":
            out.append("".join(str(rng.integers(0, 10)) for _ in range(4)))
        else:
            out.append(ch)
    return "".join(out)


def _clean(text):
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def student_email(template, matric, first="", last=""):
    if not template:
        return ""
    return (template.replace("{matric}", _clean(matric)).replace("{first}", _clean(first))
            .replace("{last}", _clean(last)))


def staff_domains(text):
    return [d.strip().lower().lstrip("@") for d in (text or "").split(",") if d.strip()]


def staff_email_ok(domains_text, email):
    domains = staff_domains(domains_text)
    if not domains:
        return True
    email = (email or "").strip().lower()
    return any(email.endswith("@" + d) or email.endswith("." + d) for d in domains)


def check_formats(matric_format, matric_example, email_template, staff_domain_text):
    """Validate settings entered by an administrator. Returns a list of problems."""
    problems = []
    if matric_format:
        if len(matric_format) > 40:
            problems.append("Matric format is too long.")
        if matric_example and not matric_matches(matric_format, matric_example):
            problems.append(f"The example matric number '{matric_example}' does not fit the format "
                            f"'{matric_format}'.")
    if email_template:
        if "@" not in email_template or email_template.count("@") != 1:
            problems.append("Student email format must contain exactly one @.")
        if not any(p in email_template for p in ("{matric}", "{first}", "{last}")):
            problems.append("Student email format needs a placeholder: {matric}, {first} or {last}.")
    for d in staff_domains(staff_domain_text):
        if "." not in d or " " in d:
            problems.append(f"'{d}' is not a valid email domain.")
    return problems


def derive_pattern(example):
    """Suggest a format from one example: digits become #, everything else is kept literally.
    NOU212289345 -> NOU#########     DIG/CSC/24/1045 -> DIG/CSC/##/####"""
    return re.sub(r"\d", "#", (example or "").strip())


def from_samples(sample_matric, sample_student_email, sample_staff_email):
    """Work out an institution's formats from one sample of each.
    Returns (matric_format, student_email_format, staff_email_domain, notes)."""
    notes = []
    matric = (sample_matric or "").strip()
    s_email = (sample_student_email or "").strip().lower()
    staff = (sample_staff_email or "").strip().lower()
    m_fmt = derive_pattern(matric) if matric else ""
    s_fmt = ""
    if s_email:
        if s_email.count("@") != 1:
            notes.append("The sample student email is not a valid email address.")
        else:
            local, domain = s_email.split("@")
            if matric and _clean(local) == _clean(matric):
                s_fmt = "{matric}@" + domain
            else:
                notes.append("Student emails could not be linked to the matric number, so they will not be "
                             "filled in automatically. You can set the format later under Settings.")
    staff_domain = ""
    if staff:
        if staff.count("@") != 1 or "." not in staff.split("@")[-1]:
            notes.append("The sample staff email is not a valid email address.")
        else:
            staff_domain = staff.split("@")[-1]
    return m_fmt, s_fmt, staff_domain, notes
