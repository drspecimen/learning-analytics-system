"""
business/onboarding.py
The institution set-up flows, each scoped to ONE institution:

  load_demo         fill the institution with demo data, train, score (new registrations)
  reset             wipe the institution's academic data (accounts are kept)
  load_own_data     reset -> import the institution's file -> train on its history -> score
  add_data          import more rows without wiping, then re-score (and retrain if asked)
"""

from business import analytics, importer
from database import db
from ml import train
from scripts.generate_institution_data import generate


def _noop(fraction, text):
    pass


def load_demo(iid, user_id=None, n_students=400, progress=None):
    step = progress or _noop
    db.wipe_institution_data(iid)
    generate(iid, n_students=n_students, verbose=False, progress=lambda f, t: step(f * 0.6, t))
    report = train.train(iid, cutoff_week=8, verbose=False, user_id=user_id,
                         progress=lambda f, t: step(0.6 + f * 0.35, t))
    step(0.97, "Scoring current students")
    n = analytics.score_current(iid, user_id)
    db.insert_log(iid, user_id, "DEMO_LOADED", f"{n} enrollments scored")
    step(1.0, "Done")
    return report, n


def reset(iid, user_id=None):
    db.wipe_institution_data(iid)
    db.insert_log(iid, user_id, "SYSTEM_RESET", "All academic data and models deleted")


def load_own_data(iid, df, file_name, cutoff_week, user_id=None, progress=None):
    """df must already be validated by importer.validate."""
    step = progress or _noop
    step(0.02, "Wiping previous data")
    reset(iid, user_id)
    step(0.08, f"Importing {len(df):,} rows")
    res = importer.import_dataframe(iid, df, user_id, file_name, cutoff_week)
    db.set_institution_mode(iid, "own")
    out = {"import": res, "report": None, "scored": 0, "train_error": None}
    try:
        out["report"] = train.train(iid, cutoff_week=cutoff_week, verbose=False, user_id=user_id,
                                    progress=lambda f, t: step(0.3 + f * 0.6, t))
        step(0.93, "Scoring current students")
        out["scored"] = analytics.score_current(iid, user_id)
    except ValueError as exc:
        out["train_error"] = str(exc)
    step(1.0, "Done")
    return out


def add_data(iid, df, file_name, cutoff_week, user_id=None, retrain=False, progress=None):
    step = progress or _noop
    step(0.05, f"Importing {len(df):,} rows")
    res = importer.import_dataframe(iid, df, user_id, file_name, cutoff_week)
    if db.get_institution(iid)["data_mode"] == "empty":
        db.set_institution_mode(iid, "own")
    out = {"import": res, "report": None, "scored": 0, "train_error": None}
    try:
        if retrain or not analytics.ml_service.model_available(iid):
            out["report"] = train.train(iid, cutoff_week=cutoff_week, verbose=False, user_id=user_id,
                                        progress=lambda f, t: step(0.3 + f * 0.6, t))
        step(0.93, "Scoring current students")
        out["scored"] = analytics.score_current(iid, user_id)
    except (ValueError, FileNotFoundError) as exc:
        out["train_error"] = str(exc)
    step(1.0, "Done")
    return out
