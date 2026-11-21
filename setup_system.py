"""
setup_system.py
Optional local helper. The app needs no setup — open it and register an institution.
This script just prepares things for local testing.

  python setup_system.py                 # create the database tables
  python setup_system.py --demo          # also create a demo institution you can sign in to
  python setup_system.py --reset --demo  # delete EVERY institution first (local only!)

Demo sign-in: admin@demo.las / Demo12345
"""

import argparse

from business import onboarding
from business.auth import register_institution
from database import db

DEMO_EMAIL, DEMO_PASSWORD = "admin@demo.las", "Demo12345"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="delete ALL institutions and data")
    ap.add_argument("--demo", action="store_true", help="create a demo institution with data")
    args = ap.parse_args()
    if args.reset:
        if input("This deletes EVERY institution in the database. Type DELETE to continue: ") != "DELETE":
            return
        db.drop_all()
    db.init_db()
    print(f"Database ready ({db.backend()}).")
    if args.demo:
        user = db.get_user_by_username(DEMO_EMAIL)
        if user:
            iid, uid = user["institution_id"], user["user_id"]
        else:
            iid, uid = register_institution("Demo University", "DEMO", "University", "Nigeria",
                                            "Demo Administrator", DEMO_EMAIL, DEMO_PASSWORD)
        onboarding.load_demo(iid, uid, progress=lambda f, t: print(f"  {f:4.0%}  {t}"))
        print(f"Demo institution ready. Sign in with {DEMO_EMAIL} / {DEMO_PASSWORD}")
    print("Start the app:  streamlit run app.py")


if __name__ == "__main__":
    main()
