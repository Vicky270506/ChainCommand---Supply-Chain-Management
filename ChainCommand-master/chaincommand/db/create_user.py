"""Create users from the command line (the first admin has to be created this way).

    python -m chaincommand.db.create_user --email you@example.com --name "Your Name" --role admin
    python -m chaincommand.db.create_user --demo        # admin/manager/staff demo accounts
"""

from __future__ import annotations

import argparse
import getpass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..security import hash_password
from .base import get_engine, init_db
from .models import Role, User

_DEMO = (("Demo Admin", "admin@example.com", Role.ADMIN),
         ("Demo Manager", "manager@example.com", Role.MANAGER),
         ("Demo Staff", "staff@example.com", Role.STAFF))


def _add(session: Session, name: str, email: str, role: str, password: str) -> bool:
    email = email.strip().lower()
    if session.scalar(select(User).where(func.lower(User.email) == email)):
        print(f"  skipped {email} (already exists)")
        return False
    session.add(User(name=name, email=email, role=role, password_hash=hash_password(password)))
    print(f"  created {email} ({role})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Create inventory-system users")
    parser.add_argument("--email")
    parser.add_argument("--name")
    parser.add_argument("--role", choices=[Role.ADMIN, Role.MANAGER, Role.STAFF], default=Role.STAFF)
    parser.add_argument("--demo", action="store_true", help="create demo admin, manager and staff accounts")
    args = parser.parse_args()
    if not args.demo and not (args.email and args.name):
        parser.error("give --email and --name, or use --demo")

    password = getpass.getpass("Password (min 8 characters): ")
    if len(password) < 8:
        raise SystemExit("Password must be at least 8 characters.")

    engine = get_engine()
    init_db(engine)
    with Session(engine) as session:
        if args.demo:
            for name, email, role in _DEMO:
                _add(session, name, email, role, password)
        else:
            _add(session, args.name, args.email, args.role, password)
        session.commit()


if __name__ == "__main__":
    main()
