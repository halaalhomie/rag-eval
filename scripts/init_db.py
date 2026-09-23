"""Initialize the database schema.

Usage:
    python scripts/init_db.py          # create extension, tables, indexes (idempotent)
    python scripts/init_db.py --drop   # drop and recreate everything (destroys data)
"""

import argparse

from app.config import get_settings
from app.db.init_db import init_db
from app.utils.logging import configure_logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--drop", action="store_true", help="drop all tables first")
    args = parser.parse_args()
    configure_logging(get_settings().observability.log_level)
    init_db(drop=args.drop)
