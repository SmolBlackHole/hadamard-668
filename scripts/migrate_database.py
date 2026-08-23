"""Backfill versioned state identities without replacing stored payloads."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.output import migrate_database
from src.verify import audit_database


def main() -> int:
    """Migrate one result database and audit the resulting records.

    Returns:
        Zero when the post-migration audit passes, otherwise one.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, nargs="?", default=Path("data/hadamard.db"))
    args = parser.parse_args()
    migration = migrate_database(args.database)
    audit = audit_database(args.database)
    print(
        f"Migrated {migration.checked}: {migration.valid} valid,"
        f" {migration.quarantined} quarantined"
    )
    print(
        f"Audit {audit.valid}/{audit.checked} valid, {audit.quarantined} quarantined,"
        f" {len(audit.failures)} failures"
    )
    return 0 if audit.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
