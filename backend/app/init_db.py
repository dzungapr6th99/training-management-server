"""Create the studio database and fill the index tables from the project folders.

    python -m app.init_db            # create the DB if missing, then (re)index whatever data exists (nothing is fine)
    python -m app.init_db --reset    # delete the studio DB first (loses robot/model edits, assignments, job history)

Safe to run repeatedly: it only writes backend/data/studio.db (or $STUDIO_DB). The project folders are read, never modified.
With no datasets/outputs/results/design folders the index simply stays empty; the built-in robot and task list is still loaded."""
import argparse
import sys

from . import config as C
from . import db, indexer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reset", action="store_true", help="delete the existing studio DB first")
    a = ap.parse_args(argv)
    if a.reset and C.DB_PATH.exists():
        C.DB_PATH.unlink()
        print(f"[init_db] deleted {C.DB_PATH}")
    new = not C.DB_PATH.exists()
    db.init()
    r = indexer.reindex()
    print(f"[init_db] {'created' if new else 'updated'} {C.DB_PATH}")
    print(f"[init_db] project root: {C.ROOT}")
    for name, path in (("datasets", C.DATASETS), ("outputs", C.OUTPUTS), ("results", C.RESULTS), ("design", C.DESIGN)):
        print(f"[init_db]   {name:9s} {'found' if path.exists() else 'not found (fine: index stays empty)'}")
    print(f"[init_db] indexed {r['counts']} in {r['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
