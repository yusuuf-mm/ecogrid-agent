"""
scripts/seed_vector_db.py

Idempotent seeding of the grid policy documents into Qdrant.
Usage:
    python scripts/seed_vector_db.py

Safe to run multiple times — point IDs are deterministic, so the second
run replaces the same points with identical content.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.rag.config import settings  # noqa: E402
from services.rag.ingest import ingest_policies  # noqa: E402


def main() -> int:
    policies_dir = Path(ROOT) / settings.POLICIES_DIR
    txt_files = sorted(policies_dir.glob("*.txt"))
    total = ingest_policies(str(policies_dir), settings.QDRANT_COLLECTION)
    print(
        f"Ingested {total} chunks from {len(txt_files)} documents into collection "
        f"'{settings.QDRANT_COLLECTION}'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
