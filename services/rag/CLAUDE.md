# CLAUDE.md — services/rag

You are working on the `feat/vector-rag` branch of EcoGrid-Agent.

Your scope is `services/rag/`, `data/policies/`, and `scripts/seed_vector_db.py`.
Do not touch any other service directory.

Read `CONTEXT.md` and `shared/contracts.py` before writing any code.
The `PolicyChunk` and `PolicyQueryResult` types you return are defined in `shared/contracts.py`.

This branch is independent of the solver, API, and ML branches.
Build and test it in complete isolation.

---

## Repo State Awareness

Before writing any code, run these checks and adjust if the reality differs:

```bash
# Confirm shared/contracts.py exists and has PolicyChunk, PolicyQueryResult
grep -n "PolicyChunk\|PolicyQueryResult" shared/contracts.py

# Confirm data/policies/ exists and has .txt files
ls data/policies/

# Confirm Qdrant is reachable (if running Docker)
curl -s http://localhost:6333/collections | python -m json.tool
```

The solver's public API is `optimize_battery_schedule(SolverConstraints)` — not
`GridSolver().solve()`. Do not import from `services.solver` in this branch.
You have no dependency on the solver.

---

## Known Environment Constraint

`sentence-transformers` requires `torch` (~2GB). On a disk-constrained machine,
install with the CPU-only torch wheel to save ~1.5GB:

```bash
# Install CPU-only torch first, then sentence-transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu
poetry add sentence-transformers --no-cache
```

If disk space is still insufficient, use the lightweight alternative:
`fastembed` (no torch dependency, ~50MB, compatible 384-dim vectors):

```bash
poetry add fastembed
```

If using `fastembed`, use model `"BAAI/bge-small-en-v1.5"` (384 dimensions,
same as `all-MiniLM-L6-v2`). The Qdrant collection schema is identical.
Document whichever you use in a comment at the top of `embedder.py`.

---

## What to Build

### `services/rag/config.py`
Settings via `pydantic-settings`:
```python
QDRANT_URL: str = "http://localhost:6333"
QDRANT_COLLECTION: str = "grid_policies"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # or "BAAI/bge-small-en-v1.5" if fastembed
VECTOR_SIZE: int = 384
POLICIES_DIR: str = "data/policies"
TOP_K_RETRIEVAL: int = 1
```

### `services/rag/embedder.py`
One responsibility: embed text into 384-dim vectors.

```python
# If using sentence-transformers:
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# If using fastembed:
from fastembed import TextEmbedding
model = TextEmbedding("BAAI/bge-small-en-v1.5")
```

Public interface (same regardless of backend):
```python
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns list of 384-dim vectors."""
```

Load the model once at module level, not per call.

### `services/rag/ingest.py`

**`chunk_document(text: str, doc_id: str, doc_title: str) -> list[dict]`**

Split on double newlines (`\n\n`). Filter chunks shorter than 30 characters.
Each chunk dict: `{text, doc_id, doc_title, chunk_idx}`.
Do not exceed 300 tokens per chunk — if a paragraph is longer, split on
single newlines as a fallback.

**`ingest_policies(policies_dir: str, collection_name: str) -> int`**

1. Create Qdrant collection if it doesn't exist:
   - vector size: 384, distance: Cosine
2. For each `.txt` file in `policies_dir`:
   - Extract `doc_id` from filename (stem, e.g. `grid_safety_sop_03`)
   - Extract `doc_title` from the first non-empty line of the file
   - Chunk the document
   - Embed all chunks in one batch call
   - Upsert into Qdrant — use a deterministic point ID derived from
     `hash(doc_id + str(chunk_idx))` so re-running is idempotent
3. Return total chunks ingested.

### `services/rag/retriever.py`

**Class: `PolicyRetriever`**

```python
class PolicyRetriever:
    def __init__(self, collection_name: str = settings.QDRANT_COLLECTION):
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.collection = collection_name

    def retrieve(self, query: str, top_k: int = 1) -> list[PolicyChunk]:
        """Embed query, search Qdrant, return top_k PolicyChunk objects."""
```

Return `list[PolicyChunk]` from `shared.contracts`.
If Qdrant is unreachable, raise a clear `ConnectionError` with the URL — do not
return an empty list silently.

### `services/rag/parser.py`

**`extract_buffer_constraint(text: str) -> float | None`**

Parse the minimum SoC buffer from policy text. Handle all of:
- `"30%"` → `0.30`
- `"0.30"` → `0.30`
- `"30 percent"` → `0.30`
- `"thirty percent"` → `0.30` (word-to-number for 5, 10, 20, 30, 35)
- `"5%"` → `0.05`

The key phrase to search for is `"minimum state of charge buffer"` — all five
policy documents contain this phrase followed by the percentage.

Return `None` if no match found. The caller (agent layer) falls back to `0.10`.

### `scripts/seed_vector_db.py`

```python
"""
Run once after Docker services are up:
    python scripts/seed_vector_db.py

Idempotent — safe to run multiple times.
"""
```

Calls `ingest_policies(settings.POLICIES_DIR, settings.QDRANT_COLLECTION)`.
Prints: `Ingested N chunks from M documents into collection 'grid_policies'`.

### `services/rag/tests/test_retriever.py`

Mark tests that need Qdrant with `@pytest.mark.integration` — these are skipped
in CI without Docker.

**Unit tests (no Qdrant needed):**
1. `extract_buffer_constraint("minimum state of charge buffer of 30%")` → `0.30`
2. `extract_buffer_constraint("minimum state of charge buffer of 5 percent")` → `0.05`
3. `extract_buffer_constraint("minimum state of charge buffer of 0.35")` → `0.35`
4. `extract_buffer_constraint("thirty percent")` → `0.30`
5. `extract_buffer_constraint("no constraint here")` → `None`
6. `chunk_document(long_text, "doc_01", "Test Doc")` returns list of dicts
   with correct `doc_id`, `doc_title`, and `chunk_idx` fields.
7. `embed_texts(["hello world"])` returns a list with one vector of length 384.

**Integration tests (require Qdrant):**
8. After seeding, querying `"hospital power reserve"` retrieves a chunk from
   `grid_safety_sop_03` (30% buffer doc).
9. After seeding, querying `"maintenance window reduced reserve"` retrieves a
   chunk from `grid_safety_sop_05` (5% buffer doc).
10. Retrieved `PolicyChunk` has non-empty `text`, `doc_id`, and `score > 0`.

---

## Data: Policy Documents

Five `.txt` files must exist in `data/policies/` before the seed script runs.
They were partially created during the planning phase. Verify they exist:

```bash
ls data/policies/
# Expected:
# grid_safety_sop_01.txt  (standard ops, 10% buffer)
# grid_safety_sop_02.txt  (summer heat advisory, 20% buffer)
# grid_safety_sop_03.txt  (critical infrastructure/hospital, 30% buffer)
# grid_safety_sop_04.txt  (emergency anomaly, 35% buffer)
# grid_safety_sop_05.txt  (maintenance window, 5% buffer)
```

Each file must contain the exact phrase:
`"minimum state of charge buffer of X%"`
where X matches the buffer value above. This is what `extract_buffer_constraint`
parses. If any file is missing or malformed, create or fix it before seeding.

---

## Definition of Done

- All 7 unit tests pass without Qdrant running: `pytest services/rag/tests/ -m "not integration"`
- Integration tests documented as requiring Docker Qdrant in PR description
- `python scripts/seed_vector_db.py` runs without error and prints ingestion summary
- `extract_buffer_constraint` correctly handles all 5 format variants
- PR description includes: which embedding library was used and why, total chunks
  ingested per document
