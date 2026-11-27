# Enterprise AI Knowledge Platform --- Master Handover

**Repository:** `enterprise-ai-knowledge-platform`\
**Review date:** 27 November 2026\
**Branch:** `main`\
**Latest committed baseline:** `9f5e6a3` ---
`Add Docker deployment and improve LLM-first RAG pipeline`

> This handover records the repository state reviewed on 27 November
> 2026. It separates verified implementation from risks and
> recommendations. No environment-file contents, credentials, or tokens
> are included.

## 1. Executive summary

This repository contains a **backend-only enterprise document knowledge
retrieval and conversational RAG application**. It accepts PDF, DOCX,
and TXT documents, extracts and chunks text, creates embeddings, stores
chunks in ChromaDB, and exposes REST APIs for document management,
semantic search, question answering, and conversation history.

The retrieval pipeline combines: - Semantic retrieval through Sentence
Transformers and ChromaDB. - BM25 lexical retrieval. - Reciprocal Rank
Fusion (RRF). - Context filtering and context construction. -
Ollama-based LLM answer generation. - Heuristic answer validation and
extractive fallback when generation fails or produces an invalid answer.

The repository also includes SQLite conversation storage, a retrieval
evaluation framework, datasets and reports, 12 automated test modules,
Docker configuration, and GitHub Actions CI.

**Verified test result:** 123 passed, 0 failed, 1 third-party ChromaDB
deprecation warning. Run pytest from `backend`.

**There is no integrated frontend in this repository.** The CORS
`FRONTEND_URL` setting is not evidence of an implemented frontend.

## 2. Repository layout

``` text
enterprise-ai-knowledge-platform/
├── .github/workflows/tests.yml
├── backend/
│   ├── app/
│   │   ├── api/routes/
│   │   │   ├── ask.py
│   │   │   ├── conversations.py
│   │   │   ├── documents.py
│   │   │   ├── health.py
│   │   │   └── search.py
│   │   ├── core/config.py
│   │   ├── services/
│   │   │   ├── answer_service.py
│   │   │   ├── answer_validator.py
│   │   │   ├── context_builder.py
│   │   │   ├── context_filter.py
│   │   │   ├── conversation_service.py
│   │   │   ├── document_parser.py
│   │   │   ├── embedding_service.py
│   │   │   ├── hybrid_search.py
│   │   │   ├── keyword_search.py
│   │   │   ├── llm_service.py
│   │   │   ├── prompt_builder.py
│   │   │   ├── query_rewriter.py
│   │   │   ├── rag_service.py
│   │   │   ├── text_chunker.py
│   │   │   └── vector_store.py
│   │   └── main.py
│   ├── evaluation/
│   │   ├── corpus/ (5 policy documents)
│   │   ├── dataset/ (4 JSON datasets)
│   │   ├── reports/ (15 timestamped reports and v3_top5_rrf40.json)
│   │   ├── evaluator.py
│   │   └── metrics.py
│   ├── tests/ (12 pytest modules)
│   ├── uploads/ (runtime data; Git-ignored)
│   ├── data/ (runtime data; Git-ignored)
│   ├── .dockerignore
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── test_chunker.py
│   ├── test_document_management.py
│   ├── test_embeddings.py
│   ├── test_parser.py
│   ├── test_company_policy.txt
│   └── vector_store_demo.py
├── .gitignore
├── docker-compose.yml
└── README.md
```

Generated `__pycache__` and `.pytest_cache` directories are omitted. Two
environment files were found (`backend/.env` and `backend/app/.env`);
both are Git-ignored and neither is tracked. Their contents were not
inspected and must never be pasted into a chat or committed.

Two empty nested directories were previously observed under
`backend/app/services/backend/app` and `backend/app/services/backend`;
recursive inspection found no files. Their purpose is unknown. Do not
remove them without review.

## 3. Git state and preservation requirements

At the last check: - Branch: `main`. - `HEAD` and `origin/main` both
pointed to `9f5e6a3`. - No changes were staged. - No changes were
committed during this review.

Last observed `git status --short`:

``` text
 M .gitignore
 M backend/app/services/hybrid_search.py
 M backend/app/services/keyword_search.py
 M backend/app/services/text_chunker.py
 M backend/app/services/vector_store.py
 D backend/test_vector_store.py
 M backend/tests/test_hybrid_search.py
 M backend/tests/test_keyword_search.py
 M backend/tests/test_text_chunker.py
?? backend/evaluation/
?? backend/tests/test_evaluation_metrics.py
?? backend/tests/test_evaluator.py
?? backend/tests/test_vector_store.py
?? backend/vector_store_demo.py
```

This is 14 status entries: 8 modified, 1 deleted, 5 untracked. Preserve
all of them. The deleted `backend/test_vector_store.py` was the old
manual demo. Its replacement is `backend/vector_store_demo.py`; the
automated test is now `backend/tests/test_vector_store.py`. Do not
restore the old demo.

Last tracked-file diff summary:

``` text
9 files changed, 752 insertions(+), 171 deletions(-)
```

This excludes untracked evaluation and test files. `git diff --check`
found no whitespace/patch errors; it printed only a warning that
`.gitignore` uses LF and may be converted to CRLF on Windows.

**Never run `git restore`, `git reset`, `git clean`, or deletion/cleanup
commands without reviewing the exact impact and receiving approval. Do
not stage or commit unless explicitly requested.**

## 4. Dependencies and configuration

`backend/requirements.txt`:

``` text
fastapi==0.163.2
uvicorn==0.54.0
pydantic==2.14.3
pydantic-settings==2.16.3
python-dotenv==1.2.4
python-multipart==0.0.34
pypdf==7.1.0
python-docx==1.2.0
sentence-transformers==6.3.0
chromadb==1.5.9
rank-bm25==0.2.2
httpx==0.28.1
```

`requirements-dev.txt` includes the application requirements and
`pytest==9.2.0`.

Observed local runtime: Python 3.14.7, pytest 9.2.0, pluggy 1.7.0, Torch
2.15, ChromaDB 1.5.9. Ollama is an external runtime, not a Python
dependency.

`backend/app/core/config.py` uses Pydantic Settings. Reviewed
defaults: - App name: `Enterprise AI Knowledge Intelligence Platform` -
Version: `1.0.0` - Frontend origin: `http://localhost:3000` - Ollama
URL: `http://localhost:11434` - Ollama model: `llama3.2:1b` - Timeout:
120 seconds - Extra environment values ignored.

The environment file is configured as `.env` relative to the process
working directory. Running from `backend` is the documented convention.
Do not assume `backend/app/.env` is loaded. Never reveal
environment-file values.

## 5. Architecture and data flows

### 5.1 Document ingestion

The upload API: 1. Accepts PDF, DOCX, and TXT. 2. Enforces a 10 MB
limit. 3. Streams to disk in 1 MB blocks and calculates SHA-256. 4.
Checks for duplicate content using stored hashes. 5. Extracts text. 6.
Chunks text. 7. Generates embeddings. 8. Stores chunks, vectors, IDs,
and metadata in ChromaDB.

The route has failure-aware handling: it removes uploads after
processing errors, uses a tombstone rename during deletion, attempts
file restoration if vector deletion fails, and reports cleanup warnings.

Risks: - If Chroma insertion succeeds and a later step fails, the file
may be removed while vectors remain. - Duplicate checking can race
during concurrent uploads. - Extension checks are not file-signature
validation. - Synchronous CPU/database work runs inside async
endpoints. - No authentication or document ownership boundary exists.

### 5.2 Parser

`document_parser.py` supports: - PDF via `pypdf`. - DOCX via
`python-docx`, including paragraphs and tables. - TXT via UTF-8 with BOM
support and replacement behavior for decode errors.

Limitations: - No OCR for scanned PDFs. - DOCX headers, footers, text
boxes, and embedded images are not comprehensively extracted. - Legacy
encodings such as Windows-1252 may be corrupted by replacement
decoding. - Add dedicated tests for corrupt/password-protected PDFs and
legacy text encodings.

### 5.3 Chunker

`text_chunker.py` includes numbered-heading detection, boundary
priorities, short-chunk preservation, line-ending/whitespace
normalization, and heading-aware overlap.

Potential issues: - The numbered-heading regex may misclassify some
numeric sentences. - Heading-aware overlap may exceed configured overlap
in some cases. - Continue testing character offsets and content
preservation.

### 5.4 Embeddings

`embedding_service.py` uses `sentence-transformers/all-MiniLM-L6-v2`,
expected dimension 384, normalized embeddings, a one-entry `lru_cache`,
and single/batch embedding functions.

Changing the embedding model requires re-embedding the stored corpus.
Future improvements: validate actual dimensions, improve model-load
errors, and make the model configurable.

### 5.5 Vector store

`vector_store.py` uses persistent ChromaDB at `backend/data/chroma`,
collection `enterprise_knowledge`, cosine distance, and supports
add/upsert, search, listing/retrieval, deletion, and duplicate-hash
lookup.

An instance-local `_version` counter increments after successful
mutations to support BM25 cache invalidation. It resets after restart
and cannot detect changes made through another store instance or direct
Chroma calls. Use the shared singleton for mutations.

### 5.6 Hybrid retrieval

`hybrid_search.py` and `keyword_search.py` combine: - ChromaDB semantic
retrieval. - BM25 lexical retrieval. - Candidate expansion with
`CANDIDATE_MULTIPLIER = 3`. - Reciprocal Rank Fusion with current
`RRF_K = 40`. - Deduplication by ID and final top-k selection.

BM25 uses Unicode-aware `r"\b\w+\b"` tokenization with `.casefold()`, an
`RLock`, and a cache keyed by `store.version`. The cache is rebuilt
after version changes; it is not cached when version is unavailable.

Potential concurrency issue: store mutation and version increment are
not atomic relative to cache rebuild. Compare RRF K=40 with alternatives
using the same fixed evaluation dataset before changing it.

### 5.7 Context filtering and construction

`context_filter.py` uses ASCII-oriented tokenization, stopwords, lexical
overlap, default relevance threshold 0.15, sentence splitting, and
adjacent-sentence preservation for conditional language.

Limitations: German Unicode handling is incomplete; lexical matching can
discard paraphrases; sentence splitting and conditional logic are
heuristic. Evaluate false negatives before changing the filter.

`context_builder.py` uses a 6,000-character total context limit and
2,000-character per-chunk limit. It includes source labels, filenames,
content, document IDs, chunk indices, RRF scores, and optional distance.
It uses characters rather than tokens. It stops if the next source
exceeds the limit, which may exclude later smaller sources. It does not
independently guarantee deduplication.

### 5.8 Query rewriting

`query_rewriter.py`: - Maximum history: 8 messages. - Maximum query
length: 500 characters. - Uses Ollama to resolve pronouns/references and
create a standalone retrieval query. - Has a rule-based fallback
combining the latest previous user question with the current one. -
Rejects empty, multiline, excessively long, prompt-leaking, and
refusal-like outputs. - Falls back on LLM errors.

Limitations: fallback may lose context beyond the latest user question;
semantic equivalence is not checked; broad exception handling may hide
programming errors.

### 5.9 Prompt, LLM, RAG, and answer validation

`prompt_builder.py` instructs the model to answer only from supplied
excerpts, treats excerpts as untrusted data rather than instructions,
treats conversation history as reference rather than evidence, and
includes recent history (up to six messages), excerpts, and the
question. Prompt wording alone does not guarantee prompt-injection
security.

`llm_service.py` calls Ollama `/api/generate`, non-streaming, using
configurable URL/model/timeout. Temperature default is 0.2; RAG calls at
0. It uses synchronous `httpx.Client` behavior per request and handles
connection, timeout, and HTTP status errors. Future work: reusable
client lifecycle, malformed response/transport handling, model
readiness, and temperature validation.

`rag_service.py`: 1. Validates question/top-k. 2. Checks vector count.
3. Rewrites conversational query. 4. Runs hybrid retrieval. 5. Filters
context. 6. Builds context and sources. 7. Builds prompt. 8. Calls
Ollama. 9. Validates answer. 10. Uses extractive fallback on
generation/validation failure. 11. Returns an unsupported response if
evidence is insufficient.

The returned model indicator can be `ollama`, `extractive`, or `none`.

Risks: broad exception handling can disguise programming errors as LLM
failures; retrieval uses rewritten query while prompt uses original
question; grounding threshold 0.5 is heuristic; no explicit retrieval
relevance threshold; fallback quality needs separate evaluation.

`answer_validator.py` checks empty output, some prompt leakage patterns,
and lexical grounding. Grounding is overlap of answer non-stopword
tokens with context divided by answer non-stopword tokens. It is not
semantic fact verification. Default threshold is 0.35; RAG uses 0.5. It
cannot reliably catch contradiction, negation, numeric mismatch, or
unsupported individual claims.

`answer_service.py` is an extractive fallback. It uses lexical token
matching, simple suffix normalization, section-heading heuristics,
recent user-history tokens, and returns up to four top-scoring
sentences. It does not provide semantic entailment, condition/negation
verification, or sentence-level citations.

### 5.10 Conversations

`conversation_service.py` uses SQLite at
`backend/data/conversations.db`.

Tables: - `conversations`: UUID ID and creation timestamp. - `messages`:
integer ID, conversation foreign key, role, content, timestamp. - Index
on `(conversation_id, id)`.

Important correctness issue: `PRAGMA foreign_keys=ON` is set in the
delete method but not on every connection. SQLite FK enforcement is
connection-specific and defaults off. Enable it for every connection
before relying on cascade behavior.

Other limitations: no authentication/ownership, pagination, or retention
policy; SQLite concurrency constraints; UUIDs are identifiers, not
authorization.

Conversation message flow: verify conversation, load history, save user
message, generate RAG answer, save assistant message, return
answer/sources/model. A failure after saving the user message can leave
a user-only message.

## 6. API endpoints

  ----------------------------------------------------------------------------------------------------
  Method                  Endpoint                                             Purpose
  ----------------------- ---------------------------------------------------- -----------------------
  GET                     `/`                                                  Root/status

  GET                     `/health`                                            Basic health response

  POST                    `/api/v1/documents/upload`                           Upload/index document

  GET                     `/api/v1/documents/`                                 List documents

  GET                     `/api/v1/documents/{document_id}`                    Get document

  DELETE                  `/api/v1/documents/{document_id}`                    Delete document/vectors

  POST                    `/api/v1/search/`                                    Semantic search

  POST                    `/api/v1/ask`                                        Stateless question
                                                                               answering

  POST                    `/api/v1/conversations`                              Create conversation

  POST                    `/api/v1/conversations/{conversation_id}/messages`   Add conversational
                                                                               message

  GET                     `/api/v1/conversations/{conversation_id}`            Get conversation

  DELETE                  `/api/v1/conversations/{conversation_id}`            Delete conversation
  ----------------------------------------------------------------------------------------------------

Notes: - `/api/v1/ask` accepts a question of 3--2,000 characters and
top-k 1--10 (default 5); it is stateless. - `/api/v1/search/` is
semantic only, not hybrid. Similarity derived from distance is not a
calibrated probability. - `/health` checks that FastAPI responds; it
does not check ChromaDB, SQLite, or Ollama. - Some async routes call
synchronous services. - Some error handlers expose raw internal
details. - No authentication, authorization, or multi-tenant isolation
exists.

## 7. Evaluation framework and findings

### 7.1 Inventory

Evaluation code: - `backend/evaluation/metrics.py` -
`backend/evaluation/evaluator.py`

Five corpus documents: - `employee_benefits.txt` -
`expense_reimbursement.txt` - `information_security.txt` -
`it_support_policy.txt` - `remote_work_policy.txt`

Four datasets: - `retrieval_questions.json` -
`retrieval_questions_backup.json` - `retrieval_questions_v2.json` -
`retrieval_questions_v3.json`

Reports: 15 timestamped JSON reports plus `v3_top5_rrf40.json`. Preserve
all reports until a canonical baseline is selected. The backup dataset
has the same observed byte size as the original, but content equality
was not established.

### 7.2 Metrics and evaluator

`metrics.py` implements Precision@k, Recall@k, Reciprocal Rank, NDCG@k,
duplicate rate, and content normalization. NDCG uses gain `2^grade - 1`
and standard logarithmic discount.

Potential edge cases: duplicate IDs in precision, empty documents in
duplicate-rate denominator, invalid grades, empty NDCG input, and
lower-vs-casefold normalization.

`evaluator.py` loads JSON datasets, runs retrieval per question,
calculates per-question and aggregate metrics, macro/category averages,
latency, and saves timestamped JSON reports.

Improvements: dataset schema validation, RRF K validation, ensure custom
services do not produce misleading RRF metadata, and record dataset
hash/model/corpus size/git commit/configuration. Separate cold-start
from warm latency.

### 7.3 V3 dataset

Previously inspected V3 facts: - 50 questions, Q001--Q050. - Categories
include HR policy, IT support, remote work, employee benefits,
information security, expense reimbursement, and cross-domain. - 19
unique relevant chunk IDs; all existed in the inspected ChromaDB
state. - Q048 has a `relevant_document_ids` mismatch: an IT Support
document is listed but no labelled relevant chunk from it exists. - No
explicit unanswerable/out-of-scope questions. - Some repeated question
patterns. - Chunk IDs are brittle if chunking changes. -
`relevant_document_ids` is not used by current metrics.

### 7.4 Named baseline `v3_top5_rrf40.json`

-   50 questions; top_k 5; RRF K 40.
-   Mean precision: 0.276.
-   Mean recall: 0.98.
-   MRR: 0.8857.
-   NDCG: 0.8914.
-   Duplicate rate: 0.
-   First-query latency: about 1,484.7 ms.
-   Subsequent mean latency: about 14.36 ms.
-   Overall mean latency: about 43.77 ms.

Interpretation: recall/NDCG are strong on this labelled set, but
precision is comparatively low. Cross-domain was weakest but has only
four questions. Cold-start latency is much higher than warm latency. Do
not tune retrieval parameters based only on aggregate metrics; inspect
per-query failures.

Previously inspected weaker examples: Q013 (IT support hardware issue;
correct chunk fifth), Q048 (request-content question; correct chunk
fourth), Q040 (laptop breaks while remote; missing IT support chunk),
and Q010 (locked account; correct chunk third).

### 7.5 Corpus/database audit

Previously observed: - ChromaDB had 20 vectors across 7 documents. - All
seven indexed TXT files had SHA-256 hashes matching Chroma metadata. -
Three PDFs existed in uploads but not in ChromaDB; two had identical
hashes. Their origin was not established. - `company_policy.txt` and
`test_company_policy.txt` overlap. - `test_company_policy.txt` includes
a parental-leave statement saying entitlements are not specified and
employees should contact HR. This can test unsupported-answer
behavior. - One parental-leave chunk was not used by the labelled
dataset.

Do not delete or reindex files without checking current database state
and getting approval.

## 8. Manual utilities

These are not part of pytest:

  -------------------------------------------------------------------------------
  File                            Purpose                 Caution
  ------------------------------- ----------------------- -----------------------
  `test_chunker.py`               Prints chunk            Demonstration; no
                                  boundaries, sizes,      systematic assertions
                                  offsets, previews

  `test_document_management.py`   Lists documents, gets   Read-only diagnostic
                                  first doc, prints
                                  vector count

  `test_embeddings.py`            Generates single/batch  May load/download model
                                  embeddings and checks
                                  dimensions

  `test_parser.py`                Parses files in         Working-directory
                                  relative `uploads/`     dependent; no
                                                          expected-text
                                                          assertions

  `vector_store_demo.py`          Inserts sample vectors  **Mutates persistent
                                  and runs semantic       ChromaDB** using fixed
                                  search                  IDs; no cleanup
  -------------------------------------------------------------------------------

`vector_store_demo.py` uses IDs `test_doc_chunk_1` etc. Repeated runs
upsert those IDs and do not clean them up. It demonstrates semantic
search, not hybrid retrieval. Use an isolated temporary
database/collection if needed.

## 9. Docker and deployment

### 9.1 Dockerfile

`backend/Dockerfile`: - `python:3.14-slim`. - Sets
`PYTHONDONTWRITEBYTECODE=1` and `PYTHONUNBUFFERED=1`. - Installs
`build-essential`. - Installs CPU-only PyTorch. - Installs
`requirements.txt`. - Copies `app/`. - Creates `/app/data/chroma` and
`/app/uploads`. - Exposes 8000 and runs Uvicorn on `0.0.0.0:8000`.

Ollama is not installed in the image.

Hardening: run as non-root, add health check, assess whether build tools
can be removed, consider pinning base image digest, and consider
persistent Sentence Transformers model cache.

### 9.2 Docker Compose

Root `docker-compose.yml` defines only `backend`: - Build context
`./backend`. - Port `8000:8000`. - Frontend origin
`http://localhost:3000`. - Ollama URL
`http://host.docker.internal:11434`. - Model `llama3.2:1b`, timeout
120. - Bind mounts `./backend/data:/app/data` and
`./backend/uploads:/app/uploads`. - Restart policy `unless-stopped`.

Ollama runs on the host. There is no frontend service. Restrict API
network exposure before deployment outside a trusted local machine.

### 9.3 Docker ignore

Current `backend/.dockerignore`:

``` text
.venv/
__pycache__/
.pytest_cache/
.env
data/
uploads/
tests/
*.pyc
*.pyo
.git/
```

Recommended hardening is to explicitly exclude nested environment files
and variants, for example:

``` text
.env
.env.*
**/.env
**/.env.*
!.env.example
```

Validate patterns against the actual Docker build context. The nested
`backend/app/.env` deserves attention because the Dockerfile copies
`app/`. Never copy real secrets into the image.

## 10. GitHub Actions CI

`.github/workflows/tests.yml`: - Triggers on push to `main` and pull
requests targeting `main`. - Runs on `ubuntu-latest`. - Uses
`actions/checkout@v7` and `actions/setup-python@v7`. - Sets Python
3.14. - Caches pip based on backend requirements. - Installs CPU-only
PyTorch and verifies it. - Installs `backend/requirements-dev.txt`. -
Runs `python -m pytest -v` with working directory `backend`.

It does not start Ollama. Tests can run without Ollama when LLM
interactions are mocked. Workflow configuration was inspected, but a
successful remote GitHub Actions run was not independently verified.

Potential improvements: job timeout, lint/format checks, coverage,
dependency scanning, test-result artifacts, and verification that CI
does not require live model downloads.

## 11. README status

`README.md` has a useful structure but needs updates: - It lists PDF and
DOCX but omits TXT. - It says extractive answering happens before LLM
generation; current behavior is extractive fallback after LLM
failure/invalid output. - It reports 80 tests; verified suite is 123
passed. - It lists evaluation, Docker deployment, and GitHub Actions CI
as future work despite their implementation. - Project tree omits
evaluation and several current files. - Add Docker setup instructions
and clarify backend-only status.

The underlying README is valid UTF-8. A Python check found no `Â`,
`â€™`, or replacement characters, and confirmed Unicode middle-dot and
box-drawing characters. The strange display from PowerShell
`Get-Content` was a console display issue, not file corruption. Preserve
Unicode.

## 12. Test results and run commands

### Verified test run

Run from `backend`:

``` powershell
Set-Location .\backend
python -m pytest -v
```

Verified 27 November 2026:

``` text
collected 123 items
123 passed, 1 warning in 83.58s
```

No failures or skips. The warning originates in ChromaDB telemetry:
`asyncio.iscoroutinefunction` is deprecated and slated for removal in
Python 3.16. It is a third-party warning.

Running `python -m pytest .\backend\tests -v` from repository root
caused 12 collection errors (`No module named 'app'` /
`No module named 'evaluation'`). This is due to the expected working
directory, not an application test failure. CI also runs from `backend`.

### Common commands

From repository root:

``` powershell
Set-Location .\backend
```

Run all tests:

``` powershell
python -m pytest -v
```

Run a specific module:

``` powershell
python -m pytest .\tests\test_rag_service.py -v
```

Run matching tests:

``` powershell
python -m pytest .\tests\test_rag_service.py -k "fallback" -v
```

Run API locally (from `backend`):

``` powershell
python -m uvicorn app.main:app --reload
```

Endpoints: - `http://127.0.0.1:8000` - Swagger:
`http://127.0.0.1:8000/docs` - ReDoc: `http://127.0.0.1:8000/redoc`

Ollama model setup:

``` powershell
ollama pull llama3.2:1b
```

Ensure Ollama is running for real LLM requests.

From repository root, Docker:

``` powershell
docker compose build
docker compose up -d
docker compose logs -f backend
docker compose down
```

Do not add volume deletion flags unless intentionally deleting
persistent data.

## 13. Known limitations and risks

### Security

-   No authentication/authorization.
-   No document ownership or multi-tenant isolation.
-   Conversation UUIDs do not authorize access.
-   Some raw internal errors may be exposed.
-   Upload validation is not file-signature validation.
-   Prompt-injection defense relies partly on prompt instructions.

### Retrieval and answer quality

-   Context filter is lexical and can discard semantic matches.
-   Answer grounding is lexical, not semantic fact verification.
-   Numeric mismatch, negation, contradiction, and claim-level support
    are not reliably checked.
-   No dedicated reranker or calibrated retrieval threshold.
-   Evaluation dataset lacks unanswerable examples.
-   Chunk IDs are brittle across chunker changes.
-   Cross-domain sample is small.

### Reliability

-   Sync operations occur inside async routes.
-   Broad exception handling may hide programming errors.
-   SQLite foreign keys are not enabled on every connection.
-   No conversation pagination or retention.
-   Health endpoint does not check dependencies.
-   No structured observability.
-   BM25 cache versioning is process-local.
-   Ollama is an external dependency.

### Document handling

-   No OCR for scanned PDFs.
-   DOCX extraction does not cover every document element.
-   Legacy text encodings may be damaged.
-   Possible file/vector inconsistency after partial ingestion failure.
-   Duplicate upload check can race.

## 14. Recommended roadmap

### Priority 1 --- Correctness and evaluation reproducibility

1.  Enable SQLite foreign keys on every connection.
2.  Add regression tests for cascade deletion.
3.  Test upload failure after successful vector insertion and reconcile
    file/vector state.
4.  Strengthen evaluation dataset schema validation.
5.  Correct the Q048 metadata mismatch.
6.  Add unanswerable/out-of-scope questions.
7.  Add metric edge-case tests.
8.  Record dataset hash, corpus size, model, RRF K, top-k, and git
    commit in evaluation reports.
9.  Identify canonical reports without deleting history.

### Priority 2 --- Retrieval and answer quality

1.  Inspect per-question retrieval failures before changing RRF K.
2.  Compare RRF K=40 with alternatives on a fixed dataset.
3.  Add Unicode/German-aware tokenization to context filtering and
    answer validation.
4.  Evaluate lexical filter false negatives.
5.  Improve answer verification for numeric mismatch, negation,
    contradiction, and claim support.
6.  Evaluate extractive fallback separately.
7.  Consider reranking after establishing a stable baseline.

### Priority 3 --- Security and reliability

1.  Add authentication and authorization.
2.  Add document ownership/access control and tenant isolation.
3.  Avoid exposing raw internal exceptions.
4.  Validate uploads by file content/signature.
5.  Add readiness checks for ChromaDB, SQLite, and Ollama.
6.  Add structured logging and request correlation IDs.
7.  Consider background tasks for CPU-intensive ingestion.
8.  Improve HTTP client lifecycle and error handling.

### Priority 4 --- Deployment

1.  Harden `.dockerignore` for nested environment files.
2.  Add non-root Docker user and health check.
3.  Review image size/build tools.
4.  Consider resource limits and restricted port binding.
5.  Verify model caching and persistent storage.
6.  Verify GitHub CI and add linting, coverage, and security scanning.

### Priority 5 --- Product

1.  Build a frontend (not currently present in this repository).
2.  Add document-level access controls and user management.
3.  Add conversation pagination and retention.
4.  Improve source citations and explainability.
5.  Improve multilingual support.
6.  Add deployment documentation and monitoring.

## 15. Rules for the next development session

1.  Inspect actual files before editing; do not assume this document
    overrides current code.
2.  Use PowerShell, one command at a time; explain the command and wait
    for output.
3.  Preserve all current uncommitted changes.
4.  Never reset, restore, clean, delete, or overwrite without reviewing
    exact impact and receiving approval.
5.  Do not commit unless explicitly requested after review.
6.  Never ask the user to paste `.env` contents.
7.  Run targeted tests after each change and the full suite before
    completion.
8.  Distinguish proven facts from hypotheses and recommendations.
9.  Run pytest from `backend`.
10. Keep work focused on this backend repository; do not claim a
    frontend exists.

## 16. Suggested first task for the next session

Start by checking current `git status --short`, branch, and latest
commit. If the working tree is still the same, begin with the confirmed
SQLite foreign-key issue: - Inspect the connection factory in
`conversation_service.py`. - Enable foreign keys for every SQLite
connection. - Add a regression test that verifies deleting a
conversation cascades to its messages. - Run conversation/API tests,
then the full suite.

## 17. Handover checklist

-   [x] Backend architecture and API routes reviewed.
-   [x] Core RAG services reviewed.
-   [x] Retrieval/vector-store changes reviewed.
-   [x] Manual scripts reviewed.
-   [x] Evaluation code, datasets, corpus, and reports inventoried.
-   [x] Dependencies/configuration reviewed.
-   [x] Dockerfile, `.dockerignore`, and Compose reviewed.
-   [x] GitHub Actions workflow reviewed.
-   [x] README reviewed and UTF-8 checked.
-   [x] Git status, branch, and latest commit captured.
-   [x] Full automated suite rerun: 123 passed, 1 warning.
-   [x] `git diff --check` completed; only line-ending warning observed.
-   [ ] README updated.
-   [ ] Docker ignore hardening implemented and verified.
-   [ ] Priority 1 improvements implemented.

**End of handover.**
