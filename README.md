# rag-pinecone-ollama

A production-grade, locally-running RAG (Retrieval-Augmented Generation) pipeline built with a layered / SOLID architecture. Documents land in a **landing zone**, pass validation and pre-processing (including optional PII redaction), are chunked and embedded via **Ollama**, and are stored in a pluggable vector database (**Pinecone**, **ChromaDB**, or **Qdrant**). A **Streamlit UI** and **CLI** expose ingest, watch, ask, eval, debug, corpus-only testing, and SQLite chunk browsing.

---

## Tech stack

| Component | Technology | Where wired |
| --- | --- | --- |
| LLM & embeddings | Ollama (local) | `src/infrastructure/embeddings/ollama_embedding_provider.py`, `src/infrastructure/generation/ollama_answer_generator.py` |
| Vector stores | Pinecone · ChromaDB · Qdrant | `src/infrastructure/vector_store/` + `src/factories/adapter_factory.py` |
| Relational audit store | SQLite (stdlib) | `src/infrastructure/relational_store/sqlite_relational_store.py` |
| Document loaders | pypdf · PyMuPDF · pdfplumber · python-docx · python-pptx · BeautifulSoup4 · JSON · Tesseract/EasyOCR/PaddleOCR | `src/infrastructure/loaders/`, `src/infrastructure/extraction/`, `src/infrastructure/ocr/` |
| Chunking | LangChain Text Splitters + semantic router | `src/infrastructure/chunking/recursive_chunker.py`, `semantic_chunker.py`, `document_router.py` |
| PII redaction | Regex (offline) | `src/infrastructure/pii/regex_pii_anonymizer.py` |
| Landing-zone watcher | watchdog | `src/infrastructure/landing_zone/file_system_watcher.py` |
| UI | Streamlit | `app.py` |
| CLI | argparse | `main.py` |
| Logging | rich + stdlib rotating files | `src/infrastructure/logging/rich_logger.py` |
| Config | `.env` + optional YAML profiles | `src/factories/settings_factory.py`, `src/factories/yaml_config_loader.py` |

Only **`langchain-text-splitters`** is imported from the LangChain ecosystem (`recursive_chunker.py`). Packages such as `langchain-pinecone` appear in `requirements.txt` but are **not** imported by application code.

---

## System dependencies (non-pip)

These are invoked via subprocess or expected on `PATH`. Full ingest works without them only for formats that do not need them.

| Dependency | Required for | Config / env |
| --- | --- | --- |
| **Ollama** (running locally) | Embedding + generation (`ask`, full `ingest`, `eval`) | `OLLAMA_BASE_URL` (default `http://localhost:11434`) |
| **Tesseract OCR** | PDF/image OCR fallbacks, `OcrLoader` | `TESSERACT_CMD` (path to binary on Windows), `TESSERACT_LANG` (default `eng`; e.g. `eng+urd`) |
| **LibreOffice** (`soffice`) | Legacy `.ppt` → `.pptx` conversion | `LIBREOFFICE_PATH` (default `soffice`), `LIBREOFFICE_TIMEOUT_SECONDS` (default `120`) |

EasyOCR and PaddleOCR are **Python packages** (in `requirements.txt`) used as OCR fallbacks when configured in YAML — no separate binary install beyond what those wheels pull in.

---

## How to run

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama and pull models (required for ask / full ingest)
ollama pull nomic-embed-text
ollama pull gemma3

# 4. Configure secrets and defaults
#    Create a .env file in the project root (see Environment variables below).
#    PINECONE_API_KEY is required only when VECTOR_STORE_TYPE=pinecone.

# 5. (Optional) Add YAML config — see Configuration
#    config/default.yml          base profile (optional; merged first)
#    config/user_<name>.yml      per-user overrides (optional)

# 6. Add documents to the landing zone
#    data/landing_zone/

# 7a. Streamlit UI
streamlit run app.py

# 7b. CLI
python main.py ingest                              # batch ingest (default: data/landing_zone)
python main.py build-corpus                        # parse only — no Ollama / vector DB
python main.py ask "What is the Magi story about?"
python main.py eval
python main.py debug "Who cut Della's hair?"
python main.py watch                               # landing-zone watcher
python main.py watch --recursive

# Optional: select a YAML profile on the CLI
python main.py build-corpus data/test_landing_zone --config config/user_testing.yml
```

### Troubleshooting: Streamlit + Starlette

If `streamlit run app.py` fails with `ImportError: cannot import name 'DEFAULT_EXCLUDED_CONTENT_TYPES' from 'starlette.middleware.gzip'`, your environment has an incompatible `starlette` version. Streamlit 1.58 expects **Starlette ≥ 0.40**; installing `chromadb` may upgrade Starlette via FastAPI. Reconcile with:

```bash
pip install "starlette>=0.40,<0.46"
```

---

## Configuration

Settings are built once at bootstrap by `SettingsFactory` (`src/factories/settings_factory.py`).

### Precedence (highest wins)

1. **Runtime override** — e.g. `Container.bootstrap(vector_store_type=...)` or the Streamlit DB selector
2. **Per-user YAML** — path passed as `--config config/user_<name>.yml` or selected in the UI sidebar
3. **`config/default.yml`** — base YAML (optional; if missing, loader returns `{}`)
4. **Environment variables** — from `.env` via `python-dotenv`
5. **Dataclass defaults** — in `src/config/settings.py`

**Secrets** (`PINECONE_API_KEY`, etc.) are read from the **environment only** — never from YAML.

The `config/` directory is gitignored locally; create files as needed. The UI can generate `config/user_<name>.yml` from the sidebar (**Create new profile**).

### YAML keys (mirror `Settings` sections)

Examples of paths read by `SettingsFactory` (full list in that file):

| YAML path | Purpose |
| --- | --- |
| `vector_store_type` | `pinecone` \| `chroma` \| `qdrant` |
| `ollama.*` | Base URL, embed/generation models, dimension |
| `chunking.*` | `chunk_size`, `chunk_overlap`, `max_table_chunk_chars` |
| `semantic_chunking.*` | Semantic chunker thresholds |
| `retrieval.top_k`, `prompt.max_context_chars` | RAG retrieval |
| `pii.enabled`, `pii.enabled_types` | PII redaction |
| `relational_store.enabled`, `relational_store.db_path` | SQLite chunk store |
| `corpus.enabled`, `corpus.output_dir` | Markdown corpus output |
| `validation.*` | Gate + quarantine + encoding threshold |
| `document_loading.pdf.text_extraction.*` | PDF text fallback chain |
| `document_loading.pdf.table_extraction.*` | PDF table fallback chain |
| `document_loading.pdf.ocr.*` | OCR engine chain |
| `chroma.*`, `qdrant.*`, `pinecone.*` (non-secret fields) | Vector store paths/names |
| `logging.*` | Log directory and rotation |

### UI profile presets (Speed / Quality / Strict)

When creating a profile in `app.py`, choosing **Speed**, **Quality**, or **Strict** writes different `document_loading` and `validation.max_replacement_char_ratio` blocks into the saved YAML. **`profile_type` in that file is informational only** — it is not read by `SettingsFactory`.

---

## Vector database options

Switch from the **Streamlit sidebar**, set `VECTOR_STORE_TYPE` in `.env`/YAML, or pass `vector_store_type` to `Container.bootstrap()`.

| Database | Type | Credential | Default storage |
| --- | --- | --- | --- |
| **Pinecone** | Cloud | `PINECONE_API_KEY` required | Pinecone cloud index |
| **ChromaDB** | Local on-disk | None | `./data/chroma` (`CHROMA_PERSIST_DIR`) |
| **Qdrant** | Local or remote | None for local path mode | `./data/qdrant` (`QDRANT_PATH`); set `QDRANT_URL` for remote |

Each database maintains its own index/collection — **re-ingest separately** per backend you use.

Implementation: `src/infrastructure/vector_store/pinecone_vector_store.py`, `chroma_vector_store.py`, `qdrant_vector_store.py`; selection in `src/factories/adapter_factory.py` → `create_vector_store()`.

---

## Document ingestion pipeline

Actual call order (see `src/application/services/ingestion_service.py` and `streaming_ingestion_service.py`):

```
1. File validation (pre-load)     FileValidationGate.check_file()
2. Load                           IDocumentLoader via DocumentLoaderFactory
3. File validation (post-load)    FileValidationGate.check_content()
4. Image extraction (optional)    PdfImageExtractor / DocxImageExtractor
5. Pre-processing pipeline        PreProcessingPipeline.process_all()
6. Corpus write (optional)        MarkdownCorpusWriter
7. Chunk                          DocumentRouter → RecursiveTextChunker or SemanticChunker
8. Embed                          OllamaEmbeddingProvider
9. Vector upsert                  IVectorStore.ensure_index_exists() + upsert()
10. Relational store (optional)   SqliteRelationalStore.save_chunks()
```

**`build-corpus`** (`CorpusBuilderService`) runs steps **1–6** only — no chunking, embedding, or vector store (`src/application/services/corpus_builder_service.py`).

**`watch`** uses `StreamingIngestionService.ingest_file()` per new file via `FileIngestionAdapter` + `FileSystemWatcher` (2s stabilisation delay; skips `*.tmp`, `*.part`, dotfiles).

### Pre-processing order

Configured in `AdapterFactory.create_pre_processing_pipeline()`:

1. `TextSanitizer`
2. `UnicodeNormalizer`
3. `MetadataNormalizer`
4. `SchemaMapper` (drops document on `ValueError`)
5. `MetadataEnricher` (`word_count`, `has_tables`, etc.)
6. `PiiAnonymizingPreProcessor` (if `pii.enabled`) — sets `pii_redacted`, `pii_entities` in metadata

---

## Supported input formats

| Extension(s) | Loader | Chunker route | Notes |
| --- | --- | --- | --- |
| `.pdf` | `PdfDocumentLoader` | Recursive (default) | Text + table fallback chains; optional pdfplumber tables |
| `.docx` | `DocxDocumentLoader` | Recursive | Pseudo-pages ~3000 chars; tables via python-docx |
| `.pptx` | `PptxDocumentLoader` | Recursive | Speaker notes in body |
| `.ppt` | `PptDocumentLoader` | Recursive | LibreOffice converts to `.pptx` first |
| `.html`, `.htm` | `HtmlLoader` | **Semantic** | Tables → Markdown via BeautifulSoup |
| `.json` | `JsonLoader` | **Semantic** | |
| `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.gif` | `OcrLoader` | Recursive (`file_type: image`) | Tesseract via `pytesseract` |

Routing logic: `src/infrastructure/chunking/document_router.py` — `html` and `json` → `SemanticChunker`; all others → `RecursiveTextChunker`.

### PDF fallback chains & confidence

Configured under `document_loading.pdf` in YAML (defaults in `src/config/settings.py` → `DocumentLoadingSettings`).

- **Text:** primary `pypdf` → fallbacks `pymupdf`, `tesseract_ocr` (threshold default **0.7**)
- **Tables:** primary `pdfplumber` → fallbacks `pymupdf_tables`, `text_extraction` (min confidence **0.6**)
- **OCR engines:** primary `tesseract` → fallbacks `easyocr`, `paddleocr` (threshold **0.5**)

Executor: `src/domain/fallback_chain.py` + confidence heuristic `src/domain/confidence.py`. Pages that needed a non-primary strategy log **`[LOAD-003]`** (`src/infrastructure/loaders/pdf_loader.py`).

Set `fallbacks: []` in YAML to disable fallbacks (Speed preset in UI).

---

## Validation and quarantine

When `validation.enabled` is true (`VALIDATION_ENABLED`, default **true**), `FileValidationGate` runs in ingest, streaming ingest, and corpus build.

| Stage | Validators | Source |
| --- | --- | --- |
| Pre-load | `FileNotEmptyValidator` | `src/infrastructure/validation/file_not_empty_validator.py` |
| Pre-load | `DocumentProtectionValidator` (if `validation.readonly_check_enabled`) | `src/infrastructure/validation/document_protection_validator.py` |
| Post-load | `ContentNotEmptyValidator` | `content_not_empty_validator.py` |
| Post-load | `EncodingValidator` | `encoding_validator.py` |

**Important:** `validation.readonly_check_enabled` enables **native document protection** checks (Word “Protect Document”, PowerPoint “Protect Presentation”, PDF owner-password edit restrictions) — **not** the OS filesystem writable bit. `ReadOnlyFileValidator` exists in the codebase but is **not** registered in `AdapterFactory`.

On failure, files are moved under **`validation.unprocessed_dir`** (default `./data/unprocessed/<reason>/`) by `UnprocessedFileMover`:

| Subfolder | Typical trigger |
| --- | --- |
| `unprotected/` | `FILE-001` — format protection not enabled |
| `empty/` | `FILE-003` — zero-byte file |
| `load_failed/` | `LOAD-002`, `VALID-003` |
| `invalid_content/` | `VALID-001`, `VALID-002` |
| `other/` | unmapped codes |

Errors are logged with structured codes from `src/domain/errors.py` (see Logging).

---

## PII anonymization

When `pii.enabled` is true (default), `RegexPiiAnonymizer` redacts:

`EMAIL`, `URL`, `CNIC`, `IBAN`, `CREDIT_CARD`, `PHONE_PK`, `PHONE_INTL`, `IP_ADDRESS`, `DATE_OF_BIRTH`

Restrict types with `pii.enabled_types` in YAML or `PII_ENABLED_TYPES=EMAIL,CNIC` in `.env` (empty = all types).

Redaction metadata flows into corpus front matter (`pii_redacted` in `markdown_corpus_writer.py`) and SQLite `metadata_json`. Vector store adapters currently persist core chunk fields only (source, page, chunk indices, text) — not `pii_redacted`.

---

## Environment variables

Secrets and tunables read by `SettingsFactory`. YAML overrides the same keys where noted in Configuration.

```bash
# ── Vector store ─────────────────────────────────────────────
VECTOR_STORE_TYPE=pinecone          # pinecone | chroma | qdrant

# Pinecone (secret + options)
PINECONE_API_KEY=
PINECONE_INDEX_NAME=rag-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Chroma
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION=rag-collection

# Qdrant
QDRANT_URL=                         # blank = local path mode
QDRANT_PATH=./data/qdrant
QDRANT_COLLECTION=rag-collection

# ── Ollama ───────────────────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_GENERATION_MODEL=gemma3
OLLAMA_EMBED_DIMENSION=768

# ── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE=512
CHUNK_OVERLAP=64
MAX_TABLE_CHUNK_CHARS=4000
SEMANTIC_SIMILARITY_THRESHOLD=0.75
SEMANTIC_MIN_SENTENCES=2
SEMANTIC_MAX_SENTENCES=15

# ── Retrieval / prompt ───────────────────────────────────────
RETRIEVAL_TOP_K=5
MAX_CONTEXT_CHARS=6000

# ── Ingestion batches ────────────────────────────────────────
UPSERT_BATCH_SIZE=100
EMBED_BATCH_SIZE=8
EMBED_RETRIES=3
DOCX_PSEUDO_PAGE_CHARS=3000

# ── PII ──────────────────────────────────────────────────────
PII_ENABLED=true
PII_ENABLED_TYPES=                  # comma-separated; empty = all

# ── Relational SQLite store ──────────────────────────────────
RELATIONAL_STORE_ENABLED=true
RELATIONAL_STORE_DB_PATH=./data/relational/rag_chunks.db

# ── Corpus / images / tables ─────────────────────────────────
CORPUS_WRITER_ENABLED=true
CORPUS_OUTPUT_DIR=./data/corpus
TABLE_EXTRACTION_ENABLED=true
IMAGE_EXTRACTION_ENABLED=true
IMAGE_OUTPUT_DIR=./data/images

# ── Validation / quarantine ────────────────────────────────────
VALIDATION_ENABLED=true
VALIDATION_READONLY_CHECK_ENABLED=true
VALIDATION_UNPROCESSED_DIR=./data/unprocessed
VALIDATION_MAX_REPLACEMENT_CHAR_RATIO=0.01

# ── PDF extraction (primaries; fallback lists are YAML-only) ───
PDF_TEXT_PRIMARY=pypdf
PDF_TEXT_CONFIDENCE_THRESHOLD=0.7
PDF_TABLE_PRIMARY=pdfplumber
PDF_TABLE_MIN_CONFIDENCE=0.6
PDF_OCR_ENGINE=tesseract
PDF_OCR_CONFIDENCE_THRESHOLD=0.5

# ── System tool paths ────────────────────────────────────────
LIBREOFFICE_PATH=soffice
LIBREOFFICE_TIMEOUT_SECONDS=120
TESSERACT_CMD=
TESSERACT_LANG=eng

# ── Logging ──────────────────────────────────────────────────
LOG_DIR=./logs
LOG_FILENAME=rag.log
LOG_EXCEPTIONS_FILENAME=exceptions.log
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5
LOG_CONSOLE_LEVEL=INFO
LOG_FILE_LEVEL=DEBUG
```

---

## CLI commands

All commands accept **`--config path/to/user.yml`** (see `main.py` → `build_arg_parser()`).

| Command | Description |
| --- | --- |
| `ingest [source]` | Full pipeline into vector store + optional SQLite. Default source: `data/landing_zone`. |
| `build-corpus [source]` | Load → images → pre-process → corpus only. **No Ollama or vector DB required.** |
| `ask <question>` | RAG query. Flags: `--top-k`, `--no-stream`. |
| `eval` | Runs `DEFAULT_EVAL_SUITE` (6 cases). Flag: `--stream`. |
| `debug <question>` | Retrieval + prompt assembly, no LLM. Flag: `--top-k` (default 5). |
| `watch [source]` | Watch folder for new files; uses streaming ingest. Flag: `--recursive`. |

Uncaught CLI exceptions are logged to `logs/rag.log` (`main.py`).

---

## Streamlit UI (`app.py`)

| Tab | Backend services |
| --- | --- |
| **Ask** | `rag_query_service.ask()` — streaming optional; source list with PII badge when metadata available |
| **Ingest** | `ingestion_service.ingest_path()`; **Build Corpus Only** → `corpus_builder_service.build()` |
| **Watch** | Background thread → `FileSystemWatcher` + `streaming_ingestion_service` |
| **Eval** | `evaluation_service.run_eval()` |
| **Debug** | `container.retrieval_service.search()` + `container.build_prompt()` |
| **Relational DB** | `container.relational_store` — search/delete by source, lookup by vector ID |
| **Settings** | Read-only view of active profile (PII, chunking, relational store, models) |

Sidebar also selects **vector database** and **config profile** (`config/user_*.yml`).

---

## Logging and error codes

Configured in `LoggingSettings`; implemented in `src/infrastructure/logging/rich_logger.py`.

| Output | Path / sink | Content |
| --- | --- | --- |
| Console | stderr (Rich) | `LOG_CONSOLE_LEVEL` (default INFO) |
| Main log | `logs/rag.log` | `LOG_FILE_LEVEL` (default DEBUG), rotated |
| Exceptions log | `logs/exceptions.log` | **ERROR and above only** — use for “what broke” |

Structured codes are defined in `src/domain/errors.py` and formatted via `format_error()`:

| Code | Meaning |
| --- | --- |
| `FILE-001` | Document format protection not enabled (when validation gate on) |
| `FILE-002` | File not found |
| `FILE-003` | Empty file |
| `LOAD-001` | Unsupported extension |
| `LOAD-002` | Loader failure |
| `LOAD-003` | PDF fallback strategy used |
| `LOAD-004` | All PDF strategies exhausted |
| `VALID-001` | Empty content after load |
| `VALID-002` | Encoding / replacement-char ratio too high |
| `VALID-003` | Corrupt / unreadable content |
| `EXT-001` / `EXT-002` | Table / image extraction failure |
| `CONV-001` | LibreOffice conversion failure |
| `PII-001` | PII redaction failure |
| `EMBED-001` | Embedding failure |
| `STORE-001` / `STORE-002` | Vector / store connection failure |
| `SYS-001` | Unexpected error |

---

## Testing without full setup

No automated `tests/` suite ships in this repo. Manual testing path:

1. **`build-corpus`** — no Pinecone/Chroma/Qdrant/Ollama needed (corpus writer must be enabled).
2. Sample inputs: `data/test_landing_zone/` — see `data/test_landing_zone/README.md`.
3. Recommended: copy/create `config/user_testing.yml` with isolated `corpus.output_dir` and `image_extraction.output_dir` so test runs do not overwrite production corpus/images.

Example:

```bash
python main.py build-corpus data/test_landing_zone --config config/user_testing.yml
```

Inspect output under the configured corpus directory and `logs/rag.log`.

---

## Data directories

| Path | Purpose |
| --- | --- |
| `data/landing_zone/` | Production drop folder (gitignored) |
| `data/test_landing_zone/` | Manual test inputs + README |
| `data/corpus/` | Markdown corpus output |
| `data/images/` | Extracted embedded images |
| `data/relational/` | SQLite DB(s) |
| `data/unprocessed/` | Quarantined rejected files |
| `data/chroma/`, `data/qdrant/` | Local vector store persistence |
| `logs/` | Rotating log files |

Default landing zone property: `Settings.data_raw` → `project_root / data / landing_zone` (`src/config/settings.py`).

---

## Layer map

```
src/
  domain/                         # Pure data + ports. Zero SDK imports.
    entities.py                   - Document, EmbeddedChunk, SearchResult, ExtractionAttempt, ...
    errors.py                     - ErrorCode registry, PipelineError, format_error()
    confidence.py                 - score_text_confidence() for fallback chains
    fallback_chain.py             - run_fallback_chain()
    interfaces/                   - ILogger, IDocumentLoader, IVectorStore, IRelationalStore,
                                    IDocumentProcessor, ILandingZoneWatcher, IOcrEngine, ...

  config/
    settings.py                   - Frozen dataclasses (all Settings sections)

  infrastructure/                 # One adapter per external concern
    loaders/                      - pdf, docx, pptx, ppt, html, json, ocr
    extraction/                   - PDF text/table strategies
    ocr/                          - tesseract, easyocr, paddleocr engines
    conversion/                   - LibreOffice headless (.ppt)
    validation/                   - file/content validators, unprocessed mover
    pre_processing/               - sanitizer → … → PII pipeline
    images/                       - PDF/DOCX image extractors
    corpus/                       - MarkdownCorpusWriter
    chunking/                     - recursive, semantic, DocumentRouter
    embeddings/                   - OllamaEmbeddingProvider
    vector_store/                 - pinecone, chroma, qdrant + SHA-256 IDs
    relational_store/             - SQLite chunk store
    generation/                   - prompt builder, Ollama answer generator
    landing_zone/                 - watchdog watcher + ingestion adapter
    logging/                      - RichLogger + rotating files
    reporting/                    - RichEvalReporter
    pii/                          - RegexPiiAnonymizer

  application/services/           # Orchestration only; depend on ports
    file_validation_gate.py
    ingestion_service.py
    streaming_ingestion_service.py
    corpus_builder_service.py
    retrieval_service.py
    rag_query_service.py
    evaluation_service.py

  factories/                      # ONLY place concrete classes are constructed
    settings_factory.py           - .env + YAML → Settings
    yaml_config_loader.py         - deep-merge config/default.yml + user file
    adapter_factory.py            - all infrastructure adapters
    service_factory.py            - application services
    document_loader_factory.py
    image_extractor_factory.py
    sdk_client_factory.py         - Pinecone + Ollama SDK clients
    logger_factory.py

  composition/
    container.py                  - bootstrap; exposes public service properties

main.py                           - CLI entry
app.py                            - Streamlit UI
```

---

## Dependency rule

```
domain  ←  application  ←  infrastructure / factories  ←  composition  ←  main.py / app.py
```

- No file outside `factories/` and `composition/` constructs concrete adapters.
- No file outside `infrastructure/` imports SDKs (`pinecone`, `ollama`, `chromadb`, `qdrant_client`, `pypdf`, `docx`, `rich`, etc.).
- Every class receives collaborators via `__init__` — no global settings singleton.

---

## Old → New mapping

| Old file | New home |
| --- | --- |
| `config/settings.py` (singleton) | `src/config/settings.py` + `src/factories/settings_factory.py` |
| `config/pinecone_client.py` | `src/factories/sdk_client_factory.py` + `src/infrastructure/vector_store/pinecone_vector_store.py` |
| `embeddings/embedder.py` | `src/infrastructure/embeddings/ollama_embedding_provider.py` |
| `ingestion/loader.py` | `src/infrastructure/loaders/*` + `src/factories/document_loader_factory.py` |
| `ingestion/chunker.py` | `src/infrastructure/chunking/recursive_chunker.py` (+ semantic router) |
| `ingestion/upsert.py` | Vector store adapters + `sha256_vector_id_strategy.py` |
| `ingestion/pipeline.py` | `src/application/services/ingestion_service.py` |
| `retrieval/retriever.py` | `retrieval_service.py` + vector store `query()` |
| `generation/prompt_builder.py` | `src/infrastructure/generation/default_prompt_builder.py` |
| `generation/generator.py` | `src/infrastructure/generation/ollama_answer_generator.py` |
| `generation/rag.py` | `src/application/services/rag_query_service.py` |
| `utils/eval.py` | `evaluation_service.py` + `rich_eval_reporter.py` |
| `utils/logger.py` | `rich_logger.py` + `logger_factory.py` |

---

## Behavior preserved (core RAG)

- Recursive chunking separators and whitespace cleanup (`recursive_chunker.py`).
- Prompt template, system prompt, and 6000-char context budget (`default_prompt_builder.py`).
- Deterministic SHA-256 vector IDs (`sha256_vector_id_strategy.py`).
- Embedding retry/backoff `2 ** attempt` seconds, 3 attempts (`ollama_embedding_provider.py`).
- Embed batch size 8, upsert batch size 100 (defaults in `IngestionSettings`).
- Full `DEFAULT_EVAL_SUITE` keyword matching (`evaluation_service.py`).
- DOCX pseudo-page grouping (~3000 chars).

---

## Maintainability

```bash
pip install radon
radon mi src/ -s
radon cc src/ -s -a
```

The codebase targets high maintainability (single responsibility per adapter, factories as the only wiring point). Re-run `radon` after dependency or structural changes.

---

