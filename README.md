# RAG-with-Pinecone — Layered / SOLID Refactor

## How to run

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in PINECONE_API_KEY at minimum

python main.py ingest                  # ingests data/raw by default
python main.py ask "What is the Magi story about?"
python main.py eval
python main.py debug "Who cut Della's hair?"
```

## Layer map

```
src/
  domain/                  # Pure data + ports. Zero SDK imports. Nothing here depends on anything else.
    entities.py             - Document, EmbeddedChunk, SearchResult, PromptPackage, RAGResponse, EvalCase/Result
    interfaces/              - ILogger, IDocumentLoader, IDocumentLoaderResolver, ITextChunker,
                                IEmbeddingProvider, IVectorStore, IVectorIdStrategy,
                                IPromptBuilder, IAnswerGenerator, IEvalReporter

  config/
    settings.py             - Frozen dataclasses (PineconeSettings, OllamaSettings, ...). No global instance.

  infrastructure/           # One adapter per external SDK. Each implements exactly one port.
    logging/rich_logger.py          -> ILogger          (rich + stdlib logging)
    loaders/pdf_loader.py           -> IDocumentLoader   (pypdf)
    loaders/docx_loader.py          -> IDocumentLoader   (python-docx)
    chunking/recursive_chunker.py   -> ITextChunker      (langchain_text_splitters)
    embeddings/ollama_embedding_provider.py -> IEmbeddingProvider (ollama)
    vector_store/pinecone_vector_store.py   -> IVectorStore       (pinecone)
    vector_store/sha256_vector_id_strategy.py -> IVectorIdStrategy
    generation/default_prompt_builder.py    -> IPromptBuilder
    generation/ollama_answer_generator.py   -> IAnswerGenerator   (ollama)
    reporting/rich_eval_reporter.py         -> IEvalReporter      (rich)

  application/services/     # Orchestration only. Depend on ports, never on adapters or SDKs.
    ingestion_service.py     - load -> chunk -> embed -> upsert
    retrieval_service.py     - embed query -> vector search
    rag_query_service.py     - retrieve -> build prompt -> generate
    evaluation_service.py    - runs DEFAULT_EVAL_SUITE / debug_query, delegates rendering to IEvalReporter

  factories/                 # The ONLY place concrete adapters/services are instantiated.
    settings_factory.py       - reads env vars -> Settings (replaces old _require/_optional globals)
    logger_factory.py         - ILogger factory
    sdk_client_factory.py     - builds raw Pinecone/Ollama SDK client objects
    document_loader_factory.py - abstract factory: resolves IDocumentLoader by file extension
    adapter_factory.py        - abstract factory: builds every infrastructure adapter
    service_factory.py        - builds application services from adapters

  composition/
    container.py              - composition root. Calls factories, wires everything, exposes
                                 `ingestion_service`, `rag_query_service`, `evaluation_service`.

main.py                      - CLI only: argparse + Container.bootstrap(). No business logic.
```

## Dependency rule

`domain` <- `application` <- `factories`/`infrastructure` <- `composition` <- `main.py`

No file outside `factories/` and `composition/` ever calls a concrete adapter's constructor.
No file outside `infrastructure/` imports `pinecone`, `ollama`, `pypdf`, `docx`, or `rich`.
Every class receives its collaborators through `__init__` — nothing reaches for a global.

## Old -> new mapping

| Old file | New home |
|---|---|
| `config/settings.py` (`settings` singleton) | `config/settings.py` (pure dataclasses) + `factories/settings_factory.py` (builds it) |
| `config/pinecone_client.py` | `factories/sdk_client_factory.py` (client) + `infrastructure/vector_store/pinecone_vector_store.py` (logic) |
| `embeddings/embedder.py` | `infrastructure/embeddings/ollama_embedding_provider.py` |
| `ingestion/loader.py` | `infrastructure/loaders/pdf_loader.py` + `docx_loader.py` + `factories/document_loader_factory.py` |
| `ingestion/chunker.py` | `infrastructure/chunking/recursive_chunker.py` |
| `ingestion/upsert.py` | `infrastructure/vector_store/pinecone_vector_store.py` (`upsert`) + `sha256_vector_id_strategy.py` |
| `ingestion/pipeline.py` | `application/services/ingestion_service.py` |
| `retrieval/retriever.py` | `application/services/retrieval_service.py` + `infrastructure/vector_store/pinecone_vector_store.py` (`query`) |
| `generation/prompt_builder.py` | `infrastructure/generation/default_prompt_builder.py` |
| `generation/generator.py` | `infrastructure/generation/ollama_answer_generator.py` |
| `generation/rag.py` | `application/services/rag_query_service.py` |
| `utils/eval.py` | `application/services/evaluation_service.py` (logic) + `infrastructure/reporting/rich_eval_reporter.py` (rendering) |
| `utils/logger.py` | `infrastructure/logging/rich_logger.py` + `factories/logger_factory.py` |

## Behavior preserved exactly

- Chunking algorithm, separators, and the spaced-character/whitespace cleanup regexes.
- Prompt template, system prompt text, and the 6000-char context budget/truncation logic.
- Deterministic SHA-256 vector IDs (idempotent re-ingestion).
- Embedding retry/backoff (`2 ** attempt` seconds, 3 attempts).
- Batch sizes for embedding (8) and upsert (100).
- The full `DEFAULT_EVAL_SUITE` and pass/fail keyword-matching logic.
- DOCX pseudo-page grouping at ~3000 chars.

## Disclosed deviations (please review)

I changed a few things deliberately while refactoring — flagging them per your instruction to surface anything altered:

1. **No more module-level `settings` singleton.** Every class now receives `Settings` (or the relevant sub-section of it) via its constructor instead of `from src.config.settings import settings`. This was required to satisfy "no hidden dependencies" — but it means any code you still have that does `from src.config.settings import settings` directly will break and needs to go through `Container` instead.
2. **`tqdm` progress bars were dropped** from embedding and upsert batch loops (`embedder.py`/`upsert.py` had `tqdm(...)` wrapping the loops). I didn't want to bake a console progress bar directly into an application-layer/infrastructure class without a port for it. If you want progress reporting back, I can add an `IProgressReporter` port + a `TqdmProgressReporter` adapter — same pattern as `IEvalReporter` — and inject it into `IngestionService`/`PineconeVectorStore`.
3. **Streaming generation no longer calls `print()` directly.** `OllamaAnswerGenerator` takes an optional `token_sink` callback (wired to `print` in `main.py`). If you call it from a Streamlit UI or another non-CLI context, just pass a different `token_sink` instead of getting raw prints.
4. **`load_pdfs_from_dir` backward-compatible alias removed.** The original `loader.py` had `load_pdfs_from_dir = load_documents_from_dir` "for validate_loader.py and any other callers" — that file wasn't in your bundle, so I couldn't confirm it's unused. If you still have a `validate_loader.py` referencing that name, tell me and I'll add the alias back (e.g. as a second method on `DocumentLoaderFactory`).
5. **`EMBEDDING_DIMENSION = 768` hardcoded constant** (in the old `pinecone_client.py`) is now `OLLAMA_EMBED_DIMENSION` in `.env`/`OllamaSettings`, since it's a property of the embedding model, not the vector store, and your embedding model is configurable.
6. **`__pycache__/*.pyc` files** in your bundle were ignored — those are build artifacts, not source, and were excluded from the refactor.

Nothing else was intentionally removed. I re-traced every function in your original bundle against this version line by line; if I missed something, it would most likely show up as an `ImportError` or `AttributeError` at startup — `python main.py ingest` or `ask` will surface that immediately.

## Maintainability notes

Every class has one constructor-injected dependency set and one job (load one file format, embed text, build a prompt, etc.). Cyclomatic complexity per method stays low (the most complex method, `PineconeVectorStore.upsert`/`query`, is still under 10 branches). No class exceeds ~120 lines. This should comfortably clear a maintainability index of 80 on standard tooling (e.g. `radon mi`), but I'd recommend running `radon mi -s src/` yourself once your real dependencies are installed, since I verified wiring with stubs rather than the live SDKs (no network access to PyPI for `pinecone`/`ollama`/`pypdf`/`python-docx`/`rich`/`langchain_text_splitters` in this environment).
