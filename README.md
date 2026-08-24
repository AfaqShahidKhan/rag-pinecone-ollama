# rag-pinecone-ollama

A production-grade, locally-running RAG (Retrieval-Augmented Generation) pipeline built with a clean layered architecture following SOLID design principles.

## Tech Stack

| Component        | Technology                   |
| ---------------- | ---------------------------- |
| LLM & Embeddings | Ollama (local)               |
| Vector Databases | Pinecone · ChromaDB · Qdrant |
| Document Loaders | pypdf · python-docx          |
| Chunking         | LangChain Text Splitters     |
| UI               | Streamlit                    |
| CLI              | argparse                     |

---

## How to run

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull Ollama models (Ollama must be installed and running)
ollama pull nomic-embed-text
ollama pull gemma3

# 4. Configure environment
cp .env.example .env
# Edit .env — fill in PINECONE_API_KEY if using Pinecone

# 5. Add your documents
# Place PDF or DOCX files inside data/landing_zone/

# 6a. Run the Streamlit UI
streamlit run app.py

# 6b. Or use the CLI
python main.py ingest
python main.py ask "What is the Magi story about?"
python main.py eval
python main.py debug "Who cut Della's hair?"
```

---

## Vector Database Options

Switch between databases from the **sidebar in the UI**, or set `VECTOR_STORE_TYPE` in your `.env`.

| Database     | Type          | API Key                       | Data Location  |
| ------------ | ------------- | ----------------------------- | -------------- |
| **Pinecone** | Cloud         | Required (`PINECONE_API_KEY`) | Pinecone cloud |
| **ChromaDB** | Local         | None                          | `data/chroma/` |
| **Qdrant**   | Local / Cloud | None for local                | `data/qdrant/` |

> Each database maintains its own index. You must ingest your documents separately into each database you want to use.

---

## Environment Variables

```bash
# Vector store selector: pinecone | chroma | qdrant  (default: pinecone)
VECTOR_STORE_TYPE=pinecone

# Pinecone (required only when VECTOR_STORE_TYPE=pinecone)
PINECONE_API_KEY=
PINECONE_INDEX_NAME=rag-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Ollama (optional, defaults shown)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_GENERATION_MODEL=gemma3
OLLAMA_EMBED_DIMENSION=768

# Chunking (optional, defaults shown)
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# Retrieval (optional, defaults shown)
RETRIEVAL_TOP_K=5
MAX_CONTEXT_CHARS=6000

# Ingestion (optional, defaults shown)
UPSERT_BATCH_SIZE=100
EMBED_BATCH_SIZE=8
EMBED_RETRIES=3
DOCX_PSEUDO_PAGE_CHARS=3000

# ChromaDB (optional, defaults shown)
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION=rag-collection

# Qdrant (optional, defaults shown)
QDRANT_URL=           # leave blank for local on-disk mode
QDRANT_PATH=./data/qdrant
QDRANT_COLLECTION=rag-collection
```

---

## Layer Map

```
src/
  domain/                   # Pure data + ports. Zero SDK imports. Foundation every layer depends on.
    entities.py              - Document, EmbeddedChunk, SearchResult, PromptPackage, RAGResponse, EvalCase/Result
    interfaces/              - ILogger, IDocumentLoader, IDocumentLoaderResolver, ITextChunker,
                               IEmbeddingProvider, IVectorStore, IVectorIdStrategy,
                               IPromptBuilder, IAnswerGenerator, IEvalReporter

  config/
    settings.py              - Frozen dataclasses (PineconeSettings, ChromaSettings, QdrantSettings,
                               OllamaSettings, VectorStoreType...). No global singleton instance.

  infrastructure/            # One adapter per external SDK. Each implements exactly one port.
    logging/rich_logger.py             -> ILogger             (rich + stdlib logging)
    loaders/pdf_loader.py              -> IDocumentLoader      (pypdf)
    loaders/docx_loader.py             -> IDocumentLoader      (python-docx)
    chunking/recursive_chunker.py      -> ITextChunker         (langchain_text_splitters)
    embeddings/ollama_embedding_provider.py -> IEmbeddingProvider  (ollama)
    vector_store/pinecone_vector_store.py   -> IVectorStore        (pinecone)
    vector_store/chroma_vector_store.py     -> IVectorStore        (chromadb)
    vector_store/qdrant_vector_store.py     -> IVectorStore        (qdrant-client)
    vector_store/sha256_vector_id_strategy.py -> IVectorIdStrategy
    generation/default_prompt_builder.py    -> IPromptBuilder
    generation/ollama_answer_generator.py   -> IAnswerGenerator    (ollama)
    reporting/rich_eval_reporter.py         -> IEvalReporter       (rich)

  application/services/      # Orchestration only. Depend on ports, never on adapters or SDKs.
    ingestion_service.py      - load -> chunk -> embed -> upsert
    retrieval_service.py      - embed query -> vector search
    rag_query_service.py      - retrieve -> build prompt -> generate
    evaluation_service.py     - runs DEFAULT_EVAL_SUITE / debug_query, delegates rendering to IEvalReporter

  factories/                 # The ONLY place concrete adapters/services are instantiated.
    settings_factory.py       - reads env vars -> Settings
    logger_factory.py         - ILogger instances
    sdk_client_factory.py     - builds landing_zone Pinecone/Ollama SDK client objects
    document_loader_factory.py - abstract factory: resolves IDocumentLoader by file extension
    adapter_factory.py        - abstract factory: builds every infrastructure adapter,
                                routes create_vector_store() by VectorStoreType
    service_factory.py        - builds application services from adapters

  composition/
    container.py              - composition root. Calls factories, wires everything, exposes
                                ingestion_service, rag_query_service, evaluation_service.
                                Accepts vector_store_type override for UI-driven DB switching.

main.py    - CLI only: argparse + Container.bootstrap(). No business logic.
app.py     - Streamlit UI: Ask · Ingest · Eval · Debug tabs + sidebar DB selector.
```

---

## Dependency Rule

```
domain  ←  application  ←  infrastructure / factories  ←  composition  ←  main.py / app.py
```

- No file outside `factories/` and `composition/` ever calls a concrete adapter constructor.
- No file outside `infrastructure/` imports `pinecone`, `ollama`, `pypdf`, `chromadb`, `qdrant_client`, `docx`, or `rich`.
- Every class receives its collaborators through `__init__` — nothing reaches for a global.

---

## Old → New Mapping

| Old file                                    | New home                                                                                                            |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `config/settings.py` (`settings` singleton) | `config/settings.py` (pure dataclasses) + `factories/settings_factory.py` (builds it)                               |
| `config/pinecone_client.py`                 | `factories/sdk_client_factory.py` (client) + `infrastructure/vector_store/pinecone_vector_store.py` (logic)         |
| `embeddings/embedder.py`                    | `infrastructure/embeddings/ollama_embedding_provider.py`                                                            |
| `ingestion/loader.py`                       | `infrastructure/loaders/pdf_loader.py` + `docx_loader.py` + `factories/document_loader_factory.py`                  |
| `ingestion/chunker.py`                      | `infrastructure/chunking/recursive_chunker.py`                                                                      |
| `ingestion/upsert.py`                       | `infrastructure/vector_store/pinecone_vector_store.py` (`upsert`) + `sha256_vector_id_strategy.py`                  |
| `ingestion/pipeline.py`                     | `application/services/ingestion_service.py`                                                                         |
| `retrieval/retriever.py`                    | `application/services/retrieval_service.py` + `infrastructure/vector_store/pinecone_vector_store.py` (`query`)      |
| `generation/prompt_builder.py`              | `infrastructure/generation/default_prompt_builder.py`                                                               |
| `generation/generator.py`                   | `infrastructure/generation/ollama_answer_generator.py`                                                              |
| `generation/rag.py`                         | `application/services/rag_query_service.py`                                                                         |
| `utils/eval.py`                             | `application/services/evaluation_service.py` (logic) + `infrastructure/reporting/rich_eval_reporter.py` (rendering) |
| `utils/logger.py`                           | `infrastructure/logging/rich_logger.py` + `factories/logger_factory.py`                                             |

---

## Behavior Preserved

- Chunking algorithm, separators, and spaced-character/whitespace cleanup regexes.
- Prompt template, system prompt text, and 6000-char context budget/truncation logic.
- Deterministic SHA-256 vector IDs (idempotent re-ingestion).
- Embedding retry/backoff (`2 ** attempt` seconds, 3 attempts).
- Batch sizes for embedding (8) and upsert (100).
- Full `DEFAULT_EVAL_SUITE` and pass/fail keyword-matching logic.
- DOCX pseudo-page grouping at ~3000 chars.

---

## Maintainability

```bash
pip install radon
radon mi src/ -s     # maintainability index — all files grade A
radon cc src/ -s -a  # cyclomatic complexity report
```

All 48 source files score **grade A** on the Maintainability Index (target was > 80).
