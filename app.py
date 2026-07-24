"""
app.py — Streamlit UI for the RAG pipeline.

Tabs:
  💬 Ask             — Chat with streaming, source attribution, PII badge
  📥 Ingest          — Batch ingest with pre-processing stats
  👁 Watch           — Start/stop landing zone watcher with live log
  🧪 Eval            — Evaluation suite with per-case expandable results
  🔎 Debug           — landing_zone retrieved chunks + assembled prompt
  🗄 Relational DB   — Browse SQLite chunks by source file
  ⚙️ Settings        — PII configuration

Run:  streamlit run app.py
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import streamlit as st

from src.composition.container import Container
from src.config.settings import VectorStoreType
from src.infrastructure.logging.rich_logger import ILogger

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔍",
    layout="wide",
)

# ── DB metadata ────────────────────────────────────────────────────────────────
DB_OPTIONS = {
    VectorStoreType.PINECONE: {
        "label": "Pinecone", "icon": "☁️",
        "description": "Cloud-hosted. Requires PINECONE_API_KEY in .env",
    },
    VectorStoreType.CHROMA: {
        "label": "ChromaDB", "icon": "💾",
        "description": "Local on-disk. No API key needed. Persists in data/chroma/",
    },
    VectorStoreType.QDRANT: {
        "label": "Qdrant", "icon": "⚡",
        "description": "Local on-disk or remote. Persists in data/qdrant/",
    },
}

# ── PII entity types ───────────────────────────────────────────────────────────
ALL_PII_TYPES = [
    "EMAIL", "URL", "CNIC", "IBAN", "CREDIT_CARD",
    "PHONE_PK", "PHONE_INTL", "IP_ADDRESS", "DATE_OF_BIRTH",
]


# ── Container cache (per db_type) ──────────────────────────────────────────────
@st.cache_resource(
    show_spinner="Connecting to services...",
    hash_funcs={VectorStoreType: lambda v: v.value},
)
def get_container(db_type: VectorStoreType) -> Container:
    return Container.bootstrap(
        project_root=Path(__file__).parent,
        token_sink=None,
        vector_store_type=db_type,
    )


# ── Session state defaults ─────────────────────────────────────────────────────
def _init_session_state() -> None:
    defaults = {
        "watch_logs": [],
        "watch_running": False,
        "watch_thread": None,
        "watch_stop_event": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 RAG Assistant")
    st.divider()

    st.subheader("🗄️ Vector Database")
    selected_db = st.selectbox(
        "Select database",
        options=list(DB_OPTIONS.keys()),
        format_func=lambda v: f"{DB_OPTIONS[v]['icon']}  {DB_OPTIONS[v]['label']}",
        index=0,
        label_visibility="collapsed",
    )
    st.caption(DB_OPTIONS[selected_db]["description"])
    st.divider()

    container = get_container(selected_db)

    st.caption(
        f"**Model:** `{container.settings.ollama.generation_model}`  \n"
        f"**Embed:** `{container.settings.ollama.embed_model}`  \n"
        f"**DB:** `{selected_db.value}`  \n"
        f"**Top-K:** `{container.settings.retrieval.top_k}`  \n"
        f"**PII:** `{'on' if container.settings.pii.enabled else 'off'}`"
    )
    st.divider()

    top_k = st.slider(
        "Top-K results", min_value=1, max_value=15,
        value=container.settings.retrieval.top_k,
    )
    stream = st.toggle("Stream answer", value=True)
    st.divider()
    st.caption("Tabs: Ask · Ingest · Watch · Eval · Debug · DB · Settings")


# ── Tabs ───────────────────────────────────────────────────────────────────────
(
    tab_ask, tab_ingest, tab_watch,
    tab_eval, tab_debug, tab_db, tab_settings
) = st.tabs([
    "💬 Ask", "📥 Ingest", "👁 Watch",
    "🧪 Eval", "🔎 Debug", "🗄 Relational DB", "⚙️ Settings",
])


# ══════════════════════════════════════════════════════════════════════════════
# ASK TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_ask:
    st.header("Ask a question")

    db_key = selected_db.value
    msg_key = f"messages_{db_key}"
    if msg_key not in st.session_state:
        st.session_state[msg_key] = []

    for msg in st.session_state[msg_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        pii_badge = (
                            " 🔒 *PII redacted*"
                            if s.get("pii_redacted") else ""
                        )
                        st.markdown(
                            f"- **{s['source']}** · page {s['page']} · "
                            f"chunk {s['chunk_index']} · "
                            f"score `{s['score']:.4f}`{pii_badge}"
                        )

    if prompt := st.chat_input("Ask anything about your documents…"):
        st.session_state[msg_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if stream:
                token_queue: queue.Queue = queue.Queue()
                result_box: list = []

                def _run_stream():
                    tokens = []
                    def sink(t):
                        token_queue.put(t)
                        tokens.append(t)
                    c = Container.bootstrap(
                        project_root=Path(__file__).parent,
                        token_sink=sink,
                        vector_store_type=selected_db,
                    )
                    resp = c.rag_query_service.ask(prompt, top_k=top_k, stream=True)
                    result_box.append(resp)
                    token_queue.put(None)

                thread = threading.Thread(target=_run_stream, daemon=True)
                thread.start()

                placeholder = st.empty()
                collected = []
                while True:
                    token = token_queue.get()
                    if token is None:
                        break
                    collected.append(token)
                    placeholder.markdown("".join(collected) + "▌")
                placeholder.markdown("".join(collected))
                thread.join()
                response = result_box[0] if result_box else None
            else:
                with st.spinner("Thinking…"):
                    response = container.rag_query_service.ask(
                        prompt, top_k=top_k, stream=False
                    )
                st.markdown(response.answer)

            if response:
                with st.expander("Sources"):
                    for s in response.sources:
                        meta = s.__dict__ if hasattr(s, "__dict__") else {}
                        pii_badge = " 🔒 *PII redacted*" if meta.get("pii_redacted") else ""
                        st.markdown(
                            f"- **{s.source}** · page {s.page} · "
                            f"chunk {s.chunk_index} · "
                            f"score `{s.score:.4f}`{pii_badge}"
                        )

                st.caption(
                    f"⏱ {response.latency_seconds:.2f}s · "
                    f"{response.context_chunks_used} chunks used · "
                    f"model: `{response.model}` · db: `{selected_db.value}`"
                )

                sources_serialized = [
                    {
                        "source": s.source, "page": s.page,
                        "chunk_index": s.chunk_index, "score": s.score,
                    }
                    for s in response.sources
                ]
                st.session_state[msg_key].append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": sources_serialized,
                })

    if st.session_state.get(msg_key):
        if st.button("Clear chat", key=f"clear_{db_key}"):
            st.session_state[msg_key] = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# INGEST TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.header("Ingest documents")

    st.info(
        f"Documents will be ingested into "
        f"**{DB_OPTIONS[selected_db]['icon']} {DB_OPTIONS[selected_db]['label']}**. "
        "Switch the database in the sidebar if needed.",
        icon="ℹ️",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        custom_path = st.text_input(
            "Directory or file path (leave blank for data/landing_zone)",
            placeholder=str(container.settings.data_raw),
        )
    with col2:
        st.write("")
        st.write("")
        run_ingest = st.button("▶ Run Ingest", type="primary", use_container_width=True)

    if run_ingest:
        source = (
            Path(custom_path.strip()) if custom_path.strip()
            else container.settings.data_raw
        )
        if not source.exists():
            st.error(f"Path not found: `{source}`")
        else:
            with st.spinner(f"Ingesting `{source}` into **{selected_db.value}**…"):
                try:
                    total = container.ingestion_service.ingest_path(source)
                    st.success(
                        f"✅ Indexed **{total}** vectors into "
                        f"**{selected_db.value}** from `{source.name}`"
                    )
                    if container.settings.pii.enabled:
                        st.caption(
                            "🔒 PII anonymization was active during this ingestion. "
                            "Emails, phones, CNICs, IBANs and other sensitive data "
                            "were redacted before storage."
                        )
                    if container.settings.relational_store.enabled:
                        st.caption(
                            f"🗄 Chunks saved to SQLite at "
                            f"`{container.settings.relational_store.db_path}`"
                        )
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# WATCH TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_watch:
    st.header("Landing Zone Watcher")
    st.caption(
        "Automatically ingests new files dropped into a folder. "
        "The watcher runs in the background — you can switch tabs while it runs."
    )

    watch_col1, watch_col2 = st.columns([2, 1])
    with watch_col1:
        watch_path_input = st.text_input(
            "Directory to watch",
            value=str(container.settings.data_raw),
            key="watch_path",
        )
    with watch_col2:
        st.write("")
        recursive_watch = st.checkbox("Watch subdirectories", value=False)

    # ── Start / Stop ───────────────────────────────────────────────────────────
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])

    with btn_col1:
        start_clicked = st.button(
            "▶ Start",
            type="primary",
            disabled=st.session_state.watch_running,
            use_container_width=True,
        )
    with btn_col2:
        stop_clicked = st.button(
            "⏹ Stop",
            type="secondary",
            disabled=not st.session_state.watch_running,
            use_container_width=True,
        )

    if start_clicked:
        watch_dir = Path(watch_path_input.strip())
        if not watch_dir.exists():
            st.error(f"Directory not found: `{watch_dir}`")
        else:
            stop_event = threading.Event()
            log_list = st.session_state.watch_logs
            log_list.clear()

            class _UiLogger:
                """Thin logger that appends messages to session state watch_logs."""
                def _log(self, level: str, msg: str):
                    ts = time.strftime("%H:%M:%S")
                    log_list.append(f"[{ts}] {level.upper()}: {msg}")

                def debug(self, msg): pass
                def info(self, msg): self._log("info", msg)
                def warning(self, msg): self._log("warn", msg)
                def error(self, msg): self._log("error", msg)

            def _watch_thread():
                from src.infrastructure.landing_zone.file_system_watcher import FileSystemWatcher
                from src.infrastructure.landing_zone.file_ingestion_adapter import FileIngestionAdapter

                ui_logger = _UiLogger()
                streaming_svc = container.streaming_ingestion_service

                adapter = FileIngestionAdapter(
                    ingestion_service=streaming_svc,
                    logger=ui_logger,
                )
                watcher = FileSystemWatcher(
                    adapter=adapter,
                    logger=ui_logger,
                    recursive=recursive_watch,
                )
                watcher.start(watch_dir)
                while not stop_event.is_set():
                    time.sleep(0.5)
                watcher.stop()

            t = threading.Thread(target=_watch_thread, daemon=True)
            t.start()

            st.session_state.watch_running = True
            st.session_state.watch_thread = t
            st.session_state.watch_stop_event = stop_event
            st.rerun()

    if stop_clicked:
        if st.session_state.watch_stop_event:
            st.session_state.watch_stop_event.set()
        st.session_state.watch_running = False
        st.session_state.watch_stop_event = None
        st.rerun()

    # ── Status + log ───────────────────────────────────────────────────────────
    if st.session_state.watch_running:
        st.success(f"🟢 Watching `{watch_path_input}` for new files…")
    else:
        st.info("⚪ Watcher is stopped.")

    st.subheader("Event log")
    log_placeholder = st.empty()

    logs = st.session_state.watch_logs
    if logs:
        log_placeholder.code("\n".join(logs[-50:]), language="bash")
    else:
        log_placeholder.caption("No events yet. Drop a file into the watched folder.")

    col_refresh, col_clear = st.columns([1, 1])
    with col_refresh:
        if st.button("🔄 Refresh log", use_container_width=True):
            st.rerun()
    with col_clear:
        if st.button("🗑 Clear log", use_container_width=True):
            st.session_state.watch_logs = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# EVAL TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.header("Evaluation suite")
    st.caption(
        f"Runs the 6-case test suite using "
        f"**{DB_OPTIONS[selected_db]['label']}** as the vector store."
    )

    if st.button("▶ Run Eval", type="primary"):
        with st.spinner("Evaluating…"):
            all_results = container.evaluation_service.run_eval(stream=False)

        passed = sum(1 for r in all_results if r.passed)
        avg_lat = sum(r.latency_seconds for r in all_results) / len(all_results)
        avg_score = sum(r.top_score for r in all_results) / len(all_results)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Passed", f"{passed}/{len(all_results)}")
        m2.metric("Score", f"{passed / len(all_results) * 100:.0f}%")
        m3.metric("Avg latency", f"{avg_lat:.1f}s")
        m4.metric("Avg top score", f"{avg_score:.4f}")

        st.divider()
        for r in all_results:
            icon = "✅" if r.passed else "❌"
            with st.expander(f"{icon} {r.case.question}"):
                st.markdown(f"**Answer:** {r.answer}")
                st.markdown(f"**Matched keywords:** `{r.matched_keywords or 'none'}`")
                st.markdown(
                    f"**Top score:** `{r.top_score:.4f}` · "
                    f"**Latency:** `{r.latency_seconds:.2f}s`"
                )
                if r.sources:
                    st.markdown("**Sources:** " + ", ".join(
                        f"{s.source} p{s.page}" for s in r.sources
                    ))


# ══════════════════════════════════════════════════════════════════════════════
# DEBUG TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_debug:
    st.header("Debug retrieval")
    st.caption(
        "Shows landing_zone retrieved chunks and the full assembled prompt. "
        "No LLM call is made — pure retrieval + prompt building only."
    )

    debug_q = st.text_input("Query to debug", key="debug_query")
    debug_k = st.slider("Top-K", 1, 15, 5, key="debug_k")

    if st.button("▶ Debug", type="primary") and debug_q:
        with st.spinner("Retrieving…"):
            # Uses public container APIs — no private attribute access
            results = container.retrieval_service.search(debug_q, top_k=debug_k)
            package = container.build_prompt(debug_q, results)

        st.subheader(f"Retrieved {len(results)} chunks")
        for i, r in enumerate(results):
            with st.expander(
                f"[{i+1}] {r.source} · page {r.page} · "
                f"chunk {r.chunk_index} · score {r.score:.4f}"
            ):
                st.text(r.text)

        st.subheader("Assembled prompt")
        st.code(package.prompt, language="markdown")
        st.caption(
            f"{package.context_chunks_used} chunks used · "
            f"{package.context_chars} context chars"
        )


# ══════════════════════════════════════════════════════════════════════════════
# RELATIONAL DB TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_db:
    st.header("Relational Store Browser")
    st.caption(
        f"Browse chunks saved to SQLite at "
        f"`{container.settings.relational_store.db_path}`"
    )

    relational_store = container.relational_store

    if relational_store is None:
        st.warning(
            "Relational store is disabled. "
            "Set `RELATIONAL_STORE_ENABLED=true` in `.env` and re-ingest."
        )
    else:
        try:
            relational_store.ensure_schema()

            source_input = st.text_input(
                "Search by source filename",
                placeholder="e.g. FCCL-Annual-Report-2023.pdf",
                key="db_source_input",
            )

            col_search, col_delete = st.columns([1, 1])
            with col_search:
                search_clicked = st.button(
                    "🔍 Search", type="primary", use_container_width=True
                )
            with col_delete:
                delete_clicked = st.button(
                    "🗑 Delete source", type="secondary", use_container_width=True
                )

            if search_clicked and source_input.strip():
                docs = relational_store.search_by_source(source_input.strip())
                if not docs:
                    st.warning(f"No chunks found for source `{source_input}`.")
                else:
                    st.success(f"Found **{len(docs)}** chunks for `{source_input}`.")
                    st.divider()

                    for i, doc in enumerate(docs[:50]):  # cap at 50 for UI perf
                        meta = doc.metadata
                        pii_flag = "🔒 " if meta.get("pii_redacted") else ""
                        label = (
                            f"{pii_flag}Chunk {meta.get('chunk_index', i)} · "
                            f"page {meta.get('page', '?')} · "
                            f"{meta.get('word_count', '?')} words"
                        )
                        with st.expander(label):
                            st.text(doc.page_content[:800])
                            with st.expander("Full metadata"):
                                st.json(meta)

                    if len(docs) > 50:
                        st.caption(f"Showing 50 of {len(docs)} chunks.")

            if delete_clicked and source_input.strip():
                deleted = relational_store.delete_by_source(source_input.strip())
                if deleted:
                    st.success(f"Deleted **{deleted}** chunks for `{source_input}`.")
                else:
                    st.warning(f"No chunks found to delete for `{source_input}`.")

            # ── Lookup by vector ID ────────────────────────────────────────────
            st.divider()
            st.subheader("Lookup by Vector ID")
            vector_id_input = st.text_input(
                "Paste a vector ID (SHA-256 prefix)",
                key="db_vector_id",
            )
            if st.button("🔍 Lookup", key="db_lookup_btn") and vector_id_input.strip():
                doc = relational_store.get_chunk_by_vector_id(vector_id_input.strip())
                if doc is None:
                    st.warning("No chunk found for this vector ID.")
                else:
                    st.success("Chunk found!")
                    st.text(doc.page_content)
                    with st.expander("Metadata"):
                        st.json(doc.metadata)

        except Exception as e:
            st.error(f"Relational store error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_settings:
    st.header("Pipeline Settings")
    st.caption(
        "These settings are read from `.env` at startup. "
        "Changes here are informational only — edit `.env` and restart to apply."
    )

    # ── PII ───────────────────────────────────────────────────────────────────
    st.subheader("🔒 PII Anonymization")
    pii = container.settings.pii
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Status", "Enabled ✅" if pii.enabled else "Disabled ❌")
    with col2:
        active_types = list(pii.enabled_types) if pii.enabled_types else ALL_PII_TYPES
        st.metric("Active entity types", len(active_types))

    st.markdown("**Entity types being redacted:**")
    cols = st.columns(3)
    for i, entity in enumerate(active_types):
        cols[i % 3].markdown(f"- `{entity}`")

    st.caption(
        "To change: set `PII_ENABLED=true/false` and "
        "`PII_ENABLED_TYPES=EMAIL,PHONE_PK,CNIC` (blank = all) in `.env`."
    )

    st.divider()

    # ── Chunking ──────────────────────────────────────────────────────────────
    st.subheader("✂️ Chunking")
    ch = container.settings.chunking
    sc = container.settings.semantic_chunking
    c1, c2, c3 = st.columns(3)
    c1.metric("Chunk size", ch.chunk_size)
    c2.metric("Chunk overlap", ch.chunk_overlap)
    c3.metric("Semantic threshold", sc.similarity_threshold)
    st.caption(
        "PDF/DOCX → RecursiveTextChunker  |  "
        "HTML/JSON → SemanticChunker"
    )

    st.divider()

    # ── Relational store ──────────────────────────────────────────────────────
    st.subheader("🗄 Relational Store")
    rs = container.settings.relational_store
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Status", "Enabled ✅" if rs.enabled else "Disabled ❌")
    with col2:
        db_file = Path(rs.db_path)
        size_kb = f"{db_file.stat().st_size / 1024:.1f} KB" if db_file.exists() else "not created"
        st.metric("DB size", size_kb)
    st.caption(f"Path: `{rs.db_path}`")

    st.divider()

    # ── Embedding ─────────────────────────────────────────────────────────────
    st.subheader("🧮 Embedding")
    ol = container.settings.ollama
    c1, c2, c3 = st.columns(3)
    c1.metric("Embed model", ol.embed_model)
    c2.metric("Dimension", ol.embedding_dimension)
    c3.metric("Generation model", ol.generation_model)

    st.divider()

    # ── Retrieval ─────────────────────────────────────────────────────────────
    st.subheader("🔍 Retrieval")
    ret = container.settings.retrieval
    pr = container.settings.prompt
    c1, c2 = st.columns(2)
    c1.metric("Top-K (default)", ret.top_k)
    c2.metric("Max context chars", pr.max_context_chars)