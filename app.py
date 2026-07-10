"""
app.py — Streamlit UI for the RAG pipeline.
Run with: streamlit run app.py
"""

from __future__ import annotations

import threading
import queue
from pathlib import Path

import streamlit as st

from src.composition.container import Container
from src.config.settings import VectorStoreType


# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔍",
    layout="wide",
)


# ── DB metadata for UI display ─────────────────────────────────────────────────
DB_OPTIONS = {
    VectorStoreType.PINECONE: {
        "label":       "Pinecone",
        "icon":        "☁️",
        "description": "Cloud-hosted. Requires PINECONE_API_KEY in .env",
    },
    VectorStoreType.CHROMA: {
        "label":       "ChromaDB",
        "icon":        "💾",
        "description": "Local on-disk. No API key needed. Data persists in data/chroma/",
    },
    VectorStoreType.QDRANT: {
        "label":       "Qdrant",
        "icon":        "⚡",
        "description": "Local on-disk or remote server. Data persists in data/qdrant/",
    },
}


# ── Container — cached per (db_type, embed_model, gen_model) ──────────────────
@st.cache_resource(
    show_spinner="Connecting to services...",
    hash_funcs={VectorStoreType: lambda v: v.value},
)
def get_container(db_type: VectorStoreType) -> Container:
    return Container.bootstrap(
        project_root=Path(__file__).parent,
        token_sink=None,          # streaming uses its own thread-local sink
        vector_store_type=db_type,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 RAG Assistant")
    st.divider()

    # ── Database selector ──────────────────────────────────────────────────────
    st.subheader("🗄️ Vector Database")
    selected_db_label = st.selectbox(
        "Select database",
        options=list(DB_OPTIONS.keys()),
        format_func=lambda v: f"{DB_OPTIONS[v]['icon']}  {DB_OPTIONS[v]['label']}",
        index=0,
        label_visibility="collapsed",
    )
    st.caption(DB_OPTIONS[selected_db_label]["description"])

    st.divider()

    container = get_container(selected_db_label)

    st.caption(
        f"**Model:** `{container.settings.ollama.generation_model}`  \n"
        f"**Embed:** `{container.settings.ollama.embed_model}`  \n"
        f"**DB:** `{selected_db_label.value}`  \n"
        f"**Top-K:** `{container.settings.retrieval.top_k}`"
    )
    st.divider()

    top_k = st.slider(
        "Top-K results", min_value=1, max_value=15,
        value=container.settings.retrieval.top_k,
    )
    stream = st.toggle("Stream answer", value=True)
    st.divider()
    st.caption("Tabs: **Ask · Ingest · Eval · Debug**")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_ask, tab_ingest, tab_eval, tab_debug = st.tabs(
    ["💬 Ask", "📥 Ingest", "🧪 Eval", "🔎 Debug"]
)


# ══════════════════════════════════════════════════════════════════════════════
# ASK TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_ask:
    st.header("Ask a question")

    db_key = selected_db_label.value
    msg_key = f"messages_{db_key}"
    if msg_key not in st.session_state:
        st.session_state[msg_key] = []

    for msg in st.session_state[msg_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"- **{s.source}** · page {s.page} · "
                            f"chunk {s.chunk_index} · score `{s.score:.4f}`"
                        )

    if prompt := st.chat_input("Ask anything about your documents…"):
        st.session_state[msg_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if stream:
                token_queue: queue.Queue = queue.Queue()
                result_box: list = []

                def _run():
                    tokens = []
                    def sink(t):
                        token_queue.put(t)
                        tokens.append(t)

                    c = Container.bootstrap(
                        project_root=Path(__file__).parent,
                        token_sink=sink,
                        vector_store_type=selected_db_label,
                    )
                    resp = c.rag_query_service.ask(prompt, top_k=top_k, stream=True)
                    result_box.append(resp)
                    token_queue.put(None)

                thread = threading.Thread(target=_run, daemon=True)
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
                        st.markdown(
                            f"- **{s.source}** · page {s.page} · "
                            f"chunk {s.chunk_index} · score `{s.score:.4f}`"
                        )
                st.caption(
                    f"⏱ {response.latency_seconds:.2f}s · "
                    f"{response.context_chunks_used} chunks used · "
                    f"model: `{response.model}` · "
                    f"db: `{selected_db_label.value}`"
                )
                st.session_state[msg_key].append({
                    "role":    "assistant",
                    "content": response.answer,
                    "sources": response.sources,
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
        f"Documents will be ingested into **{DB_OPTIONS[selected_db_label]['icon']} "
        f"{DB_OPTIONS[selected_db_label]['label']}**. "
        "Switch the database in the sidebar before ingesting if needed.",
        icon="ℹ️",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        custom_path = st.text_input(
            "Directory or file path (leave blank for data/raw)",
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
            with st.spinner(f"Ingesting `{source}` into {selected_db_label.value}…"):
                try:
                    total = container.ingestion_service.ingest_path(source)
                    st.success(
                        f"✅ Indexed **{total}** vectors into "
                        f"**{selected_db_label.value}** from `{source.name}`"
                    )
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# EVAL TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.header("Evaluation suite")
    st.caption(
        f"Runs the 6-case test suite using "
        f"**{DB_OPTIONS[selected_db_label]['label']}** as the vector store."
    )

    if st.button("▶ Run Eval", type="primary"):
        with st.spinner("Evaluating…"):
            all_results = container.evaluation_service.run_eval(stream=False)

        passed   = sum(1 for r in all_results if r.passed)
        avg_lat  = sum(r.latency_seconds for r in all_results) / len(all_results)
        avg_score = sum(r.top_score for r in all_results) / len(all_results)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Passed",        f"{passed}/{len(all_results)}")
        m2.metric("Score",         f"{passed / len(all_results) * 100:.0f}%")
        m3.metric("Avg latency",   f"{avg_lat:.1f}s")
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
        f"Shows raw retrieved chunks and the full assembled prompt — no LLM call. "
        f"Using **{DB_OPTIONS[selected_db_label]['label']}**."
    )

    debug_q = st.text_input("Query to debug", key="debug_query")
    debug_k = st.slider("Top-K", 1, 15, 5, key="debug_k")

    if st.button("▶ Debug", type="primary") and debug_q:
        with st.spinner("Retrieving…"):
            results = container.evaluation_service._retrieval_service.search(
                debug_q, top_k=debug_k
            )
            package = container.evaluation_service._prompt_builder.build(debug_q, results)

        st.subheader(f"Retrieved {len(results)} chunks")
        for i, r in enumerate(results):
            with st.expander(
                f"[{i+1}] {r.source} · page {r.page} · score {r.score:.4f}"
            ):
                st.text(r.text)

        st.subheader("Assembled prompt")
        st.code(package.prompt, language="markdown")
        st.caption(
            f"{package.context_chunks_used} chunks used · "
            f"{package.context_chars} context chars"
        )