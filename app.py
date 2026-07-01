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


# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔍",
    layout="wide",
)

# ── container (cached for the whole session) ───────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Ollama & Pinecone...")
def get_container() -> Container:
    return Container.bootstrap(project_root=Path(__file__).parent)


container = get_container()

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 RAG Assistant")
    st.caption(
        f"**Model:** `{container.settings.ollama.generation_model}`  \n"
        f"**Embed:** `{container.settings.ollama.embed_model}`  \n"
        f"**Index:** `{container.settings.pinecone.index_name}`  \n"
        f"**Top-K:** `{container.settings.retrieval.top_k}`"
    )
    st.divider()
    top_k = st.slider("Top-K results", min_value=1, max_value=15,
                      value=container.settings.retrieval.top_k)
    stream = st.toggle("Stream answer", value=True)
    st.divider()
    st.caption("Tabs: **Ask · Ingest · Eval · Debug**")


# ── tabs ───────────────────────────────────────────────────────────────────────
tab_ask, tab_ingest, tab_eval, tab_debug = st.tabs(["💬 Ask", "📥 Ingest", "🧪 Eval", "🔎 Debug"])


# ══════════════════════════════════════════════════════════════════════════════
# ASK TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_ask:
    st.header("Ask a question")

    # chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- **{s.source}** · page {s.page} · chunk {s.chunk_index} · score `{s.score:.4f}`")

    if prompt := st.chat_input("Ask anything about your documents…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if stream:
                # stream tokens into a placeholder via a queue + thread
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
                    )
                    resp = c.rag_query_service.ask(prompt, top_k=top_k, stream=True)
                    result_box.append(resp)
                    token_queue.put(None)  # sentinel

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
                    f"model: `{response.model}`"
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                })

    if st.session_state.messages:
        if st.button("Clear chat", key="clear_ask"):
            st.session_state.messages = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# INGEST TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.header("Ingest documents")

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
        source = Path(custom_path.strip()) if custom_path.strip() else container.settings.data_raw
        if not source.exists():
            st.error(f"Path not found: `{source}`")
        else:
            log_area = st.empty()
            with st.spinner(f"Ingesting `{source}`…"):
                try:
                    total = container.ingestion_service.ingest_path(source)
                    st.success(f"✅ Indexed **{total}** vectors from `{source.name}`")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# EVAL TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.header("Evaluation suite")
    st.caption("Runs the built-in 6-case test suite against the Magi story chunks.")

    if st.button("▶ Run Eval", type="primary"):
        results = []
        progress = st.progress(0, text="Running cases…")
        status_area = st.empty()

        from src.application.services.evaluation_service import DEFAULT_EVAL_SUITE
        total_cases = len(DEFAULT_EVAL_SUITE)

        with st.spinner("Evaluating…"):
            all_results = container.evaluation_service.run_eval(stream=False)

        progress.progress(1.0, text="Done!")

        # summary metrics
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
                st.markdown(f"**Top score:** `{r.top_score:.4f}` · **Latency:** `{r.latency_seconds:.2f}s`")
                if r.sources:
                    st.markdown("**Sources:** " + ", ".join(
                        f"{s.source} p{s.page}" for s in r.sources
                    ))


# ══════════════════════════════════════════════════════════════════════════════
# DEBUG TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_debug:
    st.header("Debug retrieval")
    st.caption("Shows raw retrieved chunks and the full assembled prompt — no LLM call.")

    debug_q = st.text_input("Query to debug", key="debug_query")
    debug_k = st.slider("Top-K", 1, 15, 5, key="debug_k")

    if st.button("▶ Debug", type="primary") and debug_q:
        with st.spinner("Retrieving…"):
            from src.application.services.retrieval_service import RetrievalService
            results = container.evaluation_service._retrieval_service.search(
                debug_q, top_k=debug_k
            )
            package = container.evaluation_service._prompt_builder.build(debug_q, results)

        st.subheader(f"Retrieved {len(results)} chunks")
        for i, r in enumerate(results):
            with st.expander(f"[{i+1}] {r.source} · page {r.page} · score {r.score:.4f}"):
                st.text(r.text)

        st.subheader("Assembled prompt")
        st.code(package.prompt, language="markdown")
        st.caption(
            f"{package.context_chunks_used} chunks used · "
            f"{package.context_chars} context chars"
        )