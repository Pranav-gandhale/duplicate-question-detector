"""Streamlit interface for semantic duplicate-question search and FAQ synthesis."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import streamlit as st

from src.merger import merge_answers
from src.moderation import moderate_text
from src.retriever import QuestionRetriever

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "forum_qa.json"
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Duplicate Question Detector",
    page_icon="🔎",
    layout="wide",
)


@st.cache_data
def load_documents() -> list[dict[str, Any]]:
    with DATA_PATH.open(encoding="utf-8") as handle:
        documents = json.load(handle)
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"{DATA_PATH} must contain a non-empty JSON list.")
    return documents


@st.cache_resource
def load_base_retriever() -> QuestionRetriever:
    return QuestionRetriever(load_documents())


def _show_findings(findings: tuple[str, ...], blocked: bool) -> None:
    if blocked:
        st.error("This text was blocked by the content moderation filter.")
    elif findings:
        readable = ", ".join(item.replace("_", " ") for item in findings)
        st.warning(f"Sensitive content was redacted before processing: {readable}.")


def _search_tab(retriever: QuestionRetriever) -> None:
    st.subheader("Find similar questions")
    question = st.text_area(
        "Enter a new forum question",
        placeholder="For example: How can I keep Docker database files between container restarts?",
        height=100,
        key="search_question_input",
    )
    top_k = st.slider("Number of similar threads", min_value=1, max_value=10, value=3)
    if st.button("Search and synthesize", type="primary"):
        moderation = moderate_text(question)
        _show_findings(moderation.findings, moderation.blocked)
        if moderation.blocked:
            st.session_state.pop("last_search", None)
        elif not moderation.text.strip():
            st.error("Enter a question before searching.")
        else:
            try:
                matches = retriever.search(moderation.text, top_k=top_k)
                result = merge_answers(moderation.text, matches)
                st.session_state["last_search"] = {
                    "query": moderation.text,
                    "matches": matches,
                    "answer": result.answer,
                    "mode": result.mode,
                }
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))

    search = st.session_state.get("last_search")
    if search:
        st.markdown("### Synthesized answer")
        st.caption(f"Answer mode: {search['mode']}")
        st.markdown(search["answer"])
        st.markdown("### Closest existing threads")
        for item in search["matches"]:
            with st.expander(
                f"{item['id']} · {item.get('topic', 'Community')} · "
                f"similarity {item['score']:.3f}: {item['question']}"
            ):
                st.write(item["answer"])


def _visualization_tab(retriever: QuestionRetriever) -> None:
    st.subheader("Question embedding space")
    st.caption(
        "PCA projects the question embeddings into two dimensions. Hover over points "
        "to inspect their text; the latest query and its closest matches are highlighted."
    )
    try:
        from src.visualizer import create_embedding_plot

        search = st.session_state.get("last_search")
        query = search["query"] if search else None
        matches = search["matches"] if search else []
        query_embedding = retriever.encode_query(query) if query else None
        figure = create_embedding_plot(
            retriever.documents,
            retriever.get_embeddings(),
            query=query,
            query_embedding=query_embedding,
            neighbor_ids={item["id"] for item in matches},
        )
        st.plotly_chart(figure, use_container_width=True)
    except ImportError as exc:
        st.error(f"Install the visualization dependencies to use this tab: {exc}")
    except (RuntimeError, ValueError) as exc:
        st.warning(str(exc))


def _ingestion_tab(retriever: QuestionRetriever) -> None:
    st.subheader("Add a question and answer")
    st.write("Personal information is redacted before an entry is added to this session's index.")
    with st.form("ingestion_form", clear_on_submit=True):
        question = st.text_area("Question", key="ingest_question")
        answer = st.text_area("Existing answer", key="ingest_answer")
        topic = st.selectbox("Topic", ["Python", "Git", "Docker", "SQL", "Cloud", "APIs", "Other"])
        submitted = st.form_submit_button("Moderate and add to index")

    if not submitted:
        return
    question_result = moderate_text(question)
    answer_result = moderate_text(answer)
    all_findings = tuple(dict.fromkeys(question_result.findings + answer_result.findings))
    _show_findings(all_findings, question_result.blocked or answer_result.blocked)
    if question_result.blocked or answer_result.blocked:
        st.info("The entry was not indexed. Remove inappropriate language and try again.")
        return
    if not question_result.text.strip() or not answer_result.text.strip():
        st.error("Both a question and an answer are required.")
        return
    new_id = f"USER{len([item for item in retriever.documents if str(item['id']).startswith('USER')]) + 1:03d}"
    try:
        retriever.add_document({
            "id": new_id,
            "topic": topic,
            "cluster": f"user-{topic.lower()}",
            "question": question_result.text,
            "answer": answer_result.text,
        })
    except (ValueError, RuntimeError) as exc:
        st.error(f"Entry was not indexed: {exc}")
        return
    st.success(f"Added sanitized thread {new_id} to your session's search index.")
    if all_findings:
        st.caption("Indexed content contains redaction markers instead of the detected sensitive values.")


def _faq_tab(retriever: QuestionRetriever) -> None:
    st.subheader("Recurring themes")
    st.write(
        "K-Means groups the indexed questions. Each theme gets a concise takeaway "
        "from the existing answers and can be exported as a high-resolution PNG."
    )
    if st.button("Generate FAQ infographic"):
        try:
            from src.visualizer import cluster_faqs, generate_faq_infographic

            faqs = cluster_faqs(retriever, top_n=6, api_key=os.getenv("OPENAI_API_KEY"))
            output_path = ROOT / "faq_infographic.png"
            image_bytes = generate_faq_infographic(faqs, output_path=output_path)
            st.session_state["faq_infographic"] = image_bytes
            st.session_state["faq_summaries"] = faqs
        except ImportError as exc:
            st.error(f"Install the FAQ visualization dependencies first: {exc}")
        except (RuntimeError, ValueError) as exc:
            st.error(f"Could not generate the FAQ infographic: {exc}")

    for faq in st.session_state.get("faq_summaries", []):
        st.markdown(f"**{faq['title']}** · {faq['count']} questions")
        st.write(faq["takeaway"])
    image = st.session_state.get("faq_infographic")
    if image:
        st.image(image, caption="FAQ infographic", use_container_width=True)
        st.download_button(
            "Download faq_infographic.png",
            data=image,
            file_name="faq_infographic.png",
            mime="image/png",
        )


def main() -> None:
    st.title("Duplicate Question Detector")
    st.write(
        "Search a curated forum knowledge base, discover semantically related threads, "
        "and combine their answers into a cited response."
    )
    try:
        if "session_retriever" not in st.session_state:
            st.session_state["session_retriever"] = load_base_retriever().clone()
        retriever: QuestionRetriever = st.session_state["session_retriever"]
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(f"Could not load the forum knowledge base: {exc}")
        st.stop()

    status_left, status_right = st.columns(2)
    status_left.metric("Indexed questions", len(retriever.documents))
    status_right.metric("Embedding backend", retriever.backend)
    if retriever.backend != "sentence-transformers":
        st.info(
            "Sentence-BERT is not active. The app is using a local text-similarity "
            "fallback; install the requirements and ensure the model can be downloaded "
            "to enable semantic embeddings."
        )
    if os.getenv("OPENAI_API_KEY"):
        st.caption(
            "OpenAI is enabled: sanitized questions and retrieved forum content are sent "
            "to OpenAI for answer merging and FAQ summaries."
        )

    search_tab, viz_tab, ingest_tab, faq_tab = st.tabs(
        ["Search & answers", "Embedding map", "Add to knowledge base", "FAQ infographic"]
    )
    with search_tab:
        _search_tab(retriever)
    with viz_tab:
        _visualization_tab(retriever)
    with ingest_tab:
        _ingestion_tab(retriever)
    with faq_tab:
        _faq_tab(retriever)


if __name__ == "__main__":
    main()
