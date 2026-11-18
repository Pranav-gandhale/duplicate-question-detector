"""LLM-based answer synthesis with a deterministic local fallback."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re
import textwrap
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MergeResult:
    answer: str
    mode: str


def merge_answers(
    question: str,
    matches: list[dict[str, Any]],
    api_key: str | None = None,
) -> MergeResult:
    """Synthesize retrieved answers, citing each source thread."""
    if not matches:
        return MergeResult(
            "I could not find a sufficiently similar forum thread. Try adding more detail "
            "or submitting this as a new forum question.",
            "No matching threads",
        )

    credential = api_key or os.getenv("OPENAI_API_KEY")
    if credential:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=credential)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Synthesize one accurate, concise answer using only the supplied "
                            "forum answers. Resolve repetition, preserve important caveats, "
                            "and do not invent facts. Cite claims with source IDs in square "
                            "brackets, such as [PY001]. If sources disagree, say so."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"New question:\n{question}\n\nExisting forum threads:\n"
                            + "\n".join(
                                f"ID: {item['id']}\nQuestion: {item['question']}\n"
                                f"Answer: {item['answer']}"
                                for item in matches
                            )
                        ),
                    },
                ],
            )
            answer = response.choices[0].message.content
            if answer and answer.strip():
                return MergeResult(answer.strip(), "OpenAI")
            logger.warning("OpenAI returned an empty merged answer; using local fallback.")
        except Exception as exc:
            logger.warning("OpenAI answer merging failed; using local fallback: %s", exc)

    return MergeResult(_fallback_merge(matches), "Rule-based fallback")


def summarize_cluster(
    topic: str,
    questions: list[str],
    answers: list[str],
    api_key: str | None = None,
) -> tuple[str, str]:
    """Return a concise FAQ title and takeaway, using extractive fallback."""
    credential = api_key or os.getenv("OPENAI_API_KEY")
    if credential:
        try:
            from openai import OpenAI

            response = OpenAI(api_key=credential).chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": "Write a short FAQ title and one actionable takeaway. "
                        "Return exactly two lines labeled Title: and Takeaway:.",
                    },
                    {
                        "role": "user",
                        "content": f"Topic: {topic}\nQuestions:\n"
                        + "\n".join(f"- {item}" for item in questions[:8])
                        + "\nAnswers:\n"
                        + "\n".join(f"- {item}" for item in answers[:5]),
                    },
                ],
            )
            content = response.choices[0].message.content or ""
            title_match = re.search(r"(?im)^Title:\s*(.+)$", content)
            takeaway_match = re.search(r"(?im)^Takeaway:\s*(.+)$", content)
            if title_match and takeaway_match:
                return title_match.group(1).strip(), takeaway_match.group(1).strip()
            logger.warning("FAQ summarization response did not match the requested format.")
        except Exception as exc:
            logger.warning("OpenAI FAQ summarization failed; using extractive fallback: %s", exc)

    title = _extractive_title(topic, questions)
    first_answer = answers[0] if answers else "Review the related forum threads for practical guidance."
    sentences = re.split(r"(?<=[.!?])\s+", first_answer.strip())
    takeaway = sentences[0] if sentences and sentences[0] else first_answer
    return title, takeaway


def _extractive_title(topic: str, questions: list[str]) -> str:
    if not questions:
        return f"{topic}: common questions"
    stop_words = {
        "a", "an", "and", "are", "can", "do", "does", "for", "from", "how",
        "i", "in", "is", "it", "my", "of", "on", "or", "the", "to", "what",
        "when", "with", "you",
    }
    words = re.findall(r"[A-Za-z][A-Za-z0-9+#.-]*", questions[0])
    keywords = [
        word for word in words
        if word.lower() not in stop_words and word.lower() != topic.lower()
    ]
    excerpt = " ".join(keywords[:5]) or questions[0]
    return f"{topic}: {textwrap.shorten(excerpt, width=58, placeholder='...')}"


def _fallback_merge(matches: list[dict[str, Any]]) -> str:
    """Deduplicate source sentences and attach stable thread citations."""
    seen: set[str] = set()
    sentences: list[str] = []
    sources: list[str] = []
    for item in matches:
        source_id = str(item.get("id", "unknown"))
        sources.append(f"[{source_id}]")
        for sentence in re.split(r"(?<=[.!?])\s+", str(item.get("answer", "")).strip()):
            normalized = re.sub(r"\W+", " ", sentence.lower()).strip()
            if sentence and normalized and normalized not in seen:
                seen.add(normalized)
                sentences.append(sentence)
    source_note = ", ".join(dict.fromkeys(sources))
    return (
        "Based on the related forum threads:\n\n"
        + " ".join(sentences)
        + f"\n\nSources: {source_note}"
    )
