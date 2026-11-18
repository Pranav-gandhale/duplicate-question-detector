"""Semantic question indexing with a dependency-light local fallback."""

from __future__ import annotations

from collections import Counter
import copy
import logging
import math
import re
from typing import Any

from src.moderation import moderate_text

logger = logging.getLogger(__name__)
MODEL_NAME = "all-MiniLM-L6-v2"
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for",
    "from", "how", "i", "in", "is", "it", "my", "of", "on", "or", "the",
    "this", "to", "what", "when", "with",
})
_SYNONYMS = {
    "combine": "merge", "combining": "merge", "join": "merge", "merge": "merge",
    "duplicate": "repeated", "duplicates": "repeated", "repeated": "repeated",
    "shelve": "stash", "shelving": "stash", "stash": "stash",
    "dependencies": "packages", "dependency": "packages", "packages": "packages",
    "environment": "venv", "venv": "venv", "virtualenv": "venv",
    "publish": "port", "expose": "port", "ports": "port", "port": "port",
    "persistent": "volume", "storage": "volume", "volume": "volume",
    "permissions": "access", "permission": "access", "role": "access", "iam": "access",
    "private": "secure", "secure": "secure", "bucket": "bucket",
    "throttle": "ratelimit", "limited": "ratelimit", "rate": "ratelimit",
    "pagination": "page", "paginate": "page", "pages": "page",
    "undo": "reset", "uncommit": "reset", "commit": "commit",
}


class QuestionRetriever:
    """Question index that prefers Sentence-BERT and degrades gracefully."""

    def __init__(self, documents: list[dict[str, Any]], model_name: str = MODEL_NAME):
        if not documents:
            raise ValueError("The question index cannot be empty.")
        self.documents: list[dict[str, Any]] = []
        for document in documents:
            question = str(document.get("question", "")).strip()
            answer = str(document.get("answer", "")).strip()
            if not question or not answer:
                raise ValueError("Each forum entry must include a question and answer.")
            clean_question = moderate_text(question)
            clean_answer = moderate_text(answer)
            if clean_question.blocked or clean_answer.blocked:
                raise ValueError(f"Forum entry {document.get('id', '(unknown)')} was blocked by moderation.")
            clean_document = dict(document)
            clean_document["question"] = clean_question.text
            clean_document["answer"] = clean_answer.text
            self.documents.append(clean_document)

        self.backend = ""
        self.encoder: Any = None
        self.embeddings: Any = None
        self.vectorizer: Any = None
        self._sparse_vectors: list[dict[str, float]] = []
        self._initialize(model_name)

    def _initialize(self, model_name: str) -> None:
        questions = [item["question"] for item in self.documents]
        try:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(model_name)
            self.embeddings = self.encoder.encode(
                questions, convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.backend = "sentence-transformers"
            return
        except ImportError:
            logger.info("sentence-transformers is not installed; using a local TF-IDF fallback.")
        except Exception as exc:
            logger.warning("Could not load Sentence-BERT model %s: %s", model_name, exc)

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vectorizer = TfidfVectorizer(
                lowercase=True, stop_words="english", ngram_range=(1, 2),
                sublinear_tf=True, norm="l2",
            )
            self.embeddings = self.vectorizer.fit_transform(questions)
            self.backend = "scikit-learn-tfidf"
            return
        except ImportError:
            logger.info("scikit-learn is not installed; using the built-in sparse fallback.")

        self._sparse_vectors = [_sparse_tfidf(question, questions) for question in questions]
        self.backend = "builtin-tfidf"

    def _encode_query(self, query: str) -> Any:
        if self.backend == "sentence-transformers":
            return self.encoder.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            )
        if self.backend == "scikit-learn-tfidf":
            return self.vectorizer.transform([query])
        return _sparse_tfidf(query, [item["question"] for item in self.documents])

    def encode_query(self, query: str) -> Any:
        """Return a query vector using the active index's embedding backend."""
        clean_query = moderate_text(query)
        if clean_query.blocked:
            raise ValueError("The question was blocked because it contains inappropriate language.")
        return self._encode_query(clean_query.text)

    def clone(self) -> "QuestionRetriever":
        """Create a session-local index while reusing its immutable model."""
        cloned = copy.copy(self)
        cloned.documents = [dict(document) for document in self.documents]
        cloned._sparse_vectors = [dict(vector) for vector in self._sparse_vectors]
        if self.backend == "scikit-learn-tfidf":
            cloned.vectorizer = copy.deepcopy(self.vectorizer)
        return cloned

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return the highest-scoring questions and their cosine similarities."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        clean_query = moderate_text(query)
        if clean_query.blocked:
            raise ValueError("The question was blocked because it contains inappropriate language.")
        if not clean_query.text.strip():
            raise ValueError("Enter a question to search.")

        query_vector = self._encode_query(clean_query.text)
        if self.backend == "builtin-tfidf":
            scores = [
                _dict_cosine(query_vector, candidate)
                for candidate in self._sparse_vectors
            ]
        else:
            from sklearn.metrics.pairwise import cosine_similarity

            scores = cosine_similarity(query_vector, self.embeddings).ravel().tolist()

        ordered = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        matches = []
        for index in ordered[:min(top_k, len(ordered))]:
            matches.append({**self.documents[index], "score": float(scores[index])})
        return matches

    def add_document(self, document: dict[str, Any]) -> None:
        """Moderate, sanitize, and add a single document to the active index."""
        clean_question = moderate_text(str(document.get("question", "")))
        clean_answer = moderate_text(str(document.get("answer", "")))
        if clean_question.blocked or clean_answer.blocked:
            raise ValueError("The forum entry was blocked because it contains inappropriate language.")
        question = clean_question.text.strip()
        answer = clean_answer.text.strip()
        if not question or not answer:
            raise ValueError("Both a question and an answer are required.")

        clean_document = dict(document, question=question, answer=answer)
        self.documents.append(clean_document)
        try:
            if self.backend == "sentence-transformers":
                added = self.encoder.encode(
                    [question], convert_to_numpy=True, normalize_embeddings=True,
                    show_progress_bar=False,
                )
                import numpy as np

                self.embeddings = np.concatenate((self.embeddings, added), axis=0)
            elif self.backend == "scikit-learn-tfidf":
                self.vectorizer.fit([item["question"] for item in self.documents])
                self.embeddings = self.vectorizer.transform(
                    [item["question"] for item in self.documents]
                )
            else:
                self._sparse_vectors = [
                    _sparse_tfidf(item["question"], [doc["question"] for doc in self.documents])
                    for item in self.documents
                ]
        except Exception:
            self.documents.pop()
            raise

    def get_embeddings(self) -> Any:
        """Return a numeric embedding matrix for visualization."""
        if self.backend == "builtin-tfidf":
            raise RuntimeError(
                "Install scikit-learn to visualize embeddings when Sentence-BERT is unavailable."
            )
        return self.embeddings


def _tokens(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _STOP_WORDS:
            continue
        token = _SYNONYMS.get(token, token)
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        tokens.append(token)
    tokens.extend(
        f"{left}_{right}" for left, right in zip(tokens, tokens[1:])
    )
    return tokens


def _sparse_tfidf(text: str, corpus: list[str]) -> dict[str, float]:
    document_frequency = Counter(
        token for tokens in (set(_tokens(item)) for item in corpus) for token in tokens
    )
    term_frequency = Counter(_tokens(text))
    count = max(1, len(corpus))
    weighted = {
        token: frequency * (math.log((1 + count) / (1 + document_frequency[token])) + 1)
        for token, frequency in term_frequency.items()
    }
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    return {token: value / norm for token, value in weighted.items()} if norm else {}


def _dict_cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return float(sum(value * right.get(token, 0.0) for token, value in left.items()))
