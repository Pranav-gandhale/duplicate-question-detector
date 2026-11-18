"""Interactive embedding plots, FAQ clustering, and PNG infographic rendering."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import textwrap
from typing import Any

from src.merger import summarize_cluster


def create_embedding_plot(
    documents: list[dict[str, Any]],
    embeddings: Any,
    query: str | None = None,
    query_embedding: Any = None,
    neighbor_ids: set[str] | None = None,
) -> Any:
    """Reduce question vectors to 2D and return an interactive Plotly figure."""
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from sklearn.decomposition import PCA

    matrix = embeddings.toarray() if hasattr(embeddings, "toarray") else np.asarray(embeddings)
    labels = [str(item["id"]) for item in documents]
    topics = [str(item.get("topic", "Other")) for item in documents]
    neighbor_ids = neighbor_ids or set()
    points = _reduce_to_2d(matrix)
    frame = pd.DataFrame({
        "x": points[:, 0],
        "y": points[:, 1],
        "topic": topics,
        "question": [str(item["question"]) for item in documents],
        "id": labels,
        "role": ["Top match" if item_id in neighbor_ids else "Indexed question" for item_id in labels],
    })

    if query is not None and query_embedding is not None:
        query_matrix = (
            query_embedding.toarray()
            if hasattr(query_embedding, "toarray")
            else np.asarray(query_embedding)
        ).reshape(1, -1)
        query_point = _project_query(matrix, points, query_matrix)
        frame.loc[len(frame)] = [
            float(query_point[0, 0]), float(query_point[0, 1]),
            "New question", query, "NEW", "New question",
        ]

    figure = px.scatter(
        frame, x="x", y="y", color="topic", symbol="role",
        hover_data={"id": True, "question": True, "role": True, "x": False, "y": False},
        title="Forum question embedding space",
        category_orders={"role": ["Indexed question", "Top match", "New question"]},
    )
    figure.update_traces(marker={"size": 11, "opacity": 0.82})
    figure.update_layout(legend_title_text="Topic / role", template="plotly_white")
    return figure


def cluster_faqs(
    retriever: Any,
    top_n: int = 6,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Cluster indexed questions and summarize the most populated themes."""
    import numpy as np
    from sklearn.cluster import KMeans

    documents = retriever.documents
    if not documents:
        return []
    matrix = retriever.get_embeddings()
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    cluster_count = min(max(1, top_n), len(documents))
    labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(matrix)
    groups: dict[int, list[dict[str, Any]]] = {}
    for label, document in zip(labels, documents):
        groups.setdefault(int(label), []).append(document)

    summaries = []
    ordered_groups = sorted(groups.values(), key=len, reverse=True)
    for group in ordered_groups:
        topics = Counter(str(item.get("topic", "Community")) for item in group)
        topic = topics.most_common(1)[0][0]
        title, takeaway = summarize_cluster(
            topic,
            [item["question"] for item in group],
            [item["answer"] for item in group],
            api_key=api_key,
        )
        summaries.append({
            "topic": topic,
            "title": title,
            "takeaway": takeaway,
            "count": len(group),
            "ids": [item["id"] for item in group],
        })
    return summaries


def generate_faq_infographic(
    faqs: list[dict[str, Any]],
    output_path: str | Path = "faq_infographic.png",
) -> bytes:
    """Render FAQ theme cards into a high-resolution PNG and return its bytes."""
    from PIL import Image, ImageDraw, ImageFont

    width = 1400
    card_height = 260
    padding = 70
    height = 220 + max(1, len(faqs)) * (card_height + 24) + 70
    image = Image.new("RGB", (width, height), "#F3F6FB")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(ImageFont, 46, bold=True)
    heading_font = _load_font(ImageFont, 29, bold=True)
    body_font = _load_font(ImageFont, 23)
    small_font = _load_font(ImageFont, 18)

    draw.rounded_rectangle((40, 35, width - 40, 170), radius=28, fill="#17324D")
    draw.text((padding, 65), "Forum FAQ | Top recurring themes", font=title_font, fill="white")
    draw.text((padding, 185), f"{len(faqs)} themes from the indexed knowledge base",
              font=small_font, fill="#526579")

    palette = ["#2F6B8A", "#386641", "#7251A1", "#B45F06", "#2364AA", "#A23B55"]
    y = 225
    for index, faq in enumerate(faqs):
        color = palette[index % len(palette)]
        draw.rounded_rectangle(
            (40, y, width - 40, y + card_height), radius=24,
            fill="white", outline="#DCE4EE", width=2,
        )
        draw.rounded_rectangle((40, y, 58, y + card_height), radius=8, fill=color)
        draw.text((padding, y + 25), str(faq["title"]), font=heading_font, fill="#17324D")
        draw.text(
            (width - 210, y + 31), f'{faq["count"]} questions',
            font=small_font, fill=color,
        )
        takeaway_lines = textwrap.wrap(str(faq["takeaway"]), width=84)
        for line_number, line in enumerate(takeaway_lines[:4]):
            draw.text((padding, y + 90 + line_number * 34), line,
                      font=body_font, fill="#34495E")
        draw.text((padding, y + card_height - 37), f'Topic: {faq["topic"]}',
                  font=small_font, fill="#718096")
        y += card_height + 24

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG", dpi=(180, 180), optimize=True)
    return target.read_bytes()


def _reduce_to_2d(matrix: Any) -> Any:
    import numpy as np
    from sklearn.decomposition import PCA

    if matrix.shape[0] == 1:
        return np.zeros((1, 2))
    dimensions = min(2, matrix.shape[0], matrix.shape[1])
    coordinates = PCA(n_components=dimensions, random_state=42).fit_transform(matrix)
    if dimensions == 1:
        coordinates = np.column_stack((coordinates[:, 0], np.zeros(matrix.shape[0])))
    return coordinates


def _project_query(matrix: Any, points: Any, query_matrix: Any) -> Any:
    import numpy as np
    from sklearn.decomposition import PCA

    dimensions = min(2, matrix.shape[0], matrix.shape[1])
    reducer = PCA(n_components=dimensions, random_state=42)
    reducer.fit(matrix)
    projected = reducer.transform(query_matrix)
    if dimensions == 1:
        projected = np.column_stack((projected[:, 0], np.zeros(1)))
    return projected


def _load_font(image_font: Any, size: int, bold: bool = False) -> Any:
    candidates = (
        [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\segoeui.ttf"]
        if bold else [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"]
    )
    for candidate in candidates:
        try:
            return image_font.truetype(candidate, size)
        except OSError:
            continue
    return image_font.load_default()
