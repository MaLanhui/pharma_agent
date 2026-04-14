from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss

from pharma_agent.config import INDEX_DIR
from pharma_agent.rag.embeddings import get_embedder


def retrieve(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    index_path = INDEX_DIR / "index.faiss"
    metadata_path = INDEX_DIR / "metadata.json"
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Local FAISS index not found. Run `python -m pharma_agent.rag.build_index` first.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_path))
    embedder = get_embedder()
    query_vector = embedder.encode([query])
    scores, indices = index.search(query_vector, top_k)

    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = dict(metadata[idx])
        item["score"] = round(float(score), 4)
        results.append(item)
    return results
