from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from pharma_agent.config import DATA_DIR, INDEX_DIR
from pharma_agent.rag.embeddings import chunk_id, get_embedder


CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks: list[str] = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


def build_index(source_path: Path | None = None) -> int:
    source = source_path or (DATA_DIR / "drug_rules.txt")
    text = source.read_text(encoding="utf-8")
    chunks = split_text(text)
    embedder = get_embedder()
    vectors = embedder.encode(chunks)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))

    metadata = [
        {
            "id": chunk_id(chunk),
            "text": chunk,
            "source": str(source),
        }
        for chunk in chunks
    ]
    (INDEX_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(chunks)


if __name__ == "__main__":
    count = build_index()
    print(f"Built local FAISS index with {count} chunks.")
