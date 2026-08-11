from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from .models import ChunkRecord


class VectorStore:
    def __init__(self, path: Path, collection_name: str, model_name: str):
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed(texts)
        metadatas = []
        for chunk in chunks:
            metadatas.append(
                {
                    "chapter": chunk.chapter,
                    "semantic_label": chunk.semantic_label,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "source_path": chunk.source_path,
                    "source_sha256": chunk.source_sha256,
                    "chunk_sha256": chunk.chunk_sha256,
                    "pipeline_version": chunk.pipeline_version,
                    "extraction_methods": ",".join(chunk.extraction_methods),
                    "embedding_model": self.model_name,
                }
            )
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, text: str, top_k: int = 5) -> dict:
        embedding = self.embed([text])
        return self.collection.query(
            query_embeddings=embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        return self.collection.count()
