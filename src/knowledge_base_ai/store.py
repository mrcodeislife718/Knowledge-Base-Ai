from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from .models import ChunkRecord


class VectorStore:
    """Thin adapter around Chroma plus one explicit embedding model.

    Embeddings are generated here rather than by an implicit Chroma embedding
    function so the exact model is always visible in provenance and validation.
    """

    def __init__(self, path: Path, collection_name: str, model_name: str):
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"embedding_model": model_name, "application": "knowledge-base-ai"},
        )
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Create normalized dense embeddings suitable for cosine-style retrieval."""
        if not texts:
            return []
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Embed and upsert chunks in bounded batches.

        Chroma exposes the maximum supported mutation batch size. Respecting it
        keeps this adapter correct when the corpus grows beyond a small demo.
        """
        if not chunks:
            return

        maximum = max(1, int(self.client.get_max_batch_size()))
        for start in range(0, len(chunks), maximum):
            batch = chunks[start : start + maximum]
            texts = [chunk.text for chunk in batch]
            embeddings = self.embed(texts)
            metadatas = [
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
                for chunk in batch
            ]
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

    def query(self, text: str, top_k: int = 5) -> dict:
        """Run nearest-neighbor retrieval against the current run's collection."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        count = self.count()
        if count == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        embedding = self.embed([text])
        return self.collection.query(
            query_embeddings=embedding,
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        """Return the number of records currently stored in the collection."""
        return self.collection.count()
