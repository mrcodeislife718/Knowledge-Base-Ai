from __future__ import annotations

import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from .models import ChunkRecord

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "why",
    "with",
}


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9']+", text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


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
        """Retrieve semantic candidates, then apply deterministic lexical reranking.

        Dense retrieval supplies recall. A small query-term coverage bonus improves
        precision for named entities and concrete questions without adding another
        model or network dependency. Returned distances remain the original Chroma
        distances so downstream provenance and diagnostics stay honest.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        count = self.count()
        if count == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        candidate_count = min(max(top_k * 4, top_k), count)
        raw = self.collection.query(
            query_embeddings=self.embed([text]),
            n_results=candidate_count,
            include=["documents", "metadatas", "distances"],
        )
        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        query_terms = _terms(text)

        ranked: list[tuple[float, str, str, dict, float]] = []
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            document_terms = _terms(document or "")
            lexical_coverage = (
                len(query_terms & document_terms) / len(query_terms) if query_terms else 0.0
            )
            semantic_similarity = max(0.0, 1.0 - float(distance))
            score = 0.78 * semantic_similarity + 0.22 * lexical_coverage
            ranked.append(
                (score, chunk_id, document or "", metadata or {}, float(distance))
            )

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[:top_k]
        return {
            "ids": [[item[1] for item in selected]],
            "documents": [[item[2] for item in selected]],
            "metadatas": [[item[3] for item in selected]],
            "distances": [[item[4] for item in selected]],
        }

    def count(self) -> int:
        """Return the number of records currently stored in the collection."""
        return self.collection.count()
