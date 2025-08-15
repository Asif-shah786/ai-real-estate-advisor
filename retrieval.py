"""Retrieval utilities with cross-encoder reranking."""
from __future__ import annotations

from typing import List, Tuple

from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain.vectorstores.base import VectorStore


class CrossEncoderRerankRetriever(BaseRetriever):
    """Retriever that applies a cross-encoder reranker on vector search results."""

    vectordb: VectorStore
    top_k: int = 20
    top_n: int = 5
    model_name: str = "BAAI/bge-reranker-large"
    _model = None

    def _load_model(self) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_name)

    @classmethod
    def from_vectorstore(
        cls, vectordb: VectorStore, top_k: int = 20, top_n: int = 5, model_name: str = "BAAI/bge-reranker-large"
    ) -> "CrossEncoderRerankRetriever":
        return cls(vectordb=vectordb, top_k=top_k, top_n=top_n, model_name=model_name)

    def _get_model(self):
        if self._model is None:
            try:
                self._load_model()
            except Exception:
                self._model = False  # marker for failed loading
        return self._model

    def _search(self, query: str) -> List[Document]:
        return self.vectordb.similarity_search(query, k=self.top_k)

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        model = self._get_model()
        if model is False:
            # model failed to load; fallback to similarity scores
            return docs[: self.top_n]
        if model is None:
            # should not happen, but fallback anyway
            return docs[: self.top_n]
        pairs: List[Tuple[str, str]] = [(query, d.page_content) for d in docs]
        scores = model.predict(pairs)
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[: self.top_n]]

    def get_relevant_documents(self, query: str) -> List[Document]:  # type: ignore[override]
        candidates = self._search(query)
        return self._rerank(query, candidates)

    async def aget_relevant_documents(self, query: str) -> List[Document]:  # type: ignore[override]
        # Async wrapper simply calls the sync version for simplicity
        return self.get_relevant_documents(query)
