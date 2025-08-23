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

    # Class-level cache to share model across instances
    _shared_model = None
    _model_lock = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None

    def _load_model(self) -> None:
        from sentence_transformers import CrossEncoder
        import threading

        # Use class-level cache to avoid reloading the same model
        if CrossEncoderRerankRetriever._shared_model is None:
            # First time loading - create a lock to prevent multiple downloads
            if CrossEncoderRerankRetriever._model_lock is None:
                CrossEncoderRerankRetriever._model_lock = threading.Lock()

            with CrossEncoderRerankRetriever._model_lock:
                # Double-check pattern to prevent race conditions
                if CrossEncoderRerankRetriever._shared_model is None:
                    print(f"🔄 Loading CrossEncoder model: {self.model_name}")
                    CrossEncoderRerankRetriever._shared_model = CrossEncoder(
                        self.model_name
                    )
                    print(f"✅ CrossEncoder model loaded successfully")

        # Use the shared model instance
        self._model = CrossEncoderRerankRetriever._shared_model

    @classmethod
    def from_vectorstore(
        cls,
        vectordb: VectorStore,
        top_k: int = 20,
        top_n: int = 5,
        model_name: str = "BAAI/bge-reranker-large",
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
        if model is False or model is None:
            # model failed to load; fallback to similarity scores
            return docs[: self.top_n]

        try:
            # Type check to ensure model is a CrossEncoder
            if hasattr(model, "predict"):
                pairs: List[Tuple[str, str]] = [(query, d.page_content) for d in docs]
                # Use getattr to avoid type checker issues
                predict_method = getattr(model, "predict", None)
                if predict_method:
                    scores = predict_method(pairs)
                    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
                    return [doc for _, doc in scored[: self.top_n]]
            # fallback if model doesn't have predict method
            return docs[: self.top_n]
        except Exception:
            # fallback to similarity scores if reranking fails
            return docs[: self.top_n]

    def get_relevant_documents(self, query: str) -> List[Document]:  # type: ignore[override]
        candidates = self._search(query)
        return self._rerank(query, candidates)

    async def aget_relevant_documents(self, query: str) -> List[Document]:  # type: ignore[override]
        # Async wrapper simply calls the sync version for simplicity
        return self.get_relevant_documents(query)
