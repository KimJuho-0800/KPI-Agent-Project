# rag_storage.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import os

# PDF Loader (LangChain community)
from langchain_community.document_loaders import PyPDFLoader

# Text splitter (LangChain splitters)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings (Sentence-Transformers via LangChain HuggingFace integration)
from langchain_huggingface import HuggingFaceEmbeddings

# Chroma vectorstore (prefer langchain-chroma, fallback to legacy community import)
try:
    from langchain_chroma import Chroma
except Exception:  # pragma: no cover
    from langchain_community.vectorstores import Chroma  # type: ignore

from langchain_core.documents import Document


@dataclass(frozen=True)
class SearchHit:
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


class KPIMannualRAGStorage:
    """
    kpi_manual.pdf를 ChromaDB에 저장(임베딩/영속화)하고, 질문에 대해 관련 청크를 검색하는 클래스입니다.
    """

    def __init__(
        self,
        pdf_path: str | Path = "kpi_manual.pdf",
        persist_dir: str | Path = "chroma_db",
        collection_name: str = "kpi_manual",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.embedding_model_name = embedding_model_name

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.vector_store: Optional[Chroma] = None

    def build_and_persist(self, rebuild: bool = False) -> int:
        """
        PDF를 로드 → 청킹 → 임베딩 → ChromaDB(persist_dir)에 저장합니다.
        rebuild=False면 기존 DB가 있으면 그대로 재사용하고, 없을 때만 새로 구축합니다.
        반환값은 저장한 청크 개수입니다.
        """
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {self.pdf_path.resolve()}")

        if (not rebuild) and self._has_existing_db():
            self.load()
            return 0

        docs = self._load_pdf()
        chunks = self._split_documents(docs)
        chunks = self._annotate_chunks(chunks)

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )

        self.vector_store.add_documents(chunks)

        if hasattr(self.vector_store, "persist"):
            try:
                self.vector_store.persist()  # 일부 버전에서만 존재
            except Exception:
                pass

        return len(chunks)

    def load(self) -> None:
        """
        persist_dir에 저장된 ChromaDB를 로드합니다.
        """
        if not self._has_existing_db():
            raise FileNotFoundError(
                f"저장된 ChromaDB를 찾을 수 없습니다. 먼저 build_and_persist()를 실행하세요: {self.persist_dir.resolve()}"
            )

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )

    def search(self, question: str, k: int = 4) -> List[SearchHit]:
        """
        사용자 질문과 가장 관련 있는 문서 청크를 k개 반환합니다.
        score가 지원되는 환경이면 score도 포함해 반환합니다.
        """
        if not question or not question.strip():
            raise ValueError("question이 비어 있습니다.")

        self._ensure_loaded()

        assert self.vector_store is not None

        hits: List[SearchHit] = []

        try:
            results: List[Tuple[Document, float]] = self.vector_store.similarity_search_with_score(question, k=k)
            for doc, score in results:
                hits.append(SearchHit(content=doc.page_content, metadata=dict(doc.metadata), score=float(score)))
            return hits
        except Exception:
            docs_only: List[Document] = self.vector_store.similarity_search(question, k=k)
            for doc in docs_only:
                hits.append(SearchHit(content=doc.page_content, metadata=dict(doc.metadata), score=None))
            return hits

    def get_relevant_context(self, question: str, k: int = 4, max_chars: int = 2000) -> str:
        """
        검색된 상위 k개 청크를 이어 붙여, RAG 프롬프트에 바로 넣기 좋은 context 문자열을 반환합니다.
        max_chars로 전체 길이를 제한합니다.
        """
        hits = self.search(question=question, k=k)

        parts: List[str] = []
        total = 0
        for h in hits:
            text = h.content.strip()
            if not text:
                continue
            if total + len(text) > max_chars:
                remain = max(0, max_chars - total)
                if remain > 0:
                    parts.append(text[:remain])
                break
            parts.append(text)
            total += len(text)

        return "\n\n".join(parts)

    def _ensure_loaded(self) -> None:
        if self.vector_store is None:
            self.load()

    def _has_existing_db(self) -> bool:
        if not self.persist_dir.exists():
            return False
        return any(self.persist_dir.iterdir())

    def _load_pdf(self) -> List[Document]:
        loader = PyPDFLoader(str(self.pdf_path))
        return loader.load()

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        return splitter.split_documents(docs)

    @staticmethod
    def _annotate_chunks(chunks: List[Document]) -> List[Document]:
        for i, d in enumerate(chunks):
            md = dict(d.metadata) if d.metadata else {}
            md["chunk_index"] = i
            d.metadata = md
        return chunks


if __name__ == "__main__":
    storage = KPIMannualRAGStorage(
        pdf_path="kpi_manual.pdf",
        persist_dir="chroma_db",
        collection_name="kpi_manual",
        chunk_size=500,
        chunk_overlap=50,
        embedding_model_name="all-MiniLM-L6-v2",
    )

    added = storage.build_and_persist(rebuild=False)
    print(f"indexed_chunks_added={added}")

    q = "불량률(defect_rate) 이상치 기준은 무엇인가요?"
    hits = storage.search(q, k=3)
    for idx, h in enumerate(hits, 1):
        print(f"\n--- HIT {idx} (score={h.score}) ---")
        print(h.content[:600])
        print("metadata:", h.metadata)
