# src/retriever/__init__.py 或 factory.py
from typing import List, Dict, Any, Optional
from src.ingestion.base import Document
from src.retriever.vector_store_retriever import VectorStoreRetriever
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.hybrid_retriever import HybridRetriever
from src.embeddings.base import EmbeddingClient
from src.vector_store.base import VectorStore
from src.retriever.base import Retriever

def create_retriever(
    vector_store: 'VectorStore',
    embedding_client: 'EmbeddingClient',
    documents: List[Document],  # 新增：用于 BM25
    retriever_type: str = "hybrid",  # "vector", "bm25", or "hybrid"
    **kwargs
) -> Retriever:
    """
    创建检索器的工厂函数
    
    Args:
        vector_store: 向量数据库
        embedding_client: Embedding 客户端
        documents: 所有原始文档（用于 BM25）
        retriever_type: 检索器类型
    """
    if retriever_type == "vector":
        return VectorStoreRetriever(vector_store, embedding_client)
    
    elif retriever_type == "bm25":
        lang = kwargs.get("language", "zh")
        return BM25Retriever(documents, language=lang)
    
    elif retriever_type == "hybrid":
        vector_retriever = VectorStoreRetriever(vector_store, embedding_client)
        bm25_retriever = BM25Retriever(documents, language=kwargs.get("language", "zh"))
        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            vector_weight=kwargs.get("vector_weight", 0.6),
            bm25_weight=kwargs.get("bm25_weight", 0.4)
        )
    
    else:
        raise ValueError(f"Unsupported retriever_type: {retriever_type}")