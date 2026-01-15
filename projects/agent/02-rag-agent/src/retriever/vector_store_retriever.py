from typing import List, Dict, Any, Optional
from src.retriever.base import Retriever
from src.ingestion.base import Document
from src.embeddings.base import EmbeddingClient
from src.vector_store.base import VectorStore


class VectorStoreRetriever(Retriever):
    """
    基于向量数据库的检索器，实现了Retriever接口
    """
    
    def __init__(self, vector_store: VectorStore, embedding_client: EmbeddingClient):
        """
        初始化向量数据库检索器
        
        Args:
            vector_store: 向量数据库实例
            embedding_client: Embedding客户端实例
        """
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self._retrieval_count = 0
    
    def retrieve(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """
        根据查询文本检索最相关的文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            **kwargs: 其他检索参数
            
        Returns:
            相关文档列表，按相关性排序
        """
        self._retrieval_count += 1
        
        # 生成查询的embedding
        query_embedding = self.embedding_client.embed_text(query)
        
        # 在向量数据库中搜索
        results = self.vector_store.search_by_vector(
            vector=query_embedding,
            k=k,
            **kwargs
        )
        
        # 转换为Document对象
        documents = []
        for result in results:
            document = Document(
                id=result.get("id", ""),
                content=result.get("text", ""),
                metadata=result.get("metadata", {})
            )
            documents.append(document)
        
        return documents
    
    def retrieve_with_score(self, query: str, k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        根据查询文本检索最相关的文档，并返回相关性分数
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            **kwargs: 其他检索参数
            
        Returns:
            包含文档和分数的字典列表，按分数降序排序
        """
        self._retrieval_count += 1
        
        # 生成查询的embedding
        query_embedding = self.embedding_client.embed_text(query)
        
        # 在向量数据库中搜索，获取带分数的结果
        results = self.vector_store.search_by_vector(
            vector=query_embedding,
            k=k,
            include_scores=True,
            **kwargs
        )
        
        # 转换为包含Document和分数的字典
        documents_with_scores = []
        for result in results:
            document = Document(
                id=result.get("id", ""),
                content=result.get("text", ""),
                metadata=result.get("metadata", {})
            )
            documents_with_scores.append({
                "document": document,
                "score": result.get("score", 0.0)
            })
        
        return documents_with_scores
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        获取检索器的统计信息
        
        Returns:
            统计信息字典
        """
        base_stats = super().get_retrieval_stats()
        return {
            **base_stats,
            "retrieval_count": self._retrieval_count,
            "vector_store_type": self.vector_store.__class__.__name__,
            "embedding_model": self.embedding_client.model
        }


def create_retriever(
    vector_store: VectorStore,
    embedding_client: EmbeddingClient
) -> Retriever:
    """
    创建检索器实例的工厂函数
    
    Args:
        vector_store: 向量数据库实例
        embedding_client: Embedding客户端实例
        
    Returns:
        检索器实例
    """
    return VectorStoreRetriever(
        vector_store=vector_store,
        embedding_client=embedding_client
    )

# 为了兼容test_rag_agent.py中的调用，添加别名
def create_basic_retriever(
    vector_store: VectorStore,
    embedding_client: EmbeddingClient
) -> Retriever:
    """
    创建基础检索器实例的工厂函数（别名）
    
    Args:
        vector_store: 向量数据库实例
        embedding_client: Embedding客户端实例
        
    Returns:
        检索器实例
    """
    return create_retriever(
        vector_store=vector_store,
        embedding_client=embedding_client
    )