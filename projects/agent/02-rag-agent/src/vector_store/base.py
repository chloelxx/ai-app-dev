from typing import Dict, List, Optional, Tuple

from src.ingestion.base import Document


class VectorStore:
    """向量数据库基础接口"""
    
    def __init__(self, collection_name: str, embedding_dimensions: int):
        """初始化向量数据库
        
        Args:
            collection_name: 集合名称
            embedding_dimensions: 向量维度
        """
        self.collection_name = collection_name
        self.embedding_dimensions = embedding_dimensions
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> List[str]:
        """添加文档和对应的向量到数据库
        
        Args:
            documents: 文档列表
            embeddings: 对应的向量列表
            
        Returns:
            List[str]: 添加的文档ID列表
        """
        raise NotImplementedError
    
    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict]] = None) -> List[str]:
        """添加文本和对应的向量到数据库
        
        Args:
            texts: 文本列表
            embeddings: 对应的向量列表
            metadata: 元数据列表
            
        Returns:
            List[str]: 添加的文档ID列表
        """
        raise NotImplementedError
    
    def search_by_vector(self, query_vector: List[float], top_k: int = 4) -> List[Tuple[Document, float]]:
        """根据向量查询相关文档
        
        Args:
            query_vector: 查询向量
            top_k: 返回的文档数量
            
        Returns:
            List[Tuple[Document, float]]: 文档和相似度得分的列表
        """
        raise NotImplementedError
    
    def search_by_text(self, query_text: str, top_k: int = 4) -> List[Tuple[Document, float]]:
        """根据文本查询相关文档
        
        Args:
            query_text: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            List[Tuple[Document, float]]: 文档和相似度得分的列表
        """
        raise NotImplementedError
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """根据文档ID获取文档
        
        Args:
            document_id: 文档ID
            
        Returns:
            Optional[Document]: 文档，如果不存在则返回None
        """
        raise NotImplementedError
    
    def delete_document(self, document_id: str) -> bool:
        """根据文档ID删除文档
        
        Args:
            document_id: 文档ID
            
        Returns:
            bool: 删除成功返回True，否则返回False
        """
        raise NotImplementedError
    
    def delete_collection(self) -> bool:
        """删除整个集合
        
        Returns:
            bool: 删除成功返回True，否则返回False
        """
        raise NotImplementedError
    
    def get_collection_size(self) -> int:
        """获取集合中文档数量
        
        Returns:
            int: 文档数量
        """
        raise NotImplementedError
    
    def close(self) -> None:
        """关闭向量数据库连接
        """
        raise NotImplementedError