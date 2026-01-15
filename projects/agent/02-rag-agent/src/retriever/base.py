from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from src.ingestion.base import Document


class Retriever(ABC):
    """
    检索器基类，定义检索器的核心接口
    """
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        获取检索器的统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "retriever_type": self.__class__.__name__
        }