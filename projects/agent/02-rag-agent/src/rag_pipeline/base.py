from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.ingestion.base import Document


class RAGPipeline(ABC):
    """
    RAG Pipeline 基类，定义 RAG Pipeline 的核心接口
    """
    
    @abstractmethod
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        运行 RAG Pipeline 处理查询并生成响应
        
        Args:
            query: 用户查询文本
            **kwargs: 其他参数
            
        Returns:
            包含响应和相关信息的字典
        """
        pass
    
    @abstractmethod
    def get_context(self, query: str, **kwargs) -> List[Document]:
        """
        获取查询的上下文文档
        
        Args:
            query: 用户查询文本
            **kwargs: 其他参数
            
        Returns:
            相关文档列表
        """
        pass
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        获取 Pipeline 的统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "pipeline_type": self.__class__.__name__
        }