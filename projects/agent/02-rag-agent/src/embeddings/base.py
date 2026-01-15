from typing import List, Optional


class EmbeddingClient:
    """Embedding客户端基础接口"""
    
    def __init__(self, model: str, dimensions: Optional[int] = None):
        """初始化Embedding客户端
        
        Args:
            model: Embedding模型名称
            dimensions: Embedding向量维度
        """
        self.model = model
        self.dimensions = dimensions
    
    def embed_text(self, text: str) -> List[float]:
        """为单个文本生成Embedding向量
        
        Args:
            text: 要生成Embedding的文本
            
        Returns:
            List[float]: Embedding向量
        """
        raise NotImplementedError
    
    def embed_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """为多个文本生成Embedding向量
        
        Args:
            texts: 要生成Embedding的文本列表
            batch_size: 批量处理大小
            
        Returns:
            List[List[float]]: Embedding向量列表
        """
        raise NotImplementedError
    
    def get_model_name(self) -> str:
        """获取当前使用的Embedding模型名称
        
        Returns:
            str: 模型名称
        """
        return self.model
    
    def get_dimensions(self) -> Optional[int]:
        """获取Embedding向量维度
        
        Returns:
            Optional[int]: 向量维度
        """
        return self.dimensions