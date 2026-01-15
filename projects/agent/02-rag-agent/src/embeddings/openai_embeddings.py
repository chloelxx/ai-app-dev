import logging
from typing import List, Optional

import openai
from openai import OpenAIError

from src.config.settings import get_settings
from src.embeddings.base import EmbeddingClient


class OpenAIEmbeddingClient(EmbeddingClient):
    """OpenAI兼容的Embedding客户端"""
    
    def __init__(self, model: Optional[str] = None, dimensions: Optional[int] = None):
        """初始化OpenAI Embedding客户端
        
        Args:
            model: Embedding模型名称
            dimensions: Embedding向量维度
        """
        settings = get_settings()
        
        # 使用配置的模型和维度，如果提供则覆盖
        super().__init__(
            model=model or settings.embedding_model,
            dimensions=dimensions or settings.embedding_dimensions
        )
        
        # 初始化OpenAI客户端
        self.client = openai.OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base
        )
        
        self.batch_size = settings.embedding_batch_size
        self.logger = logging.getLogger(__name__)
    
    def embed_text(self, text: str) -> List[float]:
        """为单个文本生成Embedding向量
        
        Args:
            text: 要生成Embedding的文本
            
        Returns:
            List[float]: Embedding向量
        """
        try:
            # 去除文本中的多余空白字符
            text = " ".join(text.strip().split())
            
            if not text:
                return [0.0] * self.dimensions
            
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            return response.data[0].embedding
        
        except OpenAIError as e:
            self.logger.error(f"OpenAI API错误: {e}")
            raise RuntimeError(f"Embedding生成失败: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Embedding生成过程中发生未知错误: {e}")
            raise RuntimeError(f"Embedding生成失败: {str(e)}")
    
    def embed_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """为多个文本生成Embedding向量
        
        Args:
            texts: 要生成Embedding的文本列表
            batch_size: 批量处理大小
            
        Returns:
            List[List[float]]: Embedding向量列表
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.batch_size
        embeddings = []
        
        try:
            # 处理每个批次
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # 清理文本
                cleaned_texts = [" ".join(text.strip().split()) for text in batch_texts]
                
                # 过滤空文本
                cleaned_texts = [text if text else " " for text in cleaned_texts]
                
                response = self.client.embeddings.create(
                    model=self.model,
                    input=cleaned_texts
                )
                
                # 按输入顺序收集embedding
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            
            return embeddings
        
        except OpenAIError as e:
            self.logger.error(f"OpenAI API错误 (批量处理): {e}")
            raise RuntimeError(f"批量Embedding生成失败: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"批量Embedding生成过程中发生未知错误: {e}")
            raise RuntimeError(f"批量Embedding生成失败: {str(e)}")


# 工厂函数：创建Embedding客户端实例
def create_embedding_client(
    client_type: str = "openai",
    model: Optional[str] = None,
    dimensions: Optional[int] = None
) -> EmbeddingClient:
    """创建Embedding客户端实例
    
    Args:
        client_type: Embedding客户端类型
        model: Embedding模型名称
        dimensions: Embedding向量维度
        
    Returns:
        EmbeddingClient: Embedding客户端实例
    """
    if client_type == "openai":
        return OpenAIEmbeddingClient(model=model, dimensions=dimensions)
    
    raise ValueError(f"不支持的Embedding客户端类型: {client_type}")