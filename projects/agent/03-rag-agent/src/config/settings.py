import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""
    
    # OpenAI/DeepSeek API 配置
    openai_api_key: str = Field(default="sk-86f2f102a83d4437b7e272c1b12e4161", env="OPENAI_API_KEY")
    openai_api_base: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        env="OPENAI_API_BASE"
    )
    openai_model: str = Field(default="deepseek-r1", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=2000, env="OPENAI_MAX_TOKENS")

    # Embedding 模型配置
    # 注意：如果使用阿里云百炼 API，需要使用支持的 Embedding 模型
    # 常见选项：text-embedding-v1, text-embedding-v2, text-embedding-v3
    embedding_model: str = Field(
        default="text-embedding-v2",
        env="EMBEDDING_MODEL"
    )
    embedding_dimensions: int = Field(default=1536, env="EMBEDDING_DIMENSIONS")
    embedding_batch_size: int = Field(default=64, env="EMBEDDING_BATCH_SIZE")

    # 向量数据库配置（使用 Chroma）
    vector_store_path: str = Field(
        default="./vector_store",
        env="VECTOR_STORE_PATH"
    )
    vector_store_collection_name: str = Field(
        default="rag_agent",
        env="VECTOR_STORE_COLLECTION_NAME"
    )

    # RAG 配置
    rag_top_k: int = Field(default=4, env="RAG_TOP_K")
    rag_chunk_size: int = Field(default=512, env="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=128, env="RAG_CHUNK_OVERLAP")
    rag_context_limit: int = Field(default=4096, env="RAG_CONTEXT_LIMIT")

    # 文档处理配置
    document_dir: str = Field(default="./documents", env="DOCUMENT_DIR")
    document_extensions: list = Field(
        default=[".md", ".txt", ".pdf",".xlsx",".csv"],
        env="DOCUMENT_EXTENSIONS"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """获取配置实例（单例）"""
    return Settings()

