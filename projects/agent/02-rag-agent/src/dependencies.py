from typing import AsyncGenerator, Optional
from src.config.settings import get_settings
from src.embeddings.openai_embeddings import create_embedding_client
from src.vector_store.faiss_vector_store import create_vector_store
from src.retriever.vector_store_retriever import create_retriever
from src.rag_pipeline.basic_rag import create_rag_pipeline
from src.clients.llm_client import LLMClient
from src.agent.rag_agent import RAGAgentService


async def get_llm_client() -> AsyncGenerator[LLMClient, None]:
    """
    获取 LLM 客户端实例
    """
    llm = LLMClient()
    try:
        yield llm
    finally:
        await llm.aclose()


async def get_rag_agent() -> AsyncGenerator[RAGAgentService, None]:
    """
    获取 RAG Agent 服务实例
    """
    settings = get_settings()
    
    # 1. 创建 Embedding 客户端
    embedding_client = create_embedding_client(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        base_url=settings.openai_base_url
    )
    
    # 2. 创建向量数据库
    vector_store = create_vector_store(
        persist_directory=settings.vector_store_path,
        embedding_client=embedding_client,
        collection_name=settings.vector_store_collection
    )
    
    # 3. 创建检索器
    retriever = create_retriever(
        vector_store=vector_store,
        embedding_client=embedding_client
    )
    
    # 4. 创建 LLM 客户端
    llm_client = LLMClient()
    
    # 5. 创建 RAG Pipeline
    rag_pipeline = create_rag_pipeline(
        retriever=retriever,
        llm_client=llm_client
    )
    
    # 6. 创建并配置 RAG Agent
    agent = RAGAgentService(rag_pipeline=rag_pipeline)
    
    try:
        yield agent
    finally:
        await llm_client.aclose()


async def get_llm_only_agent() -> AsyncGenerator[RAGAgentService, None]:
    """
    获取只使用 LLM 的 Agent 服务实例（不包含 RAG）
    """
    llm_client = LLMClient()
    agent = RAGAgentService(rag_pipeline=None)
    
    try:
        yield agent
    finally:
        await llm_client.aclose()