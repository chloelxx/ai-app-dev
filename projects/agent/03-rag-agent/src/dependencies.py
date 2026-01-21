from typing import AsyncGenerator, Optional
from src.config.settings import get_settings
from src.embeddings.openai_embeddings import create_embedding_client
from src.vector_store.chroma_vector_store import create_vector_store
# from src.retriever.vector_store_retriever import create_retriever
from src.retriever.factory import create_retriever
from src.rag_pipeline.basic_rag import create_rag_pipeline
from src.clients.llm_client import LLMClient
from src.agent.rag_agent import RAGAgentService


# 1. 加载所有文档（用于 BM25）
from src.ingestion.document_loader import SimpleDocumentLoader
loader = SimpleDocumentLoader()
all_docs = loader.load_directory("./documents")

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
    获取 RAG Agent 服务实例（使用 Chroma 向量数据库）
    """
    settings = get_settings()
    
    # 1. 创建 Embedding 客户端
    embedding_client = create_embedding_client(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        base_url=settings.openai_api_base
    )
    
    # 2. 创建向量数据库（使用 Chroma）
    vector_store = create_vector_store(
        store_type="chroma",
        collection_name=settings.vector_store_collection_name,
        embedding_dimensions=settings.embedding_dimensions,
        vector_store_path=settings.vector_store_path,
        embedding_function=embedding_client.embed_text  # 传递给 Chroma 的嵌入函数
    )
    
    # 3. 创建检索器
    retriever = create_retriever(
        vector_store=vector_store,
        embedding_client=embedding_client,
        documents=all_docs,
        retriever_type="hybrid",
        vector_weight=0.7,
        bm25_weight=0.3,
        language="zh"
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
    agent = RAGAgentService(rag_pipeline=None)
    
    try:
        yield agent
    finally:
        if hasattr(agent, "_llm"):
            await agent._llm.aclose()

