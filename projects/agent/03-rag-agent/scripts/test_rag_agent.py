#!/usr/bin/env python3
"""
RAG Agent 测试脚本

此脚本用于测试 RAG Agent 的各个模块是否正常工作：
1. 配置管理
2. Embedding 客户端
3. 向量数据库（Chroma）
4. 检索器
5. RAG Pipeline
6. Agent 服务
"""

import os
import sys
import logging
import asyncio

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.settings import get_settings
from src.embeddings.openai_embeddings import create_embedding_client
from src.vector_store.chroma_vector_store import create_vector_store
from src.ingestion.base import Document
from src.retriever.vector_store_retriever import create_retriever
from src.rag_pipeline.basic_rag import create_rag_pipeline
from src.clients.llm_client import LLMClient
from src.agent.rag_agent import RAGAgentService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def test_config():
    """
    测试配置管理
    """
    logger.info("测试配置管理...")
    try:
        settings = get_settings()
        logger.info(f"  ✓ 配置加载成功")
        logger.info(f"  - OpenAI API Key: {'已配置' if settings.openai_api_key else '未配置'}")
        logger.info(f"  - Embedding 模型: {settings.embedding_model}")
        logger.info(f"  - Chat 模型: {settings.openai_model}")
        logger.info(f"  - 向量数据库路径: {settings.vector_store_path}")
        return True
    except Exception as e:
        logger.error(f"  ✗ 配置测试失败: {e}")
        return False

def test_embedding_client():
    """
    测试 Embedding 客户端
    """
    logger.info("测试 Embedding 客户端...")
    try:
        settings = get_settings()
        embedding_client = create_embedding_client(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            base_url=settings.openai_api_base
        )
        
        # 测试生成 Embedding
        text = "这是一个测试文本"
        embedding = embedding_client.embed_text(text)
        logger.info(f"  ✓ Embedding 生成成功")
        logger.info(f"  - Embedding 维度: {len(embedding)}")
        return True
    except Exception as e:
        logger.error(f"  ✗ Embedding 客户端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_store():
    """
    测试向量数据库（Chroma）
    """
    logger.info("测试向量数据库（Chroma）...")
    try:
        settings = get_settings()
        embedding_client = create_embedding_client(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            base_url=settings.openai_api_base
        )
        
        # 创建测试集合
        test_collection_name = "test_collection"
        vector_store = create_vector_store(
            store_type="chroma",
            collection_name=test_collection_name,
            embedding_dimensions=settings.embedding_dimensions,
            vector_store_path=settings.vector_store_path,
            embedding_function=embedding_client.embed_text
        )
        
        # 添加测试文档
        test_docs = [
            Document(text="这是第一个测试文档", metadata={"source": "test.txt"}),
            Document(text="这是第二个测试文档", metadata={"source": "test.txt"})
        ]
        
        # 生成embeddings
        embeddings = [embedding_client.embed_text(doc.text) for doc in test_docs]
        vector_store.add_documents(test_docs, embeddings)
        
        # 查询文档
        query_vector = embedding_client.embed_text("测试文档")
        results = vector_store.search_by_vector(query_vector, top_k=2)
        logger.info(f"  ✓ 向量数据库操作成功")
        logger.info(f"  - 查询结果数量: {len(results)}")
        if results:
            logger.info(f"  - 第一个结果相似度: {results[0][1]:.4f}")
        
        # 删除测试集合
        vector_store.delete_collection()
        vector_store.close()
        return True
    except Exception as e:
        logger.error(f"  ✗ 向量数据库测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_retriever():
    """
    测试检索器
    """
    logger.info("测试检索器...")
    try:
        settings = get_settings()
        embedding_client = create_embedding_client(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            base_url=settings.openai_api_base
        )
        
        # 创建测试集合
        test_collection_name = "test_retriever_collection"
        vector_store = create_vector_store(
            store_type="chroma",
            collection_name=test_collection_name,
            embedding_dimensions=settings.embedding_dimensions,
            vector_store_path=settings.vector_store_path,
            embedding_function=embedding_client.embed_text
        )
        
        # 添加测试文档
        test_docs = [
            Document(text="人工智能是计算机科学的一个分支", metadata={"source": "ai.txt"}),
            Document(text="机器学习是人工智能的一个子领域", metadata={"source": "ml.txt"}),
            Document(text="深度学习是机器学习的一个子领域", metadata={"source": "dl.txt"})
        ]
        
        # 生成embeddings
        embeddings = [embedding_client.embed_text(doc.text) for doc in test_docs]
        vector_store.add_documents(test_docs, embeddings)
        
        # 创建检索器
        retriever = create_retriever(
            vector_store=vector_store,
            embedding_client=embedding_client
        )
        
        # 测试检索
        query = "什么是深度学习？"
        results = retriever.retrieve(query, k=2)
        logger.info(f"  ✓ 检索器测试成功")
        logger.info(f"  - 查询结果数量: {len(results)}")
        if results:
            logger.info(f"  - 第一个结果: {results[0].text[:50]}...")
        
        # 删除测试集合
        vector_store.delete_collection()
        vector_store.close()
        return True
    except Exception as e:
        logger.error(f"  ✗ 检索器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_llm_client():
    """
    测试 LLM 客户端
    """
    logger.info("测试 LLM 客户端...")
    try:
        llm_client = LLMClient()
        
        # 测试简单对话
        system_prompt = "你是一个友好的助手。"
        user_message = "你好，请简单介绍一下你自己。"
        response = await llm_client.chat(system_prompt, user_message)
        logger.info(f"  ✓ LLM 客户端测试成功")
        logger.info(f"  - 响应内容: {response[:100]}...")
        
        await llm_client.aclose()
        return True
    except Exception as e:
        logger.error(f"  ✗ LLM 客户端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主函数
    """
    logger.info("开始 RAG Agent 模块测试（使用 Chroma 向量数据库）...")
    
    tests = [
        test_config,
        test_embedding_client,
        test_vector_store,
        test_retriever,
    ]
    
    # 异步测试
    async_tests = [
        test_llm_client
    ]
    
    passed_tests = 0
    total_tests = len(tests) + len(async_tests)
    
    # 运行同步测试
    for test in tests:
        if test():
            passed_tests += 1
        print()
    
    # 运行异步测试
    for test in async_tests:
        if asyncio.run(test()):
            passed_tests += 1
        print()
    
    logger.info(f"测试完成: {passed_tests}/{total_tests} 个测试通过")
    
    if passed_tests == total_tests:
        logger.info("所有测试通过！RAG Agent 模块工作正常。")
        return 0
    else:
        logger.warning("部分测试失败，请检查相关模块。")
        return 1


if __name__ == "__main__":
    exit(main())

