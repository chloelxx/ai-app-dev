from typing import List, Dict, Any, Optional
from src.rag_pipeline.base import RAGPipeline
from src.retriever.base import Retriever
from src.clients.llm_client import LLMClient
from src.ingestion.base import Document
from src.config.settings import get_settings


class BasicRAGPipeline(RAGPipeline):
    """
    基础的 RAG Pipeline 实现，包含检索、上下文构造和 LLM 调用三个核心步骤
    """
    
    def __init__(self, retriever: Retriever, llm_client: LLMClient):
        """
        初始化 RAG Pipeline
        
        Args:
            retriever: 检索器实例
            llm_client: LLM 客户端实例
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.settings = get_settings()
        self._pipeline_runs = 0
        self._retrieval_count = 0
        self._llm_calls = 0
    
    def _build_prompt(self, query: str, context_documents: List[Document]) -> str:
        """
        构建包含上下文和查询的提示词
        
        Args:
            query: 用户查询
            context_documents: 检索到的上下文文档
            
        Returns:
            完整的提示词
        """
        # 构建上下文部分
        context_text = ""
        for i, doc in enumerate(context_documents, 1):
            source = doc.metadata.get("file_name", "未知来源")
            context_text += f"上下文 {i}（来源: {source}）:\n{doc.text}\n\n"
        
        # 构建完整提示词
        prompt = f"""
你是一个基于知识库的问答助手，请严格根据以下提供的上下文回答用户问题。

上下文信息:
{context_text}

用户问题: {query}

要求:
1. 回答必须基于提供的上下文，不得添加任何外部知识
2. 如果上下文没有相关信息，直接回答"根据提供的上下文，我无法回答这个问题"
3. 保持回答简洁明了，不要使用任何引导性或总结性语句
4. 使用与问题相同的语言进行回答
5. 可以在回答中引用来源（如"根据文档 XXX"）
        """
        
        return prompt.strip()
    
    def get_context(self, query: str, k: Optional[int] = None, **kwargs) -> List[Document]:
        """
        获取查询的上下文文档
        
        Args:
            query: 用户查询文本
            k: 返回的文档数量，默认使用配置文件中的值
            **kwargs: 其他检索参数
            
        Returns:
            相关文档列表
        """
        self._retrieval_count += 1
        retrieve_k = k or self.settings.rag_top_k
        return self.retriever.retrieve(query, k=retrieve_k, **kwargs)
    
    async def run(self, query: str, k: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        运行 RAG Pipeline 处理查询并生成响应
        
        Args:
            query: 用户查询文本
            k: 返回的文档数量，默认使用配置文件中的值
            **kwargs: 其他参数
            
        Returns:
            包含响应和相关信息的字典
        """
        self._pipeline_runs += 1
        
        # 1. 检索相关文档
        retrieve_k = k or self.settings.rag_top_k
        context_documents = self.get_context(query, k=retrieve_k, **kwargs)
        
        # 2. 构建提示词
        prompt = self._build_prompt(query, context_documents)
        
        # 3. 调用 LLM 生成响应
        self._llm_calls += 1
        system_prompt = "你是一个基于知识库的问答助手，严格根据提供的上下文回答用户问题。"
        response = await self.llm_client.chat(system_prompt, prompt)
        
        # 4. 准备返回结果
        result = {
            "query": query,
            "response": response,
            "context_documents": [
                {
                    "id": doc.id,
                    "content": doc.text,
                    "metadata": doc.metadata
                }
                for doc in context_documents
            ],
            "retrieved_count": len(context_documents),
            "used_context_count": min(len(context_documents), retrieve_k)
        }
        
        return result
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        获取 Pipeline 的统计信息
        
        Returns:
            统计信息字典
        """
        base_stats = super().get_pipeline_stats()
        retriever_stats = self.retriever.get_retrieval_stats()
        
        return {
            **base_stats,
            "pipeline_runs": self._pipeline_runs,
            "retrieval_count": self._retrieval_count,
            "llm_calls": self._llm_calls,
            "retriever": retriever_stats,
            "rag_top_k": self.settings.rag_top_k
        }


def create_rag_pipeline(retriever: Retriever, llm_client: LLMClient) -> RAGPipeline:
    """
    创建 RAG Pipeline 实例的工厂函数
    
    Args:
        retriever: 检索器实例
        llm_client: LLM 客户端实例
        
    Returns:
        RAG Pipeline 实例
    """
    return BasicRAGPipeline(
        retriever=retriever,
        llm_client=llm_client
    )

