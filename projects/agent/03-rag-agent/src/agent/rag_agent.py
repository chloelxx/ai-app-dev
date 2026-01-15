from typing import Dict, Any, Optional
from src.clients.llm_client import LLMClient
from src.rag_pipeline.base import RAGPipeline
from src.tools.calculator import CalculatorTool


SYSTEM_PROMPT = """
你是一个基于知识库的开发学习助手，能够和用户进行自然语言对话并提供专业的技术支持。

功能说明：
1. 当用户提出带有明显数学表达式的问题时，你可以调用计算器工具来得到结果
2. 对于其他技术相关问题，你将使用知识库中的信息进行回答
3. 你必须严格根据提供的上下文信息回答问题，不添加任何外部知识
4. 如果没有相关上下文信息，直接回答"根据提供的上下文，我无法回答这个问题"
5. 保持回答简洁明了，使用专业术语

注意：
- 如果无法调用大模型（例如缺少 API Key），就直接用你能访问到的信息给出尽量有帮助的回复
- 使用与用户问题相同的语言进行回答
"""


class RAGAgentService:
    """
    集成 RAG 能力的对话 Agent 服务
    """
    
    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        """
        初始化 RAG Agent
        
        Args:
            rag_pipeline: RAG Pipeline 实例，默认使用 BasicRAGPipeline
        """
        self._llm = LLMClient()
        self._calculator = CalculatorTool()
        self._rag_pipeline = rag_pipeline
        self._agent_calls = 0
        self._rag_used_count = 0
        self._direct_llm_count = 0
        self._tool_used_count = 0
    
    async def handle_message(self, message: str, use_rag: bool = True) -> Dict[str, Any]:
        """
        处理用户消息，根据消息类型选择不同的处理方式
        
        Args:
            message: 用户消息
            use_rag: 是否使用 RAG Pipeline，默认为 True
            
        Returns:
            包含响应和处理信息的字典
        """
        self._agent_calls += 1
        text = message.strip()
        
        # 1. 检查是否是计算器工具调用
        if text.lower().startswith("calc:"):
            self._tool_used_count += 1
            expr = text[len("calc:"):].strip()
            if not expr:
                return {
                    "response": "请在 calc: 后面输入要计算的表达式，例如：calc: 1+2*3",
                    "type": "tool",
                    "tool_name": "calculator",
                    "details": None
                }
            
            try:
                result = self._calculator.evaluate(expr)
                return {
                    "response": f"表达式 {expr} 的计算结果是：{result}",
                    "type": "tool",
                    "tool_name": "calculator",
                    "details": {
                        "expression": expr,
                        "result": result
                    }
                }
            except Exception as e:
                return {
                    "response": f"计算失败：{str(e)}",
                    "type": "error",
                    "tool_name": "calculator",
                    "details": {
                        "expression": expr,
                        "error": str(e)
                    }
                }
        
        # 2. 如果启用了 RAG 且有 RAG Pipeline 实例，则使用 RAG 处理
        if use_rag and self._rag_pipeline:
            self._rag_used_count += 1
            try:
                rag_result = await self._rag_pipeline.run(text)
                return {
                    "response": rag_result["response"],
                    "type": "rag",
                    "tool_name": None,
                    "details": rag_result
                }
            except Exception as e:
                # 如果 RAG 处理失败，回退到直接调用 LLM
                self._direct_llm_count += 1
                fallback_response = await self._llm.chat(SYSTEM_PROMPT, text)
                return {
                    "response": fallback_response,
                    "type": "fallback",
                    "tool_name": None,
                    "details": {
                        "error": str(e),
                        "message": "RAG 处理失败，已回退到直接 LLM 调用"
                    }
                }
        
        # 3. 默认直接调用 LLM
        self._direct_llm_count += 1
        reply = await self._llm.chat(SYSTEM_PROMPT, text)
        return {
            "response": reply,
            "type": "direct",
            "tool_name": None,
            "details": None
        }
    
    def set_rag_pipeline(self, rag_pipeline: RAGPipeline):
        """
        设置 RAG Pipeline 实例
        
        Args:
            rag_pipeline: RAG Pipeline 实例
        """
        self._rag_pipeline = rag_pipeline
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """
        获取 Agent 的统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "agent_type": self.__class__.__name__,
            "total_calls": self._agent_calls,
            "rag_used_count": self._rag_used_count,
            "direct_llm_count": self._direct_llm_count,
            "tool_used_count": self._tool_used_count
        }
        
        # 如果有 RAG Pipeline，添加其统计信息
        if self._rag_pipeline:
            stats["rag_pipeline_stats"] = self._rag_pipeline.get_pipeline_stats()
        
        return stats
    
    async def aclose(self):
        """
        关闭资源
        """
        if hasattr(self, "_llm"):
            await self._llm.aclose()

