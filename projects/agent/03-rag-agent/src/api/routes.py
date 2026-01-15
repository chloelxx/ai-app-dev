import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from src.schemas.chat import ChatRequest, ChatResponse, ChatResponseDetails, ContextDocument
from src.agent.rag_agent import RAGAgentService
from src.dependencies import get_rag_agent

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    agent_service: RAGAgentService = Depends(get_rag_agent)
) -> ChatResponse:
    """
    处理用户消息并返回 Agent 回复。
    如果发生错误，会返回友好的错误信息而不是 500 错误。
    """
    try:
        # 调用 Agent 处理消息
        result = await agent_service.handle_message(
            message=request.message,
            use_rag=request.use_rag
        )
        
        # 构建响应对象
        response = ChatResponse(
            reply=result["response"],
            response_type=result["type"],
            tool_name=result["tool_name"],
            details=None
        )
        
        # 如果有详细信息，构建详细信息对象
        if result["details"]:
            details = result["details"]
            
            # 转换上下文文档
            context_documents = None
            if "context_documents" in details and details["context_documents"]:
                context_documents = [
                    ContextDocument(
                        id=doc.get("id"),
                        content=doc.get("content"),
                        metadata=doc.get("metadata", {})
                    )
                    for doc in details["context_documents"]
                ]
            
            response_details = ChatResponseDetails(
                query=details.get("query"),
                context_documents=context_documents,
                retrieved_count=details.get("retrieved_count"),
                used_context_count=details.get("used_context_count"),
                error=details.get("error"),
                message=details.get("message"),
                expression=details.get("expression"),
                result=details.get("result")
            )
            response.details = response_details
        
        return response
    except Exception as e:
        logger.error(f"处理消息时发生错误: {e}", exc_info=True)
        # 返回友好的错误信息，而不是抛出 500 错误
        return ChatResponse(
            reply=f"处理消息时发生错误: {str(e)}。请稍后重试或联系管理员。",
            response_type="error",
            tool_name=None,
            details=ChatResponseDetails(error=str(e))
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_stats(
    agent_service: RAGAgentService = Depends(get_rag_agent)
) -> Dict[str, Any]:
    """
    获取 Agent 的统计信息
    """
    try:
        return agent_service.get_agent_stats()
    except Exception as e:
        logger.error(f"获取统计信息时发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

