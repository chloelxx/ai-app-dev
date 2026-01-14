import logging

from fastapi import APIRouter, HTTPException

from src.schemas.chat import ChatRequest, ChatResponse
from src.services.agent import AgentService

logger = logging.getLogger(__name__)

router = APIRouter()
agent_service = AgentService()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    处理用户消息并返回 Agent 回复。
    如果发生错误，会返回友好的错误信息而不是 500 错误。
    """
    try:
        reply = await agent_service.handle_message(request.message)
        return ChatResponse(reply=reply)
    except Exception as e:
        logger.error(f"处理消息时发生错误: {e}", exc_info=True)
        # 返回友好的错误信息，而不是抛出 500 错误
        error_message = f"处理消息时发生错误: {str(e)}。请稍后重试或联系管理员。"
        return ChatResponse(reply=error_message)


