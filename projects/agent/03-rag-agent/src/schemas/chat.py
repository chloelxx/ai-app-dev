from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List


class ChatRequest(BaseModel):
    """
    聊天请求模型
    """
    message: str = Field(..., description="用户消息")
    use_rag: bool = Field(default=True, description="是否使用RAG功能")


class ContextDocument(BaseModel):
    """
    上下文文档模型
    """
    id: Optional[str] = Field(default=None, description="文档ID")
    content: str = Field(..., description="文档内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")


class ChatResponseDetails(BaseModel):
    """
    聊天响应详细信息模型
    """
    query: Optional[str] = Field(default=None, description="原始查询")
    context_documents: Optional[List[ContextDocument]] = Field(default=None, description="使用的上下文文档")
    retrieved_count: Optional[int] = Field(default=None, description="检索到的文档数量")
    used_context_count: Optional[int] = Field(default=None, description="使用的上下文数量")
    error: Optional[str] = Field(default=None, description="错误信息")
    message: Optional[str] = Field(default=None, description="附加消息")
    
    # 工具使用相关
    expression: Optional[str] = Field(default=None, description="计算表达式")
    result: Optional[str] = Field(default=None, description="计算结果")


class ChatResponse(BaseModel):
    """
    聊天响应模型
    """
    reply: str = Field(..., description="Agent回复内容")
    response_type: str = Field(default="direct", description="响应类型：direct, rag, tool, fallback, error")
    tool_name: Optional[str] = Field(default=None, description="使用的工具名称")
    details: Optional[ChatResponseDetails] = Field(default=None, description="响应详细信息")

