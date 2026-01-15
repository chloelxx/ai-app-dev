import logging
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.api.routes import router as api_router
from src.config.settings import get_settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 创建应用上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器
    """
    logger.info("正在启动 RAG Agent 应用...")
    
    # 检查必要的环境变量
    settings = get_settings()
    if not settings.openai_api_key:
        logger.warning("未设置 OPENAI_API_KEY 环境变量，将使用占位模式")
    
    # 确保必要的目录存在
    os.makedirs(settings.vector_store_path, exist_ok=True)
    os.makedirs(settings.document_dir, exist_ok=True)
    
    logger.info("RAG Agent 应用启动成功")
    yield
    
    logger.info("正在关闭 RAG Agent 应用...")
    logger.info("RAG Agent 应用已关闭")


def create_app() -> FastAPI:
    """
    创建 FastAPI 应用实例
    """
    app = FastAPI(
        title="02 RAG Agent",
        version="0.1.0",
        description="基于 RAG 的智能对话代理 API",
        lifespan=lifespan
    )

    # 健康检查接口
    @app.get("/health")
    async def health_check() -> dict:
        return {
            "status": "ok",
            "service": "RAG Agent API",
            "version": "0.1.0"
        }

    # 包含 API 路由
    app.include_router(api_router, prefix="/agent")
    
    return app


app = create_app()