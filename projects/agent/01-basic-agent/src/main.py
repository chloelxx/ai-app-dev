import logging

from fastapi import FastAPI

from src.api.routes import router as api_router

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="01 Basic Agent", version="0.1.0")

    @app.get("/health")
    async def health_check() -> dict:
        return {"status": "ok"}

    app.include_router(api_router, prefix="/agent")
    logger.info("Application started successfully")
    return app


app = create_app()


