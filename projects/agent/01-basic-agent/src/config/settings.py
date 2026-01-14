import os
from functools import lru_cache


from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 注意：虽然变量名是 openai_*，但实际可用于任何兼容 OpenAI API 格式的服务（如 DeepSeek）
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    # DeepSeek API 地址
    openai_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        env="OPENAI_BASE_URL",
    )
    # DeepSeek 模型名称
    openai_model: str = Field(
        default="deepseek-r1",
        env="OPENAI_MODEL",
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    # 确保即使没有 .env 文件也不会报错，只是使用默认或环境变量
    return Settings(_env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", ".env"))


