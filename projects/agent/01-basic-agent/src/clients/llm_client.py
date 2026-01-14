import logging

import httpx

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    一个基于 HTTP 的最小 OpenAI Chat Completions 封装。
    说明：
    - 接口路径和字段设计参照 OpenAI 官方 Chat Completions HTTP API。
    - 如官方有更新，请对照文档调整 base_url、路径和请求体结构。
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = httpx.AsyncClient(
            base_url=self._settings.openai_base_url,
            timeout=30.0,
        )

    async def chat(self, system_prompt: str, user_message: str) -> str:
        if not self._settings.openai_api_key:
            # 未配置 API Key 时，返回一个说明性的占位文本，避免直接报错崩溃
            return "LLM 未配置（缺少 OPENAI_API_KEY 环境变量），当前为占位回复。"

        headers = {
            "Authorization": f"Bearer {self._settings.openai_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._settings.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }

        try:
            # 参考：OpenAI Chat Completions HTTP API
            # https://platform.openai.com/docs/api-reference/chat
            response = await self._client.post("/chat/completions", headers=headers, json=payload)
            
            # 检查 HTTP 状态码
            if response.status_code != 200:
                error_detail = response.text
                logger.error(
                    f"LLM API 调用失败: status={response.status_code}, "
                    f"model={self._settings.openai_model}, "
                    f"error={error_detail}"
                )
                return f"LLM API 调用失败（状态码: {response.status_code}）。请检查 API Key 和模型名称是否正确。错误详情: {error_detail[:200]}"
            
            data = response.json()
            
            # 检查返回数据格式
            if "choices" not in data or not data["choices"]:
                logger.error(f"LLM API 返回格式异常: {data}")
                return "LLM API 返回数据格式异常，请检查 API 响应。"
            
            # 按照官方返回格式，从 choices[0].message.content 中读取回复
            content = data["choices"][0]["message"]["content"]
            return content
            
        except httpx.TimeoutException:
            logger.error("LLM API 调用超时")
            return "LLM API 调用超时，请稍后重试。"
        except httpx.RequestError as e:
            logger.error(f"LLM API 网络请求错误: {e}")
            return f"LLM API 网络请求失败: {str(e)}。请检查网络连接和 API 地址。"
        except KeyError as e:
            logger.error(f"LLM API 返回数据缺少必要字段: {e}, data={data if 'data' in locals() else 'N/A'}")
            return "LLM API 返回数据格式不正确，缺少必要字段。"
        except Exception as e:
            logger.error(f"LLM API 调用发生未知错误: {e}", exc_info=True)
            return f"LLM API 调用发生错误: {str(e)}。请查看日志获取详细信息。"

    async def aclose(self) -> None:
        await self._client.aclose()


