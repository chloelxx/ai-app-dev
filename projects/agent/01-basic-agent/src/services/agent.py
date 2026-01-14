from src.clients.llm_client import LLMClient
from src.tools.calculator import CalculatorTool


SYSTEM_PROMPT = """
你是一个基础的开发学习助手，能够和用户进行自然语言对话。
当用户提出带有明显数学表达式的问题时，你可以调用一个“计算器工具”来得到结果，
然后用自然语言解释给用户听。

如果无法调用大模型（例如缺少 API Key），就直接用你能访问到的信息给出尽量有帮助的回复。
"""


class AgentService:
    def __init__(self) -> None:
        self._llm = LLMClient()
        self._calculator = CalculatorTool()

    async def handle_message(self, message: str) -> str:
        """
        最小示例逻辑：
        - 如果消息以 `calc:` 开头，则使用计算器工具。
        - 否则，将消息转发给大模型。
        """
        text = message.strip()

        if text.lower().startswith("calc:"):
            expr = text[len("calc:") :].strip()
            if not expr:
                return "请在 calc: 后面输入要计算的表达式，例如：calc: 1+2*3"
            result = self._calculator.evaluate(expr)
            return f"表达式 {expr} 的计算结果是：{result}"

        # 默认走大模型对话
        reply = await self._llm.chat(SYSTEM_PROMPT, text)
        return reply


