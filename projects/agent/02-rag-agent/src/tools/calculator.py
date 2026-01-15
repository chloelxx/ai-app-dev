import math


class CalculatorTool:
    """
    一个非常简单的“计算器工具”示例。
    为了安全起见，只支持受控的 eval 环境和有限的数学函数。
    """

    name = "calculator"
    description = "执行简单数学表达式计算，例如：1+2*3、sqrt(2)、sin(pi/2)。"

    @staticmethod
    def evaluate(expression: str) -> str:
        allowed_names = {
            k: v
            for k, v in math.__dict__.items()
            if not k.startswith("_")
        }
        allowed_names["abs"] = abs
        allowed_names["round"] = round

        try:
            result = eval(expression, {"__builtins__": {}}, allowed_names)
        except Exception as exc:
            return f"计算失败：{exc}"

        return str(result)