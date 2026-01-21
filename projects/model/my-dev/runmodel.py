from openai import OpenAI
import os

# 初始化OpenAI客户端
client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key="sk-86f2f102a83d4437b7e272c1b12e4161",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

while(True):
    # 接受用户输入
    user_input = input("请输入您的问题：")
    messages = [{"role": "user", "content": user_input}]
    completion = client.chat.completions.create(
        model="deepseek-v3.2",
        messages=messages,
        # 通过 extra_body 设置 enable_thinking 开启思考模式
        extra_body={"enable_thinking": True},
        stream=True,
        stream_options={
            "include_usage": True
        },
    )

    reasoning_content = ""  # 完整思考过程
    answer_content = ""  # 完整回复
    is_answering = True  # 是否进入回复阶段
    print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

    for chunk in completion:
        if not chunk.choices:
            print("\n" + "=" * 20 + "Token 消耗" + "=" * 20 + "\n")
            print(chunk.usage)
            continue

        delta = chunk.choices[0].delta

        # 只收集思考内容
        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
            if not is_answering:
                print(delta.reasoning_content, end="", flush=True)
            reasoning_content += delta.reasoning_content

        # 收到content，开始进行回复
        if hasattr(delta, "content") and delta.content:
            if not is_answering:
                print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                is_answering = True
            print(delta.content, end="", flush=True)
            answer_content += delta.content

