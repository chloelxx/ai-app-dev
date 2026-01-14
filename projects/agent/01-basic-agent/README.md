# 01 Basic Agent - 工具增强型聊天智能体

本项目是 AI 应用工程师学习路径中的第一个基础 Agent 实战项目，目标是搭建一个**工具增强型聊天智能体**，包含：

- 基于 FastAPI 的 Web 服务
- 调用大模型 API 的 `LLMClient`
- 一个简单的计算器工具（通过特定指令触发）
- `/health` 与 `/agent/chat` 接口

## 目录结构（简要）

- `src/`
  - `main.py`：FastAPI 应用入口
  - `api/`：路由定义（如 `/agent/chat`）
  - `services/agent.py`：Agent 业务逻辑
  - `clients/llm_client.py`：大模型 HTTP 客户端封装
  - `config/settings.py`：配置与环境变量
  - `tools/calculator.py`：计算器工具
- `tests/`：测试用例（待扩展）
- `project-guide.md`：完整项目说明与学习指引
- `requirements.txt`：依赖列表

## 环境准备

```bash
cd projects/agent/01-basic-agent
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell 可使用: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

配置环境变量（以 PowerShell 为例）：

```powershell
$env:OPENAI_API_KEY = "你的_OpenAI_API_Key"
```

如需自定义模型或 Base URL，可设置：

- `OPENAI_MODEL`（默认：`gpt-4.1-mini`）
- `OPENAI_BASE_URL`（默认：`https://api.openai.com/v1`）

## 启动服务

```bash
uvicorn src.main:app --reload
```

验证服务是否正常：

- 健康检查：

```bash
curl http://127.0.0.1:8000/health
```

- 普通对话：

```bash
curl -X POST http://127.0.0.1:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{ "message": "你好，简单介绍一下你能做什么？" }'
```

- 使用计算器工具（以 `calc:` 开头）：

```bash
curl -X POST http://127.0.0.1:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{ "message": "calc: 1+2*3" }'
```

## 运行测试

在项目根目录执行：

```bash
pytest
```

> 建议在 `tests/` 目录中持续补充更多用例，如：错误处理、多轮对话行为等，以配合 `project-guide.md` 中的里程碑逐步完善功能。


