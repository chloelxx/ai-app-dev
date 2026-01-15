# 03 RAG Agent - 基于检索增强生成的智能对话代理

## 项目简介

03 RAG Agent 是一个基于**检索增强生成（Retrieval-Augmented Generation, RAG）**技术的智能对话代理系统。该系统能够从自定义知识库中检索相关信息，结合大语言模型（LLM）生成准确、可靠的回答，解决了传统LLM存在的知识时效性和准确性问题。

**本项目使用 Chroma 作为向量数据库，无需 Rust 编译，安装更简单。**

## 核心功能

- **文档处理**：支持多种格式文档（TXT、PDF、Markdown）的加载和处理
- **文本分块**：智能拆分长文档，优化检索效果
- **向量存储**：使用 **Chroma DB** 进行向量存储和相似度检索（无需 Rust）
- **检索增强生成**：将检索结果作为上下文，增强LLM的回答质量
- **工具集成**：支持计算器等工具调用
- **RESTful API**：提供HTTP接口，方便集成到其他系统

## 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| 后端框架 | FastAPI | 0.104.1 |
| 向量数据库 | **Chroma DB** | >=0.4.24 |
| LLM客户端 | OpenAI API | >=1.0.0 |
| HTTP客户端 | httpx | 0.25.2 |
| 配置管理 | python-dotenv | 1.0.0 |
| 文档处理 | PyPDF2 | 3.0.1 |
| 进度显示 | tqdm | 4.66.1 |
| 类型检查 | Pydantic | 2.5.3 |

## 项目结构

```
03-rag-agent/
├── src/                  # 源代码目录
│   ├── agent/           # Agent实现
│   ├── api/             # FastAPI接口
│   ├── clients/         # 客户端模块
│   ├── config/          # 配置管理
│   ├── embeddings/      # Embedding实现
│   ├── ingestion/       # 文档处理
│   ├── rag_pipeline/    # RAG流程实现
│   ├── retriever/       # 检索器
│   ├── schemas/         # 数据模型
│   ├── tools/           # 工具函数
│   └── vector_store/    # 向量数据库（Chroma）
├── scripts/             # 辅助脚本
│   ├── build_index.py   # 知识库索引构建
│   └── test_rag_agent.py # 功能测试
├── documents/           # 知识库文档目录
├── vector_store/        # 向量存储目录（Chroma自动创建）
├── requirements.txt     # 依赖列表
├── .env                 # 环境变量配置
└── README.md            # 项目说明文档
```

## 快速开始

### 1. 安装依赖

```bash
cd projects/agent/03-rag-agent
# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖（使用 Chroma，无需 Rust）
pip install -r requirements.txt
```

**注意**：本项目使用 Chroma 作为向量数据库，安装过程简单快速，无需安装 Rust 工具链。

### 2. 配置环境变量

创建并编辑 `.env` 文件，设置必要的环境变量：

```env
# OpenAI/DeepSeek API 配置
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-reasoner
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=2000

# Embedding 模型配置
EMBEDDING_MODEL=deepseek-chat-embed
EMBEDDING_DIMENSIONS=1024
EMBEDDING_BATCH_SIZE=64

# 向量数据库配置（Chroma）
VECTOR_STORE_PATH=./vector_store
VECTOR_STORE_COLLECTION_NAME=rag_agent

# RAG 配置
RAG_TOP_K=4
RAG_CHUNK_SIZE=512
RAG_CHUNK_OVERLAP=128
RAG_CONTEXT_LIMIT=4096

# 文档处理配置
DOCUMENT_DIR=./documents
```

### 3. 构建知识库

将需要加入知识库的文档放入 `documents/` 目录，然后运行索引构建脚本：

```bash
python scripts/build_index.py
```

可选参数：
- `--document-dir`: 指定文档目录路径
- `--collection-name`: 指定向量数据库集合名称
- `--chunk-size`: 指定文档分块大小
- `--chunk-overlap`: 指定文档分块重叠大小
- `--rebuild`: 是否重建索引（删除现有集合）

示例：
```bash
python scripts/build_index.py --rebuild --chunk-size 1024
```

### 4. 启动服务

```bash
uvicorn src.main:app --reload
```

服务将在 `http://localhost:8000` 启动。

### 5. 访问API文档

启动服务后，可以通过以下地址访问自动生成的API文档：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API接口文档

### 健康检查

```
GET /health
```

返回服务健康状态。

### 对话接口

```
POST /agent/chat
```

请求体：
```json
{
  "message": "你的问题",
  "use_rag": true  // 是否使用RAG增强
}
```

响应体：
```json
{
  "reply": "回答内容",
  "response_type": "rag_answer",
  "tool_name": null,
  "details": {
    "query": "你的问题",
    "context_documents": [
      {
        "id": "doc_id",
        "content": "相关文档内容...",
        "metadata": {
          "file_name": "文档来源",
          "chunk_id": 0
        }
      }
    ],
    "retrieved_count": 4,
    "used_context_count": 4
  }
}
```

### 统计信息

```
GET /agent/stats
```

返回服务调用统计信息。

## 使用示例

### 使用 curl 测试

```bash
# 健康检查
curl http://localhost:8000/health

# 发送问题（使用RAG）
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "什么是RAG？", "use_rag": true}'

# 发送问题（不使用RAG，直接LLM）
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好", "use_rag": false}'

# 使用计算器工具
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "calc: 1+2*3"}'
```

### 使用 Python 客户端

```python
import httpx

client = httpx.AsyncClient(base_url="http://localhost:8000")

# 发送问题
response = await client.post("/agent/chat", json={
    "message": "什么是RAG？",
    "use_rag": True
})

result = response.json()
print(result["reply"])
```

## 测试

运行功能测试：

```bash
python scripts/test_rag_agent.py
```

测试脚本会检查：
- 配置管理
- Embedding 客户端
- 向量数据库（Chroma）
- 检索器
- LLM 客户端

## 开发指南

### 代码风格

- 遵循PEP 8代码风格
- 使用类型注解
- 使用Pydantic进行数据验证

### 项目扩展

1. **添加新工具**：
   - 在 `src/tools/` 目录下创建新的工具类
   - 实现工具名称、描述和执行方法
   - 在 `src/agent/rag_agent.py` 中注册工具

2. **自定义Embedding模型**：
   - 在 `src/embeddings/` 目录下创建新的Embedding客户端
   - 实现 `embed_text` 和 `embed_documents` 方法
   - 在 `src/embeddings/__init__.py` 中导出新客户端

3. **自定义向量数据库**：
   - 在 `src/vector_store/` 目录下创建新的向量数据库客户端
   - 实现向量存储和检索的基本方法
   - 在 `src/vector_store/__init__.py` 中导出新客户端

## 常见问题

### 1. Chroma 安装问题

如果遇到 Chroma 安装问题，可以尝试：

```bash
# 升级 pip
pip install --upgrade pip

# 单独安装 Chroma
pip install chromadb
```

### 2. API Key 配置

确保在 `.env` 文件中正确配置了 `OPENAI_API_KEY`，或者通过环境变量设置：

```bash
export OPENAI_API_KEY=your_api_key_here
```

### 3. 向量数据库路径

默认向量数据库存储在 `./vector_store` 目录。如果遇到权限问题，可以修改 `.env` 中的 `VECTOR_STORE_PATH`。

## 注意事项

1. **API密钥安全**：不要将API密钥硬编码到代码中，使用环境变量或配置文件
2. **文档质量**：知识库文档质量直接影响检索效果，建议使用结构化、清晰的文档
3. **性能优化**：
   - 根据文档特点调整分块大小
   - 合理设置检索结果数量（top_k）
   - 定期更新知识库索引
4. **向量数据库**：Chroma 会自动持久化数据到本地，无需手动保存

## 与 02-rag-agent 的区别

本项目（03-rag-agent）与 02-rag-agent 的主要区别：

- **向量数据库**：使用 **Chroma** 替代 FAISS，无需 Rust 编译
- **安装简单**：`pip install -r requirements.txt` 即可，无需额外工具链
- **代码结构**：保持相同的模块化设计，便于理解和扩展

## 许可证

MIT License

## 联系方式

如有问题或建议，欢迎提交Issue或Pull Request。

