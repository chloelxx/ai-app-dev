# RAG Agent - 基于检索增强生成的智能对话代理

## 项目简介

RAG Agent是一个基于**检索增强生成（Retrieval-Augmented Generation, RAG）**技术的智能对话代理系统。该系统能够从自定义知识库中检索相关信息，结合大语言模型（LLM）生成准确、可靠的回答，解决了传统LLM存在的知识时效性和准确性问题。

## 核心功能

- **文档处理**：支持多种格式文档（TXT、PDF等）的加载和处理
- **文本分块**：智能拆分长文档，优化检索效果
- **向量存储**：使用Chroma DB进行向量存储和相似度检索
- **检索增强生成**：将检索结果作为上下文，增强LLM的回答质量
- **工具集成**：支持计算器等工具调用
- **RESTful API**：提供HTTP接口，方便集成到其他系统

## 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| 后端框架 | FastAPI | 0.104.1 |
| 向量数据库 | Chroma DB | 0.4.24 |
| LLM客户端 | OpenAI API | - |
| HTTP客户端 | httpx | 0.25.2 |
| 配置管理 | python-dotenv | 1.0.0 |
| 文档处理 | PyPDF2 | 3.0.1 |
| 进度显示 | tqdm | 4.66.1 |
| 类型检查 | Pydantic | 2.5.3 |

## 项目结构

```
02-rag-agent/
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
│   └── vector_store/    # 向量数据库
├── scripts/             # 辅助脚本
│   ├── build_index.py   # 知识库索引构建
│   └── test_rag_agent.py # 功能测试
├── documents/           # 知识库文档目录
├── vector_store/        # 向量存储目录
├── requirements.txt     # 依赖列表
├── .env                 # 环境变量配置
└── README.md            # 项目说明文档
```

## 安装和配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建并编辑 `.env` 文件，设置必要的环境变量：

```env
# OpenAI API 配置
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://api.deepseek.com/v1  # 可替换为其他兼容OpenAI API的服务
OPENAI_MODEL=deepseek-v3.2
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=2000

# Embedding 模型配置
EMBEDDING_MODEL=deepseek-chat-embed
EMBEDDING_DIMENSIONS=1024

# 向量数据库配置
VECTOR_STORE_PATH=./vector_store
VECTOR_STORE_COLLECTION_NAME=rag_agent

# RAG 配置
RAG_TOP_K=4
RAG_CHUNK_SIZE=512
RAG_CHUNK_OVERLAP=128

# 文档处理配置
DOCUMENT_DIR=./documents
```

## 使用方法

### 1. 构建知识库

将需要加入知识库的文档放入 `documents/` 目录，然后运行索引构建脚本：

```bash
python scripts/build_index.py
```

可选参数：
- `--document-dir`: 指定文档目录
- `--collection-name`: 指定向量数据库集合名称
- `--chunk-size`: 指定文档分块大小
- `--chunk-overlap`: 指定文档分块重叠大小
- `--rebuild`: 是否重建索引（删除现有集合）

示例：
```bash
python scripts/build_index.py --rebuild --chunk-size 1024
```

### 2. 启动服务

```bash
uvicorn src.main:app --reload
```

服务将在 `http://localhost:8000` 启动。

### 3. 访问API文档

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
  "response": "回答内容",
  "response_type": "rag_answer",
  "tool_name": null,
  "details": {
    "query": "你的问题",
    "context_documents": [
      {
        "content": "相关文档内容...",
        "source": "文档来源",
        "score": 0.95
      }
    ],
    "retrieval_stats": {
      "total_documents": 10,
      "retrieved_documents": 4,
      "retrieval_time": 0.12
    }
  }
}
```

### 统计信息

```
GET /agent/stats
```

返回服务调用统计信息。

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
   - 实现 `embed_query` 和 `embed_documents` 方法
   - 在 `src/embeddings/__init__.py` 中导出新客户端

3. **自定义向量数据库**：
   - 在 `src/vector_store/` 目录下创建新的向量数据库客户端
   - 实现向量存储和检索的基本方法
   - 在 `src/vector_store/__init__.py` 中导出新客户端

## 测试

运行功能测试：

```bash
python scripts/test_rag_agent.py
```

## 注意事项

1. **API密钥安全**：不要将API密钥硬编码到代码中，使用环境变量或配置文件
2. **文档质量**：知识库文档质量直接影响检索效果，建议使用结构化、清晰的文档
3. **性能优化**：
   - 根据文档特点调整分块大小
   - 合理设置检索结果数量（top_k）
   - 定期更新知识库索引

## 许可证

MIT License

## 联系方式

如有问题或建议，欢迎提交Issue或Pull Request。