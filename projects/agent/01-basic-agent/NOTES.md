# 01 Basic Agent 项目开发笔记

> 本文档记录项目开发过程中的问题、思考与后续优化方向

---

## 一、开发过程中遇到的技术问题

### 1.1 模块导入路径问题

**问题描述：**
- 执行 `uvicorn src.main:app --reload` 时报错：`ModuleNotFoundError: No module named 'api'`
- 原因：在 `src/main.py` 中使用相对导入 `from api.routes import ...`，但 Python 无法找到 `api` 模块

**解决方案：**
- 将所有内部导入改为绝对导入，统一使用 `src.` 前缀
- 例如：`from api.routes` → `from src.api.routes`
- 这样当 `uvicorn` 以 `src.main:app` 启动时，Python 能正确解析模块路径

**经验总结：**
- 在 FastAPI 项目中，如果使用 `src/` 目录结构，建议统一使用绝对导入（`src.xxx`）
- 或者考虑在项目根目录添加 `__init__.py` 并将项目根目录加入 `PYTHONPATH`

---

### 1.2 Pydantic v2 兼容性问题

**问题描述：**
- 报错：`PydanticImportError: BaseSettings has been moved to the pydantic-settings package`
- 原因：Pydantic v2 将 `BaseSettings` 从核心包迁移到独立的 `pydantic-settings` 包

**解决方案：**
1. 安装 `pydantic-settings`：`pip install pydantic-settings`
2. 修改导入语句：
   ```python
   # 旧写法（Pydantic v1）
   from pydantic import BaseSettings, Field
   
   # 新写法（Pydantic v2）
   from pydantic import Field
   from pydantic_settings import BaseSettings
   ```
3. 更新 `requirements.txt`，明确指定版本：
   ```
   pydantic>=2.5.0
   pydantic-settings>=2.2.0
   ```

**经验总结：**
- 使用 Pydantic v2 时，必须单独安装 `pydantic-settings` 才能使用 `BaseSettings`
- 建议在项目初期就明确依赖版本，避免后续迁移成本

---

## 二、项目架构设计思考

### 2.1 目录结构设计

当前项目采用分层架构：

```
src/
├── api/          # API 路由层（FastAPI endpoints）
├── services/     # 业务逻辑层（Agent 核心逻辑）
├── clients/      # 外部服务客户端（LLM API 调用）
├── config/       # 配置管理（环境变量、设置）
├── tools/        # 工具定义（计算器、未来可扩展其他工具）
└── schemas/      # 数据模型（请求/响应 Pydantic 模型）
```

**设计优点：**
- 职责清晰，便于维护和测试
- 符合 FastAPI 最佳实践
- 易于扩展新功能（如新增工具、新增客户端）

**可优化点：**
- 未来可考虑引入依赖注入（如 `fastapi.Depends`）管理服务实例
- 可增加 `utils/` 目录存放通用工具函数

---

### 2.2 Agent 工具调用设计

当前实现采用简单的**关键字触发**机制：
- 如果消息以 `calc:` 开头，则调用计算器工具
- 否则，转发给大模型进行对话

**当前限制：**
- 工具选择逻辑过于简单，无法处理复杂意图
- 不支持多工具组合调用
- 工具调用结果没有反馈给大模型进行二次处理

**改进方向：**
- 引入大模型的 Function Calling / Tool Use 能力（如 OpenAI 的 `tools` 参数）
- 让模型自主决定何时调用工具、调用哪个工具
- 实现工具调用链：模型 → 工具 → 结果 → 模型整合 → 最终回复

---

## 三、当前 Basic Agent 的明显限制

### 3.1 功能限制

1. **工具调用能力有限**
   - 仅支持单一工具（计算器）
   - 工具选择逻辑基于简单字符串匹配，无法理解复杂意图
   - 不支持工具链式调用

2. **上下文管理缺失**
   - 没有会话记忆，每次请求都是独立的
   - 无法进行多轮对话
   - 无法记住用户偏好或历史信息

3. **知识库能力缺失**
   - 只能依赖大模型的通用知识
   - 无法访问私有文档、数据库或特定领域知识
   - 回答可能不够准确或不符合业务需求

4. **错误处理不完善**
   - 虽然有空值检查，但错误信息对用户不够友好
   - 缺少重试机制和降级策略
   - 没有详细的错误日志和监控

### 3.2 性能与可扩展性限制

1. **单线程处理**
   - 当前使用同步/异步混合，但未充分利用并发
   - 高并发场景下可能成为瓶颈

2. **无缓存机制**
   - 相同问题会重复调用大模型 API，增加成本和延迟
   - 工具计算结果没有缓存

3. **无批处理能力**
   - 无法批量处理多个请求，效率较低

### 3.3 安全与合规限制

1. **无身份认证**
   - API 接口完全开放，任何人都可以调用
   - 无法追踪用户行为或限制访问频率

2. **无输入验证与过滤**
   - 未对用户输入进行内容安全检查
   - 可能存在注入攻击风险（虽然当前工具使用受限的 `eval`）

3. **无审计日志**
   - 无法记录谁在什么时候调用了什么接口
   - 不符合企业级合规要求

---

## 四、扩展为 RAG Agent 的思考

### 4.1 需要新增的模块

1. **文档处理模块（`src/ingestion/`）**
   - 文档解析器：支持 PDF、Word、Markdown、TXT 等格式
   - 文档分块策略：按段落、按句子、按固定长度等
   - 元数据提取：标题、作者、创建时间等

2. **向量化模块（`src/embeddings/`）**
   - 嵌入模型封装：支持 OpenAI、本地模型（如 sentence-transformers）
   - 批量向量化：提高处理效率
   - 向量维度管理

3. **向量数据库模块（`src/vector_store/`）**
   - 向量数据库客户端封装（Chroma、Milvus、Pinecone 等）
   - 索引构建与更新
   - 相似度搜索接口

4. **检索模块（`src/retriever/`）**
   - 查询重写：将用户问题转换为更适合检索的形式
   - 混合检索：结合关键词检索和向量检索
   - 重排序（Re-ranking）：对检索结果进行二次排序

5. **RAG Pipeline 模块（`src/rag/`）**
   - 检索增强生成流程编排
   - 上下文拼接策略：如何将检索到的文档片段组合成提示词
   - 引用溯源：标记答案来源的文档片段

### 4.2 架构调整建议

```
src/
├── api/              # 保持不变
├── services/
│   ├── agent.py      # 扩展：集成 RAG Pipeline
│   └── rag_service.py # 新增：RAG 核心服务
├── clients/          # 保持不变
├── config/           # 扩展：新增向量数据库配置
├── tools/            # 保持不变
├── schemas/          # 扩展：新增文档上传、检索请求模型
├── ingestion/        # 新增：文档处理
├── embeddings/       # 新增：向量化
├── vector_store/     # 新增：向量数据库
├── retriever/        # 新增：检索逻辑
└── rag/              # 新增：RAG Pipeline
```

### 4.3 关键设计决策

1. **向量数据库选型**
   - **Chroma**：轻量级，适合小规模项目，易于本地部署
   - **Milvus**：企业级，支持大规模数据，需要独立服务
   - **Pinecone**：云服务，无需运维，但需要付费

2. **检索策略**
   - 初期：简单向量检索（Top-K）
   - 进阶：混合检索（BM25 + 向量检索）
   - 高级：多轮检索、查询扩展

3. **上下文窗口管理**
   - 控制检索文档片段的总长度，避免超出模型上下文限制
   - 实现智能截断和摘要

---

## 五、支持多 Agent 协作的设计思路

### 5.1 Agent 角色划分

假设构建一个"智能开发助手系统"，可以设计以下 Agent：

1. **规划 Agent（Planner）**
   - 职责：理解用户需求，拆解任务，制定执行计划
   - 输入：用户原始需求
   - 输出：任务列表、执行顺序、依赖关系

2. **执行 Agent（Executor）**
   - 职责：执行具体任务（代码生成、文档检索、工具调用等）
   - 输入：任务描述、上下文信息
   - 输出：执行结果、状态（成功/失败）

3. **审阅 Agent（Reviewer）**
   - 职责：检查执行结果质量，提供反馈
   - 输入：执行结果、原始需求
   - 输出：质量评估、改进建议

4. **协调 Agent（Coordinator）**
   - 职责：管理 Agent 之间的通信，控制工作流
   - 输入：各 Agent 的输出
   - 输出：下一步指令、最终结果整合

### 5.2 通信机制设计

1. **消息总线模式**
   - 使用消息队列（如 Redis、RabbitMQ）或内存事件总线
   - Agent 通过发布/订阅机制通信

2. **共享状态存储**
   - 使用 Redis 或内存字典存储共享上下文
   - 每个 Agent 可以读取和更新共享状态

3. **工作流编排**
   - 使用框架如 LangGraph、CrewAI 或自建状态机
   - 定义 Agent 之间的依赖关系和执行顺序

### 5.3 架构设计

```
src/
├── services/
│   ├── planner_agent.py    # 规划 Agent
│   ├── executor_agent.py   # 执行 Agent
│   ├── reviewer_agent.py   # 审阅 Agent
│   └── coordinator.py      # 协调器
├── workflows/              # 新增：工作流定义
│   └── dev_assistant_workflow.py
└── messaging/             # 新增：消息通信
    └── message_bus.py
```

---

## 六、企业级部署需要的补充能力

### 6.1 安全与认证

1. **身份认证**
   - API Key 认证
   - OAuth 2.0 / JWT Token 认证
   - 集成企业 SSO（单点登录）

2. **权限控制**
   - 基于角色的访问控制（RBAC）
   - 细粒度权限：哪些用户可以访问哪些功能
   - 数据隔离：多租户支持

3. **输入验证与过滤**
   - 内容安全检查：防止注入攻击、恶意输入
   - 输入长度限制、频率限制
   - 敏感信息过滤（如 PII 数据脱敏）

### 6.2 监控与可观测性

1. **日志系统**
   - 结构化日志（JSON 格式）
   - 日志级别管理（DEBUG、INFO、WARNING、ERROR）
   - 日志聚合与分析（如 ELK Stack）

2. **指标监控**
   - 请求量、响应时间、错误率
   - 大模型 API 调用次数、成本统计
   - 工具调用成功率
   - 使用 Prometheus + Grafana 可视化

3. **链路追踪**
   - 分布式追踪（如 OpenTelemetry）
   - 追踪每个请求的完整生命周期
   - 识别性能瓶颈

4. **告警机制**
   - 错误率超过阈值时告警
   - API 响应时间异常告警
   - 成本超预算告警

### 6.3 性能优化

1. **缓存策略**
   - 对话结果缓存（Redis）
   - 向量检索结果缓存
   - 工具计算结果缓存

2. **异步处理**
   - 长时间任务异步化（如文档索引构建）
   - 使用 Celery 或类似框架处理后台任务

3. **负载均衡**
   - 多实例部署
   - 使用 Nginx 或云负载均衡器
   - 健康检查与自动故障转移

4. **数据库优化**
   - 向量数据库索引优化
   - 查询性能调优
   - 连接池管理

### 6.4 部署与运维

1. **容器化**
   - Docker 镜像构建
   - Docker Compose 本地开发环境
   - Kubernetes 生产部署

2. **CI/CD 流程**
   - 自动化测试（单元测试、集成测试）
   - 自动化构建与部署
   - 版本管理与回滚机制

3. **配置管理**
   - 环境变量管理（开发、测试、生产）
   - 密钥管理（使用密钥管理服务，如 AWS Secrets Manager）
   - 配置热更新

4. **备份与恢复**
   - 向量数据库定期备份
   - 配置与日志备份
   - 灾难恢复预案

### 6.5 成本控制

1. **API 调用优化**
   - 模型选择策略（简单问题用便宜模型）
   - 请求批处理
   - 智能重试与降级

2. **资源监控**
   - 实时成本统计
   - 预算告警
   - 使用量分析报告

---

## 七、后续学习与优化计划

### 7.1 短期优化（1-2 周）

- [ ] 实现会话记忆功能（使用 Redis 或内存存储）
- [ ] 集成 OpenAI Function Calling，让模型自主选择工具
- [ ] 增加更多工具（如时间查询、天气查询）
- [ ] 完善错误处理和日志记录
- [ ] 编写单元测试和集成测试

### 7.2 中期目标（1-2 个月）

- [ ] 完成 `02-rag-agent` 项目，实现知识库问答
- [ ] 学习向量数据库（Chroma/Milvus）的使用
- [ ] 掌握文档处理和嵌入技术
- [ ] 优化 RAG 检索质量

### 7.3 长期目标（3-6 个月）

- [ ] 完成 `03-multi-agent-workflow` 项目
- [ ] 学习工作流编排框架（LangGraph/CrewAI）
- [ ] 完成 `04-enterprise-agent-service` 项目
- [ ] 掌握容器化部署和监控体系
- [ ] 具备独立设计和交付企业级 AI 应用的能力

---

## 八、参考资料

- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [Pydantic v2 迁移指南](https://docs.pydantic.dev/2.12/migration/)
- [OpenAI API 文档](https://platform.openai.com/docs/api-reference)
- [LangChain 文档](https://python.langchain.com/)（后续学习 RAG 和多 Agent 时参考）

---

**最后更新时间：** 2025-01-09  
**项目状态：** ✅ 基础功能已完成，可正常运行

