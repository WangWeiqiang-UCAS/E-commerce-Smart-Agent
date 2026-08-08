你可以使用conda，poetry等环境，选择最为合适的即可。

可以改造成成熟系统，但不建议继续在现有 Python 数据模型上扩建商城。当前仓库本质是“客服 Agent 演示原型”，合理路线是：

以成熟 Java 开源商城承接商品、库存、订单、支付、售后等交易能力；保留并重构 Python Agent；新增 TypeScript 客服工作台、聊天组件和 BFF。

以下方案按“自营 B2C 商城”设计。如果目标是多商户入驻、分账和商家结算，需要另选商城底座或增加结算域。

一、仓库现状评估

当前仓库约 75 个受控文件、6 张业务表、7 个 API，已经实现了：

FastAPI、LangGraph、Qwen/OpenAI 兼容接口。

简单政策 RAG。

登录、订单查询、退款申请。

Redis Checkpoint、Celery、SSE、WebSocket。

Gradio 用户端和审核端。

基础用户数据隔离。

这些代码适合作为 Agent PoC，不适合作为商城交易核心。

上线阻断问题

当前代码无法正常导入：app/graph/tools.py:203 的 f-string 含反斜杠，Python 编译直接报错。

普通用户可以调用管理员审核接口。两个接口使用的是 get_current_user_id，而不是已存在的管理员鉴权方法：

app/api/v1/admin.py:48

app/api/v1/admin.py:89

Gradio 管理端在客户端本地使用服务端密钥自行签发管理员 JWT，等同于认证绕过：app/frontend/admin_dashboard.py:17。

Compose 定义的是 db，应用却依赖不存在的 postgres，同时仓库没有 Dockerfile：

docker-compose.yaml:5

docker-compose.yaml:43

CORS 使用 * 并同时允许凭证，不符合生产安全要求：app/main.py:20。

退款在 Agent 节点中直接写库，没有幂等键、并发保护、订单明细级退款和库存/支付一致性处理：app/graph/nodes.py:390。

退款审核 Service 仍是空实现：app/services/refund_service.py:247。

订单使用 JSON 保存商品明细、金额使用 float，没有 SKU、库存流水、优惠拆分、支付单、发货包裹、售后明细、发票等正式商城模型。

因此，不建议“修补后继续堆商城功能”，应进行业务归属重构。

二、目标架构

flowchart LR
    U[商城用户] --> STORE[商城前台 Vue/React]
    U --> WIDGET[TS 客服聊天组件]
    CS[人工客服] --> DESK[TS 客服工作台]

    STORE --> GW[Java API Gateway]
    GW --> MALL[Java 商城核心]
    MALL --> MYSQL[(MySQL)]
    MALL --> REDIS[(Redis)]
    MALL --> SEARCH[(OpenSearch)]
    MALL --> MQ[Kafka/RabbitMQ]

    WIDGET --> BFF[TS Agent BFF]
    DESK --> BFF
    BFF --> AGENT[Python Agent Runtime]

    AGENT --> PG[(Agent PostgreSQL + pgvector)]
    AGENT --> LLM[LLM Gateway]
    AGENT --> TOOL[受控 Commerce Tools]
    TOOL --> GW

    MQ --> SYNC[Python 知识/事件同步]
    SYNC --> PG

    AGENT --> OBS[Langfuse + OpenTelemetry]
    BFF --> OBS
    MALL --> OBS

核心原则：

Java 商城是用户、商品、价格、库存、订单、支付、退款的唯一真相源。

Python 不再复制商城订单和用户表，只保存会话、知识、Agent 执行记录和评测数据。

TypeScript 负责聊天 UI、客服工作台、渠道接入和前端协议适配。

LLM 不能直接访问数据库，只能调用白名单业务 API。

所有产生资金、库存、订单状态变化的操作必须幂等、可审计、可确认。

三、开源项目选型

1. Java 商城底座：推荐 macrozheng/mall

建议 fork macrozheng/mall 的 master 分支作为交易底座。它已经包含商品、搜索、购物车、订单、会员、促销、内容、权限和后台管理，当前主分支基于 Spring Boot 3.5 和 JDK 17；许可证是 Apache License 2.0，适合二次开发和商业使用，但必须保留许可证、版权和修改说明。

直接复用：

mall-common

mall-security

mall-mbg

mall-admin

mall-portal

mall-search

mall-admin-web

mall-app-web

建议先使用模块化单体，不要一开始拆成十几个微服务。订单、库存、支付和退款存在大量事务边界，过早微服务化会显著增加一致性成本。

当日订单量、团队规模或独立扩缩容需求明确后，再参考官方的 mall-swarm 拆分网关、认证、商品、订单和搜索服务。

需要在原项目上补强：

SKU 库存预占、释放和库存流水。

支付单、支付回调验签、退款单。

订单状态机和售后状态机。

订单明细级、数量级、部分金额退款。

优惠券和促销金额按明细分摊。

发货包裹、物流轨迹、签收时间。

Outbox 事件表和消息投递补偿。

幂等键、乐观锁、重复回调保护。

数据权限、操作审计、敏感字段加密。

微信/支付宝支付适配器、对象存储、短信适配器。

2. Python Agent

继续使用 FastAPI + LangGraph，但重建领域边界。LangGraph 官方支持持久化、流式输出、故障恢复和人工审批；其 HITL 可以暂停工具调用并在批准、编辑或拒绝后恢复，适合退款、补偿、取消订单等敏感动作。LangGraph HITL 文档、持久化文档。

保留并重构：

app/graph 中的意图路由、RAG 和流式响应思路。

PostgreSQL + pgvector。

FastAPI API 层。

LangGraph Checkpoint。

用户隔离测试场景。

现有政策 ETL 脚本中的文档处理思路。

淘汰：

Python User、Order、RefundApplication 作为业务真相表。

Agent 直接写订单/退款数据库。

Celery 模拟支付、短信和退款。

用金额阈值代替正式风控规则。

依赖正则提取订单号的单轮退款流程。

Gradio 生产前端。

建议拆成五类 Agent 能力：

导购 Agent：自然语言搜商品、规格对比、推荐、库存和活动说明。

订单 Agent：订单状态、物流、发票、地址修改资格。

售后 Agent：退货资格、退款草稿、补充凭证、进度查询。

政策 Agent：运费、会员、退换货、发票等带引用问答。

客服辅助 Agent：会话总结、建议回复、工单分类、风险提示。

不必做多个自由协作的“多 Agent”。使用一个总路由加多个确定性子图，更容易评测、追踪和控制成本。

3. TypeScript 层

建议使用：

Next.js + TypeScript。

assistant-ui 作为聊天 UI 底座。

NestJS 或 Next.js Route Handler 作为 Agent BFF。

React Query、Zod、OpenAPI 生成客户端。

Playwright 做端到端测试。

assistant-ui 是 MIT 许可证的 TypeScript/React 聊天组件库，原生支持流式响应、工具 UI、人工审批，并提供 LangGraph 适配器，适合直接构建商城内嵌客服组件。

TypeScript 工程提供：

customer-widget：嵌入商城的悬浮客服。

agent-desk：人工客服工作台。

agent-bff：鉴权、会话协议、SSE/WebSocket、文件上传。

shared-contracts：从 OpenAPI 生成的请求和事件类型。

tool-ui：订单卡片、物流卡片、退款确认卡片、人工审批卡片。

如果后续需要邮件、WhatsApp、Telegram、社交媒体等全渠道客服，可以旁路部署 Chatwoot。它已经提供统一收件箱、分配、标签、客服协作和多渠道能力；但核心包含 Ruby，因此建议作为独立客服平台通过 Webhook 集成，不要把其源码混入 py+ts Agent 主工程。

4. Agent 可观测性

使用 Langfuse OSS 管理：

Trace、模型耗时和 token 成本。

Prompt 版本。

数据集与离线评测。

用户反馈和会话质量。

不同模型、检索策略 A/B 测试。

Langfuse 核心代码为 MIT，但 ee/ 等企业目录有单独许可证，二次开发时必须隔离检查。许可证说明。

跨 Java、Python、TypeScript 的基础链路统一使用 OpenTelemetry；它可以统一生成和导出 traces、metrics、logs。OpenTelemetry 官方文档。

四、业务数据归属

数据

唯一归属

Agent 获取方式

会员、地址、权限

Java 商城

JWT/JWKS + 用户上下文 API

SPU、SKU、价格、库存

Java 商城

商品查询 API

购物车、促销、优惠券

Java 商城

商城 API

订单、支付、物流

Java 商城

只读 Tool API

售后、退款、退货

Java 商城

草稿/确认/审批 API

会话、消息、附件

Agent PostgreSQL

Python/TS 读写

Agent checkpoint

Agent PostgreSQL

LangGraph Saver

政策知识与向量

Agent PostgreSQL/pgvector

RAG

商品检索索引

OpenSearch

Java 更新、Agent 查询

Trace、评测、Prompt

Langfuse

SDK/OTel

MySQL 和 Agent PostgreSQL 禁止跨库 Join。跨系统展示通过 API 聚合，异步同步通过事件完成。

五、Java 与 Agent 接口设计

不要直接把现有商城内部 API 暴露给模型，新增独立的 agent-commerce-api 防腐层。

只读工具

search_products(query, filters, user_context)

get_product_detail(sku_id)

get_user_orders(cursor, status)

get_order_detail(order_sn)

get_logistics(order_sn)

get_after_sale_status(service_no)

calculate_refund_eligibility(order_item_ids, quantities)

写操作

采用“预览—确认—执行”三步：

POST /agent/actions/refund/preview

前端展示退款范围、金额、运费、原因。

用户显式确认。

POST /agent/actions/refund/confirm

确认请求必须包含：

action_token：短期、单次使用。

idempotency_key。

user_id 和订单所有权校验。

预览结果哈希。

当前价格/订单版本号。

Agent trace ID。

用户确认时间。

Agent 只能创建退款申请，不能自行调用支付退款。真实资金退款由 Java 售后状态机在审核、收货、验货等条件完成后调用支付渠道。

事件

建议先用 RabbitMQ；已有 Kafka 基础设施时再选 Kafka。

product.changed

policy.published

order.created

order.status_changed

shipment.updated

after_sale.created

after_sale.status_changed

refund.completed

每个事件包含 event_id、event_type、schema_version、occurred_at、aggregate_id、trace_id。消费端建立 inbox 表去重，生产端使用 outbox 保证数据库事务和消息的一致性。

六、Agent 流程改造

推荐状态图：

身份确认
  → 安全/越权检测
  → 结构化意图识别
  → 参数收集
  → 调用只读工具
  → 基于证据生成答案
  → 是否涉及写操作
      ├─ 否：回答并记录引用
      └─ 是：生成操作预览
              → 用户确认
              → 风险规则
                  ├─ 低风险：调用受控写 API
                  └─ 高风险：LangGraph interrupt → 人工审批 → 恢复执行

关键约束：

订单所有权必须由 Java 根据登录身份判断，不能信任模型传来的 user_id。

意图识别使用结构化输出 Schema，不解析自由文本标签。

RAG 回答必须携带 document_id/version/chunk_id。

商品价格、库存、物流等实时信息必须来自 Tool，不进入长期向量知识库。

Prompt 中检索到的文档视为不可信输入，防止提示注入。

Tool 名称、参数、调用次数、超时和重试次数全部白名单化。

模型失败时降级到订单卡片、FAQ 搜索或转人工。

不把身份证、手机号、地址、支付信息原文发送给模型。

对取消订单、修改地址、退款等写操作采用策略引擎，不让 LLM 决策资格。

七、客服工作台

TypeScript 客服工作台至少包含：

多队列会话列表和 SLA。

用户画像、历史订单、物流、售后信息。

AI 推荐回复与引用来源。

会话总结、意图、情绪和风险标签。

人工接管/归还机器人。

退款预览与审批。

内部备注、转派、协同。

快捷回复和知识库搜索。

附件与退款凭证查看。

操作审计和会话回放。

CSAT、首次响应时长、解决时长、转人工率。

AI 推荐回复默认由客服点击后发送，不要直接代替客服发送高风险内容。

八、仓库改造后的目录

E-commerce-Smart-Agent/
├── apps/
│   ├── commerce-java/          # fork macrozheng/mall
│   ├── agent-runtime-py/       # FastAPI + LangGraph
│   ├── agent-bff-ts/           # NestJS/Next.js BFF
│   ├── customer-widget-ts/     # 商城聊天组件
│   └── agent-desk-ts/          # 客服工作台
├── packages/
│   ├── api-contracts/          # OpenAPI/AsyncAPI
│   ├── tool-schemas/           # Tool JSON Schema
│   └── ui-components/
├── deploy/
│   ├── compose/
│   ├── helm/
│   └── observability/
├── evals/
│   ├── datasets/
│   ├── graders/
│   └── regression/
├── docs/
│   ├── architecture/
│   ├── adr/
│   ├── api/
│   └── licenses/
└── third_party/
    ├── NOTICE
    └── SBOM

推荐 monorepo，但 Java、Python、TS 分别保留 Maven、Poetry/uv、pnpm 的独立构建链。

九、实施阶段

阶段 0：冻结和修复原型，1 周

修复 Python 语法错误和 Compose。

封堵管理员鉴权漏洞。

禁止前端签发 JWT。

添加 .env.example、Dockerfile、健康检查。

固定依赖和建立最小 CI。

将现有系统标记为 prototype，不再增加商城表。

验收：本地一条命令启动，单元测试和 API 冒烟测试通过，无普通用户审批漏洞。

阶段 1：引入 Java 商城，2～3 周

fork macrozheng/mall 并记录上游 commit。

初始化商品、SKU、会员、购物车、订单、后台。

接入统一用户认证。

补充支付、售后、物流的接口抽象。

定义 OpenAPI 和领域事件。

实现 Agent 防腐 API。

验收：可以完成注册、浏览、加购、下单、模拟支付、发货、查询订单。

阶段 2：重构 Python Agent，3 周

删除订单、用户、退款双写。

实现 Java API Tool Client。

拆分导购、订单、政策、售后子图。

使用正式 LangGraph interrupt/resume。

建立知识文档版本、引用和增量同步。

加入 PII 脱敏、注入防护、限流和超时。

验收：越权查询为零；所有回答可追踪到 Tool 或知识引用；写操作必须确认。

阶段 3：TypeScript 用户端和客服端，3 周

替换 Gradio。

开发商城客服 widget。

开发人工客服工作台。

支持 SSE、断线续传、消息持久化、附件。

支持机器人/人工切换和审核卡片。

订单、物流、退款使用结构化 UI。

验收：刷新页面不丢会话，人工客服可以完整接管，所有审批有审计记录。

阶段 4：真实业务集成，3～4 周

微信/支付宝支付。

物流查询。

对象存储。

短信/邮件。

库存预占与释放。

Outbox/Inbox、补偿和对账任务。

售后收货、验货、退款闭环。

验收：重复回调、重复消费和重复确认不会产生重复订单或退款。

阶段 5：生产加固，2 周

Langfuse、OpenTelemetry、Prometheus、Grafana。

压测、故障注入、安全扫描。

备份恢复演练。

灰度发布和模型降级。

Prompt/模型版本回滚。

SBOM 和开源许可证审计。

合理工期约 12～16 周，需要 5～7 人并行：2 名 Java、1～2 名 Python/LLM、2 名 TS 前端/全栈、1 名测试/DevOps。单人完成更现实的周期是 5～8 个月。

十、测试和上线指标

必须新增：

Java：JUnit、Testcontainers、支付回调和并发库存测试。

Python：pytest、Tool 合约测试、LangGraph checkpoint/resume 测试。

TypeScript：Vitest、Playwright。

合约：OpenAPI diff、AsyncAPI、Pact。

Agent：固定数据集离线回归、越权测试、提示注入测试。

全链路：下单—支付—发货—咨询—申请售后—审批—退款。

建议上线门槛：

订单/支付/退款接口幂等通过率 100%。

他人订单越权测试通过率 100%。

有依据问题的引用正确率 ≥95%。

关键 Tool 参数准确率 ≥98%。

不支持问题的拒答率 ≥98%。

P95 首 token 小于 2.5 秒。

P95 只读客服请求小于 6 秒。

写操作未经确认执行次数为 0。

Agent 故障时商城下单不受影响。

所有退款状态变更可完整审计。

最终建议

最稳妥的第一版组合是：

商城：macrozheng/mall + Spring Boot 3.5 + MySQL + Redis + OpenSearch。

Agent：现有 FastAPI/LangGraph 重构 + PostgreSQL/pgvector。

前端与客服：Next.js/TypeScript + assistant-ui。

消息：RabbitMQ + Outbox/Inbox。

可观测：Langfuse OSS + OpenTelemetry + Prometheus/Grafana。

全渠道客服：第二阶段按需旁路接入 Chatwoot。

现有仓库大约可保留 25%～35% 的 Agent 思路和测试场景，但商城模型、Gradio 前端、模拟支付退款及鉴权体系不应进入生产版本。