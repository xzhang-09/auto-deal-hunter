# 项目质量评估报告(QUALITY.md)

- **评估日期**:2026-07-02
- **评估基线**:分支 `harden-and-document`(工作区含未提交改动),commit `3c7dcea`
- **代码规模**:约 4,350 行 Python(含测试);测试 139 个,全部通过(0.44s,离线运行);`ruff check .` 无告警
- **评估方式**:通读全部源码 + 实际运行测试套件与 lint。本次仅评估,未修改任何代码。

---

## 一、总分一览

| 维度 | 得分 (1-10) | 一句话结论 |
|------|:---:|------|
| 可维护性 | **8** | 分层清晰、注释质量罕见地高;有少量死代码和三处重复的颜色常量 |
| 可靠性 | **7** | 139 个离线测试覆盖核心逻辑;但并发运行无保护、去重有一处静默降级 |
| 工程化程度 | **6** | Docker 一键起、密钥管理干净;但**没有 CI**、依赖未锁定版本 |
| 演进能力 | **7** | domain/ingest/core 边界干净;但 DealNews 假设渗入 domain 层,"按包拆单价"逻辑散布 4 处 |

---

## 二、可维护性:8/10

> 判断标准:一个陌生开发者半年后能否看懂并安全修改。

**结论:能。** 这是本项目最强的维度。

### 做得好的地方

- **目录结构与 README 完全对应**,且分层是真实的而非摆设:
  - `domain/` — 纯数据模型(Pydantic),无 I/O,可安全 import;
  - `ingest/` — 抓取与解析(I/O 层 `scraper.py` 和纯解析 `list_price.py`、`identity.py` 刻意分开,后者可离线测试);
  - `core/` — 业务规则(评分 `scoring.py`、可定价策略 `identity_policy.py`、持久化 `opportunity_store.py`);
  - `agents/` — LLM 调用封装;`app/` — 编排与 UI;`infra/` — 配置/缓存/计量等横切关注点;
  - `evaluation/`(可测的纯指标函数)与 `scripts/`(薄 CLI 壳)分离,是教科书式做法。
- **注释解释"为什么"而非"是什么"**。例如 `agents/pricer_agent.py:11-17` 解释了为什么 prompt 占位价从 123.45 改成 0.00(模型会在 RAG 上下文无用时原样复读占位数字,伪造出看似真实的估价);`ingest/identity.py:14-27` 逐字符解释正则里每个断言防的是哪种真实误判(如把型号 "CT-90325" 读成 90325 件装)。半年后的陌生人能直接读懂决策背景。
- **防"静默漂移"设计**:向量库构建时把 embedding 模型名和距离度量写进 collection 元数据,查询侧(`agents/pricer_agent.py:34-71`)校验不一致就拒绝启动——避免"换了 embedding 模型后检索结果悄悄变成垃圾"这类半年后最难查的坑。

### 问题清单(均为小问题)

1. **死代码(已无人调用)**:
   - `app/orchestrator.py:63-65` 的 `write_memory()` 与它唯一调用的 `core/opportunity_store.py:78-87` 的 `replace_all()` —— 生产代码路径无人调用,仅测试引用。`replace_all` 还是"先 DELETE 全表再重插"的危险写法,留着是隐患。
   - `core/opportunity_store.py:89-98` 的 `update()` —— 全项目零调用。
   - `domain/identity.py:31,33` 的 `bundle_components`、`source` 字段 —— 为未来 LLM 提取层预留,当前无消费者。
   - `agents/agent.py` 中大部分颜色常量(RED/GREEN/YELLOW/MAGENTA)未被任何 Agent 使用。
2. **重复实现:ANSI 颜色常量定义了三份** —— `agents/agent.py:5-13`、`infra/log_utils.py:3-12`、`app/orchestrator.py:21-23`。且 `log_utils.py` 的 `mapper` 必须与 agent 颜色隐式保持同步(agent 换颜色,UI 日志渲染就悄悄失效),这是一处无声耦合。
3. **轻微的"大杂烩"文件**:`app/ui.py`(371 行)混合了 CSS 字符串、日志队列管道、表格格式化、Gradio 布局、以及一个隐藏行为——点击表格行会直接发推送通知(`do_select`,`app/ui.py:287-295`),README 未提及,属于"改布局的人不会想到会触发外部副作用"的暗坑。
4. **跨越私有边界**:`ingest/scraper.py:21` 从 `list_price` 导入下划线开头的 `_higher_than_deal` —— 私有函数被兄弟模块依赖,重构时容易被误删。
5. **命名遗留**:`Orchestrator.memory` / `read_memory` / `MEMORY_FILENAME` 沿用了早期"JSON 记忆"时代的叫法,实际底下已是 SQLite 存储;`read_memory` 里 `getattr(self, "opportunity_store", ...)` 的防御式回退(`app/orchestrator.py:60`)让人困惑——正常构造路径下永远不会触发。

---

## 三、可靠性:7/10

### 测试覆盖

核心业务逻辑有真实覆盖,且**测试全部离线可跑**(网络、OpenAI、SentenceTransformer 都被 stub),0.44 秒跑完 139 个:

| 覆盖点 | 测试文件 |
|---|---|
| 确定性选优(guardrail 封顶、多件装总折扣) | test_deal_scoring.py |
| 商品身份识别(35 个用例,含各种误判反例) | test_identity.py |
| 价格/原价正则提取 | test_extract_list_price.py |
| SQLite 存储(含 schema 迁移、TTL 清理) | test_opportunity_store.py |
| MCP 客户端配对/交接逻辑 | test_mcp_client.py |
| token 计量、HTTP 缓存、eval 指标 | test_usage / test_http_cache / test_eval_* |

未覆盖的主要是`app/ui.py` 的 Gradio 交互流(仅 helper 有测)和真实的端到端 MCP 往返(可理解,需要子进程)。

### 错误处理与日志

总体完善:抓取单条失败只跳过该条并 warning(`ingest/scraper.py:154-155`);估价异常"响亮失败"而不是带病输出(`pricer_agent.price` 对 ≤0 的估价抛错);usage 汇总失败不影响主流程(`app/mcp_client.py:211-226`);Pushover 失败降级为日志。每次运行输出 token 用量与美元成本。

### 风险点(按严重程度)

1. **并发运行无互斥(最大的可靠性风险)**。Gradio 的 5 分钟定时器(`app/ui.py:358-363`)、手动 "Scan now" 按钮、以及**每个打开的浏览器会话各自的定时器**都会调用 `run_with_logging` → `Orchestrator.run()`,没有任何锁。两次 run 重叠时:`usage.TRACKER.reset()`(`app/orchestrator.py:77`)会把另一次运行中途的计数清零(成本报告错乱);同一批 deal 被扫描两次,LLM 费用翻倍;可能重复推送同一 deal。
2. **去重的静默降级**。`app/mcp_server.py:51-58` 中 `scan_deals` 解析 `memory_json` 失败时 `except Exception: memory = []` —— 一旦历史记录反序列化失败(如模型 schema 演进),整轮去重悄悄失效,已推送过的 deal 会被再次推送,且无任何日志。
3. **去重键不一致**。存储主键是完整 URL(`core/opportunity_store.py:30`),而扫描去重用的是 `deal_id(url)`(URL 中的数字 id,slug 变化不影响)。正常路径下扫描侧已挡住,但任何绕过扫描去重的路径(如上一条的静默降级)会让同一商品以不同 slug 存成两行。
4. **数据丢失风险:低**。SQLite 单文件、upsert 写入、TTL 清理是有意设计,`data/` 目录里还有手动备份文件。唯一的全表删除写法在无人调用的 `replace_all` 里(见死代码)。真正的数据"错误"风险在上游:估价配对靠 `product_description` 精确匹配(`app/mcp_client.py:68-93`),模型转述描述则配对失败——但代码已注释说明并有兜底(退回模型自己的 notify 选择)。
5. **无重试机制**。OpenAI 调用无 retry/backoff,瞬时网络错误会让整轮扫描失败(UI 侧有 catch,不至于崩,但该轮作废)。

---

## 四、工程化程度:6/10

- **一条命令跑起来:基本达标**。`docker compose up --build` 可起(需先跑一次向量库构建,README 对两步流程、快速小库参数都写清楚了);本地 `pip install -e .` + `python -m app.ui` 同样两步。`.claude/launch.json` 不存在,但 README 的 Quick Start 足够。
- **自动化测试:有;CI/CD:没有**。这是本维度最大的失分点——`.github/` 目录不存在。讽刺的是这个项目的测试离线、秒级、零外部依赖,是最容易接 CI 的形态;甚至 `eval_pricers.py` 都内置了 `--max-mae` 等阈值失败参数(明说"useful in CI"),但没有任何 CI 在用它。**现状等于:回归只能靠人肉记得跑测试。**
- **密钥与配置管理:干净**。全项目 grep 无硬编码 key/token;`.env` 在 `.gitignore` 中;`.env.example` 只含占位符;配置集中在 `infra/config.py` 单一出口(含 .env 加载顺序问题的修复说明)。`git ls-files` 确认无数据文件、无 egg-info 泄入版本库。
- **依赖管理:半分**。`pyproject.toml` 只有下限约束(仅 gradio 锁了 `<6.0`),没有 lockfile。README 甚至专门写了一段解释为什么不生成 lockfile(怕从 Anaconda 环境导出污染)——理由针对的是错误做法,而不是 lockfile 本身。半年后新装环境拉到不兼容的新版依赖(如 chromadb、mcp 这类 API 变动频繁的库)是大概率事件。
- **Lint**:ruff 已配置且通过。

---

## 五、演进能力:7/10

> 判断标准:需求变化时能否增量修改,哪里牵一发动全身。

### 增量友好的部分

- 换 LLM 模型/embedding 模型:改 `.env` 一处即可,`infra/config.py` 是单一事实来源,且向量库有元数据校验兜底。
- 改选优规则/guardrail:集中在 `core/scoring.py` + `domain/deal.py` 的 `effective_value`,一处改、全链路(通知、UI、排序)生效。
- 新增识别规则(如新的"非全新"关键词、新的多件装写法):`ingest/identity.py` / `scraper.py` 的正则表 + 现成的 35 个测试用例,增量安全。
- MCP 工具边界清晰,新增一个工具(如"查历史价")只动 `mcp_server.py` 一处。

### 牵一发动全身的地方

1. **"换/加数据源"是最贵的改动**。DealNews 的假设不只在 `ingest/`:`domain/deal.py:12-16` 的 `deal_id()` 硬编码了 DealNews 的 URL 格式(`/(\d+)\.html`)——**domain 纯模型层知道具体网站的 URL 结构**,这是分层里唯一一处明显的渗漏。`deal_id` 被 scanner、mcp_client、scoring 链路广泛用作去重/配对键,加第二个数据源(如 Slickdeals)时它会静默退化成"整个 URL 当 id",去重语义随之改变。
2. **"按包拆单价"(per-unit rebasing)逻辑散布 4 处**:`core/identity_policy.per_unit_fields`(拆)、`agents/pricer_agent._to_comparables`(比价侧拆)、`evaluation/retrieval._per_unit_price`(评估侧拆)、`app/ui.table_for`(展示时乘回去,还要字符串替换掉 `per_unit_note` 后缀)。任何对 quantity 语义的修改必须同时想到这 4 处,UI 靠字符串拼接/替换传递语义尤其脆弱。
3. **估价配对靠自然语言精确匹配**:`app/mcp_client.candidate_from_estimate` 用 `product_description` 全等匹配把估价挂回 deal。Scanner 的 prompt 一改(比如让描述更短),配对率就会悄悄下降,系统退化为信任模型的 notify 选择——行为变了但没有任何测试会红。
4. **两套 LLM 客户端栈并存**:scanner/pricer/mcp_client 直接用 `openai` SDK,`agents/messaging_agent.py:5` 单独用 `litellm`。想统一切换 provider 或加重试时要改两种调用方式;litellm 这个重依赖只服务一个函数。
5. **usage 跨进程合并是隐式契约**:MCP server 子进程的 `get_run_usage` 工具与 client 的 merge 逻辑必须成对演进(两边注释都写了,算是补救),新增一个在 server 侧调 LLM 的工具时,忘记这层契约不会报错,只会让成本报告悄悄少算。

---

## 六、有效性一致性检查

### 目标画像(从 README 与代码反推)

- **目标用户**:开发者本人(单用户),关注美国电子产品捡漏;同时是一个展示 MCP/RAG/评估工程能力的作品集项目。
- **核心问题**:DealNews 的"原价对比"不可信,需要一个**独立的**公允价值估计(RAG over Amazon 参考集)来判断折扣是否真实,并过滤二手/翻新/捆绑/订阅等不可比价的干扰项。
- **关键场景**:① 定时扫描 → 估价 → 确定性选出最佳 deal → Pushover 推送;② Gradio 面板浏览已存机会与守卫指标;③ 离线评估估价器/检索器质量。

### 实现与目标的一致性:高

护栏设计(估价不看列表价、savings 封顶于列表价、多件装拆单价、可疑商品宁可弃权)全部直接服务于"不推假折扣"这一核心,且都有测试。评估脚本(pricer/retrieval 分离)服务于"知道系统好不好"。一致性很好。

### 疑似过度开发(目标之外的功能)

1. **MCP 子进程架构本身**。单进程就能完成的 scan→estimate→notify 被拆成 stdio 子进程 + OpenAI 工具调用循环,代价是:PYTHONPATH 注入、跨进程 usage 合并、description 精确匹配配对这三个全项目最脆的机制**全都是这个架构衍生出来的**。README 的辩护(工具可被任意 MCP 客户端复用)成立,作为作品集展示也合理,但要清楚:本项目约三分之一的复杂度在为这个选择买单。更进一步,**agent 循环里的 LLM 实际不做任何决策**——选优是确定性的,模型的 notify 选择还会被覆盖(`app/mcp_client.py:229-241`)。一个 `for deal in scan(): estimate(deal)` 的普通循环功能等价、成本更低、故障面更小。
2. **3D t-SNE 参考图**(`app/ui.py:224-261` + orchestrator.get_plot_data):纯观赏性,UI 启动时同步计算,拖慢每次打开;与"找便宜货"无关。
3. **litellm 依赖**:只为 `craft_message` 一个函数引入的整套多 provider 抽象。
4. **`domain/identity.py` 的预留字段**与 `core/scoring.rank_opportunities` 的公开排序 API:为未来阶段铺路,当前无人用。

属于合理范围的"多":evaluation/audit 脚本虽超出最小可用产品,但直接支撑核心质量目标,不算过度。

### 关键场景遗漏

- **估价置信度**:检索距离、可比商品数量都已在手,但低置信估价照样推送(README 已列入 Future Improvements,是清单里最值得先做的一条)。
- **推送去重的最后防线**:如上文,memory 反序列化失败时静默重推。
- **"这个 deal 真的还在吗"**:TTL 只能被动过期,推送时不校验 deal 是否已下架(可接受,但用户点开死链会流失信任)。

### 用户量 ×10,哪里先崩

按"10 个浏览器会话同时开着 UI"推演,崩溃顺序:

1. **每个 Gradio 会话有独立的 5 分钟定时器**,10 个会话 = 每 5 分钟 10 次完整扫描,且无互斥锁 → LLM 费用 ×10、DealNews 抓取 ×10(可能触发封禁)、`usage.TRACKER` 被互相 reset、同一 deal 重复推送。**这是第一个崩的点,而且在 2-3 个会话时就会出现。**
2. 每次 run 都 spawn 一个新的 MCP server 子进程并重新加载 SentenceTransformer 模型(`_get_agents` 缓存只在子进程生命周期内有效)——进程创建 + 模型加载开销随运行次数线性放大。
3. 单一 Pushover 账号:系统本质是单用户的,10 个"用户"收到的是同一个人的通知,多用户需求需要账号/偏好模型,属于架构级改动。
4. SQLite 与 ChromaDB 在这个量级都不是瓶颈,不用担心。

---

## 七、全项目最严重的 3 个问题

> 术语说明:**竞态条件(race condition)**指两段代码并发执行、执行顺序不受控,导致结果取决于"谁恰好先跑完"的缺陷;**CI(持续集成)**指每次提交代码后自动运行测试的机器人。

### 1. 扫描运行无并发保护(竞态条件)

- **位置**:`app/ui.py:352-363`(按钮 + 定时器)、`app/orchestrator.py:75-86`(`run()` 无锁)、`app/orchestrator.py:77`(`TRACKER.reset()`)
- **实际损失场景**:你早上开着 UI 忘了关,又在手机浏览器开了一个;两个会话的定时器错开 2 分钟各自触发扫描。结果:OpenAI 账单翻倍;同一个"史低价显示器"推送了两次;日志里的成本报告因为 `reset()` 互相清零而完全不可信——当你月底想知道"这个项目每月花我多少钱"时,数字是错的。
- **修法成本**:一个 `threading.Lock` + "已在运行则跳过"约 10 行。

### 2. 没有 CI,依赖不锁版本 —— 质量护栏没有通电

- **位置**:`.github/` 不存在;`pyproject.toml:9-27` 全部为下限约束
- **实际损失场景**:半年后你(或接手者)改了 `ingest/identity.py` 的一个正则,本地忘了跑测试就 push——35 个身份识别用例里红了 3 个没人知道,下周开始订阅制服务被当成商品估价,推送出"Adobe 全家桶年费省 $400"的假 deal。或者:新机器 `pip install -e .` 拉到 chromadb 2.x,API 变了,项目直接起不来,而你无法复现出当年能跑的依赖组合。
- **修法成本**:一个 30 行的 GitHub Actions workflow(测试是现成的、离线的、秒级的)+ 一份 `pip freeze` 或 uv lock。这是全项目性价比最高的一处修复。

### 3. `scan_deals` 的记忆解析静默失败,去重防线整体失效

- **位置**:`app/mcp_server.py:51-58`(`except Exception: memory = []`,无日志)
- **实际损失场景**:某次重构给 `Deal` 模型加了个必填字段,旧库里的历史记录反序列化开始抛错——被这个裸 except 吞掉,`memory` 变成空表。从此每一轮扫描都"从没见过任何 deal",同一批商品每 5 分钟重复估价(费用)、重复推送(每天几十条相同通知,用户直接关掉推送,产品失去存在意义)。**最坏的是全程无一行日志,你只会觉得"最近通知怎么这么吵"。**
- **修法成本**:把裸 except 收窄 + `logging.warning` 两行;顺手统一 `deal_id` 与存储主键。

---

## 八、改进清单(按性价比排序)

> 工作量:S ≈ 半天内,M ≈ 1-2 天,L ≈ 3 天以上。

| # | 改进项 | 工作量 | 预期收益 |
|---|--------|:---:|----------|
| 1 | **加 GitHub Actions CI**:跑 `python -m unittest discover -s tests` + `ruff check .`(全离线,秒级) | S | 回归当天可见,不再依赖人肉记忆;是其他所有改进的安全网 |
| 2 | **给 `Orchestrator.run()` 加互斥锁**,运行中则跳过本轮并打日志 | S | 消除最严重的竞态:费用翻倍、重复推送、成本报告错乱一并解决 |
| 3 | **收窄 `scan_deals` 的记忆解析异常并打 warning**(`app/mcp_server.py:57`) | S | 去重防线失效从"无声"变"有声",避免重复推送风暴 |
| 4 | **锁定依赖**:提交 `requirements.lock`(pip freeze 或 uv)供部署用,pyproject 保持宽松供开发用 | S | 半年后仍能一键复现可运行环境;Docker 构建可重复 |
| 5 | **删除死代码**:`write_memory`/`replace_all`/`update`、未用颜色常量、`domain/identity` 预留字段(或加 TODO 注明) | S | 消除 `replace_all` 全表删除隐患;减少阅读噪音 |
| 6 | **合并三处 ANSI 颜色定义到 `infra/log_utils.py`**,agent 引用之 | S | 消除"改颜色 UI 渲染悄悄失效"的无声耦合 |
| 7 | **OpenAI 调用加重试**(指数退避,2-3 次;openai SDK 自带 `max_retries` 参数) | S | 瞬时网络抖动不再报废整轮扫描 |
| 8 | **统一 LLM 客户端**:`messaging_agent` 改用 openai SDK,移除 litellm 依赖 | S | 少一个重依赖;换 provider/加重试只需改一处 |
| 9 | **估价配对改用结构化键**:`estimate_value` 工具增加可选 `url` 参数,配对优先用 `deal_id`,描述匹配仅作回退 | M | 拆掉"prompt 措辞一变、确定性选优悄悄失效"的暗雷 |
| 10 | **估价置信度门槛**:利用已有的检索距离/邻居数,低置信只入库不推送(README Future Improvements 第一条) | M | 直接提升核心价值——减少假 deal 推送,是用户可感知的质量提升 |
| 11 | **`deal_id` 抽象化**:把 DealNews URL 规则移出 `domain/`,按来源注册 id 提取器;存储主键改用 deal_id | M | 为多数据源铺路;修复分层唯一的明显渗漏;统一去重语义 |
| 12 | **t-SNE 图改为懒加载/后台计算**(或按钮触发) | S | UI 秒开;移除启动路径上最大的无关开销 |
| 13 | **评估 MCP 架构的替代形态**:保留 MCP 工具服务器(对外复用),但默认扫描路径改为进程内直调,agent 循环留作 demo 模式 | L | 砍掉 PYTHONPATH 注入、跨进程 usage 合并、描述配对三个最脆机制;仅当维护成本已实际发生时再做 |

**建议执行顺序**:1-8 可在一天内全部完成(纯 S 项,先 1 后 2/3),之后按需求节奏做 9-11;12 随手;13 仅在项目从"作品集"转向"日常长期使用"时考虑。

---

## 八点五、改进进展(2026-07-02 落地)

> **改进清单 1–13 已全部落地。** 分三批:第一批 S 级项(1-8、12),第二批 M 级项(9、10、11),第三批 L 级项(13,MCP 架构瘦身)。改动后 `python -m unittest discover -s tests` **157 个用例全绿**(较初始 139 净增 18 个新测试)、`ruff check .` 无告警(在与 CI 一致的干净 venv 中验证)。

| # | 状态 | 落地说明 |
|---|:---:|------|
| 1 | ✅ | 新增 `.github/workflows/ci.yml`:Python 3.10/3.12 矩阵,跑 `ruff check` + `unittest`。CI 只装测试真正 import 的轻量子集(无 torch/chromadb/sentence-transformers),秒级完成 |
| 2 | ✅ | `Orchestrator` 增加 `threading.Lock`;`run()` 非阻塞抢锁,已在运行则打日志并返回当前 memory([orchestrator.py](app/orchestrator.py)) |
| 3 | ✅ | `scan_deals` 的裸 `except` 收窄为带 `logging.warning`,去重失效不再无声([mcp_server.py](app/mcp_server.py)) |
| 4 | ✅ | pyproject 为 churny 依赖(chromadb/mcp/openai/pydantic/numpy/huggingface_hub/sentence-transformers/plotly)加了下一 major 上限;README 新增"Reproducible installs"配方(在目标环境生成 lockfile,而非 anaconda freeze)。**注**:未提交 `requirements.lock`——本机 pip 指向被污染的 anaconda 环境,且 macOS/py3.13 的 freeze 不适用于 Linux 容器,故只提供配方,由部署环境生成 |
| 5 | ✅ | 删除 `write_memory`/`replace_all`/`update`(消除全表删除隐患);颜色常量随第 6 项收敛;`domain/identity` 预留字段保留(属设计中的模型,无害)。原 `write_memory` 测试改写为等价的 `read_memory` 测试,保留覆盖 |
| 6 | ✅ | ANSI 码统一到 [infra/log_utils.py](infra/log_utils.py);`agents/agent.py` 与 `app/orchestrator.py` 改为 import,消除三处重复与"改色 UI 悄悄失效"的无声耦合 |
| 7 | ✅ | 新增 `LLM_MAX_RETRIES`(默认 3),所有 `OpenAI()` 构造点(scanner/pricer/mcp_client/messaging)统一传入,瞬时错误自动退避重试 |
| 8 | ✅ | `messaging_agent` 由 litellm 改用 openai SDK;从 pyproject 移除 litellm 依赖;清理测试中多余的 litellm stub |
| 12 | ✅ | t-SNE 参考图改为 `ui.load` 懒加载,移出初始渲染路径,UI 秒开([ui.py](app/ui.py)) |
| 9 | ✅ | `estimate_value` 工具新增 `url` 参数;`candidate_from_estimate` 优先按 `deal_id(url)` 配对,描述全等仅作回退——prompt 措辞变化不再让确定性选优静默失效([mcp_client.py](app/mcp_client.py), [mcp_server.py](app/mcp_server.py)) |
| 11 | ✅ | `deal_id` 及各来源 URL 规则移出 `domain/`,迁入注册表 [core/source_ids.py](core/source_ids.py)(domain 层不再有站点 URL 知识);`OpportunityStore` 主键改为 `deal_id`,并带旧 url-PK 表的迁移(按 id 合并重复行),统一去重语义([opportunity_store.py](core/opportunity_store.py)) |
| 10 | ✅ | `PricerAgent` 由最近邻余弦距离算检索置信度(`estimate_with_confidence`);`estimate_value` 按 `deal_id` 暂存置信度,`notify_deal` 对低于 `RAG_MIN_CONFIDENCE`(默认 0.15)的估价**只入库不推送**。阈值可经环境变量调,默认保守只拦截近乎无可比商品的情况([pricer_agent.py](agents/pricer_agent.py), [mcp_server.py](app/mcp_server.py)) |
| 13 | ✅ | 默认扫描路径改为进程内直调 [app/pipeline.py](app/pipeline.py)(scan→估价→确定性选优→通知),砍掉子进程 `PYTHONPATH` 注入、跨进程 usage 合并、工具参数重配对三个最脆机制;MCP 工具服务器保留供外部复用,LLM agent 循环降级为 `SCAN_MODE=agent` 的演示模式。usage 现全部进程内直接累加,估价天然挂在对应 deal 上 |

---

## 九、评分依据小结

- **可维护性 8**:结构、命名、注释均高于同规模项目常见水准;扣分在死代码、三处颜色常量重复、`ui.py` 混杂职责与隐藏副作用。
- **可靠性 7**:核心逻辑测试扎实且离线秒跑、失败模式多为"响亮失败";扣分在并发无保护、一处静默 except、无重试。数据丢失风险低。
- **工程化 6**:能一(两)条命令跑起来、密钥管理无瑕疵、lint 就位;但 CI 完全缺失 + 依赖不锁版本,把现成的质量资产晾在一边,失分最重。
- **演进能力 7**:多数可预期的需求(换模型、调规则、加识别模式)都是单点修改;扣分在数据源假设渗入 domain、per-unit 逻辑散布 4 处、估价配对依赖自然语言全等、双 LLM 栈并存。
