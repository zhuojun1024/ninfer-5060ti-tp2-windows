# TP-2 前缀复用层重设计

状态：**阶段 0–2 已实现；执行层已改接到 ledger，§13 的删除清单大部分已落地**（未删项与理由见 §13 各条，剩余项见 §19）。
已落地部分：`runtime/engine/prefix_reuse.h` 的 `ConversationLog`/`Anchor`/`ReuseLedger`（阶段 1，提交 `123a93b6`）、
`decide()` 接管 core 的复用决策（提交 `96f017ee`，行为中性，与出厂 v0.1.2 在同一捕获上逐位一致）、lane 局部
边界图与 `RecallState::Boundary`（提交 `fd9d6ded`）；此后执行层（两份扫描、walk 的取点、宿主镜像池、session
catalog 的 store/recall）都改为向 ledger 取槽与取决策，`§18.7` 记录了这一路的四个缺陷与它们的修复。§14.1 的六项
决策保持不变；§15、§16 给出文件划分、纯函数边界与测试处置。本文只覆盖 prefix reuse 的**记账与锚点层**；
prefill 执行、Program、图捕获、KV page 物理合同和协议层不在范围内。

实测证据与夹具脚本放在工作区 `_temp/`，属于本地复现材料、不入库；文中引用的早期行号以提交 `fd9d6ded` 为准。

**权威边界**：本文是 TP-2 路线 prefix-reuse 层的维护者权威——记账（`ConversationLog`/`Anchor`）、决策
（`ReuseLedger::decide`）、镜像池与召回规则、以及精确定律都在这里。[资源调度与上下文缓存]
(resource-scheduling-and-context-cache.md) 定义单 GPU 路线的 admission 与资源选择规则，两者按路线分工，
不互相重复：那边的 prefix-reuse 段落只保留概念与指向本文的链接。

---

## 1. 目的

回答一个问题：

> 给定会话 S 和它这一次的 prompt，从哪个位置开始算，代价由谁付，出错时谁负责保证正确？

当前实现答不上来：它只能报告"看存货"（`host=0/36`、`reach=0`），然后把代价变成一次 43 秒的全量
prefill。重设计的目标是让这个问题有**单一、可断言、可观测**的答案。

非目标：不追求更高命中率的数值极限，也不引入新的模型能力。目标是把五套并行机制收敛成两个概念加一个阶段，
并让每一种失败都有名字。

---

## 2. 现状与问题

### 2.1 五套机制

| 机制 | 位置 | 语义 |
|---|---|---|
| device snapshot | `cached_boundaries` + `state_snapshots` | 血缘在设备上时的零拷贝边界 |
| host grid | ring 的 grid 子环 | 按固定 stride（当前 32,768）覆盖长上下文 |
| host tail | ring 的 tail 子环 | prefill 最后几个 chunk 末端 |
| divergence / block anchor | ring 的 divergence、block 子环 | 观测到的分歧点、稳定块末端 |
| session catalog | `SessionEntry` + host KV 板 | 跨 lane 退休/设备驱逐的会话级存活 |

外加 **adoption/splice**：把客户端重渲染的 assistant 轮折回引擎自己的 token。

### 2.2 已观测到的三类失败

| 事故 | 缺的条件 | 代价 |
|---|---|---|
| 重渲染差异在上一轮 answer 内（req#8） | 客户端字节与引擎字节不一致，splice 只修"最后一轮" | 18,410 token 重算 |
| 分歧点在上一轮 prompt 内部（req#10） | 该位置没有状态镜像，entry 也提供不出边界 | 73,205 token 全量，TTFT 43.4 s |
| ring 锚点写入但不可见 | 镜像按 lane 切片，请求在 lane 间轮转 | 锚点全废（`host=0/36`） |

三类失败的共同点：**没有一处代码负责"保证某个位置上有镜像"**。机制各自尽力，缺失由请求承担。

---

## 3. 设计原则

1. **按约束设计，不按机制设计。** 五套机制只作为需求证据；能被一条策略表达的就删掉。
2. **单坐标系。** 全系统只有一种位置空间：会话 token 位置。客户端 token 只作为输入，规整后不再参与索引。
3. **锚点属于会话，不属于 lane。** lane 只是执行资源；跨 lane 恢复是一等场景。
4. **失败必须命名。** 每次复用决策输出"用了哪个锚点"或"缺哪一类东西"，不允许静默退化。

---

## 4. 数据模型

### 4.1 ConversationLog

一段对话的不可变 token 日志加上它需要的身份。

```
ConversationLog {
  key:            optional<SessionKey>       // 客户端声明的会话名
  tokens:         vector<TokenId>            // 规范流（规整后的引擎字节）
  media:          vector<MediaSpan>          // 媒体身份，token 相等不足以区分图片
  anchors:        AnchorTable                // 该会话的锚点
  device_extent:  u32                        // 设备分页池为它连续覆盖到的位置
  host_extent:    u32                        // host KV 板（slab）覆盖到的位置
}
```

`tokens` 是**规范流**：一旦某轮被 splice 成引擎字节，日志里就保存引擎字节。客户端下一轮发来的渲染与它
不一致是预期内的，由第 5 节处理，不再让每个下游模块各自判断。

### 4.2 Anchor

```
Anchor {
  position:      u32         // 会话 token 位置，镜像描述的是"处理完 [0, position) 之后"的状态
  key:           ContentKey  // 内容身份，见 4.4：hash(规范流[0, position) ‖ media[0, position))
  state:         StateImage  // GDN 状态；storage=device 时是设备上的 slot
  draft_extent:  u32         // DFlash2 draft ring 覆盖到的位置；无 draft 时为 0
  storage:       Device | Host
  provenance:    PromptEnd | Divergence | Block | Generated | Grid
  policy:        Pinned | Lru
  lane:          optional<u32>   // storage=Device 时绑定到某条 lane
  kv_source:     Device | HostSlab  // 该锚点依赖的 KV 落在哪一份存储上，见 9.1
}
```

不变量（debug 构建断言）：

- `storage=Host → draft_extent == position`（有 draft 时必须逐 token 对齐）
- `position <= kv_extent(log, kv_source)`（镜像不得声明它不覆盖的位置，按存储分别判定，见 9.1）
- `storage=Device → lane 有效且该 lane 当前承载此会话`

### 4.3 AnchorTable

每个会话一张表，按 `position` 升序。容量由全局预算决定，不按 lane 分配。

- 查找：`floor(shared)` 取 ≤ shared 的最大 position；本表无命中时再查 §4.4 的全局索引（跨会话）。
- 插入：同 position 覆盖。
- 淘汰顺序：先取"冗余"的 `Lru` 锚点——同一会话在它下面不足一个 stride 处还有另一张图，这一张就不增加覆盖
  （§7.3 的覆盖规则，2026-10-25 修订）；同属冗余时取 position 最小者，没有冗余锚点时退回"position 最远者优先"。
  `Pinned` 只有显式解除才淘汰。

### 4.4 跨会话共享：key 即内容，不设策略

锚点的身份是**它覆盖的那段规范流**，不是它属于哪个会话。这个身份就是 §4.2 的 `key`，与 `position` 一起构成锚点
的唯一标识。规则：

- `key` 相同且 `position` 相同 ⇒ 同一个 `StateImage`，引用计数共享。第二个会话命中时不再拍图。
- **KV 归属锚点，不借用会话的板。** 共享锚点持有自己的 KV slab 引用：源会话被淘汰时锚点仍可用，锚点被淘汰时
  它的 KV 一起释放。否则锚点的 `kv_source` 会指向别人的板，`position <= kv_extent` 就变成一句假话。
- **不做 pin、不做独立预算、不做共享专用的淘汰层次。** 共享只有一种存在方式：表里恰好有这张图，而且还没被 LRU
  淘汰。命中算白赚，没命中与今天完全一致。
- 依据：单个共享前缀最多省下 `前缀 token / 1,687` 秒；8 个会话共享 6K token 约 28 s，2K 以下属噪音。为一个噪音级
  收益引入"钉住 + 两级淘汰"，成本正好落在 §7.1 最脆弱的预算模型上。
- 升级条件（写进阶段 4）：如果 `[reuse]` 行统计出的跨会话命中率非零且重算代价可观，再把它提升为 `policy=Pinned`
  的一档独立预算。在此之前不写这段策略。

### 4.5 与旧机制的映射

| 旧 | 新 |
|---|---|
| device snapshot | `Anchor{storage=Device, provenance=Generated}` |
| grid 子环 | `provenance=Grid` 且 `policy=Lru` 的等距锚点 |
| tail 子环 | `policy=Pinned` 的近期锚点 |
| divergence / block 子环 | `provenance=Divergence / Block` 且 `policy=Pinned` |
| prompt-end 槽 | `provenance=PromptEnd, policy=Pinned`，每会话一个 |
| `SessionEntry` 的若干字段 | 由 `tokens`、两份 extent 和锚点表推导，不再单独存储 |

### 4.6 槽分配器

锚点表是逻辑视图，槽是物理单位。两者之间需要一个分配器，这是现在没有的东西。

本设计里**一张镜像占一个槽**，所以 `--host-state-images` 就是池容量；后文不再区分"张"与"槽"，对外一律用"张"
（§13.1 解释了为什么改这个词）。

物理池已在 `tp2_generation_core.cpp:955` 存在：启动时一次性分配 `lanes_ * per_lane + lanes_` 个 `HostBuffer`，
每张 = 一条 lane 的状态镜像（73.4 MiB），pageable。当前槽号是算出来的（`lane * per_lane + offset`），
所以"锚点属于会话、可跨 lane、可跨会话共享"这件事没有地方落地。

```
HostPool {
  slots:  vector<Slot>       // 固定容量，启动分配
  free:   vector<u32>        // 空闲槽
}
Slot { buffer: HostBuffer, refs: u32 }
```

- **容量**：`capacity = host_state_images`（原名 `host_state_slots`，见 §13.1）。旧的 `+2` 消失：divergence/block
  不再各占一个预留槽，所有锚点从同一个池里分配。子环消失（§13），槽不再预留给某一类锚点。
- **分配**：`alloc()` 取空闲槽；没有空闲槽时按 §7 的取舍顺序淘汰一张 `Lru` 锚点并回收其槽。共享镜像
  （§4.4）`refs > 1` 时不能因单个会话淘汰而释放，只有引用归零才回到 `free`。
- **耗尽**：整池都是 `Pinned` 时 `alloc()` 失败。失败是**显式结果**，不是异常路径：记 `degraded=pool_exhausted`，
  这一张图不拍，当前请求照常继续。禁止为了腾槽去清 `Pinned`。
- **设备槽**：`storage=Device` 的槽仍归 lane 所有，但索引也由分配器给出，不再是
  `lane * per_lane + offset`；lane 退休时按 §8 的规则转存或释放，转存成功才归还 host 槽。
- **单属主**：分配与回收只在 `restore()` / `commit()` 路径上发生，必须在同一条调度线程上串行。阶段 1 要先确认
  两条 lane 的 flush 不同时进入分配器；若确实并行，就给池加一把锁而不是加第二份池。
- 不变量（debug 断言）：`Σ_{slots} refs == #{a : a.storage = Host}`——共享镜像被两个会话引用时 `refs = 2`，
  但只占一张槽；`free` 与"被引用的槽"不重叠。

---

## 5. 输入规整（adoption 的替代）

目标：把客户端发来的 messages 变成"与日志可比较的 token 流"。

流程：

1. 渲染客户端 messages → `incoming`（不做任何 splice）。
2. 与 `log.tokens` 求公共前缀 `shared`。
3. 若 `shared == log.tokens.size()` 或其长度关系表明只是尾部被重渲染：逐轮比较
   `rendered_turn(incoming, round)` 与 `rendered_turn(log, round)`；逐字节等价才允许替换。
4. 替换后得到**规范流**，此后所有下标都在这一套坐标里。替换记录为 `splice[round] = {begin, end}`，
   只用于可观测性，不作为索引。

硬性约束：

- 每轮替换必须重新验证；不允许"记住上次替换然后无条件重放"。
- 媒体会话（`has_media()`）不做替换。
- 规整失败时**不回退到猜测**：直接使用客户端流，并把日志标记为 `rewrite`，让第 6 节的决策显式暴露代价。
  这是硬决定（§14.1），不是可调默认值：信任 session key 强制续写会静默改写历史，正确性风险大于一次全量 prefill。

---

## 6. 复用决策

```
decide(log, incoming):
  shared = common_prefix(log.tokens, incoming)      # 已含媒体 identity 截断
  cand   = log.anchors.floor(shared)
  if cand == null:
      cand = shared_index.match_prefix(incoming, shared)   # §4.4：只查 key，不查归属
  if cand == null:
      return {reuse: 0, reason: NoAnchorAtOrBefore(shared)}
  if not identity_ok(log, cand):                    # tokens/media 一致性
      return {reuse: 0, reason: IdentityMismatch}
  return {reuse: cand.position, anchor: cand}
```

失败原因是**枚举**，不是自由文本：`NoAnchorAtOrBefore`、`IdentityMismatch`、`ConversationMiss`。日志一行即可判定
属于哪一类。跨会话命中不是新原因，只在 `src=` 上体现为 `shared`。

**没有"候选超出 prompt"这个原因**：候选是"≤ shared 的最深锚点"，而 shared ≤ prompt 长度，所以它不可能越界——
第一版实现里写了这个分支，单测证明它恒为假，已删除。反过来，`reuse == prompt 长度` 是**合法**的（整段已处理过，
请求退化成纯 decode），所以不能加 `>=` 形式的保护。

`shared_index` 是 §4.4 的全局 key → 锚点索引。它是**可选命中**，不参与正确性判断；因此它慢或失效只会让命中率
下降，不会让任何请求变错。

---

## 7. 谁负责保证"≤P 处有镜像"

这是整套设计的核心规则，替代原来五套机制各自的"尽力而为"：

1. **prefill 结束**：必拍 `PromptEnd`，`policy=Pinned`，替换该会话上一张。
2. **分歧点在历史内部**（`shared < log.tokens.size()` 且 shared > 0）：在 `shared` 处必拍一张
   `provenance=Divergence`。这张图是**本次请求自己**为下一次铺的路；本次付全价是设计的一部分，不是缺陷。
3. **全程等距采样，间距固定 8192**：`[0, prompt_end)` 段记 `provenance=Grid`，从 `prompt_end` 起的生成段记
   `provenance=Generated`。两段间距相同，区别只有来源标签和优先级。`Generated` 那张是"重渲染答案"场景的唯一救兵。
4. 采样间距是**契约**：`stride = 8192`，不随预算缩放（理由见 7.1）。历史上"网格按预算反推 stride"的做法一并取消，
   于是整个会话只有一种采样粒度。

预算不足时的取舍顺序：`Pinned` 不动 → 其余先淘汰**冗余**的（同一会话在它下面不足一个 stride 处还有另一张图），
再按"离前沿最远"淘汰。冗余优先是关键：prefill 会在末尾一个 stride 内按 chunk 宽度写一小段稠密图，如果按"位置最远"
淘汰，这一小段会把后面的网格图全部挤掉，存活集合虽然连续，却只是最近几十个 chunk（2026-10-25 实测有效覆盖
~45k token，而账面是 139k）。按覆盖淘汰后，池子收敛成"每个 stride 一张 + 尾部一小段稠密"，覆盖 = 槽数 × stride。
任何情况下不允许"某会话把表填满导致别的会话无锚点"。

### 7.1 预算反推与降级

覆盖粒度是契约，张数是结果。

- **`stride = 8192` 固定。** 它直接等于用户能感知的代价：prefill 实测 1,687 tok/s，`8192 / 1,687 ≈ 4.9 s`
  是最坏重复计算时间，与内存参数无关。因此同一请求在不同 `--host-state-images` 下 TTFT 保持可比，bench 数字有解释力。
- **需要多少张由需求反推**：`images = ceil(prompt_len / 8192) + ceil(gen_len / 8192) + 3`（+3 = `PromptEnd`、
  `Divergence`、`Block`）。一张 = 两条 shard 各一份镜像 = 146.8 MiB（73.4 MiB/shard）。于是：
  - 64K 提示 + 64K 生成 ≈ 19 张 ≈ 2.7 GiB —— **默认值 20 就落在这里**；
  - 128K + 128K = 35 张 ≈ 5.0 GiB；
  - 256K + 256K = 67 张 ≈ 9.6 GiB（也回答了"契约化网格"的价格）。
  对照当时现状：`--host-state-images 32` 实际分配 36 张（5.2 GiB），而单个会话**够得着的只有 7 张**（57K 覆盖）——
  另外 7 张绑在另一条 lane 的切片上，网格 16 张、divergence/block/prompt-end 各若干。同样的内存已经花掉，
  却只有约五分之一能用于"重渲染答案"这个实际场景。
- **DFlash2 路线每张还要多存一份 draft ring**：打包日志实测 36 张时 shard 0 占 4082.6 MiB、shard 1 占
  2642.6 MiB，差额 1440 MiB / 36 ≈ 40 MiB per 张。所以 DFlash2 下 20 张 ≈ 3.7 GiB，不是 2.9 GiB；
  上表数字是"状态镜像"的价格，draft 按路线另计。
- **余额不足时降级，不缩粒度**：保留 `n` 张网格图（`covering = n × stride` 的覆盖，按 §7.3 的覆盖规则挑选，
  实际形状是"每个 stride 一张 + 尾部一小段稠密"），超出覆盖范围的分歧退回 `PromptEnd`（本次付全价）。日志报
  `cover=<tokens>` 与 `degraded=window_truncated`。契约永远成立，只是适用范围收缩，而且收缩是一个可直接读的数。
- **反向方案被否决**：把间距放大到刚好铺满上下文。以 256K 提示 + 256K 生成为例：36 张时 `stride` 会变成
  ≈15,887（9.4 s），默认 8 张时 ≈104,858（62 s）。那种配置形式上"覆盖了全部上下文"，实际把最坏代价抬高一倍到
  十几倍，还会让延迟随内存参数漂移。

### 7.2 可见性：位置已提交即可见

§7 第 3 条要求在生成段每 8192 拍一张，而 §9 的 `commit()` 是唯一接收 `new_anchors` 的地方——两者需要一个明确的准则，
否则实现者会造出一套 staged/pending 机制。

准则：**锚点在写入的那一刻就可见，条件是 `position` 不超过已提交 token。** 镜像描述的是"处理完 `[0, position)`
之后"的确定性状态，与当前请求是否结束无关；而 `position` 在拍摄时已经 ≤ 已提交位置，所以它既不会被撤回，也不依赖
当前请求的结局。

推论：

- `commit()` 只做三件事：写 `log.tokens`、更新两份 extent、执行预算淘汰。它**不发布锚点**。
- 取消或失败：已写入的锚点保留（它们本来就有效），唯一需要回滚的是 `log.tokens` 和 extent——而它们只在
  `commit()` 里改，所以不存在半写。
- 例外：`PromptEnd` 与 `log.tokens` 同时成立，prefill 中途取消就不写它。
- 写入顺序固定：先写完镜像 → 再更新池元数据（`refs`、占用）→ 最后入表。前两步之间失败只会泄漏一个槽，
  不会产生一个可见的坏锚点。

### 7.3 覆盖优先的淘汰规则（2026-10-25）

池子装满时，Lru 锚点的取舍顺序决定"池子到底覆盖了多长一段历史"，而不是"最近写过什么"：

- **冗余先走**：若同一会话在某个锚点下面不足一个 `stride` 处还有另一张图，这张锚点不增加任何覆盖——
  任何在它之下分叉的 prompt 都能用更浅的那张站住——所以它先被淘汰。同属冗余时取 position 最小的那张，
  于是尾部稠密段从底部变薄，最深的几张留下。
- **没有冗余时退回旧规则**：position 最小者先走，也就是文档原先写的"离前沿最远先淘汰"——此时每一个
  stride 只剩一张，覆盖确实从最浅处开始收缩，这是 §7.1 的降级。
- **为什么需要这条**：prefill 在末尾一个 stride 内按 chunk 宽度写一小段稠密图（`kReuseTailWindow` +
  `kReuseTailCheckpointCount`），它覆盖的 token 数很小但张数很多。按"位置最远"淘汰时，这一小段每写一张
  就挤掉一张更浅的网格图，存活集合虽然连续，覆盖却退化成一个很窄的窗口（实测 165k 的历史只剩 ~45k，
  账面是 139k）。客户端把历史中段裁短（DSH 的 `compaction/prune`）时，分叉点通常落在这个窗口之外，
  召回退化成全量 prefill——§19.9 有完整证据。
- **不变的部分**：`Pinned` 仍只有显式解除才淘汰；共享镜像（`refs > 1`）仍不能被单个会话淘汰；
  降级仍然是显式的（拿不到槽就跳过这次捕获，日志 `pool exhausted`）。

---

## 8. 跨 lane

- 锚点属于会话；`storage=Host` 的锚点**任何 lane 都可恢复**：恢复时把镜像按目标 lane 的 geometry 拷进其 slot。
- `storage=Device` 的锚点只在其 `lane` 承载该会话时可复用（零拷贝快路径）；lane 退休时它要么转存为 Host，
  要么随该 lane 失效——**由规则决定，而不是由切片索引偶合**。
- 由此，`valid[lane]` 这种按 lane 寻址的写法整体消失。

### 8.1 恢复合同

`decide()` 只给出一个位置；把它变成"从 P 继续"是一串有顺序的操作，顺序不能换：

```
restore(log, anchor, lane) -> Result<P, RestoreFailure>
  1. 校验  anchor.position <= kv_extent(log, anchor.kv_source)，且该份 KV 仍可及
  2. KV    拷到 kv_extent，不是 anchor.position：host slab → 设备分页池对应页；kv_source=Device 时无需搬运
  3. 状态  镜像按目标 lane 的 lane_state_geometry 拷进其槽（跨 lane 就是这一步）
  4. draft draft ring 置到 anchor.draft_extent，且必须 == anchor.position，否则拒绝
  5. 记账  token 历史 = log.tokens[0, P)，position = P
  6. 返回  P；后续 prefill 从 P 开始
```

- **`kv_extent` 与 `anchor.position` 是两个量，不能合成一个**（2026-10-24 实测）。`position` 是状态站在哪，
  `kv_extent` 是这份 KV 覆盖到哪；恢复只拷到 `position` 会让**写那块 slab 的那次预填**留在环里、位置在
  `position` 之上的镜像全部落空（扫描按 `position > device_kv_limit` 拒收）。prompt 在上一次历史的尾部之内分歧
  时，这条差异就是"复用 0"和"复用一个大 chunk"的差别（§19.7 的 8,192 → 82,944）。`kv_extent` 取
  `min(entry.host_kv_end, shared)`：slab 的覆盖范围与这条 prompt 真正复现的前缀取小。
- **`anchor.position` 必须落在 prefill 的 chunk 网格上**（见 §18.7 的精确性定律）。`kv_extent` 不需要在网格上：
  它是"设备上有多长的前缀属于这条 prompt"的上界，落点由 `decide()` 在网格上挑。

- 顺序约束：4 必须在 3 之后（draft 环与 target 状态必须同一时刻）；2 必须在 6 之前；1 失败时一步都不能执行。
- 失败处理：任何一步失败 → 放弃复用、回到"从零 prefill"，原因为 `RestoreFailed(<step>)`。
  **不允许**恢复了一半的 lane 进入生成：半恢复的 lane 的 KV/状态/draft 三者不同步，症状会远晚于故障点出现。
- 恢复期间不拍图、不 `commit()`；若要中途取消，按"从零"重建 lane，而不是回滚到半途。
- MTP 无 draft ring：跳过第 4 步，`draft_extent = 0`。

---

## 9. 提交规则（唯一）

```
register(log, history_prefix, anchors):
  log.tokens = history_prefix   # 本请求与自身谱系真正一致的那一段
  log.anchors.clear()           # 每个请求从各存储的真实内容重建
  log.anchors |= anchors        # 每个锚点都来自一张真的持有该位置的镜像
```

三条性质：

- **没有事后裁剪这一步**。锚点只在"某张镜像真的持有这个位置"时才被登记（§15.4、§15.6），表里不可能出现超出存储
  覆盖的位置，所以没有"提交时按 extent 裁一遍"的动作，也就不会发生"设备侧变短、把 host 侧仍然有效的锚点一起删掉"。
- **每次请求重建**。可用镜像随 lane、ring 内容、catalog 每个请求都在变，重建比增量维护更难出错；重建也顺带把
  "这个会话上次记住但没有镜像支撑的边界"清掉。
- **一条真正的边界**：`offer()` 只登记 `position <= shared_prefix` 的位置——超过提示词与自身谱系一致点的锚点，
  对**这个**提示词没有任何意义（见 §9.1）。这条比"按 extent 裁剪"更早生效，也更难写错。

### 9.1 两份存储，各自覆盖

`[0, P)` 的状态与 KV 可以落在两处，两者的覆盖范围不同步：

| 存储 | 提供哪些位置 | 什么时候变短 |
|---|---|---|
| 设备状态与快照 | 本 lane 谱系与提示词共同前缀之内的边界 | 分歧、会话退休 |
| host 检查点环（ledger 池） | 环里当前有效、指纹与这条 prompt 的前缀一致的那些镜像位置 | 池淘汰、被后写覆盖 |

第二条的"属于本 lane"是 §18 之前的措辞：镜像属于**对话**，不属于 lane（§15.5），所以判据是池仍持有该槽
（`holds_slot`）加上指纹匹配，而不是 `writer == lane`。§9.1 的正文把这层独立性说清楚了，表格随正文改。

两条都在**登记阶段**成立，所以不需要两个 extent 数字：

- 位置来自存储，不是先声明后补：没有"先声明 `HostSlab`、内容后补"的窗口；
- 分歧只影响设备侧能提供的位置；host 侧由环里还在的镜像决定，不受设备谱系变短的影响。§9.1 原本要防的事故
  （一次分歧把深历史唯一能用的 host 锚点全删掉）现在就防在这里。

**2026-10-23 的调整**：`ReuseLedger::commit()` 与 `ConversationLog::device_extent`/`host_extent` 已删除。核对发现
`commit()` 从未被生产路径调用（只有 `test_prefix_reuse` 用），因为扫描每个请求都会重建锚点表并只登记真实位置，
它那条"按 store 裁剪"的规则在实现上永远不会触发。原有那句"唯一提交规则"描述的是更早的代码形状；规则本身
（两份存储各自覆盖、分歧不得连累 host 锚点）保留，落点从提交阶段搬到登记阶段。单测里对应的保证改成行为断言：
更深的 host 锚点在提示词覆盖它时仍然被选中，而提示词只覆盖到 1500 时决策不会越过 1500。

---

## 10. 与现有约束的接口

| 约束 | 处理 |
|---|---|
| host KV 板 | 布局不变；`host_extent` 是它覆盖到的位置，锚点不得越过。共享锚点（4.4）持有一个 slab 引用，可以活得比登记它的会话久 |
| 设备分页池 | 分歧裁剪只缩短 `device_extent`，不影响 `host_extent` 与依赖它的锚点（9.1） |
| DFlash2 draft ring | 每个 `Host` 锚点必须带 `draft_extent == position`；decode 中途拍图前先 flush pending |
| MTP 链 | 无 draft ring，`draft_extent = 0`，规则不变 |
| 媒体身份 | `identity_ok` 里检查 `media` 前缀；不依赖 token 相等 |
| offline CausalScoring | 复用同一 `decide()`；只要求"从 0 或从锚点"两种来源，不新增机制 |

---

## 11. 可观测性

一次决策一行，结构化、可直接被脚本解析：

```
[reuse] conv=<key|-> shared=<n> anchors=<a>/<cap> pick=<pos|none> src=<own|shared> reason=<enum>
        cands=[<pos>:<prov>:<storage>:<kv_source>...] kv=<device|host>:<extent> draft=<extent>
        stride=8192 cover=<n> degraded=<none|window_truncated|pool_exhausted>
```

`kv=<store>:<extent>` 报的是**这份存储能给这条 prompt 多长的 KV**（§8.1 的 `kv_extent`），不是被选中的边界：
边界是 `pick`，两者可以不同，而差异正是"环里还有更深的镜像可用"的信号。

要求：能从这一行直接回答"为什么没命中"，不需要再开别的 trace。`cover` 与 `degraded` 是 §7.1 的读数：
预算够不够不再是问题，而是一个可以直接比较的数字。

---

## 12. 迁移与验收

**阶段 0：捕获与差分夹具（先做，不写复用代码）**

原计划"把已记录的 trace 转成回放输入"不成立：`src/serve/request_log.cpp` 记的是结果（token 数、复用数），
`serve-win.log` 里是 `[tp2-diverge]` 这类决策轨迹，两边都没有请求体。req#8 / req#10 的 messages 数组没有落盘。
所以阶段 0 的第一步是拿到输入：

- 加一个 opt-in 的请求捕获开关（默认关）。规格如下——**这是第一笔代码**，它本身不依赖任何未定的设计：
  - 开关：`--capture-requests <dir>`，缺省不写；与 `--no-prefix-reuse` 互不排斥，诊断时可同时开。
  - 落盘：一个请求一个文件，文件名 `<seq:06>-<session 前 8 位>.json`，`seq` 用到达顺序（与判决日志同序）。
  - 内容：`{"seq":N,"path":...,"headers":{...},"body":<原文>}`。**必须连请求头一起存**——会话身份在
    `x-session-id` / `x-bili-session-id` / `x-deepseek-harness-session-id` 里（[session_identity.cpp](D:/Documents/workbench/ninfer/src/serve/session_identity.cpp)），
    只存 body 的话回放时没有会话归属，夹具会退化成"每个请求都是新会话"。存**除凭据外的全部头**
    （`authorization` / `x-api-key` / `cookie` 剔除），比白名单更保真，`anthropic-beta` 这类影响渲染的头也不会漏。
  - `body` 存**原文**（不解析再序列化）：回放要的是字节级一致，重新序列化会改键序和数字格式，而模板渲染对字节敏感——
    这正是 req#8 的故障面。
  - 体积：一条 73K token 的请求约几百 KB，四轮会话不到 1 MB；不做轮转，用完手动删。
- 重跑一次真实会话（bili / harness 那条 73K 路径）拿到带请求体的 trace，这是阶段 1 的唯一真值；
- 差分夹具按这份 trace 回放：每个请求比较"新旧实现选中的复用点"，差异必须能用第 6 节的原因枚举解释；
- 端到端 oracle：同一 prompt 分别走"复用路径"和"从零 prefill"，判据见 §16.2。

**阶段 0 已完成（2026-10-22）**：夹具 `_temp/20261022-1515_phase0.ps1`（`-Mode capture` / `-Mode replay`），
四轮会话捕获 4 个请求，回放进全新进程后每一个复用决策都逐位复现：

| 请求 | prompt | cached | 说明 |
|---|---|---|---|
| turn1 | 1,688 | 0 | 新会话 |
| turn2 | 2,058 | 1,024 | 追加；`shared=1687`，命中来自 `shared_end=1024`（`via=shared`） |
| turn3 | 2,061 | **0** | 分歧落在上一轮 prompt 内部（`shared=2045`），`shared_end=0 reach=0 via=none` → 全量 |
| turn4 | 2,145 | 1,024 | 分歧之后继续，`via=shared` |

三个直接读数：

1. **req#8/req#10 的机制在 4 个请求、2K token 规模上就能确定性地复现**（turn3 全量），不需要真实 73K 会话；
2. 四次请求的 `host=0/12` —— host ring 一次都没参与，1024 的命中全部来自 `via=shared` 的共享前缀缓存，
   也就是重设计要删掉的机制之一；
3. **`stride=16384`**（32K 上下文、grid 2 槽）证实"网格 stride 由预算反推"，同样的 32K 配置在 256K 下会变成 32,768。

**新发现（待定）：固定 8192 会丢掉短会话的命中。** turn2 的 1,024 命中点在当前位置 1,024，而 8192 栅格在
1,688 token 的 prompt 里只有位置 0；按本设计 turn2 会变成 `reuse=0`，比现在更差。代价约 0.6 s（1,024 token
@ 1,687 tok/s），只影响每个会话的前两轮。三种处理，阶段 4 用测量挑一个：

- 接受（默认，先不动契约）；
- 早期加密：在 1,024 / 2,048 / 4,096 / 8,192 各拍一张，之后回到 8,192 —— 多 3 张（≈0.4 GiB），覆盖所有短会话；
- 把 `shared_end` 这类"上次复用的落点"也当成锚点来源（它比栅格更贴合实际分歧位置，但位置不规则）。

**修订（同日，夹具必须确定性）**：上表的 0/1024/0/1024 是 `--temperature 0.7` 下录的，而 seed 每请求随机，所以那组数字
**不可复现**——同一条捕获连跑两次，引擎自己生成的 token 不同，历史不同，复用点就不同。夹具已改为 `--greedy`，
两次运行逐位一致，得到的新基线是 **0 / 0 / 1024 / 0**（turn2 的分歧比 0.7 时更早，turn3 反而拿到 1024）。

结论：任何"改前改后"的比较都必须在 `--greedy` 下做；上表的原始数值只作为"当时观察到过全量重算"的证据，不作为基线。
另外 greedy 下 turn3 只损失 1,037 token 而非 2,061，说明**这个夹具还不足以复现"整段丢失"**——需要把填充材料加长，
让分歧落进第一张锚点之后，再去测阶段 1 的收益。

**修订二（同日，TP-2 上 lane 保留状态每请求即销毁）**：把填充加长到 11K token、`--max-context 16384`（stride 落到契约值
8192）之后，复用到 `0 / 10240 / 10240 / 11264`，但 `[tp2-reuse]` 始终是 `host=0/12`。追查结果是一个**设计事实**而非缺陷：

`release_lane_kv(lane)`（`tp2_generation_core.cpp:3283`，调用点 3578 / 4376，均在"请求退役"的 lambda 里，紧跟在
`retire_lane_session` 之后）会清空 `cached_prompt_tokens`、`cached_boundaries`、`cached_state_valid`、`live_state_valid`、
`anchor_session`，并调用 `invalidate_host_checkpoints(lane)` 整环清空。该函数第一行是 `if (lanes_ <= 1U) return;`，
所以**单卡保留、TP-2 每请求全清**。host ring 在 TP-2 上的寿命因此只有"一个请求之内"，跨请求复用全靠会话 catalog
的 `retire_lane_session` → recall（`src=live`）这条路。这解释了 req#10 的 73K 全量只能由 catalog/recall 一侧解释
（`shared_end=0`），也解释了为什么任何"把镜像放在 lane 里"的方案在 TP-2 上都不可能生效。

由此确定：§4.6「ledger 自持镜像槽」是 TP-2 上唯一可行的落点，第 2 步必须提交到 **session slab** 一侧（复用
`retire_lane_session` 已有的拷贝），不能依赖 ring；§13 删除 ring 一条得到强证据。

同时排除的假设（探针实测）：ring 不写入、publish 不发布、扫描失效循环清空、`session_invalidate_active`、状态随请求重建、
FP8 AllReduce（`NINFER_TP2_AR_CODEC=bf16` 后所有复用数字一位不差，只有模型输出长度变化）。

> **本段已被取代（2026-10-24，§19.2）**：下面描述的按 lane 边界图与 `RecallState::Boundary` 已删除。它的捕获点不在
> prefill 的 tile 网格上，经由它的召回会切出与冷走查不同的 chunk；引入它的提交自述"恢复在 logits 精度内等价、并非逐位
> 相同"，换来的是 21K 提示词上 137 个 token。分歧锚点（`anchor_divergence`，网格定位、由 ledger 决策）覆盖同一需求。
> 本段保留作为当时的判断记录。

**第 2 步：已实现（与本段原计划的差异在此对齐）**。原计划把镜像放在 `SessionEntry` 上（按会话），实施后改为**按 lane**。

问题本身没变：prompt 在 10,240（共享块）与 11,228（上一轮 prompt 末）之间仍有一段与历史一致，而那里没有状态镜像，
所以每轮重算那一段；共享块缺失时则整段重算。

**最终形态**（提交 `fd9d6ded`）：

1. 每 lane 一张边界图，目标半 + DFlash 草稿半，**首次捕获时才分配**（目标 147 MiB/卡；草稿 130.8 MiB，仅 shard 0）。
   按会话存的话 39 个会话要 2,862 MiB/shard，占 10,240 MiB 预算的 28%，并把会话数从 39 压到 30；按 lane 存是固定的 1.4%–2.7%，会话数不变；
2. 捕获点在 prefill 的 chunk 末尾，chunk 被 clamp 到**恰好终止在一致点**（`reuse_boundary_target`），因此镜像落在正确 token 上；
3. recall 把它作为**第四个候选**（`RecallState::Boundary`），与 `host_frontier_end` / `host_prompt_end` / `host_shared_end` 取最深可用者；
4. 身份用 `tokens[0..P)` 最后八个 token 的 FNV-1a 哈希，**不用会话 id**：实测 A、B 两个会话的 `active_session` 都是 0，下标不可作标签；
   而"某个前缀的状态"属于任何携带该前缀的 prompt，与谁写入无关。

**实测**（两会话交替、21K token、greedy）：`via=boundary` 生效，cached 21,504 → 21,641，重算量 334 → 197。
用 `NINFER_TP2_LOGIT_DUMP=1` 对照 ON/OFF：prompt 末列 **top-1 相同**，top-1 logit 差 0.375（该量级 BF16 步长 0.25，约一个量化步），
第 2–8 名在互相小于一步的范围内重排；对照组（同一路径跑两次）logits 完全一致。

**结论与已知行为**：状态恢复在 **logits 精度内等价**，不是逐位相同。不同复用点会让浮点结果相差一个量化步，
贪心解码在 top-2 相差小于该步时可能翻转（实测 a2 首 token 不同、a3 长度 159 vs 100）。这是同类前缀复用方案的固有性质，
**不能承诺"缓存命中与否输出一致"**；若产品需要该保证，要改的是复用策略（同一对话总落在同一复用点），不是本实现。

**已知限制**：每 lane 只有一张图，交替会话互相覆盖，只有后写者吃到更深边界（实测 b2/b3 仍停在 21,504）。
要两边都受益需按会话存（28% 预算）或每 lane 两张轮转。

不要走 ring：`release_lane_kv` 每请求清环（见上），任何 lane 内镜像都活不过一个请求。

**历史（已作废，结论保留）**：本方案先按"每会话一块 slab"设计过，因内存账不可接受（28% 预算、会话数 39→30）改为按 lane。
实施期间观察到的 "starting engine 之后 0xC0000005" 一度被判为"启动阶段的偶发失败 / 连续重启的设备释放窗口重叠"，
夹具靠启动前等 15 秒规避。

**2026-10-22 复核：那个解释不成立。** 空机冷启、等待 90 秒、隐藏第三张卡（`CUDA_VISIBLE_DEVICES=0,1`）、关掉内核内 allreduce
（`NINFER_TP2_AR_FORCE_HOST_STAGING=1`）、换 codec（`NINFER_TP2_AR_CODEC=bf16`）、关看门狗，全部照样复现；与命令行参数
无关，最小命令 `--devices 0,1 --max-context 16384` 同样崩。`--devices` 的序号也确认无误：CUDA 默认 best-first 枚举为
`0/1 = 两张 5060 Ti、2 = Tesla T10`，`nvidia-smi -L` 的 PCI 顺序不同，不能拿来判序号。

已确认的事实（`ninfer-serve` 4/4 失败，崩溃时刻 0.4 s / 0.8 s / 29 s 三种）：

- 崩溃指令在 `nvcuda64.dll` / `nvcudart_hybrid64.dll` 内部，异常恒为 `0xC0000005`：一次 `info0=8 data=0`（跳到地址 0），
  其余是读一个畸形指针（低 32 位固定 `0xa4fa`、高 32 位每次不同），形态像是驱动侧分配器的链表被打断；
- 调用链固定为 `TP2GenerationCore` → `materialize_model_tp2` → `tp::materialize_tp2` → `tp_init` →
  `DeviceArena::DeviceArena` → `cudaMalloc`；另一种落在 `build_shard` 里；
- **分开都正常，按产品顺序串起来就崩**：`test_tp2_load`（直接 `materialize_model_tp2`，不建 `DevicePair`）4/4 通过；
  `test_tp_device_pair`（建 `DevicePair` 并跑集合通信，不 materialize）通过；`ninfer-serve`（先 `DevicePair` 再 materialize）0/4 通过。

因此根因在 **`DevicePair` 构造后留下的状态：它使随后的第一次 `cudaMalloc` 在驱动内部崩溃**。未修复。下一步应在一个不经过 HTTP 的
最小程序里把这两步串起来复现，再对 `DevicePair` 里的 peer access / 映射分配 / 通道单元格逐个二分。

**2026-10-23 用最小复现把上一条推翻**（复现程序现在是正式测试 `tests/models/qwen3_5/test_tp2_startup_order.cpp`）：

| 复现程序 | 做的事 | 结果 |
|---|---|---|
| `plain` | 只 materialize | 通过 |
| `pair` | `DevicePair`，然后 materialize | 4/4 通过 |
| `mapped` | 映射分配 + 通道单元格 + materialize | 通过 |
| `core` | 复刻 `TP2GenerationCore` 构造体：`DevicePair`、`DeviceContext` 移动赋值、`Reader`、`plan_load`、`materialize_model_tp2`、两个 `Parameters` | 3/3 通过 |
| `engine` | `ninfer::Engine`，选项与失败的命令行一致 | **7/7 失败**（0.4 s 与 28 s 两档） |

所以"`DevicePair` 留下坏状态"**也不成立**：`DevicePair` + materialize 是干净的。要复现必须是一个更大的进程
（`Engine` 或 `ninfer-serve`）。同时排掉：`--no-prefix-reuse`（关掉 host checkpoint ring）照样崩、`--max-context 2048`
与 `16384` 都崩、`--spec`/`--vision`/`--embedding-host` 都无关，所以 ring 和上下文大小都不是原因。

**根因：陈旧的目标文件，即一次主机堆越界写。** 给复现程序装上守卫页堆（重写全局 `operator new`/`delete`，把每个块尾贴到
`PAGE_NOACCESS` 页之前）后，故障当场变成确定性的：两次运行都停在同一条**写**指令上，

```
写地址 0x25f07565000（正好是某堆块后面的守卫页）
#0 TP2GenerationCore::TP2GenerationCore+0x3ec
#1 Engine::Impl::Impl+0x290
#2 Engine::Engine+0x134
```

也就是这个构造函数往 `Engine::Impl` 为它申请的堆块之外写了东西：块的大小由 `engine.cpp` 按**它的** `sizeof(TP2GenerationCore)`
申请，成员却由 `tp2_generation_core.cpp` 按**它的**布局写入，而 `engine.cpp` 的目标文件还是加 `reuse_stats_` 之前的版本。
只重新编译 `engine.cpp` 后，同一个程序从 7/7 失败变成 2/2 通过；随后触摸 `include/ninfer/types.h` 与
`tp2_generation_core.h` 全量重建，`ninfer-serve --devices 0,1` 从 0/10 变成 **4/4 存活 50 秒**，真实请求也通了
（59 prompt / 8 completion token）。此前所有"设备释放窗口""`DevicePair` 留坏状态""第三张卡""FP8 codec"的结论全部作废。

**为什么会有陈旧目标文件**：中文 MSVC 把 `/showIncludes` 的输出写成"注意: 包含文件:"，CMake 只认英文前缀，
于是**一条头文件依赖都没记下来**；改头文件后增量构建什么都不做，二进制里混着两套结构体布局。同一天再修一次时
（改 `engine_core.h` 后 ninja 报 "no work to do"）确认了这一点。修法：`tools/win_port/vcvars.ps1` 设 `VSLANG=1033`，
顶层 `CMakeLists.txt` 在 `CMAKE_CL_SHOWINCLUDES_PREFIX` 为空时直接 `FATAL_ERROR`；旧对象没有依赖记录，
必须 `--clean-first` 重建一次。细节见 `docs/windows.md`。

教训：一次"重建就好"的崩溃，先怀疑构建，再怀疑引擎。`tests/models/qwen3_5/test_tp2_startup_order.cpp` 现在覆盖
"`Engine` 在两张卡上完成加载"这条产品启动路径，`NINFER_GUARD_HEAP=1` 可以在下次出现同类症状时直接定位到越界写的那条指令。

方案 B 随后改为**惰性分配**（首次捕获时才申请），实测正常：捕获到 `pos=21641 / 21667 / 21825`，正是每轮的一致点，
比共享块边界 21,504 深 137–321 token。**但 recall 尚未把它当候选**，所以复用数字仍是 21,504。
方案 B 随后改为**惰性分配**（首次捕获时才申请），实测正常：捕获到 `pos=21641 / 21667 / 21825`，正是每轮的一致点，
比共享块边界 21,504 深 137–321 token。**但 recall 尚未把它当候选**，所以复用数字仍是 21,504。

接 recall 前必须解决一个正确性问题：日志显示两个会话的 `active_session` 都是 0，**会话下标不能做标签**。正确判据是
"这张图是前 P 个 token 之后的模型状态"——只要前 P 个 token 一致，状态就正确，与哪个会话写入无关。唯一要防的是条目被改写后
图变旧，因此标签应为 `tokens[0..P)` 的哈希（FNV-1a 足够），用时比对。

**阶段 1 已开工（2026-10-22）：记账层落地。** `src/runtime/engine/prefix_reuse.{h,cpp}`（无 CUDA，编进
`ninfer_runtime_support`）+ `tests/test_prefix_reuse.cpp`（12 组断言全过）。已实现的正是 §4/§6/§7.1/§9.1 的规则：
`AnchorTable::floor`、`decide()`、`record()`（PromptEnd 替换、预算淘汰、Pinned 保护、耗尽即降级）、`commit()` 的两份
extent 裁剪、`forget()` 回收槽。

写代码时改掉了两处设计里想当然的地方：

1. **`AnchorBeyondPrompt` 恒为假** —— `floor(shared)` 的候选不可能越过 shared，而 shared ≤ prompt 长度。删除该原因；
   同时确认 `reuse == prompt 长度` 是合法命中（纯 decode），这正是"零后缀复用"那个已有测试覆盖的场景。
2. **淘汰方向必须是最小 position**（离前沿最远），我第一版写成了最大，单测立刻抓住：淘汰方向与采样方向必须同向，
   否则存活集合出现空洞，"连续窗口"这条不变量就没了。

尚未接入执行层：还没有调用方，`restore()` 与 core 的接线是阶段 1 的后半段。

**阶段 1 剩余工作：锚点表 + 槽分配器 + 会话归属**（收益最大、可独立验收）

- 引入 `ConversationLog`/`Anchor`/`AnchorTable` 与 §4.6 的池分配器，把 device snapshot、grid、tail、divergence、
  block、prompt-end 全量映射进来（**记账层已完成**，剩下的是映射与接线）；文件与所有权划分见 §15；
- **保留 session catalog 作为会话注册表**（阶段 1 不删 `SessionEntry`，只删它与锚点重复的字段），理由见 §15.3；
- 按 §8.1 实现 `restore()`，跨 lane 恢复打通；
- 验收：req#10 那类请求从 `reuse=0` 变为 `reuse=65,536`（最近的 8192 栅格点；`shared=72,327` 处没有锚点，
  因为上一个请求结束在 72,499 且从未在 72,327 分歧过——这正是 §7 要"分歧必拍"的原因：它只对**下一次**生效）。
  重算量从 73,205 降到 6,791；产出等价性判据见 §16.2。

**阶段 2：单一提交规则**，删除三分支与清零逻辑，落地 §9.1 的两份 extent。

**阶段 3：删除旧机制**：五个子环、`valid[lane]`、`restore_path` 枚举在指标上的用途（换成 §16.1 的 `reuse_source`/`reuse_reason`）、重复 trace；
catalog 的注册职责交回 `ConversationLog` 的所有者，同时删掉 `SessionEntry`。

**阶段 4：标定张数与上限**：`stride=8192` 已经是契约，不再重标；标定每会话上限、`--host-state-images`
的默认值（按 7.1 反推，现按 20 写），并重新评估 4.4 的跨会话共享是否升级为 `Pinned`、以及 `Device` 锚点的转存
代价（14.2-1）。

**parity 矩阵**（每个阶段结束都要过）：零拷贝 live 复用、跨 lane 恢复、长上下文覆盖、
Draft 配对、媒体身份、块/分歧钉住、offline CausalScoring、取消/回滚、并发 2 lane。

---

## 13. 删除清单

- ring 的 grid / tail / divergence / block / prompt-end 五套子环与其索引逻辑。**2026-10-24 已完成**：分配已是池化的
  （`reserve_image_slot`），五个子环只剩"什么时候写一张镜像"这层策略含义，不再有各自的槽配额；
- tail 的 `min(kReuseTailCheckpointCount, flexible / 2)` 硬上限，以及"先把槽分给各子环、再拿剩余槽位反推 stride"
  的分配方式。**2026-10-24 已完成**（`95587ab2`）：池 = 配置张数 + 每个请求可能冻结的角色张数，
  `host_checkpoint_stride_` 就是常量 8192，`host_checkpoint_grid_slots_`/`tail_slots_`/`divergence_slots_`/
  `block_slots_` 四个成员一并删除；tail 窗口保留自己的计数，但它是关于边界网格的陈述，不再是子环份额；
- `valid[lane]` / `position[lane]` / `prefill_id[lane]` 这套按 lane 寻址的镜像记账，包括槽号
  `lane * per_lane + offset` 的算法（改由 §4.6 的分配器给出）。**2026-10-23 已完成**：四个元数据字段都是标量，
  写路径不再算地址（所有宿主镜像都由 `reserve_image_slot()` 给槽），`HostRing` 角色枚举、固定角色槽的
  `mark_slot_fixed()` 与 `host_checkpoint_next/tail_next` 计数一并删除。布局常量只剩定容与定 stride 的用途；
- 单一的 `kv_extent` 与"分歧时把 `prompt_end`/`shared_end` 清零"的裁剪方式（改为 §9.1 的两份 extent）。
  **2026-10-23 核对后不适用**：`host_kv_end` 是 host slab 的覆盖长度，而 `host_prompt_end`/`host_shared_end`/
  `host_frontier_end` 是**三张具名镜像各自的边界**（`session_recall` 用它们选该加载哪一张），不是两个 store 的
  extent；那几处清零发生在历史被改写、镜像不再描述新历史的时候，是正确的失效而不是 §9.1 的事故。要把它们换成
  "锚点持有 slab 引用"需要 slab 引用计数与跨会话共享锚点（第 4 批），不是本轮能删的东西；
- `session_publish` 的三条分支与字段清零。**2026-10-23 核对后不适用**：三条分支是 catalog 的**发布策略**
  （驻留条目原地更新 / 延续的条目原地重写 / 新建条目 + 容量淘汰），编码的是"这条请求延续哪个条目"，
  属于会话注册职责，不是锚点记账；删掉它们等于删掉 catalog 的发布语义。**2026-10-24 补充**：判据本身后来补了
  一条（§19.5）——压缩（前缀一致、中段重写）也是"延续"，走的是第二条分支，所以这三条一条都不能删，
- `restore_path` 在**指标**上的用途，按 §17 的 `reuse_source`/`reuse_reason` 取代。原计划是连枚举一起删，**2026-10-23 改为
  收窄作用域**：该枚举在单卡路线里是程序内部的**恢复路径**（`is_rewrite_checkpoint_restore()` 与
  `PrivateEndpoint`/`SharedStablePrefix`/`PrivateLongAnchor` 在 `program/prefill.cpp`、`context.cpp`、`pressure.cpp` 里做路由判定），
  不是存储标签——§17.6 已经写明"旧枚举是角色，不是存储"。删它就得重新编码那套路由，是净增复杂度。
  已完成的删除：`record_prefix_selection()`、六个 `*_selections`、`last_selected_frontier_tokens`、`context_cache.selections`
  与 `last_selection`。**保留**：枚举本身，作为单卡程序的路由词汇；2026-10-23 已按 §17.6 改名为 `RestorePath`、
  字段改为 `restore_path`（JSON 键同名），语义未动——原名把"复用了多少"和"走了哪条恢复路径"混在一起，
  是这套指标长期被误读的根源；
- `cached_prompt_tokens` 与 `entry.tokens` 的双份维护（合并为单一 `log.tokens`）——**2026-10-23 核对后不合并**：
  这不是重复存储，而是"每次请求与自身谱系一致的那段比较流"与"持久身份"两件事，ledger 的
  `ConversationLog.tokens` 也只承担前者；删掉 `cached_prompt_tokens` 会让没有 catalog 条目的 lane（匿名请求、
  catalog 满、`--host-kv-mib 0`）连自己那条谱系都不能复用；
- device/host 双份代码路径。

### 13.1 改名：让单位诚实

`--host-state-slots` 这个名字在三处不诚实：`slots` 是内部索引单位而不是资源单位；叫 `host-state-` 却读起来像
context cache 的通用容量（TP-2 上它实际只管 prefix-reuse 的镜像池）；而且"槽数"本来是引擎该反推的量，不该由用户算。

| 旧 | 新 | 语义 |
|---|---|---|
| `--host-state-slots N` | `--host-state-images N` | host 上最多驻留多少**张**状态镜像。一张 = 两个 shard 各一份 73.4 MiB 镜像 = **146.8 MiB**；N=20 ⇒ ≈2.9 GiB |
| `--device-state-slots N` | `--device-state-images N` | 语义不变（超出活跃 lane 的额外设备镜像数），只统一词 |
| `--host-kv-mib N` | 不变 | KV 是变长的，MiB 是诚实单位；状态镜像是定长的，张数才是 |
| `--no-prefix-reuse` | 不变 | 等价于 `--host-state-images 0`，行为不变 |

同一处代码里的连带改名（分两次改没有意义）：

- `ContextCacheOptions::host_state_slots`、`ProgramSettings::host_state_slots` → `host_state_images`；
- 占用观测字段 `host_state_occupied_slots` → `host_state_occupied_images`（它报的是占用，不是容量）；
- `serve_options.cpp` 的 usage 串与 `--help` 正文；`request_log.cpp` / `operational_log.cpp` 里的 JSON 字段名；
- 常量 `kDefaultHostStateSlots` → `kDefaultHostStateImages`；
- 测试：`tests/test_serve_options.cpp`（CLI 解析、默认值、负值/超界用例）、`tests/test_engine_options.cpp`、
  `tests/test_request_log.cpp`、`tests/test_resource_manager.cpp`、`tests/models/qwen3_5/test_engine_prefix_real.cpp`
  都直接写 `host_state_slots`，必须同一次改完；
- `[mem]` 行改为**一行报总数**，不再 per-shard 打印（用户关心的是总内存）：
  `[mem] host state images 20 x 146.8 MiB = 2.9 GiB (pageable)`；
- `docs/serving.md`：第 16-17 行的示例命令、976-977 行的参数表（顺带删掉"the core adds one slot of its own for
  the divergence anchor"，那个 `+1` 在新模型里不存在）、`docs/maintainer/resource-scheduling-and-context-cache.md`；
- `tools/win_port/serve.ps1` 目前并不传这个参数，因此不需要改；`_temp/` 下的复现脚本随任务清理，不跟改。

**已执行**：代码、公共头文件、测试与当前权威文档已按上表改名；`--help` 显示 `--host-state-images N` / `--device-state-images N`。
受影响的四个测试（serve_options、engine_options、request_log、resource_manager）均通过。`docs/maintainer/` 下的 worklog 与 plan 文档按下面这条规则保持原样。

**不改**：`docs/maintainer/` 下的 worklog 与 plan 文档。它们记录的是当时实际敲过的命令，把新名字写进去等于篡改
历史记录；新名字只出现在当前权威文档里。

---

## 14. 已定决策与开放问题

### 14.1 已定（2026-10-22）

| 问题 | 决定 | 依据 |
|---|---|---|
| 采样间距是否随内存预算缩放 | **否**。`stride=8192` 是契约，张数反推，余额不足显式降级（§7.1） | 缩放会把最坏代价从 4.9 s 抬到 9.4 s（36 张）或 62 s（默认 8 张，256K 场景），并让 TTFT 依赖内存参数 |
| 历史段（网格）的 stride 是否也契约化 | **是**，同样固定 8192（§7 第 3/4 条） | 一个会话只该有一种采样粒度；代价是 256K+256K 需要 67 张 ≈9.6 GiB，属明确的资源决策，已接受 |
| 是否跨会话共享锚点 | **构造上免费，不设策略**：key 即内容、KV 归锚点、不 pin、不做独立预算（§4.4） | 收益 = 前缀 token / 1,687 秒；6K 前缀 × 8 会话 ≈ 28 s，2K 以下属噪音。pin 策略的成本落在最脆弱的预算模型上 |
| 内存参数怎么命名 | `--host-state-images` / `--device-state-images`，按"张"计；`--host-kv-mib` 保持 MiB（§13.1） | 状态镜像是定长的（146.8 MiB/张），张数才是用户能对上内存的单位；`slots` 是内部索引，不该出现在 CLI |
| `--host-state-images` 的默认值 | **20**（≈2.9 GiB，覆盖 64K 提示 + 64K 生成） | 按 §7.1 反推；旧的 8（实占 12 张 ≈1.7 GiB）只够 40K 双段覆盖 |
| 规整失败时是否允许强制续写 | **不允许**（第 5 节）。失败就用客户端流并把代价暴露出来 | 静默改写历史的正确性风险远大于一次全量 prefill；要重开需要独立开关，不作为默认路径 |
| 单卡路线是否也迁到 ledger | **不迁**。单卡保留自己的 context cache / `StateImageStore`；`RestorePath`（原名 `PrefixReusePath`）收窄为程序内部的路由词汇（§13 第 5 条） | 单卡那套是参考路线的在线特性（device/host 双副本、跨会话 shared prefix、long anchor、rewrite checkpoint、压力驱逐、MTP 桥接），是 ledger 今天能力的**真超集**；且它与准入控制（`ResourceManager`/`PressurePlanner`/`MaterializationPlanner`/`AdmissionCandidate`）耦合，迁移必须连 admission 一起重写。收益只是"少一套代码"，代价是数值参考路线与一到两周工期 |

### 14.2 仍开放

1. `Device` 锚点在 lane 退休时转存 Host 的成本，是否值得（一次 73.4 MiB × 2 shard 的 D2H）。
   先按"转存"实现，用实测的 admit 延迟决定是否保留。

---

## 15. 实现落点

### 15.1 文件划分

| 位置 | 放什么 | 为什么 |
|---|---|---|
| `src/runtime/engine/prefix_reuse.h/.cpp`（新） | `ConversationLog` / `Anchor` / `AnchorTable` / `HostPool` 的记账、`decide()`、`commit()`、预算与淘汰 | 不 include 任何 CUDA 头，可脱离模型与显卡单测 |
| `src/runtime/engine/tp2_generation_core.cpp` | 池的物理分配、镜像 D2H/H2D、`restore()` 的执行（要用 lane geometry 和 stream） | 它已经持有 catalog 与设备资源，不新增第二处所有权 |
| `src/serve/session_identity.cpp` | 只负责把请求头变成 `SessionKey` | 引擎不该认识 HTTP |

边界一句话：**记账层不碰 CUDA，执行层不碰 HTTP。** 违反它的表现是 `decide()` 里出现 `cudaMemcpy` 或 `httplib`。

### 15.2 `decide()` 必须是纯函数

```
struct DecideInput  { span<const TokenId> incoming; span<const MediaSpan> media; };
struct DecideResult { u32 reuse; const Anchor* pick; Reason reason; Source src; };
DecideResult decide(const ConversationLog&, DecideInput, const AnchorIndex& donors);
```

要求：普通结构体输入，无锁、无 CUDA、无 I/O、无日志副作用（日志由调用方打）。这样 §6 的每个原因枚举都能被一个
不加载模型的测试覆盖——这是把"40 秒加载 + 40 分钟跑完"的集成测试压成毫秒级单测的唯一办法，也是唯一能稳定复现
req#8 / req#10 判决的形态（对照 `tests/test_turn_replay.cpp` 已有的做法）。

### 15.3 阶段 1 保留 catalog 作为会话注册表

`SessionEntry` 现在同时干三件事：会话注册（key → slab、TTL、`max_private_continuations`）、持久化、锚点记账。
本设计只取代第三件。所以阶段 1：

- 保留 catalog 的注册与持久化职责，只删 `SessionEntry` 里与锚点重复的字段（边界、`prompt_end`、`shared_end`）；
- `ConversationLog` 的 owner 是 `Tp2GenerationCore`（`map<SessionKey, ConversationLog>` + LRU），不是 serve；
- catalog 整体退场排在阶段 3。

不这么做的后果很具体：`ConversationLog` 没有 owner 就没有 TTL，`max_private_continuations` 无处可落，重启后的会话
恢复会先于锚点设计被打破。

### 15.4 存储所有权：池归 ledger，镜像归 ring

一句话：**ledger 决定一张镜像落在哪个槽，ring 只是那块内存。** 这样"两个分配器管一个池"不会出现，
淘汰规则也只有一个（先取冗余的 Lru 锚点，没有冗余时取位置最远的，见 §7.3）。

契约（都在 `src/runtime/engine/prefix_reuse.h` 上）：

| 谁 | 做什么 |
|---|---|
| 池的构造 | `ReuseLedger` 的容量就是整个 ring 的槽数；没有"保留组"，任何槽都可以给任何一张镜像 |
| 写镜像（`snapshot_host_checkpoint`） | 先 `reserve_image_slot()` 拿槽，再 `cudaMemcpy` 写进去；池满且无可淘汰锚点时拿不到槽，这次捕获直接跳过（边界没有检查点，调用方本来就能容忍） |
| 记账（`record()`） | 槽只要 `reserve` 过就记在 ledger 名下：替换或淘汰这个锚点时槽回到预算。传一个**当前没人持有**的槽（淘汰后 ring 里还留着的旧镜像）会重新认领它，否则后来的镜像可能盖到一个活锚点的缓冲上 |
| 读镜像（`restore_lane_state` / `restore_lane_dflash`） | 仍用 `reuse.slot` 直接索引 `host_checkpoints`，不需要知道它是怎么分配的 |

顺序上的一个细节：**必须在拷贝入队之前 reserve**。池满时的淘汰发生在那一步，被淘汰锚点的镜像正是这次拷贝要覆盖的
那块内存——先 reserve 就把"谁被覆盖"和"谁失去锚点"绑成了同一件事。

已落地（2026-10-23）：**每一张宿主镜像都来自同一个池**——grid、tail、divergence、block、prompt-end 不再各有
地址，`HostRing` 枚举与写路径里的 `lane * per_lane + offset` 布局算术整体删除，`mark_slot_fixed()` 随之退场；
`host_checkpoint_next/tail_next` 两个轮转计数也删了。布局常量（`grid/tail/divergence/block`）只剩**定容与定
stride** 的用途：`host_checkpoint_tail_slots_` 还决定 tail 锚点的写入间隔，`host_checkpoint_grid_slots_` 还决定
位置栅格的 stride。

证据：双卡真实请求同一会话三轮，`cached_tokens = 0 / 8192 / 8192`，第二次恢复的 `slot=8` 是池分配的（不是公式算出来的）。

**`restore()` 已收成一个入口（同日）**：`restore_lane(lane, slot, zero_draft_when_none)` 是"复用变回设备状态"的
唯一去处——目标 GDN 镜像与 draft 环在同一处、按同一个 `reuse_source` 恢复，两个 shard 一起走。收口前有**四份**
实现：`restore_lane_gdn`（两个 shard）、`restore_lane_dflash`（只有 shard A）、以及 `execute_walk` 里两个各管半边
的内联 lambda。三者现在共用一条路径，所以"目标状态恢复了、draft 没恢复"这类分歧在结构上不可能再出现。

一个参数保留了两种调用语义：`zero_draft_when_none`。单机走查（`execute_walk`）传 `true`，因为它可能把 draft 环
留在另一条谱系的前沿；批处理（`execute_plain_batch` / `execute_spec_batch`）传 `false`，因为批处理 lane 会用自己的
prefill sink 重建整个 draft 环。plain 路线本来就没有 draft 环（只有两个投机后端都关掉时才走它），所以那一路是空操作。

证据：同一会话三轮 `cached = 0 / 8192 / 8192` 不变；`tp2_forward` 的"批处理 prefill 与逐 token walk 一致"通过。

### 15.5 跨 lane 复用的真正条件：镜像属于"对话"，不属于"lane"

槽不再按 lane 分片之后，"一条 lane 用另一条 lane 的镜像"看起来只剩下一行判断。**试过再回退**，两次都失败，
原因比预想的根本：

一张镜像是**某段 token 历史之后**的状态，只有产生它的那段历史才认得它。lane 只是执行位置——两条 lane 在同一
位置的两张镜像可以对应完全不同的 token。所以"任何 lane 都能用任何镜像"是**静默错误**，不是优化。

按会话归属也走不通，两个具体障碍（都实测过）：

1. **会话号在扫描之后才有**。`lane_state.active_session` 在请求开始时被重置为 `kNoSession`，到
   `session_publish()` 才绑定；而 reuse 扫描发生在两者之间。实测：写入时 `conversation_key` 是 lane 回退键
   `0x8000…`，发布时变成 `0`，于是"发布时认领 owner"永远对不上，`host=0/11`、`reused=0`。
2. **扫描只校验本 lane 的谱系**。`shared_prefix` 是拿本 lane 的 `cached_prompt_tokens` 比的；另一条 lane 的
   镜像属于别的历史，没有被校验过。要让别的 lane 也能用，镜像必须自带**它自己那段 token 的标识**，
   而不是"写它的会话号"。

所以现在的落点：镜像归属仍是 **lane 作用域**（`writer == lane`），跨 lane 复用推到"镜像自带历史标识"之后。
这一轮真正拿到的收益是**每槽单份元数据**：`valid/position/prefill_id/dflash_frontier` 从按 lane 的数组变成标量
（§13 第 ② 条），因为一个槽同一时刻只有一张镜像、一个写者。池把槽交给另一条 lane 时，写入会清掉整个描述，
`writer` 记下是谁写的。

### 15.6 持有者集合：一张镜像活到最后一个会话放手

同一段 token 历史的两个会话（例如同一会话在两条 lane 上被调度）可以共享一张镜像。**2026-10-23 起规则改为引用
计数**：每个槽记一个"持有者集合"，镜像活到集合为空。

| 事件 | 结果 |
|---|---|
| 登记（指纹通过） | 把本会话加进该槽的持有者集合（幂等：同一会话重复登记只算一次） |
| 槽当前没人持有 | 顺带认领：`used` 且 `owned`，写者不再有特殊身份 |
| 本会话放手（`forget()`、锚点被淘汰、被同位置的新锚点替换） | 只把自己从集合里移除 |
| 集合变空 | 槽才回池（`free_slot`） |
| 淘汰（`evict_one()`） | 只考虑**本会话独占**的锚点：别人也持有的镜像删了腾不出空间，还会伤到读者 |

为什么换掉 owner/borrower：那条规则是"拥有者放手时把借用者一起删掉"，而跨 lane 复用之后，**写这张镜像的会话
可能已经走了，读它的会话才是唯一用户**——借用者会跟着拥有者一起被回收，正好把该留下的一张删掉（§18.3）。

代价是释放路径多了一步判断：`release_slot()` 现在先看持有者集合是否为空，非空就什么都不做；真正清空的是
`free_slot()`，它同时把任何还指着这个槽的锚点删掉（防御性的，集合为空时理论上不该有）。

### 15.7 `SessionEntry` 的三个职责（核对结果）

`SessionEntry` 一直被当成"锚点记账"的替身，实际清点下来它承担三件互不相干的事，**没有一件该由 ledger 接管**：

| 字段 | 职责 | 归谁 | 能否删 |
|---|---|---|---|
| `tokens`, `media` | **身份**：这段历史是哪一段（前缀比对、采纳、谱系恢复都要它） | catalog | 不能。ledger 的 `ConversationLog.tokens` 只存"本请求与自身谱系一致的那一段"，是按请求重建的比较用流，不是持久身份 |
| `frontier`, `prompt_end` | **身份**：设备停在哪、上一轮 prompt 有多长（`prompt_end` 是"这是同一会话的下一轮"的判据，也是 `adopt_generated_turn` 的 `turn_begin`） | catalog | 不能 |
| `device_lane` | **注册**：这条会话此刻驻留在哪条 lane（一个条目一份驻留，配 `mark_device_lane` 维护） | catalog | 不能 |
| `session`, `retention_weight`, `lru_clock` | **注册**：客户端命名的会话、淘汰权重、LRU 次序 | catalog | 不能（`session` 的 `optional` 语义见下） |
| `host_kv`, `host_state`, `host_prompt_state`, `host_shared_state`, `host_dflash*`, `host_mtp_kv` | **持久化**：被逐出设备的会话的 KV 板与三组状态镜像（frontier / prompt-end / shared，各自配一份 draft） | catalog | 不能（跨会话 recall 的唯一载体） |
| `host_kv_end` | **持久化的覆盖**：host slab 覆盖到哪（所有与 slab 配对的操作都必须落在它之内） | catalog | 不能 |
| `host_frontier_end`, `host_prompt_end`, `host_shared_end` | **三张具名镜像各自的边界**：`session_recall` 按"要恢复到哪个位置"选该加载哪一张 | catalog | 不能 |

结论一：**catalog 没有"锚点记账"这一职责**。锚点表在 ledger 里，由图上的镜像（ring 槽、设备快照、活前沿）登记；
catalog 记的是**持久化副本**——一组具名镜像加它们的边界。两者地址空间不同：ring 槽会淘汰、会被后写覆盖，
catalog 的 slab 只在条目被逐出时消失。

结论二：因此 §13 剩下的两条（删单一 `kv_extent`、删 `session_publish` 的三条分支）**不成立**：前者的三个
`*_end` 不是 extent 而是镜像位置，后者是发布策略而不是锚点记账。真正能做的版本是"让锚点持有一个 slab 引用、
活得比登记它的会话久"（§10、§4.4 的共享锚点），那需要 slab 引用计数——属于第 4 批。

结论三：`session` 那个 `optional` 是注册语义的支点：一个具名会话在 catalog 里占且只占一个条目。名字怎么落到那个
条目上分两种情况——**延续**（包括 §19.5 的压缩）在原地改写，只有真的开了一段新谱系时名字才从旧条目移到新条目
（`bind_entry_session`）。这也是"一个会话不会用变体填满所有槽位"的实现。删任何与它配套的分支都会先破坏这条。

---

## 16. 测试处置与判据

### 16.1 逐条分类

先给数字：`tests/models/qwen3_5/test_engine_prefix_real.cpp` 里 `reused_prompt_tokens` 出现 **78** 次、
`restore_path`（原名 `prefix_reuse_path`）出现 **54** 次。后者是**公开 API**（`include/ninfer/types.h` 的枚举
`RestorePath` 与字段 `restore_path`），而 §13 曾计划删它——所以阶段 1 做完，这套测试必然红一片。必须先分清"红得对"
和"红得不对"：

| 测试 | 现在断言什么 | 处置 |
|---|---|---|
| `test_engine_prefix_real.cpp` 的 `reused_prompt_tokens`（78 处） | 追加式复用、零后缀复用、跨会话恢复、媒体前缀各自的命中 token 数 | **保留结构，逐个更新期望值**。数值会变（栅格从"按预算反推的 32,768"变成 8192），但"命中多少"仍是行为契约；每条期望值旁注明它落在哪个锚点上 |
| 同文件的 `restore_path`（54 处） | 单卡程序走到哪条恢复路径（`SharedStablePrefix`、`Root` 等） | **保留**。该枚举是单卡程序的**路由词汇**，不是存储标签（§17.6）；2026-10-23 只把它从 `PrefixReusePath` 改名为 `RestorePath`、字段改名为 `restore_path`，语义未动。双卡路线用 `reuse_source`（存储轴 `None/Device/HostSlab/LiveState`）与 `reuse_reason` 报告同一件事，定义见 §17 |
| `test_turn_replay.cpp` | `same_rendered_turn` / `adoption_candidates` 的逐字节判等与候选筛选 | **阶段 1 不动**；§5 落地时一起改写（它是 §5 的被替换对象） |
| `test_tp2_sessions.cpp` | catalog 行为、跨会话回落、adoption 读 catalog | **保留**（阶段 1 保留 catalog）；只删依赖 ring 槽位布局的断言 |
| `test_resource_manager.cpp` 的 `host_state_occupied_*` | 设备侧 context cache 占用 | **随分配器改写**；设备侧语义不变的部分保留 |
| `test_engine_options.cpp` / `test_serve_options.cpp` / `test_request_log.cpp` | 字段名与 CLI | **机械改名**（§13.1），无行为变化 |

### 16.2 判据：可执行的等价性

- **复用点**：夹具断言 `reuse == 65,536` 且 `reason` 命中枚举（不是自由文本）。
- **产出等价性**：贪心续写 N 个 token，**token id 全等**。不用 logits 容差——`restore` 之后批形状变了，reduction
  顺序不同，bit 级永远比不过；而 token id 全等正是用户能观察到的等价性。
- **反例保护**：同一条 trace 走"从零 prefill"必须得到同一串 token id。若两条路径不等，先怀疑 `restore()` 的
  KV / 状态 / draft 三者不同步（§8.1 的失败面），而不是数值精度。

---

## 17. 指标：维度与生命周期

> 代码先于定义存在：提交 `78790d9a` 落了两个枚举和结果侧字段，但没回答"这些数字各自由谁定义、何时定稿、算给谁"。
> 这一节是那份定义，同时纠正实施过程中查出的两处错判（§17.5、§17.6）。

### 17.1 一条记录 = 一个完成的请求

统计单位是请求，不是 lane，不是锚点，不是字节。

- TP-2 一个 round 批两个 lane ⇒ 两条记录。
- prefill 走完第一个 chunk 的请求**要计**，prefill 中途被取消的**不计**。口径与 `computed_prefill_tokens`
  完全一致（两者在同一个位置累加），所以"重算的 token"与"复用的 token"永远来自同一批请求，比值才有意义。
- 排队期就被丢弃的请求**不计**：它从未看过锚点。
- 一个请求只产生一条记录：扫描在第一个 chunk 之前做一次，此后的 chunk 不再重算。

### 17.2 两个轴，四个合法组合

| 轴 | 类型 | 取值 | 回答的问题 |
|---|---|---|---|
| `source` | `ReuseSourceKind` | `None` / `Device` / `HostSlab` / `LiveState` | 状态从哪种存储来 |
| `reason` | `ReuseReasonKind` | `Used` / `NoAnchorAtOrBefore` | 决策为什么是它 |

`source` 的准确含义（写死，不靠字面猜）。四个值都由 core 的扫描产生：批量路径是 `scan_lane_reuse()`，
串行路径是 `execute_walk()` 里的内联扫描，两者经同一个 `reuse_source_kind()` 映射。

| 值 | 含义 |
|---|---|
| `Device` | 状态从一张绑在本 lane 上的设备镜像恢复（`ReuseSource::DeviceSnapshot`）：KV 本来就在设备上，只有 GDN 状态做了一次 D2D |
| `HostSlab` | 状态从 host 镜像恢复（`ReuseSource::HostCheckpoint`）：有 D2H → H2D 往返 |
| `LiveState` | lane 的活状态本来就停在这个前沿，**没有从镜像槽恢复过状态** |
| `None` | 共享前缀上没有锚点，从 token 0 全量 prefill |

`LiveState` 有一条必须写下来的边界：它**不承诺零拷贝**。会话目录把别的会话召回本 lane 时，`session_restore()`
在扫描之前就把 KV 从 host slab 拷回了设备（`tp2_generation_core.cpp:6571`），随后扫描把这份活状态当作
`LiveState` 报出来。那次 H2D 已经由 `context_cache.state_transfers` 计数，这里再报一次只会让两个数字打架。
所以：`source` 只描述**状态镜像来自哪**，传输量归传输计数器。

### 17.3 不变量

合法的组合只有四对：`(None, NoAnchorAtOrBefore)`、`(Device, Used)`、`(HostSlab, Used)`、`(LiveState, Used)`。
其余五对自相矛盾，在累加处断言：

```
reused_prompt_tokens > 0   ⟺   source != None
source       != None       ⟺   reason == Used
```

今天两个轴是 1:1 的，`reason` 只多带一个 bit。**这是有意的成本**：`reason` 的存在理由是要长（超预算、
存储在覆盖范围之外、媒体身份不符——2026-10-22 量到的 boundary tag 不匹配就是这一类），而 `source` 只有三种存储。
第一个与 `source` 不再 1:1 的 reason 出现时，改写本节；在此之前**不许**从 `source` 反推 `reason`，
也不许把 reason 列表当成 `source` 的别名。

### 17.4 生命周期

| | `source` | `reason` |
|---|---|---|
| 生产 | 执行器。TP-2 两处扫描都经 `reuse_source_kind()` 映射：批量路径 `scan_lane_reuse()`、串行路径 `execute_walk()`。单卡：Program 侧 checkpoint 的驻留（**尚未接线**，见 §17.9） | 批量路径取 `ReuseLedger::decide()` 的返回值（`:2701`）；串行路径尚未接 ledger，由 `reuse != 0` 判定。单卡同样尚未接线 |
| 定稿 | admit 之后、第一个 prefill chunk 之前；此后只读 | 同左 |
| 载体 | `GenerationResult.reuse_source` | `GenerationResult.reuse_reason` |
| 消费 | 每请求日志；累计计数器 | 同左 |
| 终点 | 请求销毁 | 请求销毁 |

**累加规则：计数器在请求完成时按 `GenerationResult` 累加，不在 plan 侧按 `RequestPlanSummary` 累加。**

这是本节最重要的一条，理由可验证：

1. `RequestPlanSummary.reusable_prompt_tokens` 是 plan 的**估计**。双卡上 core 用自己的扫描覆盖它
   （`tp2_generation_core.cpp:3829`），单卡沿用 plan 值（`engine_core.h:865` ← `:1656`）：
   同一个字段名下挂着两个来源。
2. 双卡上客户端从 `GenerationStart.reused_prompt_tokens` 拿到的是 core 的实际值（`:3839`），
   而 `record_prefix_selection()` 累加的是 plan 值——同一个请求会报出两个数。
3. plan 阶段不知道 `reuse_source`（§17.5）。

代价：plan 侧不再有"预计复用"这个观测点。需要它的时候另开名字（`planned_reuse_tokens`），不要复用
`reused_prompt_tokens`。

### 17.5 现状：TP-2 上这套指标恒为零

实施时核对出来的事实，它解释了为什么"接线"反复打滑：

- `TP2GenerationCore::runtime_stats()`（`tp2_generation_core.cpp:8906`）新建一个 `RuntimeStats`，
  只填 `computed_prefill_tokens`、`committed_decode_tokens`、`decode_rounds`、`decode_row_rounds` 四个字段，
  其余全默认 0；它不读 `cumulative_stats_`。
- `Engine::runtime_stats()`（`engine.cpp:426`）直接 `return core->runtime_stats()`，**不做合并**。
- 六个 `*_selections` 唯一的累加点是 `record_prefix_selection()`，它挂在 `engine_core.h:1549` 的
  `EngineCore` admission 路径上；TP-2 由 `Tp2GenerationCore` 执行，走自己的 `PendingRequest` 队列与 driver。

所以双卡上六个 `*_selections`、`reused_prompt_tokens`、`last_selected_frontier_tokens` **全是恒零**，
它们今天只在单卡路线上有值。

结论：这次不是"改名迁移"，是**首次给主路线装表**。定义因此必须先写（本节），再写代码。

### 17.6 旧枚举是"角色"，不是"存储"

`RestorePath` 的五个非 `Root` 值区分的是**锚点的角色**（端点、轮次收尾、响应重放、长锚、共享稳定前缀），
对应设计里的 `AnchorProvenance`（`PromptEnd/Divergence/Block/Grid/Generated`），而不是 `AnchorStorage`
（`Device/Host`）。把它映射成 `Device/HostSlab/LiveState` 会丢信息，这是有意的：

- 角色留在 trace 与 `AnchorProvenance`：`[tp2-reuse] src=...`、`[tp2-session] via=frontier|prompt-end|shared|boundary`；
- 公开字段只报存储。理由：用户能观察到的是"重算了多少"，角色是引擎内部的事；而单卡的 `Private*` 词汇
  （轮次收尾 / 响应重放 / 长锚）与 TP-2 的 `AnchorProvenance` 并不一一对应，硬凑一个公开枚举会变成第三种方言。

对外映射只剩一条：

| 旧 | 新 |
|---|---|
| `Root` | `(None, NoAnchorAtOrBefore)` |
| 其余五个 | 只有 `source`，没有公开的 provenance 对等物 |

代价要写清楚：`test_engine_prefix_real.cpp` 里 20+ 处 `Private*` 断言（§16.1）失去对等物，替代判据是
"命中多少 token + 哪种存储"。这些断言在单卡上仍有约束力，所以删它们必须和单卡的 storage 接线同批做。

### 17.7 计数器的形状

`RuntimeStats` 增：

```cpp
// 复用决策：一个请求一条记录，两个轴。定义见 docs/maintainer/prefix-reuse.md §17。
std::uint64_t reuse_source_device     = 0;
std::uint64_t reuse_source_host_slab  = 0;
std::uint64_t reuse_source_live_state = 0;
std::uint64_t reuse_source_none       = 0;
std::uint64_t reuse_reason_used       = 0;
std::uint64_t reuse_reason_no_anchor  = 0;
// 最近一次决策的观测（gauge，不做区间相减）。
std::uint32_t last_reuse_tokens       = 0;
ReuseSourceKind last_reuse_source     = ReuseSourceKind::None;
ReuseReasonKind last_reuse_reason     = ReuseReasonKind::NoAnchorAtOrBefore;
```

`request_log.cpp` 用它替换 `context_cache.selections` 整块：

```json
"context_cache": {
  "reuse": {
    "by_source": {"device": 0, "host_slab": 0, "live_state": 0, "none": 0},
    "by_reason": {"used": 0, "no_anchor_at_or_before": 0},
    "reused_prompt_tokens": 0,
    "last": {"tokens": 0, "source": "none", "reason": "no_anchor_at_or_before"}
  }
}
```

三条约束：

1. 两个 partition 的和都等于区间内完成的请求数。单卡接线（§17.9）之前这条不成立，所以单卡上两个 partition
   先保持全 0，而不是填一半。
2. `last` 是 gauge；其余六个与 `reused_prompt_tokens` 是单调计数器，消费方照现有约定做区间相减。
3. `report_has_activity()`（`http_server.cpp:84`）里六个旧计数器兼作"这条区间有没有发生事"的触发器之一。
   删它们时必须换成新的六个，否则纯复用区间不再触发报告。
4. `context_cache.selections` 整块保留到单卡接线（§17.9 第 3 步）为止。过渡期内它只有单卡有值、`reuse`
   只有双卡有值，两者不会同时有意义——这是允许的并存，不是两个口径。

### 17.8 明确不做的

- **不按锚点 / 槽位 / per-lane 计数**：槽号会随分配器重排，锚点是内部对象；这些是 trace 的事。
- **不报字节**：`context_cache.state_transfers` 已有 d2h/h2d/d2d 的 count/bytes/seconds，重复一份必然不同步。
- **不按路线分桶**：一个进程只有一条路线，路线的区分在启动参数里。
- **不把 `AnchorProvenance` 提升为公开字段**（§17.6）。
- **不新增"预计复用"计数器**：真需要时叫 `planned_reuse_tokens`，别和 `reused_prompt_tokens` 混名。

### 17.9 接线顺序

1. `RuntimeStats` 加 §17.7 的字段（`include/ninfer/types.h`）。
2. 两个执行器各自在自己的"事实定稿"处累加：TP-2 在 `computed_prefill_tokens_` 旁边（`:4083`、`:5461`、`:8031`），
   单卡在 `finish_request`（`engine_core.h:853-895`）。
3. 删 `record_prefix_selection()` 与 plan 侧那两行累加。
4. 两个 log writer（`request_log.cpp`、`operational_log.cpp`）与 `report_has_activity()`。
5. 测试：`test_request_log.cpp` 的 throughput 夹具、`test_resource_manager.cpp` 的断言。

两个未定项，接线时决定：

- **单卡的 `source` 怎么填**：`StateReplicaResidency`（`DeviceOnly/HostOnly/Both`，`request_plan.cpp:652`）是现成的判别依据，
  `DeviceOnly → Device`、`HostOnly → HostSlab`；`Both` 归哪边未定（保守做法是 `HostSlab`，它一定经历过 H2D）。
  定下来之前单卡的两个 partition 保持 0。
- **catalog 召回之后算不算 `HostSlab`**：按 §17.2 的定义不算，扫描报 `LiveState`。如果运维想知道"这次请求是不是走了
  catalog 召回"，那是 `[tp2-session] recall` 这条 trace 的事，不是这个字段的事。

**已执行（2026-10-22，TP-2 侧）**：第 1、2、4 步的 TP-2 一半完成——`RuntimeStats` 的字段、两处扫描的
`ReuseSourceKind`/`ReuseReasonKind` 填充、`record_reuse()` 在三个 `computed_prefill_tokens_` 旁累加、
`request_log.cpp` 的 `context_cache.reuse` 块、`report_has_activity()` 的六个新触发器。第 3 步与单卡侧按上面
两个未定项推迟，`context_cache.selections` 因此暂时保留。

构建通过；`test_request_log` 新增的 `reuse` 断言通过，`test_resource_manager`、`test_engine_options` 通过。
**双卡端到端已跑通（2026-10-23）**：启动崩溃的根因是陈旧目标文件（§12 历史那条），重建后 `ninfer-serve --devices 0,1`
连续 4 次启动存活、并完成一次真实请求（59 prompt / 8 completion token）。这次请求是全新对话，没有可复用前缀，
所以 `context_cache.reuse` 的计数正确停在 0——**计数本身的非零路径仍未在真实双卡请求上验证**，只有单元断言覆盖。

**已执行（2026-10-23，单卡侧接线，第 3 步收口）**：

- 生产在 `request_plan.cpp` 的 `inspect_lane()`，四处复用点各设一次 `summary.reuse_source`：
  `SharedStablePrefix` 与 `PrivateLongAnchor` 取**服务该前缀的状态镜像的驻留**；`PrivateEndpoint` 是活前沿、
  不恢复任何镜像，报 `LiveState`；rewrite 检查点没有状态镜像时页从 host store 回来，报 `HostSlab`。
- **`Both` 归 `Device`**（原未定项 1）：`checkpoint_recovery.cpp:371` 只在设备副本缺失时才追加 host→device
  需求，所以两份都在时读的是设备副本。规则固定在 `reuse_storage_kind()` 一处，不再靠猜。
- 载体：`RequestPlanSummary` → `RequestControl::Prefill` → `BeginSummary` → `GenerationResult`；
  `complete_success()` 末尾 `record_reuse(result)` 累加，并断言 §17.3 的不变量（违反即抛，报出三个字段）。
- 删除：`record_prefix_selection()`、六个 `*_selections`、`last_selected_frontier_tokens`、
  `context_cache.selections` 与 `last_selection`；`report_has_activity()` 的六个旧触发器换成新的六个。
  `reused_prompt_tokens` 保留——它现在是 `context_cache.reuse` 块里的单调计数器，仍在完成时按实际值累加。
- 验证：`test_request_log` 通过（删掉旧指标后它从长期失败转为通过）。单卡真实请求同一会话连跑三轮，
  `prompt_tokens_details.cached_tokens` 依次 **0 → 127 → 144**，不变量未被触发。这台机器的单卡路线要先过内存关：
  权重 15.2 GiB，只有 `--embedding-host` + `--max-context 512` 才装得进 16 GiB 卡，所以单卡验证只能是短上下文。
- 观测欠账：`--log-stats-interval-ms` 没吐出带 `context_cache` 的周期报告（试了两次），所以计数器的**区间相减**
  没有人眼验证过，只有 JSON 形状的单元断言与逐轮 `cached_tokens` 佐证。

---

## 18. 跨 lane 复用：让镜像自带历史指纹

这是第 4 批的第一件事，也是"两张 16 GiB 卡上跑 4 条 lane"时最实际的一处损失：会话换到另一条 lane 就失去自己在
ring 里的全部检查点。

### 18.1 现状与两个实测阻塞

`Shard::HostCheckpoint` 的归属是 `writer == lane`（§15.5）。扫描只登记本 lane 写的镜像，理由是两条：

1. **校验不足**：`shared_prefix` 是拿本 lane 的 `cached_prompt_tokens` 比的；别的 lane 的镜像属于别的历史，
   没被校验过。用它恢复等于相信"这个位置碰巧对得上"。
2. **所有权绑定失败过一次**：把归属改成按会话号键控时，写入侧看到的 key 是 `0x8000…`（无会话的匿名键），
   发布侧看到的却是真实会话号——`active_session` 在 `session_publish` 里才绑定，扫描早于它。结果 `t2 cached=0`。
   回退到 `writer == lane` 后恢复 8192。

### 18.2 指纹：镜像记下它冻结的整个前缀

位置本身不是身份，**位置之前的 token 序列**才是。所以给每张宿主镜像加一个字段：

```
HostCheckpoint.history_hash = FNV-1a(tokens[0, position))
```

- **写入侧**（`snapshot_host_checkpoint`）：写者手里有本次 prefill 的 token 流，按 frontier 增量算一遍即可，
  与拷贝本身同量级，不引入新的大 O。
- **扫描侧**：候选位置是稀疏的（每个有效槽一个），把候选位置排序后对 `cached_prompt_tokens[0, shared_prefix)`
  走一遍，在候选位置处取快照。O(提示词长度 + 槽数 log 槽数)，每请求一次，可忽略。
- **比较**：`checkpoint.history_hash == 本提示词在 position 处的前缀哈希` 才允许登记。等号意味着"这个镜像冻结的
  token 序列与本提示词到该位置为止完全一致"，与写它的 lane、会话号、`prefill_id` 都无关。

这与 `reuse_boundary_tag` 的先例一致（`tp2_generation_core.h:237`："The image is the state of that prefix whatever
conversation wrote it, so the tokens are the identity"），但**强度不同**：boundary 的 tag 只哈希边界前 8 个
token，因为它只会被同一个 lane 加载，8 个 token 足以发现"尾部被改写"；跨 lane 需要的是**整段前缀**的证据，
否则两段共享同一段样板文字、前面完全不同的历史会互相当成对方。

### 18.3 所有权：owner/borrower 不够，要引用计数

指纹解决了"能不能用"，还剩"谁让它活着"。今天的规则是 owner/borrower（§15.6）：拥有者释放时
`release_borrowers()` 把借用者一起删掉。跨 lane 之后这条会错——写它的会话已经走了，读它的会话才是唯一的用户，
却会跟着拥有者一起被删。

改为**每个槽一个持有者集合**：

| 事件 | 结果 |
|---|---|
| 登记（指纹匹配） | 把会话加进该槽的持有者集合 |
| 会话 `forget()` / 条目被逐出 | 只把自己从集合里移除 |
| 集合变空 | 槽才回池（`release_slot`） |
| 淘汰 | 只考虑持有者集合等于 {自己} 的锚点；别人也持有的槽删了腾不出空间 |
| 一个会话登记同位置的新镜像 | 先把自己从旧槽摘掉，再挂到新槽 |

`Anchor.borrowed` 这个布尔量随之消失，换成"我是不是唯一持有者"的判断（`sole_holder(slot, session)`）。

### 18.4 落地状态（2026-10-23）

四步全部落地，但**第 4 步多出一个必须先成立的不变量**，以及两个实测结论：

1. **写入侧**（`443ef091`）：`HostCheckpoint.history_hash` 由 `publish_lane_checkpoints(lane, history, tail)`
   盖章——放在"让槽变成可用"的那次扫描里，而不是 `snapshot_host_checkpoint`，因为只有扫描同时掌握"这一轮真正走过的
   token 流"和这些槽。一个槽一个位置，排序后走一遍流取快照（FNV 可增量累积）。
2. **扫描侧校验**（`7cf29038`）：两份扫描（`scan_lane_reuse` 与 `execute_walk` 自己的登记循环）都要求
   `history_hash == prefix_fingerprint(lineage, position)`；0（没盖过章）一律拒绝。反向验证的办法是把写入侧盖章
   故意改错，trace 里 `refused_hash=4`、复用掉到 0。
3. **持有者集合**（`5a823986`）：`holders_[slot]`、`hold_slot`/`drop_hold`/`free_slot`，`Anchor.borrowed` 删除。
4. **打开跨 lane**（`342192df`）：去掉 `writer == lane`，同时删掉按 lane 作废镜像的三处（退休、异常路径、召回）——
   镜像是对"它冻结的那段前缀"的事实，作废与否由指纹回答。

**新不变量：`RetentionState::device_kv_limit`。** 去掉了写者闸门之后，"镜像比它下面的 KV 还深"变成可能：
一次召回只把页恢复到它恢复的那个边界，更深的镜像即使指纹对得上，也没有 KV 托着它。所以召回把 `device_kv_limit`
设成恢复到的边界，自己 prefill 到底的路径设成无限；扫描侧只登记 `position <= device_kv_limit` 的宿主镜像。
没有这一条，去掉写者闸门就是**不健全**的。

**回归与收窄**（`cff2ddde`、本轮的会话域 + prompt-end 排除）：打开跨 lane 的那个提交之后，
`ninfer_qwen3_5_tp2_sessions_test` 的 `unaligned dialogue (plain)` 从通过变成失败。二分把责任落到池化
（`c6341b5b`）而不是跨 lane：同一份失败在 `5a823986`（跨 lane 之前）与 HEAD 上逐 token 相同，而在 `09116db5`
（池化之前）该场景 `reused 0` 且通过。机制是池化把 prompt-end 镜像从"钉住的固定槽"变成"池里的一张普通镜像"，
于是它会作为普通边界被扫描提供出去：`continued` 回退 8 个 token 到另一段对话写下的 64 位置镜像，其首采样与冷
oracle 不一致（`2752 …` vs `1605 …`）。根因不是复用了错的前缀——指纹、会话、KV 全都对——而是**边界下面的 KV 行
由另一次 prefill 写下**（88 token 一次写 vs 72 token 先写再续），归约形状不同，ulp 级差异在近似平局上翻了一个
token。两条收窄同时落地：

| 收窄 | 落点 | 理由 |
|---|---|---|
| prompt-end 镜像**不进 ring 扫描** | `HostCheckpoint.prompt_end` | prompt-end 镜像是 catalog 和活谱系的入口（重试被取消的那一轮），不是轮转镜像；把它当轮转边界用，就是上一条 ulp 类的来源 |

**试过但撤回的一条**：先做过"镜像只服务写它的那个会话"（`HostCheckpoint.conversation`，身份取请求自己的会话键的 FNV-1a，匿名请求退化为按 lane）。它对 `unaligned dialogue` 没有作用——该场景两段对话都是匿名的，按 lane 的身份分不开；真正解决问题的是 prompt-end 排除。而它**会挡住产品要的形态**：共享稳定前缀（系统提示词）本来就是跨会话复用的一等场景（§15.6 的 block 锚点、`SharedStablePrefix`）。所以该字段、身份函数和两处校验一并删掉了，镜像的身份回到"它冻结的那段前缀"（指纹）。

代价记在测试里：`dropped transport` 那条从"复用被取消那一轮的 prompt end（408）"变成"复用它下面最深的一张轮转镜像（256）"，仍然不是从零重来（事故的观测值是"复用为 0"），但少省 152 个 token。要拿回那 152 个 token，需要一个"边界 KV 与本次 prefill 同源（同 chunk 形状）"的判据——**注意它与"跨会话共享稳定前缀"是冲突的**，需要先解决下面那条。

### 18.5 实测结论（双卡）

- **走查路线（并发 1，默认）**：`0 → 8192` 不变，`refused_hash=0`。这条路线用的是本 lane 的活谱系 + ring 镜像，
  不依赖 catalog 的 slab。
- **多 lane 路线（`--max-concurrency 2`）**：两个前提都要满足才有跨请求复用，缺一个就是每轮从分歧点重来：
  1. **host KV 预算要够装至少一个整会话镜像**：`--host-kv-mib 512`（256 MiB/shard）面对 512 MiB 的单会话镜像，
     `session_capacity_` 虽然报 6，但 slab 一张都填不满，`host_kv_end=0`、`reach=0`，召回什么都不提供。
     `--host-kv-mib 2048` 之后 `frontier_end/kv_end/host_prompt_end` 都正常填上。
  2. **必须有一张 slab 镜像落在共同前缀之内**：`prompt_end=8452`、`shared=8451` 时（客户端重排答案后从 8451 分岔），
     `offered > shared` 被排除，`reach=0`——语义上正确，那条历史本来就不能再用，代价是这一轮从 8451 重 prefill。
     要让多 lane 路线真正吃到跨 lane 的收益，需要"共同前缀 +1"这类容差（prompt-end 落在分歧后一个 token），
     那是独立的设计问题，本轮没动。
- **未解决：共享系统提示词这一族在池化之后拿不到复用。** `test_tp2_sessions.cpp` 的 switch 场景（两段对话共享
  512 token 的系统提示词）在双卡上 `reused 0`，trace 里是 `[tp2-cp] skip lane=0 pos=256 pool exhausted`：
  池化让 prompt-end 镜像与轮转镜像抢同一个 11 槽池，多对话跑下来池被占满，`reserve_image_slot` 又只能淘汰
  单一持有者的锚点，于是一次镜像都冻不下来（`host=0/11`）。这条**不是**本轮引入的：在 `09116db5`（池化之前）
  同一场景复用 512 成功，但那时的失败在别处——输出与 oracle 在第 3 个 token 分岔（`16` vs `15`）。两半都要修：
  池化那一半是 eviction 策略（池满时应淘汰已完成会话的 prompt-end 锚点），再往下才是那个 token 分岔。
  **两半的进展见 §18.7**：池那半与设备深度那半都已修，`reused 512` 回来了；那个 token 分岔确认与边界选择无关。
- **镜像可以共享读**：`copy_lane_state` 的 H2D 分支只读宿主缓冲、只写本 lane 的 device 区；`load_dflash_image`
  同理；每个 shard 一条 CUDA stream，且执行被 `execution_mutex_` 串行化，所以两条 lane 读同一张镜像没有竞争。

### 18.6 不参与指纹的两件事

- **媒体身份**：`identity_ok` 已经单独检查 `media` 前缀（§15.3），token 哈希看不见图片，两者互补。
- **`prefill_id`**：它标记"这批镜像属于同一次 prefill"，是写入侧的一致性问题，不是身份的替代品；
  跨 lane 之后它仍然只在同一次 prefill 内部比较。

### 18.7 chunk 网格与设备前缀深度（2026-10-23）

§18.4 记的两半，前半已修，后半确认为独立问题并留下了判据。

**一、池那半：镜像和 prompt-end 抢槽 → 各自有槽。** `d3d37f74` 让 prompt-end 镜像占用
`capacity + lane` 这一组角色槽（`reserve_prompt_end_slot`），不再和轮转镜像抢 11 个位置的池。
`[tp2-cp] skip … pool exhausted` 消失，`switch` 场景的镜像重新冻得下来。

**二、`device_kv_limit` 必须在每条移动设备前缀深度的路径上维护。** 这个字段回答"设备 KV 里有多少
个前缀位置是本 prompt 的"，扫描用它拒绝"底下已经没有 KV"的镜像。但此前只有召回和批量 publish 写它，
两条路漏了：

| 路径 | 该写什么 | 之前 |
|---|---|---|
| walk 的 publish（写完 `[0, prompt)`） | `kNoDeviceKvLimit` | 从不写，带着上个请求的值（实测漏到 71 / 0） |
| session 切换（resident 已存盘、池子还没被覆盖） | `active_shared` | 从不写，同样带旧值 |

`60583617` 补齐之后：`dropped transport` 回到 256（它走 `continue` 分支，靠 walk 的复位），
`switch` 的**复用计数**从 0 回到 512。walk 的复用 trace 现在同时打印 `kv_limit=`，这类问题一眼可见。

**三、复用点必须落在 walk 自己的 chunk 网格上**（`83fcf8ae`）。一个 chunk 的行带着它自身宽度的归约
形状，而**最后一个 chunk 的宽度等于剩余 token 数**，随 prompt 总长变化。所以只有从 0 起切的完整 chunk
（`position % chunk_width == 0`）下面的行，才是任何一次 walk 都会算出同样结果的行；网格之外的边界是
另一次 prefill 算出来的。两份扫描都加了这道门，套件里的期望随之改成"各自边界的网格下取整"。

**精确性定律（本轮确立）**：`kPrefillChunkMaximum = 1024`，一次 prefill 的 chunk 是
`min(剩余, chunk)`。因此

- prompt ≤ 1024 时整条提示词是**单个 tile**，它的归约形状由总长度决定，**任何**跨长度的复用都不
  逐位一致；
- prompt > 1024 时只有最后一个（不完整）tile 受影响，落在完整 tile 端点上的边界是逐位一致的。

**这条定律的适用前提是两边的 chunk 切分一致**（从同一网格点、按同样的宽度往下切）。复用走路查天然做不到：
它从边界往下只切剩下的 token，冷走查要先切完整 chunk 再切尾巴，于是同样的行带着不同的归约与 all-reduce
量化历史（prefill wire 是 e4m3 + 128 元素块，见 `docs/tp2-decisions.md`），结果只能到 **logits 精度**
——这正是 §4.6 记下的"不能承诺缓存命中与否输出一致"，§18.7 下面那张深度表（256 错 / 512 对 / 1024 错）
也是它的表现，§19.8 在 TP-2 的环图起点上又复现了一次。要"任意边界逐位一致"只有两条路：fixed-shape tiles
（下面这段），或让同一对话总落在同一复用点（策略改动，不是本实现）。

要让**任意**边界都可复用，只能把最后一个 tile 补齐到固定宽度——即"fixed-shape tiles"。这条路要求
`TextContext::prefill_chunk` 接受 `nominal_length > 实际 token 数`，并在 `prefill_impl` 里同时做到：
attention 对 padding 行 mask、**GDN 递推不推进 padding 行**、**KV 不写 padding 位置**、logits 只取
最后一个真实行。后两条要穿过 `run_layers → single_layer` 进入 GDN/attention 的 op，属于模型层改动，
本轮没有动。

**四、镜像的精确性已能单点测量，结果指向镜像路径本身。** `test_tp2_sessions.cpp` 的 `check_image_equals_a_cold_prefix`
（`NINFER_IMAGE_EQUALITY=1` 打开，`NINFER_IMAGE_BLOCK=<chunks>` 选深度）把"写者"和"读者"的 chunk 序列对齐：
写者切完整块再越过去一点，读者只共享这块，所以网格能给的最深边界就是块尾，而块下面的行由**冷走查也会切出的同样
几个 chunk** 写下。两条走查若还不一致，差异只能来自镜像本身。双卡 plain 实测：

| 块深 | 边界 | 结果 |
|---|---|---|
| 1 chunk | 256 | 从第 2 个 token 起分岔 |
| 2 chunks | 512 | **与冷走查逐 token 一致** |
| 4 chunks | 1024 | 从第 1 个 token 起分岔 |

一个在某个深度精确、在相邻深度错掉的边界，**不是 tile 形状效应**（同一批行、同样的 chunk 都被两种走查切过）。

根因是**槽位索引**：一个逻辑边界是两张镜像（每 shard 一张），而 `restore_lane` 用**一个**槽号索引两个环；写入侧却是
成对调用 `snapshot_host_checkpoint`，各自从同一个 ledger 预留，两半于是落在**不同槽号**上——peer 读到的是它自己环里
那个槽位残留的镜像：位置对、前缀哈希也对（哈希只在写者那个 shard 上校验），内容是另一次 prefill 留下的。`158cfb43`
把预留提成 `snapshot_host_pair`：预留一次、两个 shard 写同一个号，配对从"每个调用点都要记得的约定"变成结构性的事
实（12 处成对调用全部改用）。修好之后三个深度都复现冷走查，`switch` 的计数与 oracle 比较也一起通过；`burst` 那条
长期"通过"的 512 复用其实踩在这个坏槽上，现在改为落回 prefill 真正写下的 256。

**五、池饥饿的两条独立原因（`ad59870e`）。** 一个 1089 token 的提示词实测复用 0，追下去是两件事叠在一起：

1. **锚点规划按位置而不是按身份跳过**：`block_anchor`/`anchor_divergence` 的守卫是
   `!has_valid_host_checkpoint_at(position)`——只要该位置上有**任何**有效镜像就不写了。而 store 与
   `capture_lane_anchor_frozen` 都是按"写它的那次 prefill"去找这个锚点，所以别的对话在同一位置留下的镜像
   并不是本次 prefill 的状态，跳过写入等于让 entry 拿不到块边界。现在规划出来的锚点照写。
2. **作废的镜像不还槽**：扫描把共享前缀之后的镜像标成 `valid = false`，但 ledger 仍占着那些槽；预留失败时也不
   向环核对到底哪些槽还有内容。于是 11 个槽全部 `used`、`valid` 为 0，池子报"满"却一张可用镜像都没有。现在
   `reserve_image_slot` 失败前会与环核对一次，把镜像已消失的槽收回来（`force_release`），与
   `retract_past` 一起保证"作废即归还"。

修完的两侧证据：家族的分歧锚点回到 512，named-block 的 1089 token 提示词复用 1024，新增的 shared block 检查
复用 1536 且与冷走查逐 token 一致；plain / mtp / dflash2 三条路线全绿。

**六、`switch` 的输出分岔不是边界选择问题。** 根因已由 §18.7 第四条的镜像槽位修复解决（`158cfb43`）——此处保留当时的判据。计数修好后该场景曾暴露

```
got      [1703 220 16 15 15 …]
expected [1703 220 15 15 15 15 …]
```

第 3 个采样翻了一个 token，前两个相同。用 `NINFER_TP2_REUSE_CAP` 把复用深度压到 256，**输出分岔逐字
相同**——所以它与"哪个边界"无关，**任何**对这段共享块的复用都会翻这个近 tie。同一个分岔在
`09116db5`（池化之前）就出现过，说明它不是池化、不是跨 lane、也不是网格规则引入的。

上面第四条把这个分岔变成了可单点判定的问题：既然 512 那块在"序列对齐"的实验里是精确的，而 256 / 1024 不是，
那么 §15 那条"缓存不改变答案"的契约是否成立就不是一句话能定的——它取决于镜像路径的这个缺陷。**先修缺陷，
再谈契约**：如果修好之后所有深度都精确，契约成立；如果只有部分深度精确，再按 fixed-shape tiles 或收窄承诺处理。

**结论**：缺陷是槽位索引（见本条前文），修好之后三个深度都精确，`switch`、`family`、named-block 全部与冷走查一致。

## 19. 剩余项（2026-10-24）

**已完成的两项清理（2026-10-24）**：

- **两份扫描合一**（`0ec844d5`）。批量路径的 `scan_lane_reuse` 与 walk 的内联取点是同一问题的两份实现，且已经分叉
  （指纹比的是不同的谱系，规则只改了一边）。walk 现在用自己 lane 的历史调用同一份扫描，注册可达锚点、ledger 决策、
  报告来源是同一段代码。合并暴露了两处隐性耦合：扫描曾顺带写 `reuse_boundary_target`（批量路径用它做一次额外的分块
  切割，walk 的分块循环也读它，于是共享扫描会改变 walk 的 chunk 形状，破坏与冷走查的逐位一致）；以及"清空锚点表"不能
  同时释放它命名的槽（环里镜像还在、紧接着的注册会重新持有，而一次失败走查留下的镜像正是下一轮要站立的地方——释放它
  会把 256 的复用变成 0）。前者归位到批量调用方，后者保持"只换表、不换所有权"。
- **固定 stride 与单一池**（`95587ab2`）。宿主环原本按旧子环的方式给自己配额：先给分歧锚点、块锚点和 tail 各留一份，
  再拿剩下的槽反推 grid 张数，最后由张数反推 stride。它喂的分配器是池化的，配额什么也没买到，而且依赖方向反了——
  stride 是复用层承诺的分辨率，应当固定，张数才该由服务上下文反推。现在池 = 配置张数 + 每个请求可能冻结的角色张数，
  stride 就是常量；tail 窗口仍有自己的计数，但它是关于边界网格的陈述（提示词末尾最后几个 chunk 端点，不超过配置窗口），
  不再是某个子环的份额。

### 19.1 为什么不做 fixed-shape tiles（2026-10-24 实测结论）

复用的边界必须落在 tile 网格上，这一点 padding 改变不了：冷走查 L=1089、W=256 的最后一块是 65 行，从 1024 召回时
的最后一块也是 65 行（宽度只取决于 L 和绝对网格起点）。所以"倍数于 W 的边界"今天已经逐位精确（§18.7 的深度表就是
实测证据），而 padding 并不会造出新的小粒度边界——粒度由 W 决定，不由 padding 决定。它唯一能买到的性质（tile 形状
与本次 run 的起点无关）现在已经成立。

真正能改变粒度的是把 W 调小，代价实测如下（双卡，24362 token 冷 prefill，每次 3 个样本）：

| W | 吞吐 | 相对 1024 |
|---|---|---|
| 1024 | 2270 tok/s | 基线 |
| 512 | 2061 tok/s | −9% |
| 256 | 1752 tok/s | −23% |

这 23% 曾被认为是主机发射开销，可以通过 prefill CUDA Graph 收回（图会同时让 fixed-shape tiles 变得有意义）。
两个测量否掉了这条链：

1. **不是调度串行。** `overlap_ok` 要求 `tokens % overlap == 0 && tokens / overlap >= 2`，默认 `--prefill-overlap 256`
   使 W=256 退化成锁步。把 overlap 降到 128 让它重新进入流水线，吞吐 1728 tok/s——与锁步的 1752 没有区别；
   overlap 64 更差（1168）。W=512 配 overlap 128 也比配 256 差（1892 vs 2061）。
2. **没有主机空档可回收。** 24k 冷 prefill 期间 5060 Ti 平均利用率 95%（中位 98–99%），W=1024 与 W=256 一样。
   也就是说主机最多占 5%，抓图省不出 23%。

结论：小 tile 的代价是设备侧 GEMM 效率与每 tile 的 all-reduce 次数，抓图无能为力；fixed-shape tiles 的唯一正当理由
（抓图前提）随之消失。保持 W=1024 默认；`--prefill-chunk` 是一个带实测价格的粒度旋钮，不是免费参数。

如果将来要做 prefill 图，正确的触发条件不是"小 tile 慢"，而是"设备利用率明显不满"——今天不满足。

### 19.2 已删除：lane 局部的边界图（`3f87700e`）

`fd9d6ded` 在每条 lane 上保存一张"提示词与历史开始分歧"处的状态图，并把它作为召回的第四个来源。问题在于那个位置是
精确分歧点、**不在 prefill 的 tile 网格上**：经由它的召回会切出与冷走查不同的 chunk，于是一次近平分叉就可能翻转；
引入它的那条提交自己也写明了——它换来的是 21K 提示词上 137 个 token 的复用，而"恢复在 logits 精度内等价，并非逐位
相同，贪心续写仍可能在候选分差小于该步时分歧"，并且为此保留了 `NINFER_TP2_REUSE_BOUNDARY=0` 这个"精确性检查需要的
A/B 开关"。

分歧锚点（`anchor_divergence`）覆盖了同一需求，而且是网格定位、由 ledger 决策的，所以这张图连同它的 147 MiB/card
惰性分配、那个环境开关、一次额外的 chunk 切割和召回路径上的第四个分支一起删除。三条路线上会话套件的数字一个都没变
（recall 71 / anchored at 512 / 1024 of 1089 / shared block 1536）——这正是"锚点早就在做这件事"的证据。

### 19.3 批内一致性：声称的是 op 级，不是整轮

`test_tp2_lanes.cpp` 曾断言"同一 prompt 四条 lane 必须逐 token 一致"，一直是红的。根因不是串扰：四条同时提交的请求
**从不共享批形状**——先到的那条在别人还在被接纳时就开始解码，轮次按 1 → 2 → 3 → 4 增长（`NINFER_TP2_LANE_TRACE=1`
会打印 `[tp2-lane] round columns=… slots=…`），而列数是解码数值路径的一部分。前四个 token 一致，之后一次近平分叉翻转。
`test_tp2_forward.cpp` 的 verify 窗口用例对同一件事写得很清楚：批窗口与单条窗口**不可能逐位相同**，能保证的是"同一次
批量调用内两条相同的 lane 逐位一致"。那条断言现在改为检查首 token（四条各自的 prefill 是同一形状，必须一致，串扰会
立刻失败），逐位一致的部分留在 logits 层可观测的 `test_tp2_forward.cpp` 里。`tp2-concurrency-headroom.md` §8.7 的
P2 行也据此收窄为 op 级措辞。

### 19.4 答案里的锚点：间距与落点（`f07facd2`）

一次真实 agent 会话（单会话 75 轮、4,673,476 个 prompt token）把复用做到了 97.1%，但剩下的 137,156 个
prefill token 里有 106,121 个（77%）是**重算已经算过的 token**。原因不在召回，而在写入：

- 生成阶段的锚点间隔是固定常量 `kReuseGeneratedTailStride = 8192`，而复用粒度是 chunk 宽度（默认
  1,024）。75 个回答里只有 1 个超过 8,192 token，其余 74 个**从没写过任何锚点**，于是下一轮只能从"上一轮
  prompt 的最后一个 chunk 末尾"起复用，整段回答被重算。
- 即便写下来也不一定能用：写入点是当轮的 frontier，而 `offer()` 只接受落在 chunk 网格上的位置，笔试
  不过一处就被拒；串行 walk 更彻底——它只在 prefill 结束时盖章（`publish_lane_checkpoints`），decode
  之后写下的镜像只被标成 `valid` 而没有指纹，扫描一律以"无从证明自己的前缀"拒绝。

现在的规则是两条，缺一不可：

1. **间距 = chunk 宽度**：每跨过一个网格倍数写一张，回答背后的回退上界从"整段回答"降到"一个 chunk"；
2. **只在正好落在倍数上的那一轮写**：状态只能在 frontier 真正站的地方冻结，所以三个执行路径都在提交前
   把窗口切到刚好落点——spec 批把接受窗口截到 `target - position`，walk 把窗口截到 `target - label`
   （walk 的 frontier 比生成计数小一，最后一个采样 token 未前向，所以落点要从那个名字算，而不是从计数
   算；这个 off-by-one 让 MTP 路线第一次没落上）。被截掉的 token 下一轮重新起草，**提交序列不变**——
   接受是前缀决策，不是调度。
3. walk 在 decode 结束后补一次盖章，回答里的锚点这才有了指纹。

代价：每 1,024 个生成 token 一张镜像（147 MiB 两份 host 拷贝）≈ 15 ms，按 ~100 tok/s 算约 0.15% 的
decode 墙钟；同时占共享池的一个槽（默认 36 槽，可按 §19.1 的公式调整）。

实测：同一会话的浪费从 106,121 降到理论下界 42,633（可回收 63,488，占全部 prefill 工作的 46%）；三个
5.6/5.8/4.3 秒的 TTFT 尖峰正是三个长回答，落点正确后降到亚秒级。证据：`sessions` 三条路线全绿（burst
场景的期望值从"prompt 末的网格点"改为"回答末的网格点"），以及一次 A/B——站在回答锚点上的那一轮与
`--no-prefix-reuse` 之外的全新 prefill 输出逐 token 一致。

### 19.5 压缩（fold）：让分歧图活过它被写下的那一轮（2026-10-24）

一次真实 agent 会话（两个具名会话、69 个请求、prompt 合计 6,884,259 token）把缓存做到了 91.4%，但剩下的
593,827 个 prefill token 里有 **177,152 个（29.8%）是重算已经算过的 token**，而且全部落在三次压缩上：

| 压缩 | prompt | 与本条目共享 | 实际复用 | 浪费 |
|---|---|---|---|---|
| 第一次 | 66,169 | 15,657 | 8,192 | 7,168 |
| 第二次 | 116,135 | 68,972 | 8,192 | 60,416 |
| 第三次 | 137,940 | 117,952 | 8,192 | 109,568 |

三次 TTFT 是 34.7 / 77.3 / 100.3 s，合计 212.3 s，占这次会话全部 TTFT（530.7 s）的 40%，其中约 132 s 是
上面的重算。其余 66 个请求的浪费是 0——追加型回合的复用是精确的，缺陷只在压缩。

**根因在发布策略，不在召回。** 压缩轮的分歧落在历史内部（`shared` 远小于条目 span），`parent` 判据
（`shared + kSessionRewrittenTailTokens >= span`）认不出来，于是 `publish ... bind=0` 另起了一条匿名条目；
同一回合末的第二次 publish 又让这条新条目认领了名字（`adopt=1`），`session_prune_keyed` 因此删掉原来的
条目——连同它刚刚在 `shared` 处冻结的分歧图（trace 的 `anchor entry=1 position=15360 / 68608 / 117760`
正是那三张）。下一轮压缩只看到自己条目上 `shared_end=0`，回落到别的会话更浅的共享块（`via=shared`，8,192）。

规则（`session_publish`）：**命名的请求，如果与自己那条条目共享一个真实前缀（≥
`session_retention_floor_tokens`，且不为 0），它就是那条会话的压缩，不是会话旁边的一次性调用**——认这条
条目为 `parent`，走已有的原地改写分支。那条分支的注释原本写的就是 "a conversation that **folded** or
re-rendered its tail"：它一直是这个用法的实现，缺的只是入口判据。判据里的"真实前缀"是它与标题/摘要调用
的分界线，服务端默认的 floor 就在那条线之上。同一判据还顺带纠正了两件事：名字在"它重现得更多"时压过内容
匹配（否则会话旁边一条短条目会凭 `bind_name` 把名字抢走），以及内容匹配本身要求"重现了大部分"（
`2 * shared >= span`，否则 `span < 256` 的条目对任何重叠都成立，见 §19.6）。

但只改发布策略不够：这条链上还有一个前置缺陷会让刚冻结的那张图在任何扫描里都读不到。实测（`sessions` 的
`check_folded_conversation`，`NINFER_TP2_CHECKPOINT_TRACE` 与 `NINFER_TP2_REUSE_TRACE`）：压缩轮的分歧图
写得最早、位置最浅，本轮后面的 tail 写入把镜像池填满后，回收通道把这张"valid=false 但被池持有"的图当成垃圾
释放了（`[tp2-cp] retire lane=0 slot=8 pos=512: the pool released it`）。发布之后它 `valid=true`，可池不再
认它，下一次扫描以同一条理由退役——见 §19.6 的第一条。

修好这一条之后，trace 才是这次要的形状：`anchor entry=0 position=512` → `rewrite ... shared_end=512` →
（一段无关流量）→ `recall frontier=512` → `-> reuse=512 slot=0 src=live`，两次压缩的输出都与冷走查逐 token
一致。

收益、代价与边界：

- **收益晚一轮**：触发重写的那次压缩仍付全价——复用边界由扫描在 prefill 之前定，发布发生在之后。
  只压缩一次的会话拿到 0，压缩两次时第二次才受益。
- **落点受网格限制**：能把召回推到 15,657，能站的复用点还是 15,360。
- **第一轮压缩救不回来**：`[0, shared)` 里从没有过镜像，那张图只能由更早的一次分歧写下来。
- **匿名客户端不受影响**：判据要求名字。没有名字时条目不会被 prune，旧条目带着图活着、下一次压缩按内容
  匹配到它——那条路本来就没有这个缺陷，代价是会攒变体（§15.7 结论三为什么要求一个名字一个条目）。
- **不让名字决定复用边界**：边界仍由 `shared_history_prefix` 决定，§5 的禁令不受影响；名字只决定
  "哪个条目活下来"。
- 一次重写只救到上一张分歧图，救不到"摘要每次前移"本身；要一路省下去，每一轮的分歧点都得被冻结并活下来
  ——这条规则的全部内容就是让冻结生效。

证据：`tests/models/qwen3_5/test_tp2_sessions.cpp` 的 `check_folded_conversation`（三条路线）。同一个具名
会话先长后短地重写两次，中间夹一段与它无关的流量——那段流量的扫描会退役本 lane 在环里的全部镜像，于是第二次
压缩只能靠会话条目自己的分歧图。断言：第一次压缩复用 0，第二次落到第一次冻结的边界（一个 chunk 的容差），
两次输出都与冷走查逐 token 一致。最后一条是这次改动的关键：改的是 catalog 里谁活下来，复用边界没动，所以
"与冷走查逐位相同"的要求不变。A/B 已做过：把发布判据关掉，第二次压缩复用 0、场景红；打开，复用 512、逐
token 一致。

### 19.6 同批修掉的五个前置/独立缺陷（2026-10-24）

第一条是 §19.5 的链路自己暴露出来的：只改发布策略，压缩轮的分歧图仍然读不到。第二条是同一次真实会话测试
暴露出来的，而且我第一版修错了方向，记在条目里。

- **回收通道把本轮刚写的槽当成垃圾**（`snapshot_host_pair` 的 reconcile）：池满时它遍历"`valid=false` 但被
  池持有"的槽并 `force_release`，而本轮预填在 publish 之前写下的镜像**正是 `valid=false`**。压缩轮的分歧图
  写得最早、位置最浅，于是被自己后面的 tail 写入挤掉：publish 之后它 `valid=true`、池已不认它，下一次扫描以
  "池已释放"退役（新增的 `[tp2-cp] retire` 就是这条）。修复：只回收**不属于任何 lane 正在进行的预填**的槽
  （`prefill_id` 不等于任何一个 lane 的 `host_checkpoint_live_id`）。这条与压缩无关，长预填填满池时同样会
  丢弃它自己前面写下的 stride/tail 图。
- **驻留谱系与召回的比较只看镜像**（进程内路线）：`session_recall` 在"要不要为了宿主 slab 放弃驻留谱系"时只
  比一个 `resident_depth`，而它只来自三处**镜像**（live 状态的 frontier、`cached_boundaries` 的快照、环里的
  checkpoint）。驻留谱系真正能站多深不需要任何镜像——设备池就是上一次预填的 KV，只要这个 prompt 完整包含它就
  能接着走。于是一条更浅的宿主 slab 会赢过更深的驻留谱系，而召回分支是**破坏性**的（`cached_boundaries.fill(0)`），
  被顶掉的深度再也回不来。修复：prompt 完整包含驻留条目的 `prompt_end` 时，把
  `min(active_shared, device_kv_limit)` 也算进 `resident_depth`。高报是安全的——扫描仍会自己挑一个能证明的
  起点；低报不安全，因为它顶掉的召回无法恢复。服务端路线每轮请求开头 `cached_state_valid` 已是 false（
  `release_lane_kv`），这一条在那里不参与判断，真实的塌陷原因见 §19.7。
  - **一次修错的方向**（值得记）：我先给 ring 那一项加了"池仍持有该槽"的判断（理由是"与扫描对齐"），恰好把它
    变成了低报，于是"浅召回顶掉深驻留"从偶发变成常态——环里的镜像被池回收，并不说明设备上的那截 KV 没了。
    这个判断已撤回。
- **短条目的内容匹配退化**：`shared + kSessionRewrittenTailTokens >= span` 在 `span < 256` 时恒真，等于"任何
  重叠都算延续"。测试里一条 114 token 的一次性调用条目就被下一条只共享 8 token 的请求认作 `parent`，并借
  `bind_name` 把名字抢过去、把真正带图的条目 prune 掉。修复：加 `2 * shared >= span`（"重现了大部分"）；对
  `span >= 512` 它已被原式蕴含，只收窄短条目这一段。

另外两条与压缩无关：

- **ledger 的角色槽越界**（潜在，此前不可达）：`used_slots_` 按 `capacity + prompt_end_capacity` 分配，
  `owned_slots_`/`holders_` 只按 `capacity`，而 `allocate_slot`、`free_slot`、`release_slot`、`record` 的
  认领分支都按"整个环"的下标寻址。池恰好满、且同一次预填认领了 prompt-end 角色槽时才会踩到。现在三个数组
  统一按整个环定容，`allocate_slot` 只遍历 rotation 段，"role 槽被占用但不被拥有"的语义写进了代码。
- **`publish_lane_checkpoints` 的 64 项截断**（可达）：它用 `std::array<PendingStamp, 64>` 收集本轮写下的
  镜像，数组满了就 `continue`，超出的镜像**永远没有指纹**（扫描一律以"无从证明自己的前缀"拒绝），却仍然占着
  槽。一次 262K 冷预填要写 32 张 stride 图 × 两个分片，再加 tail 窗口与分歧/块图，必然越过 64。现在按实际
  写入量收集。

### 19.7 客户端在回答末尾加换行：召回只填到边界，把环里更深的图挡住了（2026-10-24）

第二次真实会话测试（同一个 agent，68 条消息，**没有触发压缩**）暴露出另一类塌陷。这次会话共 23 个请求、
prompt 合计 1,848,284 token、命中 1,425,408（77.1%）、prefill 422,876、TTFT 合计 313.8 s。其中三个请求
（req#14/#16/#20）合计重算 294,529 个 token、TTFT 203.8 s，占全会话 TTFT 的 65%。

**触发条件在客户端侧。** `NINFER_TP2_REUSE_TRACE` 的 `[tp2-diverge]` 行四次记录了同一件事：客户端回放的上一
条回答与模型生成的内容**只差一个结尾换行**（`generated[1550]="…ledger 没有的。"` vs
`replayed[1550]="…ledger 没有的。\n"`）。
回放是 prompt 的一部分，所以每一轮"把上一轮回答带回来"的请求都会在上一轮 prompt 的末尾附近与自己的历史
分歧——833、94、1,632 个 token，四次分别是 83,484 / 92,066 / 110,355 / 128,384。这个位置落在 tail 窗口
（2,048）之内，`host_frontier_end` 和 `host_prompt_end` 都在它之上，条目上唯一更浅的图是 8,192 的稳定块：

| 请求 | prompt | 与本条目共享 | 可用边界 | 实际复用 | TTFT |
|---|---|---|---|---|---|
| #14 | 88,091 | 83,484 | `shared_end` 8,192 | 8,192 | 52.8 s |
| #16 | 97,096 | 92,066 | 无（`reach=0`） | 0 | 63.1 s |
| #20 | 125,726 | 110,355 | `shared_end` 8,192 | 8,192 | 87.9 s |
| #23 | 135,314 | 128,384 | `shared_end` 125,952 | 125,952 | 11.4 s |

**根因（已修）**：召回把 KV 只恢复到**状态边界**（`pages_for_tokens(boundary)`），并把 `device_kv_limit` 收窄
到同一个值；于是**写入这块 slab 的那次预填留在环里的 stride/tail 图**——它们的位置都在这条 prompt 自己的
前缀里——在注册时被 `position > device_kv_limit` 一律拒绝。trace 里 `host=32/36 lane_slots=31 refused_hash=0`
就是这个形状：32 张有效镜像、指纹没拒绝任何一张，扫描却只认了 8,192。修复：召回按
`min(entry.host_kv_end, shared)` 恢复 KV，`device_kv_limit` 也用这个值。安全性由构造保证——KV 只由 token 与
位置决定，`shared` 是这条 prompt 与条目逐 token 比出来的公共前缀，而落点仍由扫描在 chunk 网格上挑，§7 的
精确性要求不受影响。

**根因二（已修，与上面同一批）**：req#16 那一轮 `reach=0`——条目的三张图（`frontier_end` 93,379、
`prompt_end` 92,694、`shared_end` 92,160）全在分歧点 92,066 之上，最深的那张只差 94 个 token。表面看是
`host_shared_end` 只有一个槽、被 req#15 的 tail 锚点顶掉了 8,192 的块图；**实际是"状态站在哪"和"KV 覆盖到哪"
被压成了一个数**：条目自己的 slab 明明覆盖到 93,379，环里还有 36 张有效网格图，`reach` 却只从三张有名图里取，
于是 `reach=0` 连召回都不进，`active_shared=0` 又让扫描的 `shared_prefix` 归零，整段重算。

修法就是 §8.1 原本写下的形状：召回判据改成"这份 slab 能给这条 prompt 多长的 KV"（`min(host_kv_end, shared)`），
状态则取"该 extent 之内最深的网格图"——**环是第四个来源**（§9.1 把它与 slab 并列），不是重新引入 §19.2 那张
非网格的 lane 局部图。落地细节：

- 环候选的判据：`valid` + 池仍持有该槽（`holds_slot`）+ `position % prefill_chunk == 0` + 指纹对这条 prompt
  的前缀匹配（§18 的 FNV，一次遍历算完所有候选）；DFlash2 还要 draft 半边在同一 frontier。
- 状态在**召回内**就 H2D 拷进设备（和 slab 图一样），不在扫描里补：条目一旦被标成驻留，它声称的状态必须
  真的在设备上——这是 `session_store_active` 的前提，也是 §8.1 第 3 步的位置。
- `session_restore` 拆成 `session_restore_kv`（按 extent 搬 KV）与状态部分；`RecallState` 里那个从 §19.2 起
  就没用过的 `Boundary` 换成 `Ring`。

判据：`test_tp2_sessions` 新增 `tail-divergence`——两条 lane，第二份请求只在最后一条 message 末尾多一个换行
（分歧落在上一次 prompt 的尾部之内，三张图都在它之上），断言复用 ≥ `grid_floor(上一次 prompt)` 且 ≤ 同一个
值。A/B：把环这个来源关掉，复用 0（就是 req#16 的形状）；打开，复用 768（= 网格下取整）。**只锁深度、不锁
答案**：这条路第一次真的被走通时，续写与冷走查在 ~5% 的运行里首 token 就不同——这不是新缺陷，是 §4.6 早就
记录并接受的已知行为（复现证据见 §19.8）。

**与 §19.5 的关系**：压缩轮的发布判据在这里同样起作用——它让这条会话的条目在尾部被重写时活下来（带着
`host_kv_end` 和 slab），否则条目被 prune，`host_kv_end` 从 0 重来，`min(host_kv_end, shared)` 也就没有东西
可恢复。

### 19.8 环图起点的续写差异：§4.6 的已知行为复现（2026-10-25，非缺陷）

§19.7 的修法让召回第一次可以拿**环里的网格图**当起点状态（`RecallState::Ring`）。深度与判据都对，但续写与
冷走查不是每次逐 token 一致：plain 上 20 次错 3 次、dflash2 上 12 次错 2 次，每次错的还是同一串
`[760 42787 3425 5577 220 15 11 5577]`（冷走查是 `[2578 220 15 383 279 33619 3626 25]`）。

**这一差异落在 §4.6 已经记录并接受的范围内**，不是这次改动新引入的承诺缺口。那一节的结论原话是：状态恢复在
**logits 精度内等价，不是逐位相同**；不同复用点会让浮点结果相差一个量化步，贪心解码在 top-2 相差小于该步时
可能翻转（当时实测就是"a2 首 token 不同"）；"这是同类前缀复用方案的固有性质，**不能承诺'缓存命中与否输出
一致'**；若产品需要该保证，要改的是复用策略（同一对话总落在同一复用点），不是本实现"。`docs/tp2-decisions.md`
记的 prefill all-reduce wire codec 是同一个取舍的另一半：per-layer 的 prefill delta 以 e4m3 + 每 128 元素一个
f32 scale 上线（+19~20% prefill tok/s），代价是 PPL +0.025%，并明确写下 "token-level flips persist at every
length"。

本轮把这件事测得更硬，排除掉三个"可能只是我们的 bug"的假设：

| 假设 | 证据 |
|---|---|
| 判据挑错了位置或来源 | `reach=768 via=ring kv=925 ring=2` 与 `reuse=768 src=LiveState` 在成功与失败运行里逐字段相同 |
| 状态图被"还在走"的边界污染 | 失败运行的 s0/s1 两张图 FNV 哈希与成功运行**完全相同**；再把两台设备在采图前全量 quiesce，20 次里仍错 1 次 |
| 只影响某条 route | plain 与 dflash2 都错，错误串的**前三个 token 相同**，之后因 fp8/bf16 KV 不同而分叉 |

**与 §18.7 的关系（小心，别把"没承诺"当成"已解释"）**：§18.7 的定律以"两边 chunk 切分一致"为前提。本场景
里**最后一个 chunk 的宽度两边都是 157**（冷走查 256/256/256/157，复用走查 157），所以按定律这一格是
*predicted* 逐位一致的——测出来却不是。因此定律只覆盖了 KV 行这一侧，没有覆盖"从环图恢复状态再往下走"这一
侧；两侧之间的差异还没有定位（候选：恢复路径本身，或边界以上那三块 KV 的归约历史）。定位之前，本条按 §4.6
的既定契约处理：**不承诺命中/不命中输出一致**，测试只锁深度。

因此：`tail-divergence` 只锁深度；逐 token 一致性不是这个功能的承诺，要它只能改复用策略（同一对话总落在同一
复用点），与 §4.6 的原话一致。

### 19.9 池子的淘汰规则让"stride 覆盖"名不副实（2026-10-25，已修）

**症状**：一次 66 请求、753 万 prompt token 的真实 Agent 会话里，4 个"变短"的请求吃掉了 271.1 s，占
542.2 s TTFT 的 50%；其中两个是纯亏损——prompt 只比历史短一点，引擎也**有**对应 KV，却全量重算：

| req | prompt | 上一轮 prompt | `shared` | `kv` | `reach` | 实际重算 | TTFT |
|---|---|---|---|---|---|---|---|
| #57 | 138,217 | 162,386 | 92,852 | 92,852 | **0** | 138,217 | 103.3 s |
| #65 | 120,155 | 160,519 | 119,761 | 119,761 | **0** | 120,155 | 82.4 s |
| #35 | 91,529 | 162,903 | 15,410 | 15,410 | 8,192 | 83,337 | 55.1 s |
| #66 | 61,335 | 120,155 | 9,121 | 9,121 | 8,192 | 53,143 | 30.3 s |

**触发方是客户端**：这两个"变短"来自 DSH 的上下文管理，不是引擎。解开该会话的
`session.v4.jsonl.zstd`（每条记录一个 zstd 帧，共 229 帧）后可以对上秒级：

- **27 次 `compaction/prune`**，共隐藏 101,634 个 token 的整条消息（单条最大 8,141）；
- 10:16:34 一次 prune 22 条 → req#35；10:22:08 prune 4 条 → req#57；
- 10:29:08 `compaction/start` → req#65 是**摘要请求**本身；10:31:35 `summary` 产出 → req#66 是摘要后的新历史。

也就是说：客户端达阈值就把历史中段的工具结果整条摘掉再发，prompt 在一个**任意深度**与前一次分叉。
分叉点落在环里哪张图之下，决定了这一次是"省下 92,852 token"还是"全量重算"。

**根因**：`evict_one()` 按"离前沿最远"淘汰，而 prefill 会在末尾一个 stride 内按 chunk 宽度写一小段稠密图
（`kReuseTailWindow=8192`、`kReuseTailCheckpointCount=8`）。两者相遇的结果是：**稠密尾部每写一张就挤掉一张更浅的
网格图**，存活集合虽然"连续"，覆盖却退化成一个很窄的窗口。实测反推：req#65 的分叉点 119,761 落在文档承诺的
`[165,004-139,264, 165,004)` 窗口**里面**，可 27 张有效图一张都不 ≤ 119,761，实际覆盖上界 < 45,720，账面 139,264。
`reach == 0` 又让 `active_shared = 0`，连 slab 已覆盖的那 92,852 / 119,761 个 token 的 KV 也一起丢掉——所以代价是
全量，不是"退到最近镜像重算 ≤ 8192"。

**修法（§7.3 + 一处泄漏）**：

1. `ReuseLedger::evict_one()` 改为**覆盖优先**：同一会话在它下面不足一个 `stride` 处还有图时，这张是冗余的，
   先淘汰；同属冗余取 position 最小者（稠密段从底部变薄），没有冗余时才退回"位置最远先淘汰"。
   单测 `test_prefix_reuse.cpp` 直接锁这条：容量 4、图在 8,192 / 16,384 / 25,600 / 26,624，池满后必须
   **先淘汰 26,624**（冗余且最深），8,192 必须活下来；把规则关掉，这条单测立刻红。
2. `ReuseLedger::abandon(session, slot)`：扫描重建表时不再只"清空表"，而是把没有重新登记的旋转槽归还。
   一张图被裁剪丢了锚点却留着 hold，就是 §19.6 记的那类"pool 报满、镜像全在"的隐形泄漏，而裁剪正是每条 hold
   的制造者。角色槽（prompt-end）不受影响：它属于角色，不属于最后命名它的会话。
3. 集成场景 `trimmed-history`（三条 route）：16,640 token 的历史 → 成长 64 token → **把 [12,288, 12,800) 整段摘掉**。
   断言分叉点 12,288 之下最近的 stride 图 8,192 必须被复用。A/B：关掉覆盖规则，这一次复用 **0**（`host=0/7`，
   整段重算，就是 req#65 的形状）；打开，复用 **8,192**。

**容量账（§7.1 更正）**：`--host-state-images 32`、并发 2 → 轮换池 34 槽。单会话可独占全部 34 槽 →
`34 × 8192 = 278,528` token 覆盖，大于 262,144 的 `--max-context`，**这一档已经够用**（本次会话 165k 历史全额覆盖）。
两个会话同时各跑满 262144 时才平分（各 ~17 槽 = 139k），那时把 N 提到 64（+5 GiB pageable）。

**实测（2026-10-25 修复后同型会话，71 请求）**：

| | 修复前 | 修复后 |
|---|---|---|
| prompt token 合计 | 7,539,100 | 8,242,422 |
| cached | 6,892,544（91.4%） | 7,942,144（**96.4%**） |
| 实际 prefill | 646,556 | 300,278 |
| TTFT 合计 | 542.2 s | **292.3 s**（均 4.1 s） |
| 召回深度落后 `shared` 超过一个 chunk 的请求 | 4 个（271.1 s） | **1 个**（47.4 s） |

唯一那一个（req#34）是固有代价，不是缺陷：客户端把历史从 159,529 裁到 82,284，分叉点 15,504，而 stride 是 8,192，
`floor(15,504)` 之下最近的镜像就是 8,192——§7.1 承诺的"最多重算一个 stride"，它落在承诺之内（重算 74,092 token）。

同一次运行里还能看到两件事：

- `[tp2-session] ring states=` 印证了池子的形状：裁剪前是"stride 阶梯 16,384 → 139,264（16 张）+ 尾部稠密段
  143,360 → 156,672（13 张）"，即 §7.3 期望的"每个 stride 一张 + 尾部一小段稠密"，而不是旧的"只有最近几十个 chunk"。
- 裁剪发生的那一轮（req#34），环的有效镜像从 36 张掉到 5 张、`free` 从 0 跳到 27，随后同一轮的全量 prefill 写了
  18 张网格图**而没有淘汰任何活镜像**。这不是泄漏：被释放的是分叉点以上那批图（16,384 → 156,672），它们盖在被裁剪掉的
  文本上，而扫描的比较基准永远是**上一次 prompt**，所以它们此后再也过不了指纹校验——留着只是把槽占住。这一段
  正是 `abandon()` 存在的理由，也是它唯一的触发条件（重建没有收回去的锚点）。

**留下的**：一次**冷** prefill 若写下的图多于容量，末尾最深的几张会因 `pool exhausted` 被跳过（`trimmed-history`
场景故意把容量压到 6：16,640 token 的一次 prefill 要写 9 张，最后两张 16,128 / 16,384 被跳过，所以"只增长"那一步
的复用落在 15,872）。生产档 34 槽 ≥ 一次 165k 冷 prefill 的 ~28 张（20 网格 + 8 尾部），不会触发；要彻底消除得让
写入路径在"自己这次 prefill 已发布"的图上也能淘汰，属于后续优化，不影响本条的结论。
