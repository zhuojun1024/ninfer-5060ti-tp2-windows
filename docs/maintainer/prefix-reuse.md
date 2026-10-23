# TP-2 前缀复用层重设计

状态：**记账与决策层已实现**（阶段 0–2），执行层改造与删除清单未开工。已落地部分：`runtime/engine/prefix_reuse.h`
的 `ConversationLog`/`Anchor`/`ReuseLedger`（阶段 1，提交 `123a93b6`）、`decide()` 接管 core 的复用决策
（提交 `96f017ee`，行为中性，与出厂 v0.1.2 在同一捕获上逐位一致）、lane 局部边界图与 `RecallState::Boundary`
（提交 `fd9d6ded`）。§14.1 的六项决策保持不变；§15、§16 给出文件划分、纯函数边界与测试处置。本文只覆盖 prefix reuse 的
**记账与锚点层**；prefill 执行、Program、图捕获、KV page 物理合同和协议层不在范围内。

实测证据与夹具脚本放在工作区 `_temp/`（`20261022-1515_phase0.ps1` 及捕获目录），属于本地复现材料、不入库；
文中引用的行号以提交 `fd9d6ded` 为准。最终的归属仍是
[资源调度与上下文缓存](resource-scheduling-and-context-cache.md) 的 prefix-reuse 章节——本文暂作为该章节的完整草稿。

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
- 淘汰顺序：`Lru` 中 position 最远者优先；`Pinned` 只有显式解除才淘汰。

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

预算不足时的取舍顺序：`Pinned` 不动 → 其余按"离前沿最远"逐个淘汰。因为淘汰顺序和采样顺序同向，任何一类锚点存活
下来都自然是一个**连续窗口**，不存在"中间有空洞"的状态。任何情况下不允许"某会话把表填满导致别的会话无锚点"
（每会话上限，超限按同一顺序淘汰）。

### 7.1 预算反推与降级

覆盖粒度是契约，张数是结果。

- **`stride = 8192` 固定。** 它直接等于用户能感知的代价：prefill 实测 1,687 tok/s，`8192 / 1,687 ≈ 4.9 s`
  是最坏重复计算时间，与内存参数无关。因此同一请求在不同 `--host-state-images` 下 TTFT 保持可比，bench 数字有解释力。
- **需要多少张由需求反推**：`images = ceil(prompt_len / 8192) + ceil(gen_len / 8192) + 3`（+3 = `PromptEnd`、
  `Divergence`、`Block`）。一张 = 两条 shard 各一份镜像 = 146.8 MiB（73.4 MiB/shard）。于是：
  - 64K 提示 + 64K 生成 ≈ 19 张 ≈ 2.7 GiB —— **默认值 20 就落在这里**；
  - 128K + 128K = 35 张 ≈ 5.0 GiB；
  - 256K + 256K = 67 张 ≈ 9.6 GiB（也回答了"契约化网格"的价格）。
  对照现状：`--host-state-slots 32` 实际分配 36 张（5.2 GiB），而单个会话**够得着的只有 7 张**（57K 覆盖）——
  另外 7 张绑在另一条 lane 的切片上，网格 16 张、divergence/block/prompt-end 各若干。同样的内存已经花掉，
  却只有约五分之一能用于"重渲染答案"这个实际场景。
- **DFlash2 路线每张还要多存一份 draft ring**：打包日志实测 36 张时 shard 0 占 4082.6 MiB、shard 1 占
  2642.6 MiB，差额 1440 MiB / 36 ≈ 40 MiB per 张。所以 DFlash2 下 20 张 ≈ 3.7 GiB，不是 2.9 GiB；
  上表数字是"状态镜像"的价格，draft 按路线另计。
- **余额不足时降级，不缩粒度**：保留最近的 `n` 张形成连续窗口 `[end - n·8192, end)`，窗口之外的分歧退回
  `PromptEnd`（本次付全价）。日志报 `cover=<tokens>` 与 `degraded=window_truncated`。契约永远成立，只是适用范围
  收缩，而且收缩是一个可直接读的数。
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
  2. KV    host slab → 拷进设备分页池对应页；kv_source=Device 时无需搬运
  3. 状态  镜像按目标 lane 的 lane_state_geometry 拷进其槽（跨 lane 就是这一步）
  4. draft draft ring 置到 anchor.draft_extent，且必须 == anchor.position，否则拒绝
  5. 记账  token 历史 = log.tokens[0, P)，position = P
  6. 返回  P；后续 prefill 从 P 开始
```

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
| host 检查点环（ledger 池） | 环里当前有效、且属于本 lane 的那些镜像位置 | 池淘汰、被后写覆盖 |

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

**阶段 3：删除旧机制**：五个子环、`valid[lane]`、`reuse_path` 枚举（换成 §16.1 的 `reuse_source`）、重复 trace；
catalog 的注册职责交回 `ConversationLog` 的所有者，同时删掉 `SessionEntry`。

**阶段 4：标定张数与上限**：`stride=8192` 已经是契约，不再重标；标定每会话上限、`--host-state-images`
的默认值（按 7.1 反推，现按 20 写），并重新评估 4.4 的跨会话共享是否升级为 `Pinned`、以及 `Device` 锚点的转存
代价（14.2-1）。

**parity 矩阵**（每个阶段结束都要过）：零拷贝 live 复用、跨 lane 恢复、长上下文覆盖、
Draft 配对、媒体身份、块/分歧钉住、offline CausalScoring、取消/回滚、并发 2 lane。

---

## 13. 删除清单

- ring 的 grid / tail / divergence / block / prompt-end 五套子环与其索引逻辑；
- tail 的 `min(kReuseTailCheckpointCount, flexible / 2)` 硬上限（8 张），以及"先把槽分给各子环、再拿剩余槽位反推
  stride"的分配方式；张数改为由 §7.1 的需求反推，`stride` 固定 8192（§7 第 4 条），`host_checkpoint_stride_`
  那条预算反推路径整体删除；
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
  属于会话注册职责，不是锚点记账；删掉它们等于删掉 catalog 的发布语义；
- `reuse_path` 在**指标**上的用途，按 §17 的 `reuse_source`/`reuse_reason` 取代。原计划是连枚举一起删，**2026-10-23 改为
  收窄作用域**：该枚举在单卡路线里是程序内部的**恢复路径**（`is_rewrite_checkpoint_restore()` 与
  `PrivateEndpoint`/`SharedStablePrefix`/`PrivateLongAnchor` 在 `program/prefill.cpp`、`context.cpp`、`pressure.cpp` 里做路由判定），
  不是存储标签——§17.6 已经写明"旧枚举是角色，不是存储"。删它就得重新编码那套路由，是净增复杂度。
  已完成的删除：`record_prefix_selection()`、六个 `*_selections`、`last_selected_frontier_tokens`、`context_cache.selections`
  与 `last_selection`。**保留**：枚举本身，作为单卡程序的路由词汇（要更诚实可改名 `RestorePath`，与 §17.6 一致，属可选整理）；
- `cached_prompt_tokens` 与 `entry.tokens` 的双份维护（合并为单一 `log.tokens`）——不需要 ledger，可独立做；
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
| 单卡路线是否也迁到 ledger | **不迁**。单卡保留自己的 context cache / `StateImageStore`；`PrefixReusePath` 收窄为程序内部的路由词汇（§13 第 5 条） | 单卡那套是参考路线的在线特性（device/host 双副本、跨会话 shared prefix、long anchor、rewrite checkpoint、压力驱逐、MTP 桥接），是 ledger 今天能力的**真超集**；且它与准入控制（`ResourceManager`/`PressurePlanner`/`MaterializationPlanner`/`AdmissionCandidate`）耦合，迁移必须连 admission 一起重写。收益只是"少一套代码"，代价是数值参考路线与一到两周工期 |

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
淘汰规则也只有一个（按位置最远的 Lru 先走）。

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

### 15.6 borrower：一张镜像可以被几个会话同时读到，但只有一个会话让它活着

同一段 token 历史的两个会话（例如同一会话在两条 lane 上被调度）可以共享一张镜像。规则：

| 情况 | 结果 |
|---|---|
| `record()` 的槽当前没人持有 | 认领：`used` 且 `owned`，记下 owner 会话 |
| 槽由**本会话**持有 | 正常记账（重复登记自己的镜像） |
| 槽由**别的会话**持有，或是一个固定角色的槽 | 这个锚点标 `borrowed`：它读这张镜像，但不拥有它 |
| 槽被释放（拥有者被淘汰） | `release_borrowers()` 把借用的锚点一起删掉——镜像即将被覆盖 |
| 淘汰 | 只考虑非 `borrowed` 的锚点：删一个借用者腾不出任何空间 |

`evict_one()`、`forget()`、以及替换同位置的锚点，都只释放**非 `borrowed`** 的槽——否则一个借用者会把别人还在用的
缓冲还给池子。

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

结论三：`session` 那个 `optional` 是注册语义的支点：一个具名会话在 catalog 里占且只占一个条目，新的一轮把名字从旧
条目移到新条目（`bind_entry_session`），这也是"一个会话不会用变体填满所有槽位"的实现。删任何与它配套的分支都会
先破坏这条。

---

## 16. 测试处置与判据

### 16.1 逐条分类

先给数字：`tests/models/qwen3_5/test_engine_prefix_real.cpp` 里 `reused_prompt_tokens` 出现 **78** 次、
`prefix_reuse_path` 出现 **54** 次。后者是**公开 API**（`include/ninfer/types.h:763` 的枚举、`:872` 的字段），
而 §13 要删它——所以阶段 1 做完，这套测试必然红一片。必须先分清"红得对"和"红得不对"：

| 测试 | 现在断言什么 | 处置 |
|---|---|---|
| `test_engine_prefix_real.cpp` 的 `reused_prompt_tokens`（78 处） | 追加式复用、零后缀复用、跨会话恢复、媒体前缀各自的命中 token 数 | **保留结构，逐个更新期望值**。数值会变（栅格从"按预算反推的 32,768"变成 8192），但"命中多少"仍是行为契约；每条期望值旁注明它落在哪个锚点上 |
| 同文件的 `prefix_reuse_path`（54 处） | 复用来源枚举（`SharedStablePrefix`、`Root` 等） | **改写**。`GenerationResult` 保留 `reuse_source`（存储轴 `None/Device/HostSlab/LiveState`）与 `reuse_reason`，定义见 §17；旧枚举的五个非 `Root` 值描述的是**锚点角色**，不提升为公开字段（§17.6），替代判据是"命中多少 token、命中哪种存储" |
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

`PrefixReusePath` 的五个非 `Root` 值区分的是**锚点的角色**（端点、轮次收尾、响应重放、长锚、共享稳定前缀），
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
