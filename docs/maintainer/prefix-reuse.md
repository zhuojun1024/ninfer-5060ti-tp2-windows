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
commit(log, history, device_extent, host_extent, new_anchors):
  log.tokens        = history          # 规范流
  log.device_extent = device_extent
  log.host_extent   = host_extent
  log.anchors      |= new_anchors
  # 唯一的一条裁剪规则，按锚点依赖的那份存储分别判定（见 9.1）
  log.anchors.drop(a.position > kv_extent(log, a.kv_source))
  log.anchors.drop_over_budget()       # 见 7 的取舍顺序
```

删除现行 `session_publish` 的三条分支与其中"把 `prompt_end`/`shared_end` 清零"的逻辑：那些字段不再存在，
边界由锚点表本身表达。

### 9.1 两份 KV，两个 extent

`[0, P)` 的 KV 可以落在两处，两者的覆盖范围不同步，所以不能共用一个数字：

| 存储 | extent | 什么时候变短 |
|---|---|---|
| 设备分页池 | `device_extent` | 分歧裁剪（`drop(position > shared)`）、会话退休 |
| host KV 板（slab） | `host_extent` | slab 被 catalog 淘汰或容量回收 |

`kv_extent(log, source)` 按 `Anchor.kv_source` 取对应的那一个。裁剪必须分开做，理由是一个具体的事故：
分歧裁剪只缩短**设备侧**，host slab 仍然覆盖着旧位置。若用单一 extent 裁剪，一次分歧就会把 host 侧仍然完全有效、
而且是深历史复用唯一救兵的锚点全部删掉——这正是现在 `session_publish` 里 `prompt_end = 0` 清零的同构错误，
换了个位置重现。

另外两条：

- host slab 落后于设备时（还没转存），新拍的锚点一律 `kv_source = Device`。不允许"先声明 `HostSlab`、内容后补"：
  那样 `position <= kv_extent` 会在一段时间内是假话。
- `host_extent` 由 slab 容量决定，与锚点预算无关。这两个数字不要合并：一个管 KV 存得下多少 token，
  一个管状态镜像存得下几张。

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
  `lane * per_lane + offset` 的算法（改由 §4.6 的分配器给出）；
- 单一的 `kv_extent` 与"分歧时把 `prompt_end`/`shared_end` 清零"的裁剪方式（改为 §9.1 的两份 extent）；
- `session_publish` 的三条分支与字段清零；
- `reuse_path` 及 `PrefixReusePath` 枚举，由 §17 的 `reuse_source`/`reuse_reason` 取代。**注意：这一项不是"只服务日志"的清理。**
  实施时核对发现，该枚举驱动 `record_prefix_selection()` 里六个累计计数器（`root_selections`、`private_endpoint_selections`、
  `private_turn_closure_selections`、`private_response_replay_selections`、`private_long_anchor_selections`、
  `shared_stable_prefix_selections`），它们进 `RuntimeStats`、只在单卡路线上被累加（双卡恒零，见 §17.5）、
  并被 `test_resource_manager` 与 `test_engine_prefix_real` 断言。删枚举等于删掉这六个复用选择指标，
  因此必须**先加替代再删旧**：新指标按 §17 定义（`source` × `reason` 两个轴），
  双卡先装上（它今天一个都没有），单卡接线后再删旧计数器、枚举与那个 switch；
- `cached_prompt_tokens` 与 `entry.tokens` 的双份维护（合并为单一 `log.tokens`）；
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
**双卡端到端未验证**：复现脚本在这台机器上启动即崩（见 §12 历史那条的 2026-10-22 复核：与本次改动无关，
但也没有修好），所以这次改动在双卡上没有跑过一次真实请求。
