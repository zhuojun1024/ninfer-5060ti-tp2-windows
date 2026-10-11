# NInfer

[English](README.md) | **简体中文**

> 双 RTX 5060 Ti 张量并行 2 路服务，原生运行于 Windows。

本仓库是 [Neroued/ninfer](https://github.com/Neroued/ninfer) 的双卡 fork。它用张量并行 2 路
（TP-2）路线在两张 RTX 5060 Ti（16 GiB）上服务 Qwen3.8-27B NVFP4，并支持在 Windows 上原生构建；
同一份源码也能在 Linux 与 WSL2 上构建运行。上游 NInfer 是为单张 RTX 5090 从零编写的
C++/CUDA 推理引擎，那条单卡路线仍归上游。

服务器通过 OpenAI 与 Anthropic 兼容的 HTTP API 处理文本、图像与视频提示词。运行时刻意专一：
一对 TP-2 显卡、一份常驻模型、启动时固定的 1 到 8 个活跃请求容量，以及一个紧凑的解码轮——
每轮最多合批四个排队请求。CLI 是单卡路线，因此在两张 16 GiB 卡上根本放不下这个模型
（27B NVFP4 权重需要约 20.2 GiB 常驻显存）；本 fork 使用的是服务器路线。

## 硬件与工件

验证过的配置是一台机器上的两张相同 RTX 5060 Ti 16 GiB（计算能力 12.0，`sm_120a`），
系统为 Windows 11，配 Visual Studio 2022（MSVC 19.44）与 CUDA 13.3。正是这对卡让模型装得下：
27B NVFP4 工件需要约 20.2 GiB 常驻权重，即每个 TP-2 分片 10.1 GiB。构建会拒绝 `sm_120a`
以外的 CUDA 架构，两张卡不一致时 TP-2 核心会立即失败。

下面列出六个工件：五个上游官方工件与一个社区合并版本。下面的配方使用 Qwen3.8-27B NVFP4——
在这对卡上验证过的工件；其他工件只有在其单个 TP-2 分片能装进 16 GiB 时才适用于同一对卡。

| 模型 | 权重 | 工件 | 下载与模型卡 |
|---|---|---|---|
| Qwen3.8-27B Coder390（社区合并） | `nvfp4` W4A4-W8A8 | `Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-MTP-DFlash2.ninfer` | [Qwen3.8-27B-Coder390 NVFP4](https://huggingface.co/nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-NVFP4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-27B † | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B † | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B † | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

† 未在双 RTX 5060 Ti 上测试；Qwen3.6 系列工件仅为完整性列出。

社区合并件在本仓库有一份随版本维护的卡片，记录本 fork 实际验证的那个包：
[模型卡](model-cards/Qwen3.8-27B-Coder390-NVFP4-NInfer/README.md)。

每个 v3 `.ninfer` 工件都携带模型配置、编码后的权重、逻辑绑定与前端资源。运行时执行会用这些
事实，配合已实现的模型与算子能力。你也可以[转换自己的权重](docs/weight-conversion.md)，
复用官方 recipe，或选择另一种受支持的格式混合。

当前引擎要求 v3 工件。已有的官方 v2 下载可以
[本地升级](docs/weight-conversion.md#upgrade-an-existing-v2-artifact)，无需重新下载权重。

## 快速开始（原生 Windows）

前置条件：

- Visual Studio 2022，含 C++ 桌面工作负载（`vcvars64.bat`）；
- CUDA 工具链 13.3（`nvcc`），以及 CMake 3.28 或更新版本与位于 `PATH` 上的 Ninja；
- 一个 FFmpeg **开发版**前缀（头文件加导入库）用于媒体解码，以及一个 libcurl 前缀用于远程
  媒体获取。`tools/win_port/fetch_ffmpeg.ps1` 与 `tools/win_port/fetch_curl.ps1` 可以下载
  两者，各自会打印前缀路径以及把它设为默认值的环境变量。

[Windows 原生构建](docs/windows.md) 指南给出完整的前置条件清单、移植带来的平台说明与测试命令。

构建产品二进制：

```powershell
git clone https://github.com/zhuojun1024/ninfer-5060ti-tp2-windows.git
cd ninfer-5060ti-tp2-windows

pwsh -File tools/win_port/build.ps1 -Configure     # Release, sm_120a -> build-win/apps/*.exe
pwsh -File tools/win_port/build.ps1                # incremental build
```

`build.ps1` 自己导入 `vcvars64` 环境，因此不需要 cmd 包装脚本。`-Apps 0` 只构建核心库，
`-Tests 1 -Benchmarks 1` 加上测试套件与基准，而单纯 `-Configure` 会关闭它们。
`build-win/apps/*.exe --version` 会打印这些二进制构建自哪个发行版本。Python 工具独立于 CMake
运行；独立的 HBM 探针有它自己的[构建命令](tools/README.md#standalone-hbm-probe)。

发行版还会附带自包含的 `ninfer-<version>-win-x64.zip`：三个可执行文件加上它们加载的 CUDA、
FFmpeg、curl 与 MSVC 运行时 DLL，因此不需要安装 CUDA 工具包或 FFmpeg 前缀；
`tools/win_port/package.ps1` 可以从本地构建重新生成它（见
[发行打包](docs/windows.md#release-packaging)）。

下载工件并校验：

```powershell
pip install -U huggingface_hub     # provides the `hf` CLI
hf download neroued/Qwen3.8-27B-nvfp4-NInfer --local-dir models
(Get-FileHash models\qwen3_8_27b_nvfp4.ninfer -Algorithm SHA256).Hash
# 74D2C57145E6FF11D1D2FAA79594477F9BC903A611AF1FB20218189FBBB77D82
```

仓库是公开且无需授权的。工件为 23.7 GB，同一次下载也会带来它的 `SHA256SUMS`、
`artifact-manifest.json` 与模型卡；只取工件时传文件名而不是整个仓库。

启动 TP-2 服务器：

```powershell
$env:NINFER_MODEL        = "models\qwen3_8_27b_nvfp4.ninfer"
$env:NINFER_FFMPEG_ROOT  = "D:/ffmpeg-dev/expanded/ffmpeg-master-latest-win64-lgpl-shared"
$env:NINFER_LIBCURL_ROOT = "C:/curl"     # fetch_curl.ps1 prints the prefix it unpacked

pwsh -File tools/win_port/serve.ps1            # 127.0.0.1:8099, 131072-token KV, MTP K=2, vision
pwsh -File tools/win_port/serve.ps1 -Status    # processes, health, VRAM
pwsh -File tools/win_port/serve.ps1 -Stop      # stop every ninfer-serve process
```

请保持那个终端打开：`-Background` 子进程属于启动它的 shell 的作业对象，shell 一退它就跟着退。
脚本会把 FFmpeg 与 libcurl 的 `bin` 目录加到 `PATH` 上，因为要加载的 DLL 就在那里；已经在跑
服务器时它会拒绝启动。`-MaxConcurrency 4` 最多合批四个排队请求，`-Context 262144` 提高 KV
容量，`-Plain` 去掉 MTP 与视觉，`-Spec dflash2 -DraftTokens 7` 选择掩码草稿路线，
`-HostKvMiB 0` 关闭跨会话的 KV 保留。`-Model`（或 `NINFER_MODEL`）选择另一个工件。

发送一个 OpenAI 风格的请求：

```bash
curl http://127.0.0.1:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

### 不用包装脚本的同一台服务器

```powershell
build-win\apps\ninfer-serve.exe models\qwen3_8_27b_nvfp4.ninfer `
  --devices 0,1 `
  --max-context 262144 `
  --kv-capacity auto `
  --kv-dtype fp8 `
  --spec mtp --draft-tokens 2 --lm-head-draft `
  --temperature 0.7 --top-k 20 --top-p 0.80
```

`--devices 0,1` 是 CUDA 运行时索引，不是 `nvidia-smi` 的 PCI 总线顺序；传错一对时会打印
CUDA 可见的设备列表。可执行的 `--help` 是当前选项的权威，[HTTP 服务](docs/serving.md)
记录请求契约。

### Linux 与 WSL2

用 Linux 工具链构建同样的产品：那里经验证的开发工具链是 CUDA 13.1，外加 `pkg-config`、
FFmpeg 开发库（`libavformat`、`libavcodec`、`libavutil`、`libswscale`）与 `libcurl >= 7.85`。

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --devices 0,1 --kv-dtype fp8 --max-context 262144 --kv-capacity auto \
  --temperature 0.7 --top-k 20 --top-p 0.80 \
  --spec mtp --draft-tokens 2 --lm-head-draft
```

`cmake --preset release` 配置同样的产品构建；`cmake --preset dev` 还会启用测试与基准，并寻找
Python 3 解释器。两个 preset 都用 `build/`，且会显式重置构建选项；机器相关的编译器与 Python
路径属于被忽略的 `CMakeUserPresets.json`。见[构建组织与配置](docs/maintainer/build-system.md)。

## 显存与配置

`--max-context` 是每条序列的逻辑上限，`--kv-capacity` 决定活跃请求与保留前缀共用的主文本
KV 池大小；`auto` 在启动时按权重之后剩余的内存解析出最大的合法容量，并留出 1 GiB 的容量
余量。TP-2 的各个 lane（请求通道）动态共享这一个池：只要该请求自己的提示词加输出预算所需的
页可用，它就被接纳，因此单独运行的 lane 可以使用整个上限，而不是固定的 1/lanes 切片；
只有比池的接纳上限更长的提示词才会以 `context_length_exceeded` 被拒绝。

两张 16 GiB 卡在 fp8 KV 下实测的余量：

| 配置 | KV | 权重之后的空闲 |
|---|---:|---|
| `--max-context 262144`，无视觉 | 4,128 MiB | 697 MiB（分片 0）/ 1,217 MiB（分片 1） |
| `--max-context 131072`，`--vision` | 2,322 MiB | 2,002 MiB（分片 0）/ 1,580 MiB（分片 1） |
| `--max-context 8192`，`--vision` | 145 MiB | 4,180 MiB（分片 0）/ 3,930 MiB（分片 1） |

fp8 KV 每 token 每分片花费 16.125 KiB。262,144 token 的配置达到工件完整的
`max_position_embeddings`，也是 [TP-2 说明](docs/tp2-dual-5060ti.md) 推荐的配置；随仓库发布的
Windows `serve.ps1` 配方用 131,072 加 `--vision`，因为 262,144 token 的 KV 池之上再放视觉
工作区，在 WDDM 下剩下的显存太少，不够稳妥。

### DFlash2 草稿权重

DFlash2 工件在目标权重旁边还带一个第二套的、掩码草稿模型。所有随仓库发布的转换 recipe 都把
该草稿的投影存为 `q4_g64_fp16`、把它的选择器码本存为 BF16，因此草稿占用 1,155.3 MiB，
而不是全 Q8 草稿的 2,123.6 MiB。同一模型只差草稿对象的两次转换，文件相差 968.3 MiB
（0.946 GiB）。

这些字节是在承载掩码草稿的那个分片上省下来的，而正是该分片限制了上下文容量，且它是两者中
较重的一个，于是这对卡变得更均衡，省下的显存直接变成 KV：按每 token 16.125 KiB 计，
968.3 MiB 在那里约等于多出 61,000 个 fp8 token。如果某个工件的 `dflash2/` 绑定仍指向
`q8_g32_fp16` 对象，那它带的是旧草稿，这些显存是白付的；用随仓库发布的 recipe 重新转换即可
收回。不需要任何开关或额外脚本，因为格式集合由 `tools/convert/official_recipes.py` 固定：

- Q4：融合注意力 QKV 父对象（每层一个共享的 `[6144, 5120]` 对象，`context_key`/`context_value`
  是它的视图）、注意力输出、MLP gate/up/down、两个动态卷积核投影以及
  `dflash2/feature_projection`——共 31 个存储对象，909.8 MiB；
- BF16：两个选择器码本（242.5 MiB）与各类 norm；
- Q8：DFlash v1 与 MTP 草稿。

这个集合是那轮战役的合格结论，而不是逐 recipe 的随意选择。每一步都在这对卡上做过 A/B：
合并后的工件在合并采样（pooled sampling）接受率上比它替换掉的草稿高 +1.15pp（K=7）与
+3.72pp（K=5），提升进 recipe 的版本在 30 次重复中落在 1σ 以内。贪婪（greedy）探针根本无法
比较两个工件，因为草稿一变，提议的文本就变，目标模型的近似并列（near-tie）也随之改变。

转换与检查：

```bash
python3 -m tools.convert --model /path/to/NVFP4/W4A4-W8A8 --recipe qwen3_8_27b \
  --source dflash2=/path/to/NVFP4/W4A4-W8A8/DFlash2-FP8 --components text,vision,mtp,dflash2 \
  --proposal --out models/my_qwen.ninfer
python3 -m tools.artifact.inspect models/my_qwen.ninfer --objects --bindings
```

把选择器码本也量化能再省 178.09 MiB，但在 33k token 提示词、K=7 下实测接受率低约 2pp，
而且这些字节落在选择器所在的分片、而不是掩码草稿的分片，所以换不来上下文。随仓库发布的
recipe 让码本保持 BF16。

### 自己量化 DFlash2 草稿

同一个杠杆也以显式 override 的形式提供：`tools/convert/overrides/dflash2_draft_q4.py`，
用于基础 recipe 尚未选择该格式的转换——比如你用自己的 recipe 重建的工件，或想与全 Q8 草稿
做对比的工件。先跑一份不写任何东西的计划：

```bash
python3 -m tools.convert --model /path/to/NVFP4/W4A4-W8A8 --recipe my_recipe.py \
  --source dflash2=/path/to/NVFP4/W4A4-W8A8/DFlash2-FP8 --components text,vision,mtp,dflash2 \
  --override tools/convert/overrides/dflash2_draft_q4.py --dry-run
```

`--dry-run` 会应用基础 recipe、提议头与 override，把每条选择打印成
`name [begin,end) format method`，不写任何文件；只有在不用它时 `--out` 才是必需的。去掉该
开关并加上 `--out models/my_qwen.ninfer` 即可转换。默认入口函数 `configure` 把 Q4 指派给草稿
投影，其余参数保持基础 recipe 的选择；码本杠杆是第二个入口：

```bash
  --override tools/convert/overrides/dflash2_draft_q4.py:configure_with_codebooks
```

当模型没有 `dflash2/feature_projection`、没有草稿层，或某一层缺少它所量化的九个投影之一时，
该 override 会明确报错，因此不完整的转换不可能悄悄通过。通用机制见
[修改 recipe 的一部分](docs/weight-conversion.md#change-part-of-a-recipe)。

## 性能

下面每个数字都是在这对卡（`--devices 0,1`）上用 `ninfer-serve` 测得的；除非注明并发，
否则一次只有一个请求。完整的矩阵、显存账本与调优记录在
[双 RTX 5060 Ti TP-2](docs/tp2-dual-5060ti.md) 说明里；[性能索引](docs/performance.md)
里是上游在 RTX 5090 上的单卡测量。

### 单请求服务

在 2026-10-11 的构建上用 [`tools/win_port/bench_serve.ps1`](tools/win_port/bench_serve.ps1)
测得，工件为 `Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-Vision.ninfer`（即上文的
Coder390 工件加上视觉组件），参数 `--max-context 225280 --kv-dtype fp8`，一次一个请求。
预填充是在填充提示词上发一个 token 的请求；解码是在 7,200 token 提示词之后生成 128 token 的
答案。所有数字都取自服务器自身的 `timings` 块，且每个提示词都带 nonce，因此前缀复用无法伪造
预填充。每个档位跑四次。

| 路线 | 预填充 1.9k | 7.2k | 28.6k | 解码 | 相对 plain |
|---|---:|---:|---:|---:|---:|
| plain，无草稿（`-Plain`） | 1,796 | 1,943 | 1,772 | 35.9 | 1.0x |
| MTP K=3 `--lm-head-draft` | 1,797 | 1,917 | 1,753 | 71-79 | ~2.1x |
| DFlash2 K=5 `--lm-head-draft` | 1,618 | 1,629 | 1,557 | 81-96 | ~2.5x |
| DFlash2 K=7 `--lm-head-draft` | 1,595 | 1,626 | 1,556 | 87-90 | ~2.5x |

所有数字单位都是 tok/s。预填充各格是多次运行的均值（1.9k 列为四次，其余为两次），离散度低于
2%；解码各格是那些完整产出 128 token 的运行的区间。这个区间来自随文本变化的草稿接受率，
而不是测量噪声：短提示词的答案实测为 66-84（MTP K=3）、84-104（DFlash2 K=5）与
85-97（DFlash2 K=7）tok/s，输出长度 43-181 token。草稿是用预填充换解码——DFlash2 自身的前向
在每种提示词长度下都要花掉约 15% 的预填充吞吐，MTP 约 1%——而 DFlash2 K=5 与 K=7 落在彼此的
离散区间内。

该容量下的显存，`nvidia-smi` 每卡：plain 档 13,688 MiB（`-Plain` 同时关掉视觉与采样覆盖），
MTP K=3 为 14,820 / 14,638 MiB，DFlash2 K=5 为 14,954 / 14,884 MiB，K=7 为
14,960 / 14,886 MiB。引擎加载是一次性的 40 s（plain）到 44 s（DFlash2），不计入上述数字。
更早、在别的协议下取得的数字——16,057 token 的 TTFT、前缀复用延迟以及 llama.cpp 跨引擎
对比——都在 [TP-2 说明](docs/tp2-dual-5060ti.md) 里，那份文档把每个测量与它所属的协议放在
一起。

### 多请求服务

`--max-concurrency`（本路线 1..4）决定核心把多少个排队请求合批进一个解码轮；默认是 1。
单个请求在 3 ms 的组批窗口后即被接纳，在这个窗口内到达的突发请求共享同一轮，因为窗口会一直
开着，直到本轮装满或 50 ms 过去。在一轮运行期间到达的请求，会在某个 lane 退出后加入同一
批次的后续轮。

| 路线 | C=2 时的聚合解码 | C=4 时 |
|---|---:|---:|
| plain，int8 KV，131,072 | 1.76-1.91x | 2.80x |
| `--spec dflash2` | — | 2.07x |
| `--spec mtp --draft-tokens 5` | 1.48x | 2.26x |

一轮的开销等于它最长成员的开销，所以聚合收益取决于批内成员结束得有多齐：输出长度相差两倍的
一对实测 1.41x。把轮加宽本身很便宜——四个等长 lane 在单 lane 时是 26.5 ms、四个 lane 时
32.0 ms，即理想 4x 的 3.31x，每列边际开销约 1.85 ms（单 lane 轮的 7%）。四个 lane 在
`--max-context 262144 --kv-dtype fp8` 加 `--spec mtp --draft-tokens 2 --lm-head-draft` 下全部
完成，剩余 829 MiB（分片 0）与 1,789 MiB（分片 1），每个 lane 每卡约花 147 MiB 的线性注意力
状态。

正在预填充的 lane 不会阻塞旁边的 lane：每完成一个预填充块，核心就为已持有 token 的 lane 跑
一次解码轮，因此一个 54 token 的请求在 15,310 token 的请求之前 400 ms 被接纳时，用
`--prefill-chunk 256` 在 2.8 s 内完成、8.5 tok/s（默认 1024 时是 7.5 s 与 3.1 tok/s）。

## 资源感知的长上下文复用

可复用的前缀检查点包含 KV，以及其确切提示词前沿的完整续写状态。驻留设备端的检查点可直接
恢复。显存紧张时，规划器会按即时恢复工作量与后续复用成本，在设备端保留、锁页主机端
State/KV 与驱逐之间权衡。活跃请求保有自己的完成预留。在 TP-2 路线上，检查点环与会话目录按
lane 切片，因此设备端驻留的复用取决于请求落在哪个 lane，而落到别的 lane 上的召回会退回
主机端驻留的会话。

算法见[资源调度与上下文缓存](docs/maintainer/resource-scheduling-and-context-cache.md)；
通过公开 HTTP 接口对热复用、主机端恢复、驱逐、共享前缀、调度边界与多模态负载的覆盖见
[Serve TTFT 基准](tools/bench/ttft/)。

## 评测

能力分数是通过 NInfer 的 OpenAI 兼容服务路线测得的，启用思考、MTP3 与 EvalScope 1.9.0
（0-shot、规则打分、每题一个样本）：

| 模型档位 | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

Qwen3.6 各行用 temperature 0.6 与 presence penalty 1.0；Qwen3.8 各行用 temperature 1.0 与
presence penalty 0.0。多模态评测使用 `--vision` 与 81,920 token 的上下文上限。这些是上游单卡
运行得到的模型能力分数；这对 TP-2 卡服务同样的工件，并能达到工件完整的 262,144 token 上限。
每个分数是每题一个样本；正确/总数与评测说明在模型卡里。

## Docker

镜像构建同一份源码，因此 TP-2 路线可以在其中运行，但双卡容器路径在此未经验证。在装有
NVIDIA Container Toolkit 的主机上构建：

```bash
docker build --tag ninfer:local .
```

传入两张卡与同样的 TP-2 配置：

```bash
docker run --rm \
  --gpus '"device=0,1"' \
  --publish 8099:8099 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --port 8099 \
  --devices 0,1 \
  --max-context 262144 \
  --kv-capacity auto \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 2 --lm-head-draft \
  --temperature 0.7 --top-k 20 --top-p 0.80
```

## 能力与限制

官方工件提供以下能力，可选组件在启动时启用：

- 带思考与非思考提示模式的文本生成；
- 图像、多图、视频与混合多模态消息；
- 分块预填充、精确批 CUDA Graph 解码，以及启动时限定规模的合批解码；
- 草稿窗口为 1 到 5 的 MTP 投机解码；
- BF16、INT8、FP8、NVFP4 与 K8V4 KV 存储；
- 离线因果困惑度打分；
- 私有与共享的精确前缀复用，含设备端/主机端 State 与 KV 保留；
- 感知模型的采样默认值与显式采样器覆盖；
- OpenAI Responses Core、OpenAI Chat Completions 与 Anthropic Messages，含流式、工具调用、
  本地响应状态、token 计数与用量统计。

用 `qwen3_8_27b_gguf` 转换的工件则把每个张量保持在其 GGUF 发行版自有的 ggml 块类型中；
见 [GGUF 块格式](docs/gguf.md)。

35B-A3B 目标还支持 DFlash，草稿窗口 1 到 15，适用于文本与图像/视频视觉提示词。
带 DFlash2 伴随权重的 Qwen3.8-27B 工件支持 `--spec dflash2 --draft-tokens 7` 走同样的
文本/视觉引擎路径，草稿数量 1..15，提议头可选完整或优化版本。

产品边界刻意保持很小：

- 在 TP-2 服务器路线上是两张相同的 RTX 5060 Ti 16 GiB 卡（或上游单卡路线上的一张
  RTX 5090），每个引擎一份常驻模型；
- 启动时固定的 1 到 8 个活跃请求容量，入口为有界 FIFO（这对 TP-2 卡在 plain、DFlash2 与
  MTP 轮上最多合批四个排队请求，其他情况一次只跑一个）；
- 没有请求抢占、优先级/QoS、活跃请求换出、权重卸载或分布式服务；多卡仅限于双卡 TP-2
  服务器路线；
- 活跃请求与保留前缀共享一个启动时固定的 KV 池；
- 模型架构与格式/形状组合都走显式实现的原生路径；
- 解析出的工具调用返回给客户端；NInfer 不执行工具；
- 树内 C++ 头文件不作为安装式 SDK 分发。

## 启动注意事项

GPU 驻留在进程启动时固定，因此 `--devices 0,1`、`--spec` 与 `--vision` 都只选一次并在整个
进程生命周期内保持。Qwen3.6-35B-A3B 的 DFlash 可以与视觉组合；它加速的是多模态预填充之后的
生成文本解码，而不是视觉编码本身。

## 文档

- [文档索引](docs/README.md)
- [Windows 原生构建](docs/windows.md)
- [双 RTX 5060 Ti TP-2](docs/tp2-dual-5060ti.md)
- [已定案的 TP-2 决策](docs/tp2-decisions.md)
- [HTTP 服务](docs/serving.md)
- [命令行](docs/cli.md)
- [性能](docs/performance.md)
- [困惑度评测](docs/perplexity.md)
- [权重转换与自定义 recipe](docs/weight-conversion.md)
- [GGUF 块格式](docs/gguf.md)
- [资源调度与上下文缓存](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT 基准](tools/bench/ttft/)
- [CLI 示例](examples/cli/)
- [贡献指南](CONTRIBUTING.md)

运行相关的 `--help` 以获得当前确切的选项契约。

## 与上游的关系

本仓库是 [Neroued/ninfer](https://github.com/Neroued/ninfer) 的 fork。它保留上游引擎、
`.ninfer` 工件格式、转换 recipe、模型卡与已发布的工件，并新增面向两张 16 GiB 卡的双卡张量
并行 2 路（TP-2）路线以及原生 Windows 构建。引擎本身的权威在上游；TP-2 的增补、其测量与
完整适配记录见[双 RTX 5060 Ti TP-2 调优](docs/tp2-dual-5060ti.md)、
[已定案的 TP-2 决策](docs/tp2-decisions.md) 与
[TP-2 适配工作日志](docs/maintainer/tp2-adaptation-worklog.md)。

## 许可证

NInfer 采用 [Apache License 2.0](LICENSE) 许可；[NOTICE](NOTICE) 记录本 fork 所源自的上游项目，
以及 `third_party/` 下每个 vendored 组件。

已发布的工件派生自 [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)、
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 与
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)。Qwen3.6-27B NVFP4 工件
还使用了 [rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm)
的固定打包权重。Qwen3.8-27B NVFP4 工件还使用了
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) 的固定
FP8/NVFP4 混合权重。这些源仓库以 Apache-2.0 分发。vendored 依赖保留各自在 `third_party/`
下的许可证文件。发行归档在二进制旁附带其打包的每个组件的许可证文本，清单见
`THIRD-PARTY-NOTICES.txt`。
