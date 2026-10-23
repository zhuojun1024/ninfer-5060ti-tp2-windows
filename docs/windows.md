# Native Windows build and run

NInfer builds and runs natively on Windows with MSVC and the CUDA toolkit: the engine, the CLI and the
HTTP server are the same sources as the Linux build, with platform branches confined to file I/O, host
process queries, media acquisition, and one CUDA kernel-parameter detail.

Verified configuration: Visual Studio 2022 (MSVC 19.44) + CUDA 13.3 + Ninja, two RTX 5060 Ti 16 GiB
cards in a TP-2 pair (`--devices 0,1`), artifact `qwen3_8_27b_nvfp4.ninfer` (23.7 GB).

## Prerequisites

- Visual Studio 2022 with the C++ desktop workload (`vcvars64.bat`).
- CUDA toolkit 13.3 (`nvcc`); CMake and Ninja on `PATH`.
- An FFmpeg **development** prefix (headers plus import libraries) for media decoding. The shared
  BtbN build works; extract it anywhere and pass the prefix:
  `-DNINFER_FFMPEG_ROOT=D:/ffmpeg-dev/expanded/ffmpeg-master-latest-win64-lgpl-shared`.
- A libcurl prefix for remote media acquisition (`tools/win_port/fetch_curl.ps1` downloads one). The
  import library is used, not the static archive: MSVC must link `libcurl.dll.a` and load
  `libcurl-x64.dll` at run time. Only the CLI/server targets need it.

## Build

```powershell
pwsh -File tools/win_port/build.ps1 -Configure        # configure and build (Release, sm_120a)
pwsh -File tools/win_port/build.ps1                  # incremental build
pwsh -File tools/win_port/build.ps1 -Configure -Apps 0   # core libraries only
pwsh -File tools/win_port/build.ps1 -Configure -Tests 1 -Benchmarks 1   # add the suite and benchmarks
```

`-Apps`, `-Tests`, and `-Benchmarks` set `NINFER_BUILD_APPS`, `BUILD_TESTING`, and
`NINFER_BUILD_BENCHMARKS` on every `-Configure` run, so a plain `-Configure` turns the suite and the
benchmarks **off**. Pass the switches for the configuration you want; the three are additive.

Products land in `build-win/apps/{ninfer,ninfer-serve,ninfer-perplexity}.exe`. `build.ps1` imports the
`vcvars64` environment into the PowerShell session (see `tools/win_port/vcvars.ps1`), so no cmd wrapper
is needed.

Every object file's header dependencies come from cl.exe's localized `/showIncludes` note, and CMake
decodes that note with the ANSI code page. On a localized toolchain the two disagree (`注意: 包含文件:`
on the compiler side, `娉ㄦ剰: 鍖呭惈鏂囦欢:` in `rules.ninja`), so ninja matches no line and records **zero**
header dependencies: the build looks healthy, an incremental build does nothing after a header edit,
and the binary then mixes objects compiled against two different struct layouts. Nothing fails to
link — the translation unit that owns an object allocates it at one size while a fresh constructor
fills it at another, and the first symptom is an access violation inside the CUDA driver
(`cudaMalloc`, `cudaGetDeviceProperties`), a `0xC0000374` heap-validation break, or a thread that
jumps to address zero, always with a stack that names innocent code.

`tools/win_port/build.ps1` repairs this on every run: `Repair-MsvcDepsPrefix` asks cl.exe what it
prints, writes those bytes back into `CMakeFiles/rules.ninja`, drops `.ninja_deps`, and touches the
C++ sources so the objects relearn their dependencies. That is why it, and not a raw
`cmake -S . -B build-win`, must be the entry point: **a raw configure regenerates the mis-decoded
prefix**, which is exactly how a tree ends up in the state above. When a driver-side crash disappears
after a rebuild, suspect this before looking for a memory bug in the engine;
`NINFER_GUARD_HEAP=1` on `tests/models/qwen3_5/test_tp2_startup_order.cpp` puts every allocation
against a guard page and prints the faulting instruction, which localizes such an overrun in one run.

## Run

```powershell
pwsh -File tools/win_port/serve.ps1                  # foreground TP-2 server on 127.0.0.1:8099
pwsh -File tools/win_port/serve.ps1 -Context 32768   # smaller KV capacity
pwsh -File tools/win_port/serve.ps1 -Plain           # no MTP or vision
pwsh -File tools/win_port/serve.ps1 -EmbeddingHost   # +606 MiB VRAM per shard, mapped embedding table
pwsh -File tools/win_port/serve.ps1 -Status          # processes, health, VRAM
pwsh -File tools/win_port/serve.ps1 -Stop            # stop every ninfer-serve process
```

The script puts the FFmpeg and libcurl `bin` directories on `PATH` before starting the server, because
that is where the loading DLLs live. It refuses to start when a server is already running, and it only
ever manages `ninfer-serve`: no llama.cpp process is started or stopped.

Run it from a terminal you keep open. A `-Background` child belongs to the launching shell's job
object, so an agent harness that starts the server inside one tool call will kill it as soon as the call
returns (the engine then never reaches the GPU).

## Release packaging

The release version has one source of truth: `project(ninfer VERSION x.y.z ...)` in `CMakeLists.txt`,
compiled into every executable as `NINFER_VERSION` and printed by `--version`:

```powershell
build-win/apps/ninfer-serve.exe --version    # ninfer-serve 0.1.2
```

`tools/win_port/package.ps1` turns a finished build into a self-contained archive:

```powershell
pwsh -File tools/win_port/build.ps1 -Configure          # the three executables
pwsh -File tools/win_port/package.ps1                   # dist/ninfer-<version>-win-x64.zip
pwsh -File tools/win_port/package.ps1 -RequireTag       # refuse unless HEAD carries v<version>
```

The script reads the version from the executables and refuses to package when the three disagree, so
the archive cannot be labeled from anything but the binaries themselves. It walks the
`dumpbin /dependents` closure of all three, copies every DLL it resolves into `bin/`, and adds
`LICENSE`, `NOTICE`, a generated `README.txt`, `THIRD-PARTY-NOTICES.txt` and the license text of
every component in the archive: `licenses/` for the CUDA, FFmpeg and curl prefixes,
`licenses/vendored/` for the sources compiled into the executables, and `licenses/curl-deps/` for
the libraries linked into `libcurl-x64.dll`.

Vendor directories win over `System32` while resolving that closure: `msvcp140.dll` is in `System32`
on a machine with the VC redist installed, but the archive has to carry the redistributable copy so it
also runs where the redist is absent. `nvcuda.dll` always comes from the display driver and is never
bundled, so a current NVIDIA driver is still required.

The bundled cuBLAS libraries are what let the archive run without a local CUDA toolkit; Attachment A
of the CUDA toolkit license lists them as redistributable. `THIRD-PARTY-NOTICES.txt` names every
component and says where its license text sits, including components that are not separate DLLs: the
six MIT sources vendored under `third_party/` and compiled into the executables, and the TLS,
compression and HTTP/2 and HTTP/3 libraries that the curl prefix links into `libcurl-x64.dll` (the
script records that list verbatim from `curl -V`).

FFmpeg needs one decision per prefix, and the verified prefix is the LGPL shared build
(`ffmpeg-master-latest-win64-lgpl-shared`). LGPL is sufficient here because NInfer only calls the
decoders in `avcodec`, `avformat`, `avutil`, `swscale` and `swresample`
(`src/media/decode/decode.cpp`). The notice entry records the build identifier from
`ffmpeg -version` next to the prefix's own `LICENSE.txt`, so an archive never claims a license the
prefix does not carry; a GPL prefix still builds and packages, and its entry then also names the
source location. Keep the prefix and the archive in step: changing `-FfmpegRoot` without
re-packaging leaves the notice describing the previous build.

Publishing is manual and stays outside the script: build, package, tag the release commit `v<version>`,
push the tag, and attach the archive to the GitHub release. `package.ps1` never tags, pushes or uploads.
The archive is about 1.2 GiB: three ~280 MiB executables and 11 DLLs totaling 638 MiB, most of it
cuBLAS (`cublasLt64_13.dll` alone is 442 MiB).

## Verified behavior on Windows

These rows were recorded on an earlier build as a Linux/Windows parity check; the current four-route
throughput table is in the [README](../README.md#single-request-serving).

| Check | Linux (WSL2) | Windows |
|---|---|---|
| TP-2 plain decode | 32.8 tok/s | 32.9 tok/s |
| TP-2 MTP `--draft-tokens 2` decode | 56.8 tok/s | 56.9 tok/s |
| Image turn decode | MTP kept active | 60.4 tok/s, MTP continued (`rate=106/174`) |
| Engine load (23.7 GB artifact) | — | 36-39 s (`staging peak` 256 MiB: pinned ring) |
| Real agent traffic (OpenAI chat, streaming, tools, images) | — | served, 41-73 tok/s |

The loader starts the server faster than the old 45-47 s rows: `tp_materialize` stages every
weight piece through a four-slot 256 MiB portable-pinned ring (one event per device per slot) and
overlaps the artifact read with the shard uploads instead of staging pageable payloads at all.
Six measured samples on the two builds (interleaved stash/rebuild, engine `ready` line, single
resident instance): baseline 45.0/47.0/47.1 s, pipelined 29.8/37.9/38.3 s.

KV capacity and memory headroom on the 16 GiB cards (`--vision` enabled):

```
capacity   8192 | kv  145 MiB | free 4180 MiB (shard 0) / 3930 MiB (shard 1)
capacity 131072 | kv 2322 MiB | free 2002 MiB (shard 0) / 1580 MiB (shard 1)
```

131072 is the recommended Windows capacity: 262144, which the WSL recipe used, leaves too little VRAM
for comfort under WDDM.

### TP-2 mapped-pinned spike

Peer access is unavailable on this machine (no NVLink, `nvidia-smi topo -m` reports SYS), so TP-2 uses
the in-kernel allreduce over mapped pinned host memory. `tools/win_port/spike.ps1` exercises that
primitive outside the engine:

```
peer access: 0->1 0, 1->0 0
mapped pinned: host_a=...204A00000 view_a=...204A00000 | host_b=...20AC00000 view_b=...20AC00000
peer view of host_a from device 1: ...204A00000
     bytes   fill_ms  peer_read_ms
      4096    0.0452        0.0112   OK  (0.1 + 0.3 GiB/s)
    262144    0.0529        0.1986   OK  (4.6 + 1.2 GiB/s)
   4194304    0.6784        2.0432   OK  (5.8 + 1.9 GiB/s)
  25165824    4.0219       17.2313   OK  (5.8 + 1.4 GiB/s)
```

So `cudaHostAlloc(cudaHostAllocPortable | cudaHostAllocMapped)` plus `cudaHostGetDevicePointer` work
under WDDM, the peer device addresses the same UVA address, and cross-card reads of mapped host memory
are correct and run at the same ~1.4-1.9 GiB/s order as the WSL2 path. The end-to-end consequence is the
Linux-parity decode rate above.

## Tests

```powershell
pwsh -File tools/win_port/build.ps1 -Configure -Tests 1   # configure and build the suite
pwsh -File tools/win_port/test.ps1                        # run it through ctest
pwsh -File tools/win_port/test.ps1 -Filter "artifact|nvfp4"
```

`test.ps1` puts the FFmpeg and libcurl `bin` directories on `PATH` before invoking ctest; the test executables
load those DLLs at run time, and without them the Windows loader raises a missing-DLL dialog for every
executable instead of reporting a failed test.

The suite is the Linux suite. Two platform notes:

- The CUDA fault-injection companion of `ninfer_artifact_materialization_test` relies on GNU ld `--wrap`,
  which MSVC's linker does not implement, so that source and its link options are excluded on Windows and the
  executable keeps its materialization and writer-fixture checks.
- Tests that need a real model artifact report themselves as skipped through ctest's `SKIP_RETURN_CODE`.

## Platform differences carried by the port

| Area | Windows behavior |
|---|---|
| `src/artifact/file_io.cpp` | `CreateFileW` + `ReadFile` with an `OVERLAPPED` offset for positional reads; `FILE_FLAG_SEQUENTIAL_SCAN` for the buffered handle and `FILE_FLAG_NO_BUFFERING` for the direct one. The handle is opened with `FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE` so a reader keeps the POSIX contract: another process may rewrite, replace or delete the artifact underneath it |
| `tools/artifact/file_io.py` | page-cache discarding is a Linux facility, so `posix_fadvise` and `fdatasync` degrade to a no-op and `fsync`; `read_at`/`write_at` emulate `os.pread`/`os.pwrite`, which Windows lacks |
| `src/core/uint128.h` | 128-bit accounting without `__int128`: `_umul128` with saturating add, shift and compare |
| `src/core/host_process.{h,cpp}` | `_getpid`, `_isatty`, `localtime_s`/`gmtime_s`, and `GetConsoleScreenBufferInfo` for the progress width |
| `src/product/media_acquire/acquire.cpp` | `winsock2`/`ws2tcpip` name resolution with a one-time `WSAStartup`; links `ws2_32` |
| NVFP4 TMA kernels | the 64-byte-aligned tensor-map aggregate cannot be passed by value under the MSVC parameter ABI (C2719), so the kernel takes a descriptor pointer while `Nvfp4TmaDescriptorStaging` publishes the tensor maps through a mapped-pinned host arena. A device-side copy is not an option: this op also runs inside captured CUDA graphs, where a stream-ordered allocation is a memory node settled at instantiation and a pageable host source is gone by replay. The kernel admits each descriptor with `fence.proxy.tensormap::generic.acquire.gpu` because the tensor-map proxy has its own view of the tensor map |
| `CMakeLists.txt` | MSVC needs `/Zc:preprocessor` (CUDA 13's CCCL rejects the traditional preprocessor), `/utf-8` (bundled fmt), and `NOMINMAX`/`WIN32_LEAN_AND_MEAN` |

## Limitations

- The CLI is single-GPU. This 27B NVFP4 artifact needs about 20.2 GiB of resident weights (10.1 GiB per
  TP-2 shard), so it cannot be loaded onto one 16 GiB card; use `ninfer-serve --devices 0,1` on this
  machine. The CLI's own Windows paths are covered: the binary builds, parses arguments and reports usage
  errors correctly, and `ninfer_cli_options_test` passes, while everything below the CLI (artifact load,
  engine, KV, graphs, sampling) is the engine verified end to end by the TP-2 server. What this machine
  cannot demonstrate is a single-card generation run: forcing it would not fail cleanly, because WDDM would
  page the missing ~4 GiB into system memory instead of rejecting the allocation.
- The bundled libcurl is a mingw import library. It links and runs, but a native MSVC build (vcpkg) or a
  WinHTTP implementation would remove that dependency.
- Windows performance equals Linux here; the port's value is deployment convenience, not speed.
