# tp_bootstrap probes

Standalone CUDA probes used to establish the TP-2 facts on two consumer cards
(2× RTX 5060 Ti 16 GB, PCIe Gen4 x4, SYS topology, no P2P) before and during the TP-2
implementation. They are independent of the engine build unless noted, and they are the
reproducible evidence behind the measurements quoted in
[docs/tp2-dual-5060ti.md](../../docs/tp2-dual-5060ti.md) and
[docs/tp2-decisions.md](../../docs/tp2-decisions.md).

The probe sources are the only files kept from the development session. The one-off driver
scripts, artifact dump tools and log analyzers that produced the round-by-round numbers
(`r35_*` … `r89_*`, `dump*.cpp`, `analyze*.py`, `run_*.sh`, `build_*.sh`) were removed when the
repository was published; the rounds themselves, their commands and their results are recorded
in [tp2-adaptation-worklog.md](../../docs/maintainer/tp2-adaptation-worklog.md).

## Standalone probes

Build any of them with `nvcc` (CUDA 13.x accepts `-arch=sm_120a`):

```
nvcc -O3 -std=c++17 -arch=sm_120a -o bench_ar_overlap bench_ar_overlap.cu
```

| Probe | Question it answers |
|---|---|
| `cuda_devlist.cu` | Which CUDA device index maps to which physical card. In WSL2 the CUDA enumeration order can differ from `nvidia-smi`'s PCI order (observed: `nvidia-smi` `[0=5060Ti,1=T10,2=5060Ti]` vs CUDA `[0=5060Ti,1=5060Ti,2=T10]`), and `ninfer-serve --devices a,b` takes CUDA indices. Each device's UUID is printed in `nvidia-smi`'s 8-4-4-4-12 layout so the CUDA index can be mapped back for the memory-free check. |
| `p2p_probe.cu` | Whether `cudaDeviceCanAccessPeer` allows peer access between the two cards, and — if the driver says yes — the two-way 10 KiB peer-copy latency. On this hardware the driver query reports no P2P, which is why the TP-2 route stages through mapped host memory instead. |
| `mapped_probe.cu` | Whether the `cudaHostAllocMapped` + `cudaHostGetDevicePointer` route works when the peer is another device, i.e. whether both cards see the *same* device pointer into one host allocation, and whether a kernel on card B can read what a kernel on card A wrote. Includes the cross-device in-kernel allreduce (write local, spin on the peer's arrival id, read peer, add) that the engine uses. |
| `probe_mem.cu` | Whether a given device really has free memory: try to `cudaMalloc` 14 GB. A stale-accounting card reports success, a genuinely occupied one fails. Useful before launching a second engine. |
| `bench_allreduce.cu` | Latency of the *host-staging* allreduce path — the four copies, four synchronizations and CPU add that `DevicePair::allreduce` performs per call — at the real per-layer payload (hidden 5120 bf16 = 10,240 B), for comparison against the in-kernel path. |
| `bench_pcie_stage.cu` | The PCIe host-staging bandwidth the collective rides on: a GPU writes into mapped pinned host memory and reads the peer's buffer back, both directions, at the decode payload (10 KiB) and at the T=256 prefill payload (2.5 MiB). |
| `bench_ar_protocol.cu` | Whether slice-pipelining recovers the unused half of the PCIe link. The production protocol writes then reads serially, so a full-duplex link is busy half the time; the sliced variant orders the writes (`slice k` only after `slice k-1` landed) so the peer can read early slices while later ones are still in flight. |
| `bench_ar_overlap.cu` | Whether the in-kernel collective can co-run with a GPU-filling compute kernel. Reproduces the production protocol on a second stream next to a compute kernel and reports the collective's start offset against the compute end (starvation) plus how much the collective's own transfer inflates. `style 0` is FP32-FMA bound, `style 1` is HBM-streaming bound. |
| `bench_ar_tp2.cu` | Per-call latency and achieved PCIe bandwidth of the production in-kernel `DevicePair` path against payload size. |
| `bench_ar_diag.cu` | Minimal construction/teardown diagnostic for `DevicePair`: device contexts, pair creation, P2P availability, then small and large allreduce rounds. |

All probes take CUDA device indices and print their own results; run each with no arguments to
get its defaults.

## Engine-linked probes

`bench_ar_tp2.cu` and `bench_ar_diag.cu` include `core/tp/device_pair.h` and therefore need the
engine headers and the TP-2 translation unit:

```
nvcc -O3 -std=c++17 -arch=sm_120a -Iinclude \
     bench_ar_tp2.cu ../src/core/tp/device_pair.cu -o bench_ar_tp2
```

## Constraints

- Every probe here requires two CUDA devices and exits with a message when fewer are visible.
- The mapped-pinned route these probes measure requires `cudaHostAllocMapped | cudaHostAllocPortable`;
  it works under WSL2 and under native Windows/WDDM at the same order of magnitude.
- Hardware facts these probes established are CPU-independent: the SYS topology and absent P2P mean
  all cross-card traffic goes over the host, which is the design constraint the TP-2 route is built
  around.
