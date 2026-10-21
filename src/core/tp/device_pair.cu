#include "core/tp/device_pair.h"

#include "core/arena.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// The host-staging fallback's BF16 add is the one hot-path arithmetic this file performs on the
// host, so it carries an SSE2 path of its own (baseline on every x86-64 host). A host without it
// keeps the scalar __hadd loop.
#if defined(_M_X64) || defined(_M_IX86) || defined(__SSE2__) || defined(__x86_64__)
#define NINFER_TP2_HOST_ADD_SSE2 1
#include <emmintrin.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

namespace ninfer::tp {
namespace {

// Elementwise BF16 add of two 16-byte groups (eight BF16 values).
__device__ __forceinline__ uint4 add_bf16x8(uint4 a, uint4 b) {
    const __nv_bfloat162 a0 = *reinterpret_cast<const __nv_bfloat162*>(&a.x);
    const __nv_bfloat162 a1 = *reinterpret_cast<const __nv_bfloat162*>(&a.y);
    const __nv_bfloat162 a2 = *reinterpret_cast<const __nv_bfloat162*>(&a.z);
    const __nv_bfloat162 a3 = *reinterpret_cast<const __nv_bfloat162*>(&a.w);
    const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&b.x);
    const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&b.y);
    const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(&b.z);
    const __nv_bfloat162 b3 = *reinterpret_cast<const __nv_bfloat162*>(&b.w);
    uint4 r;
    *reinterpret_cast<__nv_bfloat162*>(&r.x) = __hadd2(a0, b0);
    *reinterpret_cast<__nv_bfloat162*>(&r.y) = __hadd2(a1, b1);
    *reinterpret_cast<__nv_bfloat162*>(&r.z) = __hadd2(a2, b2);
    *reinterpret_cast<__nv_bfloat162*>(&r.w) = __hadd2(a3, b3);
    return r;
}

// Payload past which the reduction is split across host threads. Thread creation is what sets this
// bound, not the arithmetic: measured on the 2x RTX 5060 Ti host, the reduction's own cost is
// 2.1-2.5 ms at 10 MiB and 12.0 ms at 50 MiB single-threaded, against 4.1-4.4 ms at 10 MiB and
// 8.4 ms at 50 MiB across eight threads - the split loses about 2 ms to spawning at the prefill
// payload width (10 MiB) and wins about 3.5 ms past this threshold. A host with cheaper threads
// (Linux) wins earlier; the bound only decides when the threads are worth creating.
constexpr std::size_t kParallelAddMinimumBytes = 32ULL << 20;
// Threads one reduction uses at most. The pass is memory-bound, and past this the strips stop
// paying for their own spawn and join.
constexpr unsigned kMaxAddThreads = 8;

#if defined(NINFER_TP2_HOST_ADD_SSE2)
// Four FP32 sums (as bit patterns) rounded to BF16, nearest-even, with NaN canonicalized to 0x7fff:
// bit for bit what cuda_bf16.hpp's __float2bfloat16 does on the host, which is the rounding the
// device's __hadd goes through.
__m128i bf16_round_x4(__m128i bits) {
    const __m128i magnitude = _mm_and_si128(bits, _mm_set1_epi32(0x7fffffff));
    const __m128i is_nan    = _mm_cmpgt_epi32(magnitude, _mm_set1_epi32(0x7f800000));
    const __m128i kept      = _mm_srli_epi32(bits, 16);
    // The dropped half moved to the top of the word, so the two rounding tests are comparisons
    // against half. Unsigned compares are signed compares after flipping the sign bit.
    const __m128i dropped =
        _mm_xor_si128(_mm_slli_epi32(bits, 16), _mm_set1_epi32(static_cast<int>(0x80000000u)));
    const __m128i over_half = _mm_cmpgt_epi32(dropped, _mm_setzero_si128());
    const __m128i half      = _mm_cmpeq_epi32(dropped, _mm_setzero_si128());
    const __m128i tie_up =
        _mm_and_si128(half, _mm_cmpgt_epi32(_mm_and_si128(kept, _mm_set1_epi32(1)),
                                           _mm_setzero_si128()));
    // The masks are all-ones, so subtracting adds one.
    const __m128i rounded = _mm_sub_epi32(kept, _mm_or_si128(over_half, tie_up));
    return _mm_or_si128(_mm_and_si128(is_nan, _mm_set1_epi32(0x7fff)),
                        _mm_andnot_si128(is_nan, rounded));
}

// Four BF16 pairs summed in FP32. The widening is a left shift: a BF16 value is the high half of the
// FP32 bit pattern it converts to, so the conversion is exact and needs no exponent arithmetic.
__m128i bf16_sum_x4(__m128i a_bits, __m128i b_bits) {
    const __m128 fa = _mm_castsi128_ps(_mm_slli_epi32(a_bits, 16));
    const __m128 fb = _mm_castsi128_ps(_mm_slli_epi32(b_bits, 16));
    return _mm_castps_si128(_mm_add_ps(fa, fb));
}
#endif

// Elementwise in-place BF16 add of two host halves: out[i] = a[i] + b[i] with the device transport's
// exact rounding (see bf16_round_x4). This is the host-staging fallback's whole arithmetic and used
// to be its whole cost: the scalar __hadd loop it replaces ran at 0.31 GB/s, an order of magnitude
// under the link it feeds, which is what made the fallback a performance cliff rather than a slower
// transport. Eight values per iteration now, and about 10x the retired loop's rate end to end.
void add_bf16_host(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
                   std::size_t elements) {
#if defined(NINFER_TP2_HOST_ADD_SSE2)
    const auto* au = reinterpret_cast<const std::uint16_t*>(a);
    const auto* bu = reinterpret_cast<const std::uint16_t*>(b);
    auto* ou       = reinterpret_cast<std::uint16_t*>(out);
    const __m128i zero = _mm_setzero_si128();
    std::size_t i      = 0;
    for (; i + 8 <= elements; i += 8) {
        const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(au + i));
        const __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(bu + i));
        const __m128i lo = bf16_round_x4(
            bf16_sum_x4(_mm_unpacklo_epi16(va, zero), _mm_unpacklo_epi16(vb, zero)));
        const __m128i hi = bf16_round_x4(
            bf16_sum_x4(_mm_unpackhi_epi16(va, zero), _mm_unpackhi_epi16(vb, zero)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ou + i), _mm_packus_epi32(lo, hi));
    }
    for (; i < elements; ++i) { out[i] = __hadd(a[i], b[i]); }
#else
    for (std::size_t i = 0; i < elements; ++i) { out[i] = __hadd(a[i], b[i]); }
#endif
}

// out[i] = a[i] + b[i] over `elements`, split across `threads` strips handed out by an atomic
// counter, in place (out may alias a). The counter is what makes the split total and single-visit:
// a thread the OS refused (emplace_back throwing under thread pressure) only narrows the reduction,
// it never leaves a strip unadded, and no strip is handed out twice - which the in-place add
// requires, since adding a strip twice would add its operand twice.
void add_bf16_host_split(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
                         std::size_t elements, unsigned threads) {
    if (threads <= 1 || elements < 8) {
        add_bf16_host(a, b, out, elements);
        return;
    }
    // Strips are whole 16-byte groups (eight BF16), so every strip but the last runs the vector loop
    // over its whole range and the last covers the remainder.
    const std::size_t groups    = (elements + 7) / 8;
    const std::size_t strips    = std::min<std::size_t>(threads, groups);
    const std::size_t per_strip = ((groups + strips - 1) / strips) * 8;
    // Strip 0 is nobody's in particular: this thread and every worker take the next one the counter
    // hands out, so the split covers the payload whatever number of workers the spawn managed.
    std::atomic<std::size_t> next{0};
    const auto reduce = [&] {
        for (;;) {
            const std::size_t strip = next.fetch_add(1, std::memory_order_relaxed);
            if (strip >= strips) { return; }
            const std::size_t begin = std::min(elements, strip * per_strip);
            const std::size_t end   = std::min(elements, begin + per_strip);
            add_bf16_host(a + begin, b + begin, out + begin, end - begin);
        }
    };
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads) - 1);
    try {
        for (unsigned t = 1; t < threads; ++t) { workers.emplace_back(reduce); }
    } catch (const std::system_error&) {
        // Fewer threads than asked for; the counter still covers every strip.
    }
    reduce();
    for (std::thread& worker : workers) { worker.join(); }
}

// Host threads one reduction uses: one below the parallel threshold, otherwise what the machine has,
// capped. `requested` is the NINFER_TP2_AR_ADD_THREADS override, or 0 for the automatic count.
unsigned add_threads(std::size_t count_bytes, unsigned requested) {
    if (count_bytes < kParallelAddMinimumBytes) { return 1; }
    const unsigned available = std::max(1U, std::thread::hardware_concurrency());
    return std::min<unsigned>(kMaxAddThreads, requested != 0 ? requested : available);
}

// In-kernel all-reduce (WSL2, no P2P): both devices run this kernel on their
// own compute streams. Phase 1 writes the local delta into this device's
// mapped pinned host buffer; phase 2 has one thread per block publish an arrival
// token and spin (with __nanosleep) until the peer's token matches; phase 3
// reads the peer's buffer and writes the elementwise sum back in place.
// Cross-GPU synchronization happens entirely inside the kernel via the
// host-memory arrival tokens, so the host never calls cudaStreamSynchronize.
// The token is a monotonically increasing call number (never reset), so a stale
// token from a previous call can never satisfy the spin.
//
// A batched prefill all-reduces a [hidden, T] delta (megabytes), so the payload is moved in
// 16-byte groups when every pointer is group-aligned ('groups' is then count/8, zero otherwise and
// the scalar loops below cover the whole payload). Phase 3 reads four groups per thread per step
// because a dependent single-group loop over a mapped host buffer is latency-bound long before it
// is bandwidth-bound: the four loads issue before any of them is consumed.
//
// Slices: a decode-sized payload rides a single block, because its cost is the peer round trip, not
// the transfer. A prefill-sized payload is split over several blocks, each owning one contiguous
// slice and one arrival slot, so the slices reduce independently once their data has landed. On
// this host the payload's device link (device 1 sits behind a Gen4 x4 chipset link, ~7.9 GB/s in
// both directions combined) is the limit, and moving a slice out of the write queue early is what
// lets the peer read it while the remaining slices are still going out: see ar_slices. Block b
// additionally waits for its predecessor's slice to become system-visible before it writes, so the
// slices land in index order instead of interleaving into one long drain.
// block.b write order chain: block b only starts writing once block b-1 published its arrival, so
// each slice becomes system-visible in index order and the peer can read early slices while later
// ones are still in flight. Without it the concurrent writes interleave and no slice is complete
// until the whole transfer drains.
// The arrival token is read from device memory rather than passed by value: a CUDA Graph records
// kernel arguments, so a value-baked token would be replayed unchanged and a peer that is a call
// ahead would satisfy its spin from the previous replay. A device counter that every replay
// increments on both devices keeps the monotonic-token invariant inside a captured sequence.
// kAdd selects the reduction: true is the in-place all-reduce the TP-2 routes run, false is a
// byte-exact exchange where `out` receives the peer's staged bytes unchanged. Both share the
// staging, arrival-token and slice machinery; only phase 3 differs. The copy mode preserves every
// 16-bit lane, so a payload whose lanes spell a signaling NaN - an I32 id, an FP32 score - travels
// intact where the BF16 add would quiet it.
// Rendezvous ids are host-authored (see DevicePair::create_ar_channel): the base cell is written
// before the launch, a captured graph reading it through its own memcpy node and an eager call
// through the host's mapped cell. The ordinal within the block is baked at capture, where it is a
// constant of the graph. Ids therefore only ever increase and are never reused, which is what lets
// the arrival slots be compared for equality at all: a reused id would let a stale slot satisfy the
// spin before the peer wrote anything.
//
// Wall-clock nanoseconds for the bounded spin: %globaltimer is a device-wide clock every SM agrees
// on, unlike clock64(), whose per-SM counters drift apart under boost.
__device__ __forceinline__ unsigned long long ar_now_ns() {
    unsigned long long t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// True once the deadline taken at kernel entry has passed. A zero deadline disables the bound, so a
// timeout of 0 keeps the transport's behavior exactly as it was before the bound existed.
// A collective that gave up must leave its output in a defined state. This staging buffer is reused
// by every round, and a replayed captured graph reaches the same arena memory, so leaving the
// previous round's values in place lets a downstream consumer read stale data as an index and fault
// inside the window - where no host-side check can intervene. Zero is always a valid token id,
// candidate index and logit, so the round can finish and the convergence check turns the trip into
// the intended failure instead of an illegal address.
__device__ __forceinline__ void ar_abandon_output(uint4* out_groups, int begin, int end, int stride,
                                                  int thread) {
    for (int i = begin + thread; i < end; i += stride) { out_groups[i] = make_uint4(0, 0, 0, 0); }
}

__device__ __forceinline__ bool ar_expired(unsigned long long deadline) {
    return deadline != 0ULL && ar_now_ns() >= deadline;
}

// Pulls one block's slice of a prefetch hint into L2. Nothing reads the result: the point is the
// cache line, not a value. The loop stops as soon as the rendezvous no longer needs the wait, so a
// hint can never hold a collective past the time its exchange would have taken - the stop edge is
// the peer's arrival, published before this side reads its data.
__device__ __forceinline__ void ar_prefetch_window(const char* base, unsigned long long bytes,
                                                   unsigned int slice, unsigned int slices,
                                                   const int* stop) {
    if (base == nullptr || bytes == 0ULL || slice >= slices) { return; }
    constexpr unsigned long long kLine = 128ULL;
    const unsigned long long begin     = (bytes * slice) / slices;
    const unsigned long long end       = (bytes * (slice + 1)) / slices;
    const unsigned long long span      = kLine * static_cast<unsigned long long>(blockDim.x);
    for (unsigned long long offset = begin + kLine * threadIdx.x; offset < end; offset += span) {
        if (*reinterpret_cast<const volatile int*>(stop) != 0) { return; }
        asm volatile("prefetch.global.L2 [%0];" ::"l"(base + offset));
    }
}

// E4M3 wire codec (ArCodec::Fp8E4m3): one f32 scale per 128-element block, carried after the
// payload in the same staging slot. The producer stages encode(v * 448 / amax); the receiver
// rebuilds the identical quantized value from its own bf16 half, so both shards consume
// Q(local) + Q(peer) and stay bit-identical (see DevicePair::allreduce). A whole-zero block encodes
// to zero and stores a zero scale, which decodes back to zero.
// One warp owns one block and each lane stages four consecutive elements, so the wire store is a
// single coalesced 128-byte run per warp. That mapping is load-bearing: a byte-per-thread mapping
// writes 32 scattered sectors per warp into mapped host memory and measured about four times slower
// end to end than bf16, so both the producer and the receiver stride by lane, not by block.
constexpr int kArWireBlockElements = 128; // one E4M3 byte per element
constexpr int kArWireLaneElements  = kArWireBlockElements / 32;

// The block's scale and its reciprocal, from the amax the warp agrees on.
__device__ __forceinline__ float ar_wire_scale(float amax) { return __fdiv_rn(amax, 448.0f); }

__device__ __forceinline__ float ar_wire_inv(float amax) {
    return amax > 0.0f ? __fdiv_rn(448.0f, amax) : 0.0f;
}

// This lane's four elements of a block and the amax all 32 lanes agree on.
__device__ __forceinline__ float4 ar_wire_lane_values(const __nv_bfloat16* block, int lane) {
    const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(block + lane * kArWireLaneElements);
    const float2 lo   = __bfloat1622float2(pairs[0]);
    const float2 hi   = __bfloat1622float2(pairs[1]);
    return make_float4(lo.x, lo.y, hi.x, hi.y);
}

__device__ __forceinline__ float ar_wire_amax(float4 values) {
    float amax = fmaxf(fmaxf(fabsf(values.x), fabsf(values.y)),
                       fmaxf(fabsf(values.z), fabsf(values.w)));
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    return amax;
}

// Stages one block: four elements per lane, packed into one 32-bit store.
__device__ __forceinline__ void ar_stage_block(const __nv_bfloat16* block, __nv_fp8_e4m3* payload,
                                               float* scale_out, int lane) {
    const float4 values = ar_wire_lane_values(block, lane);
    const float amax    = ar_wire_amax(values);
    const float inv     = ar_wire_inv(amax);
    const __nv_fp8_storage_t bytes[4] = {
        __nv_cvt_float_to_fp8(__fmul_rn(values.x, inv), __NV_SATFINITE, __NV_E4M3),
        __nv_cvt_float_to_fp8(__fmul_rn(values.y, inv), __NV_SATFINITE, __NV_E4M3),
        __nv_cvt_float_to_fp8(__fmul_rn(values.z, inv), __NV_SATFINITE, __NV_E4M3),
        __nv_cvt_float_to_fp8(__fmul_rn(values.w, inv), __NV_SATFINITE, __NV_E4M3)};
    *reinterpret_cast<std::uint32_t*>(payload + lane * kArWireLaneElements) =
        static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8) |
        (static_cast<std::uint32_t>(bytes[2]) << 16) | (static_cast<std::uint32_t>(bytes[3]) << 24);
    if (lane == 0) { *scale_out = ar_wire_scale(amax); }
}

// Q(local) + Q(peer) for two consecutive elements, rounded exactly as the producer staged it.
__device__ __forceinline__ __nv_bfloat162 ar_wire_add_pair(float mine0, float mine1, float inv,
                                                           float scale, const __nv_fp8_e4m3* peer,
                                                           int index, float peer_scale) {
    // Every product and the final add are explicitly rounded: an FMA from Q(local)*scale + Q(peer)
    // would disagree with the producer's separately rounded product and the two shards would then
    // differ by one ulp on exact ties.
    const float mine_q0 =
        __fmul_rn(static_cast<float>(__nv_fp8_e4m3(__fmul_rn(mine0, inv))), scale);
    const float mine_q1 =
        __fmul_rn(static_cast<float>(__nv_fp8_e4m3(__fmul_rn(mine1, inv))), scale);
    const float peer0 = __fmul_rn(static_cast<float>(peer[index]), peer_scale);
    const float peer1 = __fmul_rn(static_cast<float>(peer[index + 1]), peer_scale);
    return __floats2bfloat162_rn(__fadd_rn(mine_q0, peer0), __fadd_rn(mine_q1, peer1));
}

// Adds one block of the peer's payload to the local half, four elements per lane.
__device__ __forceinline__ void ar_consume_block(const __nv_bfloat16* mine,
                                                 const __nv_fp8_e4m3* peer,
                                                 const float* peer_scales, __nv_bfloat16* dst, int b,
                                                 int lane) {
    const float4 values    = ar_wire_lane_values(mine, lane);
    const float amax       = ar_wire_amax(values);
    const float inv        = ar_wire_inv(amax);
    const float scale      = ar_wire_scale(amax);
    const float peer_scale = peer_scales[b];
    const __nv_fp8_e4m3* staged = peer + b * kArWireBlockElements + lane * kArWireLaneElements;
    auto* out = reinterpret_cast<__nv_bfloat162*>(dst + b * kArWireBlockElements +
                                                  lane * kArWireLaneElements);
    out[0] = ar_wire_add_pair(values.x, values.y, inv, scale, staged, 0, peer_scale);
    out[1] = ar_wire_add_pair(values.z, values.w, inv, scale, staged, 2, peer_scale);
}

template <bool kAdd, ArCodec kCodec = ArCodec::Bf16>
__global__ void ar_exchange(const __nv_bfloat16* local, __nv_bfloat16* out,
                            char* host_mine_base, const char* host_other_base, int count,
                            unsigned long long* arrival_mine, unsigned long long* arrival_other,
                            unsigned long long* order_mine, const unsigned long long* base_dev,
                            unsigned int call_index, unsigned long long base_value, int slot_bytes,
                            int groups, int* stall_mine, int* stall_peer,
                            unsigned long long timeout_ns, unsigned int bank_mask,
                            unsigned long long hold_ns, const char* prefetch_mine,
                            unsigned long long prefetch_bytes, int ar_blocks) {
    // The id is a device load of a cell the host filled before the launch, plus the ordinal this
    // kernel was captured with. It is fixed for the whole call, so one read per block, published
    // through shared memory, is enough.
    __shared__ unsigned long long shared_token;
    // Cooperative give-up: thread 0 is the only spinner, so a timeout is published through shared
    // memory and every thread leaves the block together. Returning from one thread alone would
    // deadlock the block at its next __syncthreads.
    __shared__ int stalled;
    // Set by the spinner once this block's exchange is over; the prefetching threads read it to stop.
    __shared__ int prefetch_stop;
    const unsigned long long deadline = timeout_ns == 0ULL ? 0ULL : ar_now_ns() + timeout_ns;
    const int stride                 = blockDim.x;
    const int lane                   = threadIdx.x & 31u;
    const int warp                   = static_cast<int>(threadIdx.x >> 5);
    const int warps                  = static_cast<int>(blockDim.x >> 5);
    const int blocks                 = ar_blocks;
    // This block's contiguous slice of the 16-byte groups. Slice bounds are derived from the block
    // index alone, so every block on this device and its peer agree on which slot covers which data.
    // Computed before the entry check below because an abandoned call still has to fill its slice.
    const int begin = static_cast<int>((static_cast<long long>(groups) * blockIdx.x) / blocks);
    const int end =
        static_cast<int>((static_cast<long long>(groups) * (blockIdx.x + 1)) / blocks);
    auto* out_groups = reinterpret_cast<uint4*>(out);
    // A tripped pair is one call out of step, so every later rendezvous would wait out its whole
    // deadline: without this entry check a desynchronized round pays one timeout per allreduce
    // (about 128 of them) before the host can react. Both flags are read, because either side may
    // have been the one that gave up.
    if (threadIdx.x == 0) {
        stalled       = (*stall_mine != 0 || *stall_peer != 0) ? 1 : 0;
        prefetch_stop = 0;
        if (stalled == 0) {
            // A captured collective reads the cell its graph's memcpy node filled and adds the
            // ordinal baked into this kernel; an eager launch carries its id in the launch
            // arguments, which are fresh on every launch.
            shared_token = base_dev != nullptr ? *base_dev + call_index : base_value;
        }
    }
    __syncthreads();
    // A block past the exchange slices owns no part of the payload as this call defines it: it must
    // not touch the bank slots or the output, only fill the rendezvous window. Slice 0's arrival is
    // published by the first exchange block as soon as the peer's data is readable and the remaining
    // slices land inside the same window, so that arrival is the window's closing edge.
    if (blockIdx.x >= static_cast<unsigned int>(blocks)) {
        if (stalled != 0) { return; }
        const int bank = static_cast<int>(shared_token & static_cast<unsigned long long>(bank_mask));
        const auto* peer_cell =
            arrival_other + static_cast<std::size_t>(bank) * DevicePair::kArSliceSlots;
        if (threadIdx.x == 0) {
            while (*(const volatile unsigned long long*)(peer_cell) != shared_token) {
                if (ar_expired(deadline)) { break; }
                __nanosleep(100);
            }
            *(volatile int*)(&prefetch_stop) = 1;
            return;
        }
        ar_prefetch_window(prefetch_mine, prefetch_bytes, blockIdx.x, gridDim.x, &prefetch_stop);
        return;
    }
    if (stalled != 0) { ar_abandon_output(out_groups, begin, end, stride, threadIdx.x); return; }
    const unsigned long long token = shared_token;
    constexpr bool kFp8            = kCodec == ArCodec::Fp8E4m3;
    static_assert(!kFp8 || kAdd, "the fp8 codec has no byte-exact copy form");
    // Fp8 stages whole 128-element blocks (one E4M3 byte each) with a trailing f32 scale array, so
    // its ownership bounds are block indices; the bf16 path keeps the 16-byte group bounds below.
    const int wire_blocks = kFp8 ? count / kArWireBlockElements : 0;
    const int wire_begin =
        kFp8 ? static_cast<int>((static_cast<long long>(wire_blocks) * blockIdx.x) / blocks) : 0;
    const int wire_end = kFp8 ? static_cast<int>((static_cast<long long>(wire_blocks) *
                                                 (blockIdx.x + 1)) /
                                                blocks)
                              : 0;
    const int tail   = kFp8 ? 0 : groups * 8;
    // This collective's arrival bank. Consecutive collectives rotate banks, so the id a slow side is
    // waiting for is still in the bank it published under when it finally looks, even though the peer
    // has already published the next collective into another bank.
    const int bank = static_cast<int>(token & static_cast<unsigned long long>(bank_mask));
    const auto bank_offset = static_cast<std::size_t>(bank) * DevicePair::kArSliceSlots;
    auto* arrival_mine_bank        = arrival_mine + bank_offset;
    const auto* arrival_other_bank = arrival_other + bank_offset;
    auto* order_mine_bank          = order_mine + bank_offset;
    char* const mine_base =
        host_mine_base + static_cast<std::size_t>(token & 1) * static_cast<std::size_t>(slot_bytes);
    const char* const other_base =
        host_other_base + static_cast<std::size_t>(token & 1) * static_cast<std::size_t>(slot_bytes);
    auto* host_mine        = reinterpret_cast<__nv_bfloat16*>(mine_base);
    const auto* host_other = reinterpret_cast<const __nv_bfloat16*>(other_base);
    auto* mine_groups         = reinterpret_cast<uint4*>(host_mine);
    const auto* local_groups  = reinterpret_cast<const uint4*>(local);
    const auto* other_groups  = reinterpret_cast<const uint4*>(host_other);
    if (blocks > 1 && blockIdx.x > 0 && threadIdx.x == 0) {
        while (*(const volatile unsigned long long*)(order_mine_bank + blockIdx.x - 1) != token) {
            if (ar_expired(deadline)) {
                stalled    = 1;
                *(volatile int*)stall_mine = 1;
                break;
            }
            __nanosleep(100);
        }
    }
    if (blocks > 1) {
        __syncthreads();
        if (stalled != 0) { ar_abandon_output(out_groups, begin, end, stride, threadIdx.x); return; }
    }
    if constexpr (kFp8) {
        auto* payload = reinterpret_cast<__nv_fp8_e4m3*>(mine_base);
        auto* scales  = reinterpret_cast<float*>(mine_base + count);
        for (int b = wire_begin + warp; b < wire_end; b += warps) {
            ar_stage_block(local + b * kArWireBlockElements, payload + b * kArWireBlockElements,
                           scales + b, lane);
        }
    } else {
        for (int i = begin + threadIdx.x; i < end; i += stride) { mine_groups[i] = local_groups[i]; }
        if (blockIdx.x == 0) {
            for (int i = tail + threadIdx.x; i < count; i += stride) { host_mine[i] = local[i]; }
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *(volatile unsigned long long*)(order_mine_bank + blockIdx.x)  = token;
        *(volatile unsigned long long*)(arrival_mine_bank + blockIdx.x) = token;
        __threadfence_system();
        // Diagnostics: delay this side's arrival poll, so the peer completes this collective and
        // publishes the next one before this side ever looks. That is the skew the banks survive, and
        // the hold is taken after the arrival is published, so the peer can always finish this one.
        if (hold_ns != 0ULL) {
            const unsigned long long until = ar_now_ns() + hold_ns;
            while (ar_now_ns() < until) { __nanosleep(1000); }
        }
        while (*(const volatile unsigned long long*)(arrival_other_bank + blockIdx.x) != token) {
            if (ar_expired(deadline)) {
                stalled    = 1;
                *(volatile int*)stall_mine = 1;
                break;
            }
            __nanosleep(100);
        }
        *(volatile int*)(&prefetch_stop) = 1;
    } else if (prefetch_bytes != 0ULL) {
        // The wait is the window: the memory system is idle on this side while the peer's arrival is
        // in flight, and the next kernel's weight stream is independent of this reduction, so it is
        // fetched here instead of on that kernel's critical path. The stop flag is this block's own
        // arrival, so the hint can never hold the collective past its rendezvous.
        ar_prefetch_window(prefetch_mine, prefetch_bytes, blockIdx.x, gridDim.x, &prefetch_stop);
    }
    __syncthreads();
    if (stalled != 0) { ar_abandon_output(out_groups, begin, end, stride, threadIdx.x); return; }
    __threadfence_system();
    if constexpr (kFp8) {
        const auto* peer_payload = reinterpret_cast<const __nv_fp8_e4m3*>(other_base);
        const auto* peer_scales  = reinterpret_cast<const float*>(other_base + count);
        for (int b = wire_begin + warp; b < wire_end; b += warps) {
            ar_consume_block(local + b * kArWireBlockElements, peer_payload, peer_scales, out, b,
                             lane);
        }
    } else {
        int i = begin + threadIdx.x;
        for (; i + 3 * stride < end; i += 4 * stride) {
            const uint4 o0 = other_groups[i];
            const uint4 o1 = other_groups[i + stride];
            const uint4 o2 = other_groups[i + 2 * stride];
            const uint4 o3 = other_groups[i + 3 * stride];
            out_groups[i]              = kAdd ? add_bf16x8(local_groups[i], o0) : o0;
            out_groups[i + stride]     = kAdd ? add_bf16x8(local_groups[i + stride], o1) : o1;
            out_groups[i + 2 * stride] = kAdd ? add_bf16x8(local_groups[i + 2 * stride], o2) : o2;
            out_groups[i + 3 * stride] = kAdd ? add_bf16x8(local_groups[i + 3 * stride], o3) : o3;
        }
        for (; i < end; i += stride) {
            out_groups[i] = kAdd ? add_bf16x8(local_groups[i], other_groups[i]) : other_groups[i];
        }
        if (blockIdx.x == 0) {
            for (int j = tail + threadIdx.x; j < count; j += stride) {
                out[j] = kAdd ? __hadd(local[j], host_other[j]) : host_other[j];
            }
        }
    }
}

// Staging capacity per device, per parity slot. A batched prefill reduces BF16 payloads per layer:
// the mixer's and the FFN's [hidden, T] deltas and, for a GGUF block checkpoint whose mixer output
// projection gathers its input columns, a [value_width, T] activation (value_width=6144 is the
// widest). 48 MiB carries the maximum prefill chunk (1024 tokens: a 6144-wide payload is 12 MiB)
// with a factor of four to spare, and the TP-2 core narrows the chunk to whatever the buffer holds
// (TP2GenerationCore::prefill_chunk_width) rather than letting a wider one fall back to the
// host-staging transport, which synchronizes both compute streams on every collective.
constexpr std::size_t kInKernelArBytes = 48ULL << 20; // 48 MiB staging per device, per buffer

// The staging ladder, largest rung first. The reservation is per device and two parity slots deep,
// so a host whose mapped-host window cannot hold the full size keeps the in-kernel transport at the
// largest rung both devices can map instead of dropping every collective to the host-staging
// fallback, which synchronizes both compute streams per collective.
// The floor is 12 MiB, the capacity at which nothing in this product narrows. The widest all-reduce
// the TP-2 core issues is a full prefill chunk at the widest activation the route reduces (1024
// tokens x 6144 BF16 columns = 12 MiB exactly), and the widest payload a *captured* graph can
// contain is the vocabulary-split head merge over a whole verify window (vocab x (drafts + 1) x 2 =
// 7.6 MiB for the 27B's 248320-row head at K = 15). A rung below the floor would have to narrow the
// prefill chunk (TP2GenerationCore::prefill_chunk_width) or refuse a capture (see
// require_capturable_payload), so the ladder stops there and such a host keeps the fallback.
constexpr std::size_t kInKernelArLadder[] = {kInKernelArBytes, kInKernelArBytes / 2,
                                             kInKernelArBytes / 4};
constexpr int kArThreads               = 1024;
constexpr int kArMaxSlices             = 8;
// A slot holds one rendezvous id as an unsigned long long: the ids are host-authored and never
// repeat over the pair's lifetime, which is only expressible in 64 bits - a 32-bit slot would let a
// stale value from one id range match a future id from another. One bank is a cache line of slice
// slots; the pair lays out its arrival banks and their write-order chains through ar_banks(),
// ar_order_base() and ar_token_bytes(), and the kernel indexes them with the launch's bank mask.
static_assert(DevicePair::kArSliceSlots == static_cast<std::size_t>(kArMaxSlices),
              "the kernel's slice stride and the pair's arrival-bank geometry have to agree");
static_assert(DevicePair::kArBankBytes ==
                  static_cast<std::size_t>(kArMaxSlices) * sizeof(unsigned long long),
              "one bank is a cache line of slice slots");
// One cache line per side's bounded-spin flag: the mapped host path is not coherent between peers,
// so two flags sharing one line would let a peer's line write-back drop the other's store.
constexpr std::size_t kArTokenStrideBytes = 64;

// Decode-sized staging for the size-keyed transport. A single token's [hidden, 1] delta is a few
// KiB: the 24 MiB prefill slots are almost all waste, and sharing them would let a small call's
// parity slot alias a prefill payload the peer is still reading. This buffer holds one small
// payload per parity instead.
constexpr std::size_t kArSmallBytes = 64ULL << 10; // staging per device, per parity

// Slices per allreduce. A decode-sized payload is a latency problem and rides one slice. A
// prefill-sized payload is split into 512 KiB slices whose writes are ordered by the kernel's own
// chain, which lets the peer read early slices while the rest is still in flight; measured on the
// RTX 5060 Ti pair (T=1024: 10 MiB delta, device 1 behind a Gen4 x4 chipset link) this took the
// allreduce from 3.18 ms to 2.70 ms, essentially the link's two-way bandwidth bound (21 MiB at
// 7.9 GB/s = 2.66 ms). More slices lose to the per-slice handshake latency (16 slices: 3.84 ms,
// 32 slices: 6.24 ms), so the count stays small.
int ar_slices(std::size_t count_bytes) {
    if (count_bytes <= (256ULL << 10)) { return 1; }
    const std::size_t want = (count_bytes + (512ULL << 10) - 1) / (512ULL << 10);
    return static_cast<int>(std::min<std::size_t>(std::max<std::size_t>(want, 2),
                                                  static_cast<std::size_t>(kArMaxSlices)));
}

// Blocks the rendezvous window may add on top of the exchange slices. Each one streams one slice of
// the hint into L2, a line per thread per pass, so a handful already saturates the memory system
// while the exchange block waits on the mapped-host round trip. Zero disables the window.
constexpr int kArPrefetchBlocks = 4;

int ar_prefetch_blocks(std::size_t bytes) { return bytes == 0 ? 0 : kArPrefetchBlocks; }

void require_bytes(std::size_t count_bytes) {
    if (count_bytes == 0) { return; }
    if (count_bytes % 16 != 0) {
        throw std::invalid_argument(std::string("tp allreduce: byte count ") + std::to_string(count_bytes) + " must be a multiple of 16");
    }
}

} // namespace

// Watched state for the spin diagnostics: the host-visible token counters and the two arrival/order
// arrays of the in-kernel transport. The watchdog thread reads them with memcpy because the devices
// write them concurrently.
struct DevicePair::ArWatchState {
    const std::atomic<unsigned long long>* id = nullptr;
    // The two bounded-spin trip flags, read exactly as ar_stalled() reads them. They are what
    // separates a give-up from the idle gap between requests, which also stops the call counter.
    const int* stall = nullptr;
    const unsigned char* arrival_a = nullptr;
    const unsigned char* arrival_b = nullptr;
    // Arrival banks the pair publishes under: the dump reports the newest bank's slots, so its
    // one-line format keeps meaning what the interpreter reads it as.
    unsigned int banks = 1;
    std::atomic<unsigned long long> calls      = 0;
    std::atomic<unsigned long long> last_bytes = 0;
    std::atomic<bool> stop = false;
    std::thread thread;
};

// Stops and joins this pair's watchdog before the mapped arrays are freed. The thread holds only a
// raw pointer to the state block, which outlives it because the join happens first.
void DevicePair::stop_ar_watchdog() {
    if (!watch_) { return; }
    watch_->stop.store(true);
    if (watch_->thread.joinable()) { watch_->thread.join(); }
}

namespace {

// One arrival/order slot: an id per slice at the front of the bank, the write-order chain for the
// same slices at the same bank one bank-stride further in (see DevicePair::ar_order_base). Read with
// memcpy because the devices write them while the watchdog thread reads.
unsigned long long watch_slot(const unsigned char* base, int index) {
    unsigned long long value = 0;
    std::memcpy(&value, base + static_cast<std::size_t>(index) * sizeof(value), sizeof(value));
    return value;
}

// Slot index of the newest bank a side published into: the banks rotate per collective, so the one
// whose slice-0 slot holds the largest id is the last one written.
int newest_bank(const unsigned char* base, unsigned int banks) {
    int head = 0;
    for (unsigned int bank = 1; bank < banks; ++bank) {
        const int candidate = static_cast<int>(bank) * kArMaxSlices;
        if (watch_slot(base, candidate) > watch_slot(base, head)) { head = candidate; }
    }
    return head;
}

// Renders eight slots for the watchdog's single-line dump.
void format_slots(const unsigned long long* values, char* out, std::size_t size) {
    int written = 0;
    for (int i = 0; i < 8; ++i) {
        written += std::snprintf(out + written, size - static_cast<std::size_t>(written), "%s%llu",
                                 i == 0 ? "" : " ", values[i]);
    }
}

} // namespace

void DevicePair::note_ar_call(std::size_t count_bytes) {
    if (!watch_) { return; }
    watch_->last_bytes.store(count_bytes, std::memory_order_relaxed);
    watch_->calls.fetch_add(1, std::memory_order_relaxed);
}

// Fault injection: only an armed process counts collectives, so an unarmed process pays nothing.
// The skip hook counts every in-kernel collective, captures included, as it always has. The hold
// counts eager collectives only: a hold recorded into a captured graph would be replayed by every
// launch of that graph (the kernel carries no serial), and the skew it reproduces is the eager one
// the production dumps show.
DevicePair::ArFaults DevicePair::ar_faults() {
    ArFaults out;
    if (ar_fault_skip_call_ != 0) {
        ++ar_call_serial_;
        out.skip_peer = ar_call_serial_ == ar_fault_skip_call_;
        if (out.skip_peer) {
            std::fprintf(stderr, "[ar-fault] collective %llu: skipping the peer launch\n",
                         ar_call_serial_);
        }
    }
    if (ar_fault_hold_ns_ != 0 && !capturing_) {
        ++ar_fault_hold_serial_;
        if (ar_fault_hold_call_ == 0 || ar_fault_hold_serial_ == ar_fault_hold_call_) {
            if (ar_fault_hold_peer_) {
                out.hold_peer_ns = ar_fault_hold_ns_;
            } else {
                out.hold_self_ns = ar_fault_hold_ns_;
            }
            // The landing is otherwise invisible in a healthy arm, and the acceptance has to tell "the
            // injection ran and the banks carried it" from "the injection never ran".
            std::fprintf(stderr,
                         "[ar-fault] eager collective %llu: holding the %s arrival poll for %llu ns\n",
                         ar_fault_hold_serial_, ar_fault_hold_peer_ ? "peer" : "self",
                         ar_fault_hold_ns_);
        }
    }
    return out;
}

bool DevicePair::ar_stalled() const noexcept {
    if (stall_host_ == nullptr) { return false; }
    const auto* flags = reinterpret_cast<const int*>(stall_host_);
    return flags[0] != 0 || flags[kArTokenStrideBytes / sizeof(int)] != 0;
}

std::uint64_t DevicePair::ar_last_id() const noexcept {
    return static_cast<std::uint64_t>(ar_last_id_.load(std::memory_order_relaxed));
}

void DevicePair::clear_ar_stall() noexcept {
    if (stall_host_ == nullptr) { return; }
    std::memset(stall_host_, 0, 2 * kArTokenStrideBytes);
    // The arrival and write-order slots need no repair, and deliberately get none: no id is ever
    // handed out twice, so a slot left over from the stalled round can never satisfy a later spin.
    // That is what makes a stalled round recoverable in place instead of needing a renumbering.
}

void DevicePair::reserve_ar_channels(std::size_t count) {
    if (count <= ar_channel_capacity_) { return; }
    // Without the in-kernel transport there are no id cells to grow; create_ar_channel already
    // refuses, so this is a no-op rather than an error.
    if (base_host_ == nullptr && base_dev_a_ == nullptr && base_dev_b_ == nullptr) { return; }
    if (!ar_channels_.empty()) {
        throw std::logic_error(
            "tp allreduce: the rendezvous id space cannot grow once a channel exists");
    }
    unsigned long long* host = nullptr;
    if (cudaHostAlloc(reinterpret_cast<void**>(&host), count * sizeof(unsigned long long),
                      cudaHostAllocPortable) != cudaSuccess) {
        (void)cudaGetLastError();
        throw std::runtime_error("tp allreduce: failed to reserve the rendezvous id cells");
    }
    std::memset(host, 0, count * sizeof(unsigned long long));
    a_.bind_to_current_thread();
    void* dev_a = nullptr;
    if (cudaMalloc(&dev_a, count * sizeof(unsigned long long)) != cudaSuccess) {
        (void)cudaGetLastError();
        cudaFreeHost(host);
        throw std::runtime_error("tp allreduce: failed to reserve the device-a rendezvous ids");
    }
    b_.bind_to_current_thread();
    void* dev_b = nullptr;
    if (cudaMalloc(&dev_b, count * sizeof(unsigned long long)) != cudaSuccess) {
        (void)cudaGetLastError();
        a_.bind_to_current_thread();
        cudaFree(dev_a);
        cudaFreeHost(host);
        throw std::runtime_error("tp allreduce: failed to reserve the device-b rendezvous ids");
    }
    if (base_host_ != nullptr) { cudaFreeHost(base_host_); }
    if (base_dev_a_ != nullptr) {
        a_.bind_to_current_thread();
        cudaFree(base_dev_a_);
    }
    if (base_dev_b_ != nullptr) {
        b_.bind_to_current_thread();
        cudaFree(base_dev_b_);
    }
    base_host_           = host;
    base_dev_a_          = dev_a;
    base_dev_b_          = dev_b;
    ar_channel_capacity_ = count;
    std::fprintf(stderr, "[mem] TP-2 rendezvous id channels reserved: %zu\n", count);
}

DevicePair::ArChannel DevicePair::create_ar_channel() {
    if (base_host_ == nullptr || base_dev_a_ == nullptr || base_dev_b_ == nullptr) {
        throw std::logic_error("tp allreduce: the in-kernel transport is unavailable");
    }
    if (ar_channels_.size() >= ar_channel_capacity_) {
        throw std::logic_error("tp allreduce: the captured graphs exceed the reserved rendezvous id channels");
    }
    const std::size_t index = ar_channels_.size();
    ArChannelState state;
    state.host  = base_host_ + index;
    state.dev_a = static_cast<char*>(base_dev_a_) + index * sizeof(unsigned long long);
    state.dev_b = static_cast<char*>(base_dev_b_) + index * sizeof(unsigned long long);
    // Channel k owns ids from (k + 1) << 32 upward, so the channels' blocks cannot overlap and
    // every id a channel hands out is larger than the ones it handed out before.
    state.next_base = (static_cast<std::uint64_t>(index) + 1) << 32;
    ar_channels_.push_back(state);
    return static_cast<ArChannel>(ar_channels_.size());
}

void DevicePair::begin_capture(ArChannel channel) {
    if (capturing_) { throw std::logic_error("tp allreduce: a capture is already open"); }
    if (channel == kNoArChannel || channel > ar_channels_.size()) {
        throw std::invalid_argument("tp allreduce: unknown rendezvous id channel");
    }
    capturing_            = true;
    capture_channel_      = channel;
    capture_index_        = 0;
    ArChannelState& state = ar_channels_[channel - 1];
    state.seen_a          = false;
    state.seen_b          = false;
}

void DevicePair::end_capture() {
    if (!capturing_) { return; }
    ar_channels_[capture_channel_ - 1].calls = capture_index_;
    capturing_                               = false;
    capture_channel_                         = kNoArChannel;
}

void DevicePair::arm_round(ArChannel channel) {
    if (channel == kNoArChannel || channel > ar_channels_.size()) {
        throw std::invalid_argument("tp allreduce: unknown rendezvous id channel");
    }
    ArChannelState& state = ar_channels_[channel - 1];
    *state.host           = static_cast<unsigned long long>(state.next_base);
    ar_last_id_.store(state.next_base, std::memory_order_relaxed);
    // Advance past this launch's block even when the capture numbered no collective: handing the
    // same block to a later launch would let a stale arrival slot satisfy its spin.
    state.next_base += state.calls == 0 ? 1 : state.calls;
}

DevicePair::CollectiveId DevicePair::acquire_collective_id(cudaStream_t stream_a,
                                                           cudaStream_t stream_b) {
    CollectiveId id;
    if (!capturing_) {
        // Eager: the id travels in the launch arguments, so there is nothing the next call could
        // overwrite before this kernel runs.
        id.value = ar_eager_next_++;
        ar_last_id_.store(id.value, std::memory_order_relaxed);
        return id;
    }
    ArChannelState& state = ar_channels_[capture_channel_ - 1];
    // The base has to arrive through the graph rather than as a kernel argument: a baked id would be
    // replayed unchanged, and a stale arrival slot holding it would satisfy the spin without the
    // peer having written anything.
    if (!state.seen_a) {
        a_.bind_to_current_thread();
        CUDA_CHECK(cudaMemcpyAsync(state.dev_a, state.host, sizeof(unsigned long long),
                                   cudaMemcpyHostToDevice, stream_a));
        state.seen_a = true;
    }
    if (!state.seen_b) {
        b_.bind_to_current_thread();
        CUDA_CHECK(cudaMemcpyAsync(state.dev_b, state.host, sizeof(unsigned long long),
                                   cudaMemcpyHostToDevice, stream_b));
        state.seen_b = true;
    }
    id.base_a     = static_cast<const unsigned long long*>(state.dev_a);
    id.base_b     = static_cast<const unsigned long long*>(state.dev_b);
    id.call_index = capture_index_++;
    return id;
}

void DevicePair::start_ar_watchdog() {
    if (!watch_) { return; }
    // On by default. The per-collective counter this thread reads is incremented whether or not the
    // thread runs (note_ar_call is gated only on watch_ being allocated, which the in-kernel
    // transport always does), and the thread itself only reads host memory, so the diagnostics add
    // nothing measurable to the hot path. Running them always is what turns a rare give-up into
    // evidence: every natural occurrence so far left no dump because the thread was opt-in.
    const char* env = std::getenv("NINFER_TP2_AR_WATCHDOG");
    if (env != nullptr && env[0] == '0') { return; }
    watch_->stall      = reinterpret_cast<const int*>(stall_host_);
    ArWatchState* raw = watch_.get();
    watch_->thread    = std::thread([raw] {
        // The mapped arrays are read only while the transport is provably stalled (the host-side
        // call counter not advancing for a tick), so a healthy run's token/arrival lines see no
        // extra traffic from the diagnostics themselves.
        unsigned long long last_calls = 0;
        bool stalled                  = false;
        while (!raw->stop.load(std::memory_order_relaxed)) {
            // A tripped device is usually followed by a fault and a process exit within a few
            // milliseconds, so the watchdog has to poll far faster than a human-scale interval to
            // catch the state that explains it. The thread is opt-in and only reads atomics.
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            if (raw->stop.load(std::memory_order_relaxed)) { break; }
            const unsigned long long calls = raw->calls.load(std::memory_order_relaxed);
            if (calls == last_calls) {
                // An idle gap between requests stops the call counter just like a give-up does, and
                // the model load satisfies this branch for its whole duration, so only a device that
                // actually tripped the bounded spin is worth a dump.
                const int* flags = raw->stall;
                const bool tripped =
                    flags != nullptr &&
                    (flags[0] != 0 || flags[kArTokenStrideBytes / sizeof(int)] != 0);
                if (!tripped || calls == 0) {
                    stalled = false;
                    continue;
                }
                if (stalled) {
                    // One detailed dump per stall episode. The state that classifies the failure is
                    // the state at detection, so repeating it every tick for the whole timeout only
                    // added volume: at the current 10 s bound that was about 400 lines per incident.
                    continue;
                }
                stalled = true;
                std::fprintf(stderr, "[ar-watch] stalled at calls=%llu\n", calls);
            } else {
                last_calls = calls;
                stalled    = false;
                continue;
            }
            const unsigned long long id = raw->id->load(std::memory_order_relaxed);
            // The newest bank is the one whose slice-0 slot holds the largest id (consecutive
            // collectives rotate banks), so that is the bank the dump reports: the interpreter reads
            // slot 0 as the last id a side published, and it must keep meaning that.
            const int stride = static_cast<int>(raw->banks) * kArMaxSlices;
            unsigned long long slots[4][8] = {};
            const int head_a = newest_bank(raw->arrival_a, raw->banks);
            const int head_b = newest_bank(raw->arrival_b, raw->banks);
            for (int i = 0; i < 8; ++i) {
                slots[0][i] = watch_slot(raw->arrival_a, head_a + i);
                slots[1][i] = watch_slot(raw->arrival_a, stride + head_a + i);
                slots[2][i] = watch_slot(raw->arrival_b, head_b + i);
                slots[3][i] = watch_slot(raw->arrival_b, stride + head_b + i);
            }
            char arr_a[160] = {}, ord_a[160] = {}, arr_b[160] = {}, ord_b[160] = {};
            format_slots(slots[0], arr_a, sizeof(arr_a));
            format_slots(slots[1], ord_a, sizeof(ord_a));
            format_slots(slots[2], arr_b, sizeof(arr_b));
            format_slots(slots[3], ord_b, sizeof(ord_b));
            std::fprintf(stderr,
                         "[ar-watch] calls=%llu last=%llu id=%llu arrA=[%s] ordA=[%s] "
                         "arrB=[%s] ordB=[%s]\n",
                         raw->calls.load(), raw->last_bytes.load(), id, arr_a, ord_a, arr_b, ord_b);
        }
    });}

// Reserves the staging ladder's largest rung both devices can map (see kInKernelArLadder). A failed
// rung frees the partial reservation before the next attempt: half a staging pair is worse than
// none, because the pair would look available while one side had nothing to publish into.
std::size_t DevicePair::reserve_in_kernel_staging() {
    // NINFER_TP2_AR_STAGING_MIB caps the ladder, which is how a host whose mapped window refuses a
    // rung is reproduced from a script (and how the size-keyed fallback is reached on a host whose
    // full staging takes every production payload).
    std::size_t ceiling = kInKernelArBytes;
    if (const char* cap = std::getenv("NINFER_TP2_AR_STAGING_MIB")) {
        const unsigned long long mib = std::strtoull(cap, nullptr, 10);
        if (mib != 0) {
            ceiling = static_cast<std::size_t>(mib) << 20;
        }
    }
    for (const std::size_t staging : kInKernelArLadder) {
        if (staging > ceiling) {
            continue;
        }
        a_.bind_to_current_thread();
        void* host_a = nullptr;
        void* dev_a  = nullptr;
        if (cudaHostAlloc(&host_a, 2 * staging,
                          cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer(&dev_a, host_a, 0) != cudaSuccess) {
            if (host_a != nullptr) {
                cudaFreeHost(host_a);
            }
            (void)cudaGetLastError();
            continue;
        }
        b_.bind_to_current_thread();
        void* host_b = nullptr;
        void* dev_b  = nullptr;
        if (cudaHostAlloc(&host_b, 2 * staging,
                          cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer(&dev_b, host_b, 0) != cudaSuccess) {
            if (host_b != nullptr) {
                cudaFreeHost(host_b);
            }
            (void)cudaGetLastError();
            a_.bind_to_current_thread();
            cudaFreeHost(host_a);
            continue;
        }
        host_a_ = host_a;
        host_b_ = host_b;
        dev_a_  = dev_a;
        dev_b_  = dev_b;
        return staging;
    }
    return 0;
}

DevicePair::DevicePair(int device_a, int device_b) : a_(device_a), b_(device_b) {
    if (device_a == device_b) {
        throw std::invalid_argument("tp DevicePair: devices must be distinct");
    }
    // 10 s: about 3700x the slowest legitimate collective (~2.7 ms for a full-width prefill payload;
    // decode collectives are 9-18 us). The bound exists only to break a *permanent* desync, which no
    // value resolves, so it belongs far above anything a healthy round can produce: a multi-second
    // stall is a host or driver event (WDDM scheduling, paging, driver contention), not a slow
    // kernel. The two costs are not symmetric - a false trip spends a whole request plus its
    // re-prefill, while the headroom only lengthens the wait on an incident that was going to fail
    // anyway, and natural incidents run about three in 1500 requests. Still bounded, so a genuine
    // desync cannot wedge the service indefinitely.
    ar_timeout_ns_ = 10000ULL * 1000000ULL;
    if (const char* timeout = std::getenv("NINFER_TP2_AR_TIMEOUT_MS")) {
        ar_timeout_ns_ = static_cast<unsigned long long>(std::strtoull(timeout, nullptr, 10)) *
                         1000000ULL;
    }
    // Fault injection for the bounded-spin acceptance run: skip the peer launch of one collective so
    // the pair desynchronizes exactly as a divergent call sequence would. Diagnostics only.
    if (const char* fault = std::getenv("NINFER_TP2_AR_FAULT_SKIP_PEER_CALL")) {
        ar_fault_skip_call_ = std::strtoull(fault, nullptr, 10);
    }
    // Fault injection for the skew acceptance: hold one side's arrival poll of an eager collective
    // (see set_ar_fault_hold_poll). "<eager serial>:<nanos>", with an optional ":self" for the a-side
    // launch; serial 0 holds every eager collective.
    if (const char* hold = std::getenv("NINFER_TP2_AR_FAULT_HOLD_POLL")) {
        char* split                     = nullptr;
        const unsigned long long serial = std::strtoull(hold, &split, 10);
        unsigned long long nanos        = 0;
        bool peer                       = true;
        if (split != nullptr && *split == ':') {
            char* end          = nullptr;
            nanos              = std::strtoull(split + 1, &end, 10);
            const char* suffix = end;
            if (suffix != nullptr && std::strncmp(suffix, ":self", 5) == 0) { peer = false; }
        }
        if (nanos != 0) {
            ar_fault_hold_call_ = serial;
            ar_fault_hold_ns_   = nanos;
            ar_fault_hold_peer_ = peer;
        }
    }
    // Diagnostics: exchange only 1/N of every in-kernel collective's partial (see the member).
    if (const char* divisor = std::getenv("NINFER_TP2_AR_PAYLOAD_DIVISOR")) {
        const long long requested = std::strtoll(divisor, nullptr, 10);
        if (requested >= 2) { ar_payload_divisor_ = static_cast<int>(requested > 64 ? 64 : requested); }
    }
    // Wire codec for the prefill call sites (see ArCodec). Read once: the decision changes the
    // arithmetic every collective returns, so it must not change between capture and replay. The
    // e4m3 wire is the default; "bf16" opts out and an unavailable in-kernel transport (the only
    // transport the codec has) is resolved below, once that fact is known.
    if (const char* codec = std::getenv("NINFER_TP2_AR_CODEC")) {
        const std::string_view requested(codec);
        if (requested.empty()) {
            // Unset-equivalent: keep the default.
        } else if (requested == "bf16") {
            ar_codec_          = ArCodec::Bf16;
            ar_codec_explicit_ = true;
        } else if (requested == "fp8" || requested == "fp8_e4m3") {
            ar_codec_          = ArCodec::Fp8E4m3;
            ar_codec_explicit_ = true;
        } else {
            throw std::runtime_error("NINFER_TP2_AR_CODEC must be bf16 or fp8, got '" +
                                     std::string(codec) + "'");
        }
    }
    // Diagnostics: keep every collective on the host-staging fallback (see force_host_staging_).
    // This is the only way to qualify that transport where the in-kernel one takes every call.
    if (const char* forced = std::getenv("NINFER_TP2_AR_FORCE_HOST_STAGING")) {
        force_host_staging_ = forced[0] != '0';
    }
    if (const char* threads = std::getenv("NINFER_TP2_AR_ADD_THREADS")) {
        host_add_threads_ = static_cast<unsigned>(std::strtoul(threads, nullptr, 10));
    }
    // Arrival banks: a power of two in 1..8, so the kernel can index with a mask. One bank is the
    // geometry the transport shipped with, which the skew acceptance uses as its positive control.
    if (const char* banks = std::getenv("NINFER_TP2_AR_BANKS")) {
        const long long requested = std::strtoll(banks, nullptr, 10);
        unsigned int count        = 1;
        while (count < 8U && static_cast<long long>(count) * 2 <= requested) { count *= 2U; }
        ar_bank_mask_ = count - 1U;
        std::fprintf(stderr, "[mem] TP-2 arrival banks: %u (NINFER_TP2_AR_BANKS=%lld)\n", count,
                     requested);
    }
    int can_a_to_b = 0;
    int can_b_to_a = 0;
    const cudaError_t err_a = cudaDeviceCanAccessPeer(&can_a_to_b, device_a, device_b);
    const cudaError_t err_b = cudaDeviceCanAccessPeer(&can_b_to_a, device_b, device_a);
    if (err_a == cudaSuccess && err_b == cudaSuccess && can_a_to_b && can_b_to_a) {
        a_.bind_to_current_thread();
        b_.bind_to_current_thread();
        CUDA_CHECK(cudaDeviceEnablePeerAccess(device_b, 0));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(device_a, 0));
        p2p_ = true;
    }
    if (!force_host_staging_) {
        // Set up the in-kernel allreduce path: mapped pinned staging buffers (one per device) plus
        // arrival tokens. cudaHostAllocMapped makes the host memory directly addressable by the
        // device; the staging reservation walks the ladder, so a host that cannot map the full size
        // keeps this transport at a smaller one instead of dropping to the host-staging fallback,
        // which synchronizes both compute streams per collective. When no rung maps, allreduce
        // falls back to that path.
        // The staging area is double buffered by allreduce call parity: the peer publishes its
        // arrival token before it reads its slice, so without two buffers a device could overwrite
        // the buffer the peer is still reading from the previous layer.
        const std::size_t staging = reserve_in_kernel_staging();
        if (staging != 0) {
            a_.bind_to_current_thread();
            if (cudaHostAlloc(&arrival_host_a_, ar_token_bytes(), cudaHostAllocPortable |
                                 cudaHostAllocMapped) == cudaSuccess &&
                cudaHostGetDevicePointer((void**)&arrival_a_, arrival_host_a_, 0) == cudaSuccess) {
                b_.bind_to_current_thread();
                if (cudaHostAlloc(&arrival_host_b_, ar_token_bytes(), cudaHostAllocPortable |
                                     cudaHostAllocMapped) == cudaSuccess &&
                    cudaHostGetDevicePointer((void**)&arrival_b_, arrival_host_b_, 0) == cudaSuccess) {
                    // The rendezvous id cells and the bounded-spin flags. A channel cell is
                    // host memory whose device scalars are written by the graph's memcpy node;
                    // the eager cell is mapped so the eager kernels read it directly.
                    bool ids = false;
                    if (cudaHostAlloc(reinterpret_cast<void**>(&stall_host_),
                                      2 * kArTokenStrideBytes, cudaHostAllocPortable |
                                          cudaHostAllocMapped) == cudaSuccess) {
                        std::memset(stall_host_, 0, 2 * kArTokenStrideBytes);
                        ids = cudaHostGetDevicePointer(reinterpret_cast<void**>(&stall_a_),
                                                       stall_host_, 0) == cudaSuccess;
                        if (ids) {
                            stall_b_ = reinterpret_cast<int*>(
                                reinterpret_cast<char*>(static_cast<void*>(stall_a_)) +
                                kArTokenStrideBytes);
                        } else {
                            stall_a_ = nullptr;
                        }
                    } else {
                        stall_host_ = nullptr;
                    }
                    if (ids) {
                        ids = cudaHostAlloc(reinterpret_cast<void**>(&base_host_),
                                            ar_channel_capacity_ * sizeof(unsigned long long),
                                            cudaHostAllocPortable) == cudaSuccess;
                        if (ids) {
                            std::memset(base_host_, 0,
                                        ar_channel_capacity_ * sizeof(unsigned long long));
                            a_.bind_to_current_thread();
                            ids = cudaMalloc(&base_dev_a_, ar_channel_capacity_ *
                                                               sizeof(unsigned long long)) ==
                                  cudaSuccess;
                            b_.bind_to_current_thread();
                            ids = ids && cudaMalloc(&base_dev_b_, ar_channel_capacity_ *
                                                                      sizeof(unsigned long long)) ==
                                             cudaSuccess;
                        }
                    }
                    if (ids) {
                        std::memset(arrival_host_a_, 0, ar_token_bytes());
                        std::memset(arrival_host_b_, 0, ar_token_bytes());
                        in_kernel_bytes_ = staging;
                        in_kernel_available_ = true;
                        if (staging != kInKernelArBytes) {
                            std::fprintf(stderr,
                                         "[mem] TP-2 in-kernel allreduce staging %.1f MiB of %.1f "
                                         "(larger rungs did not map)\n",
                                         static_cast<double>(staging) / 1048576.0,
                                         static_cast<double>(kInKernelArBytes) / 1048576.0);
                        }
                        if (ar_payload_divisor_ > 1) {
                            std::fprintf(stderr,
                                         "[mem] TP-2 allreduce payload divisor %d (diagnostic: only "
                                         "1/%d of every collective crosses PCIe)\n",
                                         ar_payload_divisor_, ar_payload_divisor_);
                        }
                    }
                }
            }
        }
        // Decode-sized staging, sized so its two parity slots never alias the prefill-sized ones.
        if (in_kernel_available_) {
            a_.bind_to_current_thread();
            if (cudaHostAlloc(&small_host_a_, 2 * kArSmallBytes,
                              cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess &&
                cudaHostGetDevicePointer(&small_dev_a_, small_host_a_, 0) == cudaSuccess) {
                b_.bind_to_current_thread();
                if (cudaHostAlloc(&small_host_b_, 2 * kArSmallBytes,
                                  cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess &&
                    cudaHostGetDevicePointer(&small_dev_b_, small_host_b_, 0) == cudaSuccess) {
                    small_available_ = true;
                }
            }
        }
        if (!small_available_) {
            if (small_host_a_) { cudaFreeHost(small_host_a_); small_host_a_ = nullptr; small_dev_a_ = nullptr; }
            if (small_host_b_) { cudaFreeHost(small_host_b_); small_host_b_ = nullptr; small_dev_b_ = nullptr; }
        }
        // On any failure, free whatever was allocated and stay on host staging.
        if (!in_kernel_available_) {
            if (host_a_) { cudaFreeHost(host_a_); host_a_ = nullptr; dev_a_ = nullptr; }
            if (host_b_) { cudaFreeHost(host_b_); host_b_ = nullptr; dev_b_ = nullptr; }
            if (arrival_host_a_) { cudaFreeHost(arrival_host_a_); arrival_host_a_ = nullptr; arrival_a_ = nullptr; }
            if (arrival_host_b_) { cudaFreeHost(arrival_host_b_); arrival_host_b_ = nullptr; arrival_b_ = nullptr; }
            if (stall_host_) { cudaFreeHost(stall_host_); stall_host_ = nullptr; }
            stall_a_ = nullptr;
            stall_b_ = nullptr;
            if (base_host_) { cudaFreeHost(base_host_); base_host_ = nullptr; }
            if (base_dev_a_) {
                a_.bind_to_current_thread_noexcept();
                cudaFree(base_dev_a_);
                base_dev_a_ = nullptr;
            }
            if (base_dev_b_) {
                b_.bind_to_current_thread_noexcept();
                cudaFree(base_dev_b_);
                base_dev_b_ = nullptr;
            }
        }
    }
    if (in_kernel_available_) {
        watch_            = std::make_shared<ArWatchState>();
        watch_->id        = &ar_last_id_;
        watch_->arrival_a = static_cast<const unsigned char*>(arrival_host_a_);
        watch_->arrival_b = static_cast<const unsigned char*>(arrival_host_b_);
        watch_->banks     = static_cast<unsigned int>(ar_banks());
        start_ar_watchdog();
    }
    // The codec has no host-staging form: that transport reduces on the host and would have to
    // quantize both halves there, which is exactly the fallback the prefill route must not take. An
    // unset request therefore degrades to bf16 on such a host (with the line below); an explicit
    // request is a caller error, not something to override.
    if (ar_codec_ == ArCodec::Fp8E4m3 && !in_kernel_available_) {
        if (ar_codec_explicit_) {
            throw std::runtime_error(
                "NINFER_TP2_AR_CODEC=fp8 needs the in-kernel allreduce transport; this host mapped "
                "none (see the [mem] TP-2 in-kernel allreduce staging line)");
        }
        ar_codec_ = ArCodec::Bf16;
        std::fprintf(stderr,
                     "[tp2] prefill allreduce wire codec: bf16 (the default e4m3 wire needs the "
                     "in-kernel transport, which this host did not map)\n");
    }
    if (ar_codec_ == ArCodec::Fp8E4m3) {
        std::fprintf(stderr,
                     "[tp2] prefill allreduce wire codec: fp8-e4m3, 128-element blocks "
                     "(decode, logits, embeddings and captures stay bf16)\n");
    } else if (ar_codec_explicit_) {
        std::fprintf(stderr, "[tp2] prefill allreduce wire codec: bf16 (NINFER_TP2_AR_CODEC)\n");
    }
}

DevicePair::~DevicePair() {
    stop_ar_watchdog();
    if (p2p_) {
        a_.bind_to_current_thread_noexcept();
        b_.bind_to_current_thread_noexcept();
        cudaDeviceDisablePeerAccess(b_.device);
        cudaDeviceDisablePeerAccess(a_.device);
    }
    if (host_a_) { cudaFreeHost(host_a_); }
    if (host_b_) { cudaFreeHost(host_b_); }
    if (arrival_host_a_) { cudaFreeHost(arrival_host_a_); }
    if (arrival_host_b_) { cudaFreeHost(arrival_host_b_); }
    if (stall_host_) { cudaFreeHost(stall_host_); }
    if (base_host_) { cudaFreeHost(base_host_); }
    if (base_dev_a_) {
        a_.bind_to_current_thread_noexcept();
        cudaFree(base_dev_a_);
    }
    if (base_dev_b_) {
        b_.bind_to_current_thread_noexcept();
        cudaFree(base_dev_b_);
    }
    if (small_host_a_) { cudaFreeHost(small_host_a_); }
    if (small_host_b_) { cudaFreeHost(small_host_b_); }
}

DevicePair::DevicePair(DevicePair&& other) noexcept
    : a_(std::move(other.a_)), b_(std::move(other.b_)), p2p_(other.p2p_),
      staging_(std::move(other.staging_)), staging_slot_bytes_(other.staging_slot_bytes_),
     staging_parity_(other.staging_parity_), in_kernel_available_(other.in_kernel_available_),
      in_kernel_bytes_(other.in_kernel_bytes_), host_a_(other.host_a_), host_b_(other.host_b_),
      dev_a_(other.dev_a_), dev_b_(other.dev_b_), arrival_a_(other.arrival_a_),
      arrival_b_(other.arrival_b_), arrival_host_a_(other.arrival_host_a_),
      arrival_host_b_(other.arrival_host_b_),
      stall_host_(other.stall_host_), stall_a_(other.stall_a_), stall_b_(other.stall_b_),
      ar_timeout_ns_(other.ar_timeout_ns_), ar_call_serial_(other.ar_call_serial_),
      ar_fault_skip_call_(other.ar_fault_skip_call_),
      ar_fault_hold_call_(other.ar_fault_hold_call_), ar_fault_hold_ns_(other.ar_fault_hold_ns_),
      ar_fault_hold_peer_(other.ar_fault_hold_peer_), ar_bank_mask_(other.ar_bank_mask_),
      ar_payload_divisor_(other.ar_payload_divisor_), ar_codec_(other.ar_codec_),
      ar_codec_explicit_(other.ar_codec_explicit_),
      ar_channel_capacity_(other.ar_channel_capacity_), base_host_(other.base_host_),
      base_dev_a_(other.base_dev_a_), base_dev_b_(other.base_dev_b_),
      ar_channels_(std::move(other.ar_channels_)), capturing_(other.capturing_),
      capture_channel_(other.capture_channel_), capture_index_(other.capture_index_),
      ar_eager_next_(other.ar_eager_next_), ar_last_id_(other.ar_last_id_.load()),
      force_host_staging_(other.force_host_staging_), host_add_threads_(other.host_add_threads_),
      small_available_(other.small_available_), small_host_a_(other.small_host_a_),
      small_host_b_(other.small_host_b_), small_dev_a_(other.small_dev_a_),
      small_dev_b_(other.small_dev_b_), watch_(std::move(other.watch_)) {
    other.p2p_              = false;
    other.in_kernel_available_ = false;
    other.small_available_  = false;
    other.small_host_a_     = nullptr;
    other.small_host_b_     = nullptr;
    other.small_dev_a_      = nullptr;
    other.small_dev_b_      = nullptr;
    other.host_a_           = nullptr;
    other.host_b_           = nullptr;
    other.arrival_host_a_   = nullptr;
    other.arrival_host_b_   = nullptr;
    other.stall_host_       = nullptr;
    other.stall_a_          = nullptr;
    other.stall_b_          = nullptr;
    other.base_host_        = nullptr;
    other.base_dev_a_       = nullptr;
    other.base_dev_b_       = nullptr;
    other.capturing_        = false;
    // The diagnostics thread now belongs to this object, but the state block still names other's id
    // counter. Stop it, re-point it at this object's members (the mapped arrays' addresses do not
    // move), and restart, so neither destruction order can leave it reading freed memory.
    if (watch_) {
        stop_ar_watchdog();
        watch_->id        = &ar_last_id_;
        watch_->arrival_a = static_cast<const unsigned char*>(arrival_host_a_);
        watch_->arrival_b = static_cast<const unsigned char*>(arrival_host_b_);
        watch_->banks     = static_cast<unsigned int>(ar_banks());
        watch_->stop.store(false, std::memory_order_relaxed);
        start_ar_watchdog();
    }
}

DevicePair& DevicePair::operator=(DevicePair&& other) noexcept {
    if (this == &other) { return *this; }
    stop_ar_watchdog();
    watch_.reset();
    // Release this object's own mapped/device reservations before the plain pointer assignments
    // below take other's: those assignments would otherwise strand them.
    if (p2p_) {
        a_.bind_to_current_thread_noexcept();
        b_.bind_to_current_thread_noexcept();
        cudaDeviceDisablePeerAccess(b_.device);
        cudaDeviceDisablePeerAccess(a_.device);
    }
    if (host_a_) { cudaFreeHost(host_a_); }
    if (host_b_) { cudaFreeHost(host_b_); }
    if (arrival_host_a_) { cudaFreeHost(arrival_host_a_); }
    if (arrival_host_b_) { cudaFreeHost(arrival_host_b_); }
    if (stall_host_) { cudaFreeHost(stall_host_); }
    if (base_host_) { cudaFreeHost(base_host_); }
    if (base_dev_a_) { a_.bind_to_current_thread_noexcept(); cudaFree(base_dev_a_); }
    if (base_dev_b_) { b_.bind_to_current_thread_noexcept(); cudaFree(base_dev_b_); }
    if (small_host_a_) { cudaFreeHost(small_host_a_); }
    if (small_host_b_) { cudaFreeHost(small_host_b_); }
    a_ = std::move(other.a_);
    b_ = std::move(other.b_);
    p2p_ = other.p2p_;
    staging_ = std::move(other.staging_);
    staging_slot_bytes_ = other.staging_slot_bytes_;
    staging_parity_     = other.staging_parity_;
    in_kernel_available_ = other.in_kernel_available_;
    in_kernel_bytes_     = other.in_kernel_bytes_;
    host_a_              = other.host_a_;
    host_b_              = other.host_b_;
    dev_a_               = other.dev_a_;
    dev_b_               = other.dev_b_;
    arrival_a_           = other.arrival_a_;
    arrival_b_           = other.arrival_b_;
    arrival_host_a_      = other.arrival_host_a_;
    arrival_host_b_      = other.arrival_host_b_;
    stall_host_          = other.stall_host_;
    stall_a_             = other.stall_a_;
    stall_b_             = other.stall_b_;
    ar_timeout_ns_       = other.ar_timeout_ns_;
    ar_call_serial_      = other.ar_call_serial_;
    ar_fault_skip_call_  = other.ar_fault_skip_call_;
    ar_fault_hold_call_  = other.ar_fault_hold_call_;
    ar_fault_hold_ns_    = other.ar_fault_hold_ns_;
    ar_fault_hold_peer_  = other.ar_fault_hold_peer_;
    ar_bank_mask_        = other.ar_bank_mask_;
    base_host_           = other.base_host_;
    base_dev_a_          = other.base_dev_a_;
    base_dev_b_          = other.base_dev_b_;
    ar_channels_         = std::move(other.ar_channels_);
    capturing_           = other.capturing_;
    capture_channel_     = other.capture_channel_;
    capture_index_       = other.capture_index_;
    ar_eager_next_       = other.ar_eager_next_;
    force_host_staging_  = other.force_host_staging_;
    host_add_threads_    = other.host_add_threads_;
    ar_last_id_.store(other.ar_last_id_.load(), std::memory_order_relaxed);
    ar_payload_divisor_  = other.ar_payload_divisor_;
    ar_channel_capacity_ = other.ar_channel_capacity_;
    watch_               = std::move(other.watch_);
    small_available_     = other.small_available_;
    small_host_a_        = other.small_host_a_;
    small_host_b_        = other.small_host_b_;
    small_dev_a_         = other.small_dev_a_;
    small_dev_b_         = other.small_dev_b_;
    other.p2p_              = false;
    other.in_kernel_available_ = false;
    other.host_a_           = nullptr;
    other.host_b_           = nullptr;
    other.arrival_host_a_   = nullptr;
    other.arrival_host_b_   = nullptr;
    other.stall_host_       = nullptr;
    other.stall_a_          = nullptr;
    other.stall_b_          = nullptr;
    other.base_host_        = nullptr;
    other.base_dev_a_       = nullptr;
    other.base_dev_b_       = nullptr;
    other.capturing_        = false;
    other.small_available_  = false;
    other.small_host_a_     = nullptr;
    other.small_host_b_     = nullptr;
    other.small_dev_a_      = nullptr;
    other.small_dev_b_      = nullptr;
    if (watch_) {
        stop_ar_watchdog();
        watch_->id        = &ar_last_id_;
        watch_->arrival_a = static_cast<const unsigned char*>(arrival_host_a_);
        watch_->arrival_b = static_cast<const unsigned char*>(arrival_host_b_);
        watch_->banks     = static_cast<unsigned int>(ar_banks());
        watch_->stop.store(false, std::memory_order_relaxed);
        start_ar_watchdog();
    }
    return *this;
}

// Parity slot for one host-staging collective. The previous call's uploads may still be reading the
// other slot on either device's stream, and the two streams are independent, so a shared buffer
// cannot serve back-to-back collectives: the next call's download would overwrite the half the
// previous call's upload is still sending from (observed as changed logits in a prefill, and
// downstream as an out-of-range index - see the staging note in the class header).
char* DevicePair::host_staging(std::size_t count_bytes, cudaStream_t stream_a,
                               cudaStream_t stream_b) {
    const std::size_t slot = 2 * count_bytes;
    if (!staging_ || staging_slot_bytes_ < slot) {
        // Growth releases the buffer those uploads are reading from; both streams are about to be
        // synchronized by this call anyway, so wait before reallocating.
        if (staging_) {
            CUDA_CHECK(cudaStreamSynchronize(stream_a));
            CUDA_CHECK(cudaStreamSynchronize(stream_b));
        }
        staging_            = std::make_unique<PinnedHostBuffer>(2 * slot);
        staging_slot_bytes_ = slot;
    }
    char* pinned =
        static_cast<char*>(staging_->data()) + (staging_parity_ ? staging_slot_bytes_ : 0);
    staging_parity_ = !staging_parity_;
    return pinned;
}

void DevicePair::require_capturable_payload(std::size_t count_bytes) const {
    if (capturing_ && count_bytes > in_kernel_bytes_) {
        throw std::logic_error("tp collective: a " + std::to_string(count_bytes) +
                               "-byte payload exceeds the " + std::to_string(in_kernel_bytes_) +
                               "-byte in-kernel all-reduce staging while a capture is open; the "
                               "host-staging fallback synchronizes the caller's streams, which a "
                               "capture cannot contain");
    }
}

template <ArCodec kCodec>
void DevicePair::launch_ar_side(const ArSideCall& side) {
    const int extra = ar_prefetch_blocks(side.prefetch_bytes);
    side.device->bind_to_current_thread();
    ar_exchange<true, kCodec><<<side.blocks + extra, kArThreads, 0, side.stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(side.data),
        reinterpret_cast<__nv_bfloat16*>(side.data), side.mine, side.other, side.count,
        side.arrival_mine, side.arrival_other, side.order_mine, side.base_dev, side.call_index,
        side.base_value, side.slot_bytes, side.groups, side.stall_mine, side.stall_peer,
        side.timeout_ns, side.bank_mask, side.hold_ns, side.prefetch, side.prefetch_bytes,
        side.blocks);
}

void DevicePair::allreduce(void* data_a, void* data_b, std::size_t count_bytes,
                           cudaStream_t stream_a, cudaStream_t stream_b,
                           const ArPrefetchWindow& prefetch, ArCodec codec) {
    require_bytes(count_bytes);
    require_capturable_payload(count_bytes);
    if (count_bytes == 0) { return; }
    if (codec == ArCodec::Fp8E4m3) {
        // The codec has no host-staging or captured form: prefill is eager and always fits the
        // staging, so a request that would take another path is a caller bug, not a fallback.
        if (!in_kernel_available_ || count_bytes > in_kernel_bytes_) {
            throw std::runtime_error("TP-2 fp8 allreduce needs the in-kernel transport and a "
                                     "payload within the staging buffer");
        }
        if (capturing_) {
            throw std::runtime_error(
                "TP-2 fp8 allreduce cannot run inside a CUDA Graph capture (decode stays bf16)");
        }
        if (ar_payload_divisor_ > 1) {
            throw std::runtime_error(
                "TP-2 fp8 allreduce cannot be combined with NINFER_TP2_AR_PAYLOAD_DIVISOR");
        }
        if ((count_bytes / 2) % kArWireBlockElements != 0) {
            throw std::runtime_error("TP-2 fp8 allreduce needs whole 128-element blocks");
        }
    }
    // In-kernel path (WSL2, no P2P, mapped pinned available): both devices run
    // ar_exchange<true> on their compute streams. Each writes its delta to its own
    // mapped host buffer, signals an arrival id, spins on the peer's id,
    // then reads the peer's buffer and sums in place. No host synchronization -
    // the cross-GPU barrier is the kernel's arrival-id spin. The kernel is
    // ordered after the producing kernels on the same stream, and the caller's
    // next work (residual_add) is ordered after it, so no extra barrier is
    // needed. Falls through to host staging if mapped pinned is unavailable or
    // the delta exceeds the staging capacity.
    //
    // This transport is at the hardware floor for a batched prefill: the peer's PCIe link sets the
    // pace (on this host CUDA device 1 is Gen4 x4, ~7.9 GB/s counting both directions together,
    // against device 0's Gen5 x8 ~20 GB/s each way), and a [hidden, T] delta that must be written
    // and then read back costs 2 * hidden * T * 2 bytes per layer per device. Ordered slicing and
    // copy-engine transport were both measured against the plain single-block version; slicing
    // recovers the interleave loss (see ar_slices) while the copy engines were slower.
    if (in_kernel_available_ && count_bytes <= in_kernel_bytes_) {
        note_ar_call(count_bytes);
        const CollectiveId id = acquire_collective_id(stream_a, stream_b);
        const int count = static_cast<int>(count_bytes / 2);
        const auto address =
            reinterpret_cast<std::uintptr_t>(data_a) | reinterpret_cast<std::uintptr_t>(data_b) |
            reinterpret_cast<std::uintptr_t>(dev_a_) | reinterpret_cast<std::uintptr_t>(dev_b_);
        // require_bytes already bounds the payload to whole 16-byte groups; the pointer check only
        // decides whether the vectorized group loops may be used.
        const int groups = (address % 16 == 0) ? count / 8 : 0;
        auto* order_a = ar_order_base(arrival_a_);
        auto* order_b = ar_order_base(arrival_b_);
        // Diagnostics may shrink the range a collective actually exchanges; the caller's payload and
        // the staging decision still describe the full size. See NINFER_TP2_AR_PAYLOAD_DIVISOR.
        int exchange_groups = groups;
        int exchange_count  = count;
        if (ar_payload_divisor_ > 1) {
            exchange_groups = groups / ar_payload_divisor_;
            exchange_count  = exchange_groups * 8;
        }
        const int slices      = ar_slices(count_bytes);
        const ArFaults faults = ar_faults();
        const bool small      = small_available_ && count_bytes <= kArSmallBytes;
        ArSideCall side_a;
        side_a.data           = data_a;
        side_a.mine           = static_cast<char*>(small ? small_dev_a_ : dev_a_);
        side_a.other          = static_cast<const char*>(small ? small_dev_b_ : dev_b_);
        side_a.count          = exchange_count;
        side_a.groups         = exchange_groups;
        side_a.slot_bytes     = static_cast<int>(small ? kArSmallBytes : in_kernel_bytes_);
        side_a.blocks         = small ? 1 : slices;
        side_a.arrival_mine   = arrival_a_;
        side_a.arrival_other  = arrival_b_;
        side_a.order_mine     = order_a;
        side_a.base_dev       = id.base_a;
        side_a.call_index     = id.call_index;
        side_a.base_value     = id.value;
        side_a.stall_mine     = stall_a_;
        side_a.stall_peer     = stall_b_;
        side_a.timeout_ns     = ar_timeout_ns_;
        side_a.bank_mask      = ar_bank_mask_;
        side_a.hold_ns        = faults.hold_self_ns;
        side_a.prefetch       = static_cast<const char*>(prefetch.base_a);
        side_a.prefetch_bytes = prefetch.bytes_a;
        side_a.stream         = stream_a;
        side_a.device         = &a_;
        ArSideCall side_b     = side_a;
        side_b.data           = data_b;
        side_b.mine           = static_cast<char*>(small ? small_dev_b_ : dev_b_);
        side_b.other          = static_cast<const char*>(small ? small_dev_a_ : dev_a_);
        side_b.arrival_mine   = arrival_b_;
        side_b.arrival_other  = arrival_a_;
        side_b.order_mine     = order_b;
        side_b.base_dev       = id.base_b;
        side_b.stall_mine     = stall_b_;
        side_b.stall_peer     = stall_a_;
        side_b.hold_ns        = faults.hold_peer_ns;
        side_b.prefetch       = static_cast<const char*>(prefetch.base_b);
        side_b.prefetch_bytes = prefetch.bytes_b;
        side_b.stream         = stream_b;
        side_b.device         = &b_;
        if (codec == ArCodec::Fp8E4m3) {
            launch_ar_side<ArCodec::Fp8E4m3>(side_a);
            if (!faults.skip_peer) { launch_ar_side<ArCodec::Fp8E4m3>(side_b); }
        } else {
            launch_ar_side<ArCodec::Bf16>(side_a);
            if (!faults.skip_peer) { launch_ar_side<ArCodec::Bf16>(side_b); }
        }
        return;
    }
    // Two peer copies of an in-place all-reduce cannot be correct without a device scratch: the
    // first copy overwrites the local half the second copy still has to read. Rather than exchange
    // through scratch, every payload past the mapped staging takes the host-staging fallback below,
    // whose reduction is the measured and qualified path on both P2P-capable and consumer hosts.
    // Host staging on the caller's compute streams: this call's parity slot holds [a | b], both
    // halves count_bytes, and the sum lands in the first half. The D2H copies are ordered after the
    // kernels that produced the deltas on those same streams, and the H2D copies are ordered before
    // the caller's next work (residual_add) on them, so only the two D2H transfers need a barrier.
    char* pinned = host_staging(count_bytes, stream_a, stream_b);
    // Bind before each copy: cudaMemcpyAsync orders on the stream's device, and
    // the driver validates the current device against the stream.
    a_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(pinned, data_a, count_bytes, cudaMemcpyDeviceToHost, stream_a));
    b_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(pinned + count_bytes, data_b, count_bytes,
                               cudaMemcpyDeviceToHost, stream_b));
    CUDA_CHECK(cudaStreamSynchronize(stream_a));
    CUDA_CHECK(cudaStreamSynchronize(stream_b));
    // Reduce in place across the machine's cores, then upload. The copy engines set this transport's
    // floor (both directions run at once; measured 3.1 ms for a 10 MiB payload on this host), and the
    // scalar __hadd pass this replaced ran at 0.31 GB/s, so the reduction was the fallback's dominant
    // cost and is now vectorized and split (see add_bf16_host_split).
    add_bf16_host_split(reinterpret_cast<const __nv_bfloat16*>(pinned),
                        reinterpret_cast<const __nv_bfloat16*>(pinned + count_bytes),
                        reinterpret_cast<__nv_bfloat16*>(pinned), count_bytes / 2,
                        add_threads(count_bytes, host_add_threads_));
    a_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(data_a, pinned, count_bytes, cudaMemcpyHostToDevice, stream_a));
    b_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(data_b, pinned, count_bytes, cudaMemcpyHostToDevice, stream_b));
    // No H2D barrier: the caller enqueues residual_add on stream_a/stream_b
    // immediately after, so it is ordered after these transfers.
}

void DevicePair::sendrecv(const void* send_a, void* recv_a, const void* send_b, void* recv_b,
                          std::size_t count_bytes, cudaStream_t stream_a, cudaStream_t stream_b) {
    require_bytes(count_bytes);
    require_capturable_payload(count_bytes);
    if (count_bytes == 0) { return; }
    // In-kernel transport: both devices run ar_exchange<false> on their compute streams. Each device
    // publishes its own send bytes to its mapped host staging, signals an arrival id, spins on the
    // peer's id, then copies the peer's staging slice into its receive buffer - no host
    // synchronization, and the same id/parity discipline as the all-reduce, so an allreduce and a
    // sendrecv may interleave within one round as long as both devices issue the same sequence.
    if (in_kernel_available_ && count_bytes <= in_kernel_bytes_) {
        note_ar_call(count_bytes);
        const CollectiveId id = acquire_collective_id(stream_a, stream_b);
        const int count = static_cast<int>(count_bytes / 2);
        const auto address = reinterpret_cast<std::uintptr_t>(send_a) |
                             reinterpret_cast<std::uintptr_t>(recv_a) |
                             reinterpret_cast<std::uintptr_t>(send_b) |
                             reinterpret_cast<std::uintptr_t>(recv_b) |
                             reinterpret_cast<std::uintptr_t>(dev_a_) |
                             reinterpret_cast<std::uintptr_t>(dev_b_);
        // require_bytes already bounds the payload to whole 16-byte groups; the pointer check only
        // decides whether the vectorized group loops may be used.
        const int groups = (address % 16 == 0) ? count / 8 : 0;
        auto* order_a = ar_order_base(arrival_a_);
        auto* order_b = ar_order_base(arrival_b_);
        // Diagnostics may shrink the range a collective actually exchanges; the caller's payload and
        // the staging decision still describe the full size. See NINFER_TP2_AR_PAYLOAD_DIVISOR.
        int exchange_groups = groups;
        int exchange_count  = count;
        if (ar_payload_divisor_ > 1) {
            exchange_groups = groups / ar_payload_divisor_;
            exchange_count  = exchange_groups * 8;
        }
        if (small_available_ && count_bytes <= kArSmallBytes) {
            const ArFaults faults = ar_faults();
            a_.bind_to_current_thread();
            ar_exchange<false><<<1, kArThreads, 0, stream_a>>>(
                static_cast<const __nv_bfloat16*>(send_a), static_cast<__nv_bfloat16*>(recv_a),
                static_cast<char*>(small_dev_a_), static_cast<const char*>(small_dev_b_), exchange_count,
                arrival_a_, arrival_b_, order_a, id.base_a, id.call_index, id.value,
                static_cast<int>(kArSmallBytes), exchange_groups, stall_a_, stall_b_, ar_timeout_ns_,
                ar_bank_mask_, faults.hold_self_ns, nullptr, 0ULL, 1);
            if (!faults.skip_peer) {
                b_.bind_to_current_thread();
                ar_exchange<false><<<1, kArThreads, 0, stream_b>>>(
                    static_cast<const __nv_bfloat16*>(send_b), static_cast<__nv_bfloat16*>(recv_b),
                    static_cast<char*>(small_dev_b_), static_cast<const char*>(small_dev_a_), exchange_count,
                    arrival_b_, arrival_a_, order_b, id.base_b, id.call_index, id.value,
                    static_cast<int>(kArSmallBytes), exchange_groups, stall_b_, stall_a_, ar_timeout_ns_,
                    ar_bank_mask_, faults.hold_peer_ns, nullptr, 0ULL, 1);
            }
            return;
        }
        const int slices      = ar_slices(count_bytes);
        const int slot_bytes  = static_cast<int>(in_kernel_bytes_);
        const ArFaults faults = ar_faults();
        a_.bind_to_current_thread();
        ar_exchange<false><<<slices, kArThreads, 0, stream_a>>>(
            static_cast<const __nv_bfloat16*>(send_a), static_cast<__nv_bfloat16*>(recv_a),
            static_cast<char*>(dev_a_), static_cast<const char*>(dev_b_), exchange_count, arrival_a_,
            arrival_b_, order_a, id.base_a, id.call_index, id.value, slot_bytes, exchange_groups, stall_a_,
            stall_b_, ar_timeout_ns_, ar_bank_mask_, faults.hold_self_ns, nullptr, 0ULL, slices);
        if (!faults.skip_peer) {
            b_.bind_to_current_thread();
            ar_exchange<false><<<slices, kArThreads, 0, stream_b>>>(
                static_cast<const __nv_bfloat16*>(send_b), static_cast<__nv_bfloat16*>(recv_b),
                static_cast<char*>(dev_b_), static_cast<const char*>(dev_a_), exchange_count, arrival_b_,
                arrival_a_, order_b, id.base_b, id.call_index, id.value, slot_bytes, exchange_groups,
                stall_b_, stall_a_, ar_timeout_ns_, ar_bank_mask_, faults.hold_peer_ns, nullptr, 0ULL,
                slices);
        }
        return;
    }
    // A sendrecv may alias a side's send and receive (see the header), and the two peer copies run
    // on independent streams, so a peer copy could overwrite the source the other copy has not read
    // yet. The host-staging fallback publishes both send halves before either receive is uploaded, so
    // it is correct for that aliasing; use it for every payload here.
    // Host staging: this call's parity slot holds [a's send | b's send] and each receive copies the
    // other half back. Only the two D2H transfers need a barrier, exactly as the all-reduce fallback
    // does.
    char* pinned = host_staging(count_bytes, stream_a, stream_b);
    a_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(pinned, send_a, count_bytes, cudaMemcpyDeviceToHost, stream_a));
    b_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(pinned + count_bytes, send_b, count_bytes, cudaMemcpyDeviceToHost,
                               stream_b));
    CUDA_CHECK(cudaStreamSynchronize(stream_a));
    CUDA_CHECK(cudaStreamSynchronize(stream_b));
    a_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(recv_a, pinned + count_bytes, count_bytes, cudaMemcpyHostToDevice,
                               stream_a));
    b_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(recv_b, pinned, count_bytes, cudaMemcpyHostToDevice, stream_b));
}

ShardedTensor shard_view(const Tensor& full, int dim, std::int32_t local_len, int shard) {
    if (dim < 0 || dim >= 4) { throw std::invalid_argument("tp shard_view: dim out of range"); }
    if (shard < 0 || local_len <= 0) {
        throw std::invalid_argument("tp shard_view: invalid shard geometry");
    }
    const std::int32_t full_len = full.ne[dim];
    if (local_len > full_len) {
        throw std::invalid_argument("tp shard_view: local_len exceeds full extent");
    }
    ShardedTensor out;
    out.shard  = shard;
    out.shards = 1;
    out.local  = full.slice(dim, 0, local_len);
    return out;
}

ShardedTensor column_shard(const Tensor& weight, int shard, int shards) {
    if (shards < 1 || shard < 0 || shard >= shards) {
        throw std::invalid_argument("tp column_shard: invalid shard index");
    }
    if (weight.ne[3] != 1 || weight.ne[2] != 1) {
        throw std::invalid_argument("tp column_shard: expects a rank-2 weight");
    }
    const std::int32_t cols      = weight.ne[1];
    const std::int32_t base      = cols / shards;
    const std::int32_t remainder = cols % shards;
    const std::int32_t local     = base + (shard < remainder ? 1 : 0);
    const std::int32_t offset    = shard * base + std::min(shard, remainder);
    ShardedTensor out;
    out.shard  = shard;
    out.shards = shards;
    out.local  = weight.slice(1, offset, local);
    return out;
}

ShardedTensor row_shard(const Tensor& weight, int shard, int shards) {
    if (shards < 1 || shard < 0 || shard >= shards) {
        throw std::invalid_argument("tp row_shard: invalid shard index");
    }
    if (weight.ne[3] != 1 || weight.ne[2] != 1) {
        throw std::invalid_argument("tp row_shard: expects a rank-2 weight");
    }
    const std::int32_t rows      = weight.ne[0];
    const std::int32_t base      = rows / shards;
    const std::int32_t remainder = rows % shards;
    const std::int32_t local     = base + (shard < remainder ? 1 : 0);
    const std::int32_t offset    = shard * base + std::min(shard, remainder);
    ShardedTensor out;
    out.shard  = shard;
    out.shards = shards;
    out.local  = weight.slice(0, offset, local);
    return out;
}

} // namespace ninfer::tp
