#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::tp {

// Wire format of an all-reduce. Bf16 is the qualified default: the two shards exchange their bf16
// partial sums and the receiver adds them, which is exact for this transport (both cards read the
// same host sum). Fp8E4m3 halves the payload by quantizing each 128-element block to E4M3 with one
// f32 scale per block before the transfer, then dequantizing both halves and adding in f32. Both
// halves are quantized, so both shards still consume the identical sum. Only the prefill deltas use
// it: a decode collective is a few KiB and latency-bound, where the codec costs more than it saves.
enum class ArCodec { Bf16, Fp8E4m3 };

// Two independent DeviceContexts executing the same Op sequence in lockstep.
// The owning Program (Phase 3) drives both streams; this type owns the device
// facts and the inter-device communication path only.
class DevicePair {
public:
    explicit DevicePair(int device_a, int device_b);
    ~DevicePair();

    DevicePair(const DevicePair&)            = delete;
    DevicePair& operator=(const DevicePair&) = delete;
    DevicePair(DevicePair&& other) noexcept;
    DevicePair& operator=(DevicePair&& other) noexcept;

    // Spawns the in-kernel transport's diagnostics thread, which runs unless
    // NINFER_TP2_AR_WATCHDOG=0. It prints the mapped id/arrival/order state to stderr once per stall
    // episode, so a stuck in-kernel transport can be classified after the fact: ids that stopped
    // advancing with published arrivals mean a spin failed to observe the peer's write, and a stale
    // order slot means the slice write-order chain broke. Ids are host-authored (see
    // create_ar_channel), so the two sides can no longer diverge. The thread costs nothing
    // measurable: it sleeps 25 ms per tick, reads one host counter plus, while idle, the two mapped
    // stall flags, and touches the kernel-written arrival lines only after a device has tripped.
    void start_ar_watchdog();

    const DeviceContext& a() const noexcept { return a_; }
    const DeviceContext& b() const noexcept { return b_; }
    // Whether peer access between the two devices is available. Reported only: the transport no
    // longer issues peer copies, because an in-place two-copy all-reduce aliases the half the second
    // copy must still read (see allreduce), and a sendrecv may alias a side's send and receive.
    bool p2p_available() const noexcept { return p2p_; }
    // True when allreduce runs entirely in-kernel over mapped pinned host staging with no host
    // synchronization. Only this transport may be captured into a CUDA Graph: the host-staging
    // fallback synchronizes the caller's streams.
    bool in_kernel_allreduce() const noexcept { return in_kernel_available_; }
    // Largest payload the in-kernel transport carries in one call, in bytes; 0 when that transport
    // is not in use. A caller that sizes a batched payload (the TP-2 prefill chunk) bounds it by
    // this: a payload past the staging buffer silently takes the host-staging fallback, which
    // synchronizes both compute streams on every collective.
    std::size_t in_kernel_allreduce_bytes() const noexcept { return in_kernel_bytes_; }
    // Wire codec selected by NINFER_TP2_AR_CODEC ("bf16" opts out; unset: the E4M3 prefill codec,
    // or bf16 where the in-kernel transport is unavailable). The TP-2 prefill call sites pass this
    // to allreduce; every other route keeps the bf16 default.
    ArCodec ar_codec() const noexcept { return ar_codec_; }

    // Bounded-spin safety for the in-kernel transport. Every rendezvous spin carries a wall-clock
    // deadline (NINFER_TP2_AR_TIMEOUT_MS, default 10000, 0 disables it): a spin that can never be
    // satisfied would otherwise pin both devices at 100% forever and freeze the whole process. On
    // expiry the side that gave up raises its mapped flag and the kernel returns, so the caller's
    // streams drain and the request can fail instead of the service locking up. The caller must
    // synchronize both streams and call clear_ar_stall() before the next round, and must not keep
    // using state a stalled round may have half-written.
    [[nodiscard]] bool ar_stalled() const noexcept;
    // Clears both trip flags. The arrival and write-order slots need no repair: ids only ever
    // increase (see create_ar_channel), so a slot left over from the stalled round can never
    // satisfy a later spin. Both streams must be synchronized before this call.
    void clear_ar_stall() noexcept;
    // Id published by the most recent arming or eager collective, for diagnostics and error
    // messages. Equal on both sides by construction; it exists so a stall can name its round.
    [[nodiscard]] std::uint64_t ar_last_id() const noexcept;

    // Every collective carries an id that must be unique for the lifetime of the pair: a spin is
    // satisfied by any arrival slot that holds its id, so a reused id lets a kernel read the peer's
    // staging before the peer has written it. A captured graph takes its ids from a channel. The
    // channel's pinned cell is copied into the graph by a memcpy node, so every replay reads the id
    // block the host published for *that* launch, and the ordinal within the block is baked at
    // capture - it is a constant of the graph, which is why baking it is safe where a baked id
    // would not be. Eager calls draw from a separate monotonic range. One channel per captured
    // graph: two graphs launched in one round need disjoint id blocks, and two graphs sharing a
    // cell would read each other's base before their memcpy node ran.
    using ArChannel = std::uint32_t;
    static constexpr ArChannel kNoArChannel = 0;
    ArChannel create_ar_channel();
    // Brackets the collectives a capture records. They take the channel's next ordinals, and the
    // base memcpy node is recorded ahead of the first collective on each stream.
    void begin_capture(ArChannel channel);
    void end_capture();
    // Publishes the channel's next id block into its pinned cell and advances past it, so no id is
    // ever handed out twice. Call it immediately before launching the graph that owns the channel.
    void arm_round(ArChannel channel);

    // Arrival-bank geometry (see the arrival_a_ member). One bank holds one rendezvous id per slice
    // and is one cache line; a side's write-order chains follow its arrival banks, at the same banks.
    // The kernel indexes 'arrival + (id & mask) * kArSliceSlots + blockIdx.x', so the stride is part
    // of the kernel's contract and is asserted against kArMaxSlices in device_pair.cu.
    static constexpr std::size_t kArSliceSlots = 8;
    static constexpr std::size_t kArBankBytes = kArSliceSlots * sizeof(unsigned long long);

    // Fault injection for the bounded-spin test: skip the peer-side launch of the n-th in-kernel
    // collective issued after arming (1-based; 0 disarms). It desynchronizes the two sides exactly
    // like a divergent call sequence, which is the failure the bound exists for. Diagnostics only.
    void set_ar_fault_skip_peer_call(std::uint64_t serial) noexcept {
        ar_fault_skip_call_ = serial;
        ar_call_serial_     = 0;
    }

    // Fault injection for the skew the banks exist to survive: hold one side's arrival poll for
    // 'nanos' after it has published its own arrival, so the other side completes that collective and
    // publishes the next one before this side ever looks. 'serial' counts eager (non-capture)
    // in-kernel collectives since arming, 1-based, or 0 for every one of them; 'nanos' 0 disarms.
    // 'peer' picks the b-side launch (true) or the a-side one (false). Captures are excluded because a
    // hold recorded into a captured graph would be replayed by every launch of that graph, and the
    // skew this hook reproduces is the eager one the production dumps show. NINFER_TP2_AR_FAULT_HOLD_POLL
    // ('<serial>:<nanos>[:self]') arms the same hook at construction. Diagnostics only: the payload and
    // the arithmetic are untouched.
    void set_ar_fault_hold_poll(std::uint64_t serial, std::uint64_t nanos, bool peer) noexcept {
        ar_fault_hold_call_   = serial;
        ar_fault_hold_ns_     = nanos;
        ar_fault_hold_peer_   = peer;
        ar_fault_hold_serial_ = 0;
    }

    // Sizes the rendezvous id space to hold `count` captured graphs. Must be called before the first
    // create_ar_channel (no channel may exist yet); a no-op when the in-kernel transport is
    // unavailable or `count` does not exceed the current capacity. The TP-2 core passes the number
    // of graph buckets it built, so a bucket can no longer exhaust a fixed id block.
    void reserve_ar_channels(std::size_t count);

    // In-place all-reduce of count_bytes (multiple of 16): on return both
    // buffers contain a + b elementwise. stream_a/stream_b are the compute
    // streams that produced data_a/data_b. The P2P path runs peer copies plus
    // an add kernel per device on the DevicePair's own streams and completes
    // them before returning. The host-staging path (WSL2 blocks peer access)
    // stages both halves through one pinned buffer on stream_a/stream_b: the
    // D2H copies are ordered after the producing kernels and the H2D copies
    // before the caller's next work on those streams, so only the two D2H
    // transfers are synchronized.
    // Bytes the rendezvous window may pull while the peer has not arrived yet. Decode's layer chain is
    // strictly sequential, so a collective has no compute to overlap with: what it does have is a
    // memory system that is idle for the whole wait. The next kernel's weight stream has no
    // dependency on the reduction, so each side prefetches its own hint into L2 and the following
    // kernel finds those bytes resident instead of paying for them on its critical path. A hint is
    // never part of the contract: the payload, the staging and the ids are untouched, and a
    // reduction that ignores it produces exactly the same values.
    struct ArPrefetchWindow {
        const void* base_a  = nullptr;
        std::size_t bytes_a = 0;
        const void* base_b  = nullptr;
        std::size_t bytes_b = 0;
    };

    // `codec` selects the wire format. Fp8E4m3 is only valid on the in-kernel transport with a
    // payload of whole 128-element blocks and never inside a capture; it throws otherwise rather
    // than silently falling back, because a silent fallback would mix two precisions inside one
    // layer chain.
    void allreduce(void* data_a, void* data_b, std::size_t count_bytes,
                   cudaStream_t stream_a, cudaStream_t stream_b,
                   const ArPrefetchWindow& prefetch = {}, ArCodec codec = ArCodec::Bf16);

    // Byte-exact exchange of count_bytes (multiple of 16): on return recv_a holds send_b's bytes and
    // recv_b holds send_a's bytes, bit for bit. The in-kernel route reuses the all-reduce transport -
    // same staging, arrival ids and slicing - with a copy in place of the BF16 add, so it needs no
    // host synchronization and is safe to capture into a CUDA Graph. Use it for any payload the BF16
    // add would reinterpret: integer ids, FP32 scores, or bits that must survive unchanged. A side may
    // pass the same buffer as its send and receive: it publishes its own bytes and returns the peer's,
    // because the two halves are distinct.
    void sendrecv(const void* send_a, void* recv_a, const void* send_b, void* recv_b,
                  std::size_t count_bytes, cudaStream_t stream_a, cudaStream_t stream_b);

private:
    DeviceContext a_;
    DeviceContext b_;
    bool p2p_ = false;
    // Host-staging fallback: one buffer holding two parity slots, each [a | b]. The two devices'
    // compute streams are independent, so a single shared buffer would let the next call's download
    // overwrite the half the previous call's upload is still reading from (see host_staging).
    std::unique_ptr<PinnedHostBuffer> staging_;
    std::size_t staging_slot_bytes_ = 0; // capacity of one slot: 2 * the largest payload seen
    bool staging_parity_            = false;
    // Returns this call's parity slot, growing the buffer (after synchronizing both compute streams)
    // when the payload is the largest so far.
    char* host_staging(std::size_t count_bytes, cudaStream_t stream_a, cudaStream_t stream_b);
    // One side of an in-kernel collective; kCodec selects the wire format (see ArCodec).
    struct ArSideCall {
        void* data = nullptr;
        char* mine = nullptr;
        const char* other = nullptr;
        int count = 0;
        int groups = 0;
        int slot_bytes = 0;
        int blocks = 0;
        unsigned long long* arrival_mine = nullptr;
        unsigned long long* arrival_other = nullptr;
        unsigned long long* order_mine = nullptr;
        const unsigned long long* base_dev = nullptr;
        unsigned int call_index = 0;
        unsigned long long base_value = 0;
        int* stall_mine = nullptr;
        int* stall_peer = nullptr;
        unsigned long long timeout_ns = 0;
        unsigned int bank_mask = 0;
        unsigned long long hold_ns = 0;
        const char* prefetch = nullptr;
        std::size_t prefetch_bytes = 0;
        cudaStream_t stream = nullptr;
        DeviceContext* device = nullptr;
    };
    template <ArCodec kCodec>
    void launch_ar_side(const ArSideCall& side);
    // In-kernel allreduce (WSL2, no P2P): mapped pinned host staging + arrival ids. Each device's
    // kernel writes its delta to its own mapped host buffer, signals an arrival id, spins on the
    // peer's id, then reads the peer's buffer and sums in place - all inside the kernel, so the host
    // never calls cudaStreamSynchronize. Mapped pinned memory is verified available at construction;
    // when it is not, allreduce falls back to the host-staging path.
    bool in_kernel_available_ = false;
    std::size_t in_kernel_bytes_ = 0;
    void* host_a_ = nullptr; // cudaFreeHost handle for device a's staging
    void* host_b_ = nullptr; // cudaFreeHost handle for device b's staging
    void* dev_a_  = nullptr; // device-side pointer (device a) for host_a_
    void* dev_b_  = nullptr; // device-side pointer (device b) for host_b_
    // Arrival ids, one per allreduce slice, in mapped pinned host memory (the caller decides which
    // shard drives a pair, so a device allocation could be dereferenced from the other device's
    // context). Each slice writes its id, publishes an arrival, then spins for the peer's.
    //
    // The ids live in banks: consecutive collectives publish under different banks (bank = id & mask),
    // so one bank still holds the id a slow side is waiting for after the peer has published the next
    // collective. The engine legitimately produces that one-collective skew - the mirror shard is a
    // call or two behind wherever only shard A is drained (see TP2GenerationCore::abort_if_ar_stalled)
    // - and without the rotation the waiting side spins out its whole deadline on an id the slot has
    // already overwritten. NINFER_TP2_AR_BANKS selects the count (a power of two in 1..8, default 4;
    // 1 is the pre-bank geometry the skew acceptance uses as its positive control).
    unsigned long long* arrival_a_ = nullptr; // device pointer (device a) for its arrival array
    unsigned long long* arrival_b_ = nullptr; // device pointer (device b) for its arrival array
    void* arrival_host_a_ = nullptr; // cudaFreeHost handle for arrival_a_
    void* arrival_host_b_ = nullptr; // cudaFreeHost handle for arrival_b_
    // Bounded-spin state: one mapped flag per side, a cache line apart, plus the deadline handed to
    // every kernel launch (see ar_stalled).
    int* stall_host_ = nullptr; // cudaFreeHost handle: 2 * kArTokenStrideBytes
    int* stall_a_    = nullptr; // device pointer to side a's own flag
    int* stall_b_    = nullptr; // device pointer to side b's own flag
    unsigned long long ar_timeout_ns_ = 0; // 0 disables the deadline
    std::uint64_t ar_call_serial_     = 0; // in-kernel collectives issued since the last arming
    std::uint64_t ar_fault_skip_call_ = 0; // fault injection, see set_ar_fault_skip_peer_call

    std::uint64_t ar_fault_hold_call_   = 0; // fault injection, see set_ar_fault_hold_poll
    std::uint64_t ar_fault_hold_serial_ = 0; // eager collectives seen since the hook was armed
    unsigned long long ar_fault_hold_ns_ = 0;
    bool ar_fault_hold_peer_             = true;
    // Arrival banks this pair publishes under, as a mask (banks - 1). Read once at construction from
    // NINFER_TP2_AR_BANKS; the kernel and the watchdog both index the banks through it, so the move
    // operations carry it or a moved pair would index a geometry its reservations were not sized for.
    unsigned int ar_bank_mask_ = 3U;
    // Diagnostics only: divides the bytes each in-kernel collective exchanges, without changing the
    // payload the caller passed or the staging decision. A run with a divisor above one is
    // numerically wrong by construction (the skipped region keeps whatever the destination held,
    // which for an in-place all-reduce is the local partial) and bounds what compressing the
    // exchanged payload could buy. Set by NINFER_TP2_AR_PAYLOAD_DIVISOR.
    int ar_payload_divisor_ = 1;
    // Wire codec requested by NINFER_TP2_AR_CODEC. Read once at construction: the TP-2 prefill call
    // sites query it per collective, so a pair never mixes codecs inside one layer chain unless the
    // caller explicitly asks for bf16 at sendrecv.
    ArCodec ar_codec_ = ArCodec::Fp8E4m3;
    // True when the environment chose the codec, which makes an unusable request a startup error
    // instead of a quiet bf16 fallback.
    bool ar_codec_explicit_ = false;
    // Rendezvous ids (see create_ar_channel). All channels share one pinned cell block and one
    // device scalar block per device; a channel's memcpy node copies its 8-byte cell into its own
    // scalar inside the captured graph.
    struct ArChannelState {
        unsigned long long* host = nullptr; // pinned cell: the graph's memcpy node reads it per replay
        void* dev_a = nullptr; // device scalar the a-side memcpy node writes
        void* dev_b = nullptr; // device scalar the b-side memcpy node writes
        std::uint64_t next_base = 0; // first id of the next arming; only ever increases
        std::uint32_t calls     = 0; // collectives numbered by the last capture
        bool seen_a             = false; // the base memcpy is recorded once per capture, per stream
        bool seen_b             = false;
    };
    // Rendezvous id channels: one per captured graph. The owning core reserves the count its graph
    // buckets imply (reserve_ar_channels) before the first capture; this default is the floor for a
    // caller that never reserves. The capacity is fixed once a channel exists, because a channel's
    // device scalar is the destination a graph's memcpy node was captured with.
    static constexpr std::size_t kArDefaultChannels = 16;
    std::size_t ar_channel_capacity_ = kArDefaultChannels;
    unsigned long long* base_host_ = nullptr; // cudaFreeHost handle: ar_channel_capacity_ id cells
    void* base_dev_a_ = nullptr; // cudaMalloc on device a: ar_channel_capacity_ scalars
    void* base_dev_b_ = nullptr; // cudaMalloc on device b: ar_channel_capacity_ scalars
    std::vector<ArChannelState> ar_channels_; // one per captured graph, created on demand
    bool capturing_              = false;
    ArChannel capture_channel_   = kNoArChannel;
    std::uint32_t capture_index_ = 0; // ordinal assigned to the next captured collective
    // Eager ids come from their own high range, far above every channel's, so the two can never
    // meet and a stale arrival slot can never match a later id from the other range.
    std::uint64_t ar_eager_next_ = 1ULL << 62;
    // Published id, read by the watchdog thread and by the engine's stall message.
    std::atomic<unsigned long long> ar_last_id_{0};
    // Diagnostics: NINFER_TP2_AR_FORCE_HOST_STAGING=1 keeps every collective on the host-staging
    // fallback, which is how the test suite qualifies that transport's arithmetic where the
    // in-kernel one would otherwise take every call.
    bool force_host_staging_ = false;
    // Host threads the host-staging reduction uses; 0 takes what the machine has (capped). Only
    // NINFER_TP2_AR_ADD_THREADS sets it, so the fallback's reduction can be measured at one thread
    // against the automatic split on the machine it runs on.
    unsigned host_add_threads_ = 0;
    // Decode-sized staging: a single token's [hidden, 1] delta is a few KiB, so the prefill-sized
    // slots are almost all waste, and sharing them would let a small call's parity slot alias a
    // prefill payload the peer is still reading.
    bool small_available_ = false;
    void* small_host_a_ = nullptr; // cudaFreeHost handle for device a's decode-sized staging
    void* small_host_b_ = nullptr; // cudaFreeHost handle for device b's decode-sized staging
    void* small_dev_a_  = nullptr; // device-side pointer (device a) for small_host_a_
    void* small_dev_b_  = nullptr; // device-side pointer (device b) for small_host_b_
    // Diagnostics state for the in-kernel transport (see start_ar_watchdog). Held through a
    // shared_ptr because the detached watchdog thread only reads the mapped arrays and may outlive
    // moves of the pair.
    struct ArWatchState;
    std::shared_ptr<ArWatchState> watch_;
    void note_ar_call(std::size_t count_bytes);
    void stop_ar_watchdog();
    // A capture records launches and cannot contain the fallback's per-collective stream
    // synchronization, so a payload past the staging while a capture is open is a hard error rather
    // than a quiet transport change. Called by allreduce and sendrecv.
    void require_capturable_payload(std::size_t count_bytes) const;
    // Reserves the largest staging-ladder rung both devices can map, leaving host_a_/host_b_ and
    // dev_a_/dev_b_ holding it. Returns its capacity per parity slot, or 0 when no rung fits: the
    // pair then has no in-kernel transport and every collective takes the host-staging fallback.
    std::size_t reserve_in_kernel_staging();
    // What one collective's diagnostics hooks ask for. Taken once per collective, so the shared
    // serial counter advances exactly once whatever combination is armed.
    struct ArFaults {
        bool skip_peer                  = false;
        unsigned long long hold_self_ns = 0;
        unsigned long long hold_peer_ns = 0;
    };
    ArFaults ar_faults();
    // Arrival banks this pair publishes under (a power of two).
    [[nodiscard]] std::size_t ar_banks() const noexcept {
        return static_cast<std::size_t>(ar_bank_mask_) + 1U;
    }
    // A side's write-order chain base: its arrival banks, then the chains at the same banks.
    [[nodiscard]] unsigned long long* ar_order_base(unsigned long long* arrival) const noexcept {
        return arrival + ar_banks() * kArSliceSlots;
    }
    // One side's token block: the arrival banks followed by their write-order chains.
    [[nodiscard]] std::size_t ar_token_bytes() const noexcept {
        return 2U * ar_banks() * kArBankBytes;
    }
    // What one collective runs with: the ordinal baked at capture (0 for eager, where the host
    // writes the id itself) and the base cell each side's kernel reads. Recording the base memcpy
    // node for a captured collective happens here, on its first appearance per stream.
    struct CollectiveId {
        // Captured: the base cell each side's memcpy node filled, plus the ordinal baked into this
        // kernel. Eager: no cell, and the id travels in the launch arguments, which are fresh on
        // every launch - a shared cell would be overwritten by the next call before the kernel ran.
        const unsigned long long* base_a = nullptr;
        const unsigned long long* base_b = nullptr;
        unsigned int call_index = 0;
        unsigned long long value = 0;
    };
    CollectiveId acquire_collective_id(cudaStream_t stream_a, cudaStream_t stream_b);
};

// Non-owning view of one shard of a tensor split along dim. The local tensor
// is a valid contiguous view: for dim < 3 the shard keeps the full strides of
// the upper dims, so callers must treat it as a strided window, not a fresh
// allocation. For the 27B dense model only dims 0 and 1 are split (both
// produce contiguous shards of a contiguous parent).
struct ShardedTensor {
    Tensor local;
    int shard  = 0;
    int shards = 1;
};

// View of shard index of a contiguous tensor split along dim.
ShardedTensor shard_view(const Tensor& full, int dim, std::int32_t local_len, int shard);

// Column-parallel split of a [rows, cols] weight: shard gets cols/shards
// contiguous columns.
ShardedTensor column_shard(const Tensor& weight, int shard, int shards);

// Row-parallel split of a [rows, cols] weight: shard gets rows/shards
// contiguous rows.
ShardedTensor row_shard(const Tensor& weight, int shard, int shards);

} // namespace ninfer::tp
