#include "core/tp/tp_materialize.h"

#include "artifact/framing.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/startup.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::tp {
namespace {

// Pinned staging pipeline. The single-card materializer (artifact/materializer.cpp) already
// proves the pattern: reads stage into ring slots while earlier pieces are still in flight
// and nothing synchronizes per object. This port adds the TP twist: one ring serves both
// shard streams, every slot carries one event per device, and a replicated piece enqueues
// its copy on both devices from the same slot. The resident pinned occupancy stays at four
// slots even for multi-GiB weights, and the producer only blocks when a slot has to wait
// for its own copies — the ring goes as deep as the devices allow before it fights back.
constexpr std::size_t kSlotBytes = 64ULL * 1024 * 1024;
constexpr std::size_t kSlotCount = 4;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw artifact::ArtifactError(std::string(operation) + ": " + cudaGetErrorName(status) + ": " +
                                      cudaGetErrorString(status));
    }
}

// One staging slot: exactly one ≤kSlotBytes piece, readable by both devices. Portable
// pinned is mandatory — cudaMallocHost alone pins for the allocating context only, and the
// same slot feeds a copy enqueued on the other device's stream (arena.h PinnedHostBuffer).
// Events are context-affine: an event records only into a stream of the context that
// created it, so each slot builds one event under each shard's device context explicitly,
// and restores the caller's current device.
struct Slot {
    Slot(std::size_t bytes, const DeviceContext& device0, const DeviceContext& device1)
        : buffer(bytes, /*portable=*/true) {
        int restored_device = -1;
        check_cuda(cudaGetDevice(&restored_device), "query current device for tp staging events");
        check_cuda(cudaSetDevice(device0.device), "bind tp staging event device 0");
        check_cuda(cudaEventCreateWithFlags(&events[0], cudaEventDisableTiming),
                   "create tp staging completion event 0");
        check_cuda(cudaSetDevice(device1.device), "bind tp staging event device 1");
        check_cuda(cudaEventCreateWithFlags(&events[1], cudaEventDisableTiming),
                   "create tp staging completion event 1");
        check_cuda(cudaSetDevice(restored_device), "restore current device for tp staging");
    }

    ~Slot() {
        // Drain before release: a slot destroyed while a copy is still in flight would free
        // the pinned backing under that copy. The destructor must not throw, so the drain
        // ignores the (already-failed) status here.
        for (int stream = 0; stream < 2; ++stream) {
            if (pending[stream]) {
                (void)cudaEventSynchronize(events[stream]);
                pending[stream] = false;
            }
            if (events[stream]) { (void)cudaEventDestroy(events[stream]); }
        }
    }

    // The next producer may only overwrite a slot whose enqueued copies have completed on
    // every device that reads from it.
    void wait() {
        for (int stream = 0; stream < 2; ++stream) {
            if (pending[stream]) {
                check_cuda(cudaEventSynchronize(events[stream]), "wait for tp staging transfer");
                pending[stream] = false;
            }
        }
    }

    void issue(DeviceContext& device, int stream) {
        check_cuda(cudaEventRecord(events[stream], device.transfer_stream),
                   "record tp staging completion");
        pending[stream] = true;
    }

    PinnedHostBuffer buffer;
    cudaEvent_t events[2] = {nullptr, nullptr};
    bool pending[2]       = {false, false};
};

// One opaque piece scheduled through the ring: copy the bytes out of the caller's payload
// into the next slot, then enqueue the H2D copies into every target (shard destination +
// owning device). Replication is a single stage_piece call with both shard targets; a split
// weight stages one call per shard from its own shard payload.
struct UploadTarget {
    std::byte* data = nullptr;       // device arena address for this piece
    DeviceContext* device = nullptr; // owning shard context (transfer stream + event slot)
    int stream = 0;                  // shard index; selects the slot's event half
    const char* operation = nullptr; // failure label for the cudaMemcpyAsync
};

void stage_piece(std::span<Slot> ring, std::size_t& cursor, const std::byte* source,
                 std::size_t bytes, std::span<const UploadTarget> targets) {
    if (bytes == 0 || targets.empty()) { return; }
    // One slot holds at most one piece, so a payload larger than kSlotBytes walks the ring
    // one piece at a time; every piece keeps its own offset in every destination, and a
    // replicated piece still enqueues all shard copies from the same slot.
    std::size_t offset = 0;
    while (offset < bytes) {
        const std::size_t piece = std::min(bytes - offset, kSlotBytes);
        Slot& slot = ring[cursor++ % ring.size()];
        slot.wait();
        std::memcpy(slot.buffer.data(), source + offset, piece);
        for (const auto& target : targets) {
            check_cuda(cudaMemcpyAsync(target.data + offset, slot.buffer.data(), piece,
                                       cudaMemcpyHostToDevice, target.device->transfer_stream),
                       target.operation);
            slot.issue(*target.device, target.stream);
        }
        offset += piece;
    }
}

// Read the full payload of one device object into host storage.
std::vector<std::byte> read_object_payload(const artifact::Reader& reader,
                                           artifact::ObjectHandle handle,
                                           artifact::MaterializationStats& stats) {
    auto payload = reader.read_object(handle);
    stats.read_bytes += payload.size();
    return payload;
}

// Shard geometry: recompute the encoded planes for a half-size shape (NVFP4 or BF16).
WeightGeometry shard_geometry(const WeightGeometry& full, std::int32_t n,
                               std::int32_t k) {
    WeightGeometry g = full;
    g.shape          = {static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(k)};
    g.elements       = static_cast<std::uint64_t>(n) * k;
    // padded_columns tracks k for all non-RowSplit layouts (NVFP4, FP8 RowScale, BF16);
    // group_size tracks k for FP8 RowScale (one scale per row spanning all k columns).
    g.padded_columns = static_cast<std::uint64_t>(k);
    if (full.layout == QuantLayout::RowScale) { g.group_size = static_cast<std::uint64_t>(k); }
    if (full.layout == QuantLayout::RowSplit) {
        // Grouped integer formats (Q4/Q5/Q6_G64, Q8_G32) pack code words, optional high bits and
        // one FP16 scale per group into separate planes of a row block. The layout derives entirely
        // from the format and the shape, so reuse the shared geometry function.
        const std::uint64_t shape[2] = {static_cast<std::uint64_t>(n),
                                        static_cast<std::uint64_t>(k)};
        WeightGeometry g = weight_geometry(full.format, full.layout, shape);
        g.alignment      = full.alignment;
        return g;
    }
    if (full.layout == QuantLayout::GgufBlocks) {
        // A GGUF block weight derives its planes entirely from its format and shape, exactly as the
        // RowSplit branch above does. This must run before any arena is sized: the generic
        // element-width fallback below would charge four bytes per element for a ~0.44 B/element
        // format and over-allocate a shard by about nine times, which surfaces as a cudaMalloc OOM
        // rather than as the actual condition. A shard whose K is not a whole number of blocks has no
        // representation and fails here with that reason.
        const std::uint64_t shape[2] = {static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(k)};
        WeightGeometry g = weight_geometry(full.format, full.layout, shape);
        g.alignment      = full.alignment;
        return g;
    }
    const auto align256 = [](std::uint64_t v) { return (v + 255) / 256 * 256; };
    if (full.layout == QuantLayout::BlockScaleK16M128x4) {
        g.code_bytes_per_row = static_cast<std::uint64_t>(k) / 2;
        g.code_bytes         = g.elements / 2;
        g.scale_offset       = align256(g.code_bytes);
        g.scale_bytes        = g.elements / 16;
        g.divisor_offset     = g.scale_offset + g.scale_bytes;
        g.bytes              = g.divisor_offset + 4;
    } else if (full.layout == QuantLayout::RowScale) {
        g.code_bytes_per_row = static_cast<std::uint64_t>(k);
        g.code_bytes         = g.elements;
        g.scale_offset       = align256(g.code_bytes);
        g.scale_bytes        = static_cast<std::uint64_t>(n) * 2;
        g.divisor_offset     = 0;
        g.bytes              = g.scale_offset + g.scale_bytes;
    } else {
        const std::uint64_t wb = full.format == QType::BF16 ? 2 : 4;
        g.code_bytes = g.bytes = g.elements * wb;
        g.divisor_offset       = 0;
    }
    return g;
}

// Device bytes one shard of `placement` occupies under `spec`, computed analytically from the
// split geometry (no payload read). Replicated objects count their full size on every shard.
std::uint64_t object_shard_bytes(const artifact::Reader& reader,
                                 const artifact::DevicePlacement& placement,
                                 const TPSplitSpec& spec) {
    const auto& geometry = reader.geometry(placement.object);
    const std::uint64_t n = geometry.shape.size() >= 1 ? geometry.shape[0] : 1;
    const std::uint64_t k = geometry.shape.size() >= 2 ? geometry.shape[1] : 1;
    const auto kind = spec.kind(placement.object);
    if (kind == WeightSplitKind::Replicated) { return geometry.bytes; }
    std::int32_t shard_n = static_cast<std::int32_t>(n);
    std::int32_t shard_k = static_cast<std::int32_t>(k);
    if (kind == WeightSplitKind::GatherRows || kind == WeightSplitKind::GatherCols) {
        const auto* split = spec.find(placement.object);
        if (split && !split->parts.empty()) {
            std::int32_t half = 0;
            for (const auto& part : split->parts) { half += part.row_count / 2; }
            if (kind == WeightSplitKind::GatherRows) { shard_n = half; }
            else { shard_k = half; }
        }
    } else if (kind == WeightSplitKind::ColumnParallel) {
        shard_n = static_cast<std::int32_t>(n / 2);
    } else if (kind == WeightSplitKind::RowParallel) {
        shard_k = static_cast<std::int32_t>(k / 2);
    }
    return shard_geometry(geometry, shard_n, shard_k).bytes;
}

// Total device bytes one shard arena must hold for the whole plan, plus per-object alignment slack.
// An object placed on the other shard alone contributes nothing to this shard's arena.
std::uint64_t shard_capacity_bytes(const artifact::Reader& reader,
                                   const artifact::MaterializationPlan& plan,
                                   const TPSplitSpec& spec, int shard) {
    std::uint64_t bytes = 0;
    for (const auto& placement : plan.device_objects) {
        if (!spec.on_shard(placement.object, shard)) { continue; }
        bytes += object_shard_bytes(reader, placement, spec);
        bytes += 256; // per-object alignment headroom
    }
    return bytes;
}

// Device objects this shard actually materializes (shard-local components are excluded elsewhere).
std::uint64_t shard_object_count(const artifact::MaterializationPlan& plan, const TPSplitSpec& spec,
                                 int shard) {
    std::uint64_t count = 0;
    for (const auto& placement : plan.device_objects) {
        if (spec.on_shard(placement.object, shard)) { ++count; }
    }
    return count;
}

// Build a complete Weight descriptor for the full parent, so split_weight can slice it.
Weight make_full_weight(const std::vector<std::byte>& payload,
                        const WeightGeometry& geometry, float divisor) {
    const std::int32_t n = static_cast<std::int32_t>(geometry.shape[0]);
    const std::int32_t k = static_cast<std::int32_t>(geometry.shape[1]);
    Weight weight;
    weight.qtype            = geometry.format;
    weight.layout           = geometry.layout;
    weight.group_size       = static_cast<std::uint32_t>(geometry.group_size);
    weight.group            = static_cast<std::int32_t>(geometry.group_size);
    weight.shape[0]         = n;
    weight.shape[1]         = k;
    weight.padded_shape[0]  = n;
    weight.padded_shape[1]  = k;
    weight.ndim             = 2;
    weight.n                = n;
    weight.k                = k;
    weight.payload          = payload.data();
    weight.payload_bytes    = payload.size();
    weight.qdata            = payload.data();
    weight.scales           = payload.data() + geometry.scale_offset;
    weight.scale_dtype      = geometry.format == QType::NVFP4 ? DType::FP8_E4M3FN : DType::BF16;
    weight.weight_scale_divisor = divisor;
    return weight;
}

} // namespace

std::pair<artifact::MaterializedArtifact, artifact::MaterializedArtifact> materialize_tp2(
    const artifact::Reader& reader, const artifact::MaterializationPlan& plan,
    DeviceContext& device0, DeviceContext& device1, const TPSplitSpec& spec,
    const StartupObserver* startup_observer) {
    (void)startup_observer;
    if (plan.source != &reader || plan.object_count != reader.directory().objects.size()) {
        throw artifact::ArtifactError("tp materialization plan belongs to another load session");
    }
    artifact::MaterializedArtifact a0, a1;
    // Each shard arena holds only its own shard bytes (the full-model capacity would over-allocate
    // ~13GB per GPU and OOM on 16GB cards). DeviceArena's cudaMalloc lands on the current device,
    // so bind each shard's device before its arena is created.
    const std::uint64_t capacity0 = shard_capacity_bytes(reader, plan, spec, 0);
    const std::uint64_t capacity1 = shard_capacity_bytes(reader, plan, spec, 1);
    device0.bind_to_current_thread();
    a0.tp_init(plan.object_count, capacity0);
    device1.bind_to_current_thread();
    a1.tp_init(plan.object_count, capacity1);
    a0.tp_stats().file_bytes          = reader.file_bytes();
    a1.tp_stats().file_bytes          = reader.file_bytes();
    a0.tp_stats().device_object_count = shard_object_count(plan, spec, 0);
    a1.tp_stats().device_object_count = shard_object_count(plan, spec, 1);
    auto& a0_objects = a0.tp_objects();
    auto& a1_objects = a1.tp_objects();
    auto& a0_arena   = a0.tp_arena();
    auto& a1_arena   = a1.tp_arena();
    auto& a0_stats   = a0.tp_stats();
    auto& a1_stats   = a1.tp_stats();

    // The staging ring outlives every object's pieces; a C array (not a vector — MSVC
    // instantiates _Uninitialized_move for vector<T>::reserve and demands a move ctor the
    // pinned slots cannot have) declares exactly kSlotCount portable slots, each readable by
    // both device contexts and destroyed in reverse order at scope exit, draining any copy
    // still in flight before its buffer is freed.
    Slot staging[kSlotCount] = {Slot(kSlotBytes, device0, device1), Slot(kSlotBytes, device0, device1),
                                Slot(kSlotBytes, device0, device1), Slot(kSlotBytes, device0, device1)};
    std::span<Slot> ring = staging;
    std::size_t staging_cursor = 0;
    // The ring doubles as the pipeline's touchdown footprint: report its full size the way
    // the single-card materializer reports its staging (materializer.cpp peak_staging_bytes)
    // so the load summary proves the pinned path is actually engaged.
    const std::uint64_t staged_bytes = kSlotBytes * kSlotCount;
    a0.tp_stats().peak_staging_bytes = staged_bytes;
    a1.tp_stats().peak_staging_bytes = staged_bytes;

    for (const auto& placement : plan.device_objects) {
        const auto& geometry = reader.geometry(placement.object);
        const std::uint64_t n = geometry.shape.size() >= 1 ? geometry.shape[0] : 1;
        const std::uint64_t k = geometry.shape.size() >= 2 ? geometry.shape[1] : 1;
        auto payload         = read_object_payload(reader, placement.object, a0_stats);
        const float divisor =
            geometry.format == QType::NVFP4
                ? std::bit_cast<float>(artifact::read_u32_le(payload.data() + geometry.divisor_offset))
                : 0.0F;

        const auto kind = spec.kind(placement.object);
        if (kind == WeightSplitKind::Replicated) {
            // Replication stages one piece and enqueues both shard copies from the same slot.
            UploadTarget targets[2];
            int target_count = 0;
            for (int shard = 0; shard < 2; ++shard) {
                if (!spec.on_shard(placement.object, shard)) { continue; }
                DeviceArena& arena  = shard == 0 ? a0_arena : a1_arena;
                auto& storage =
                    shard == 0 ? a0_objects.at(placement.object.index)
                               : a1_objects.at(placement.object.index);
                artifact::MaterializationStats& stats = shard == 0 ? a0_stats : a1_stats;
                DeviceContext& device                 = shard == 0 ? device0 : device1;
                auto span = arena.alloc_bytes(static_cast<std::size_t>(geometry.bytes),
                                              static_cast<std::size_t>(geometry.alignment));
                targets[target_count++] = UploadTarget{
                    static_cast<std::byte*>(span.data), &device, shard, "upload tp replicated weight"};
                storage.device =
                    WeightParent{geometry, static_cast<const std::byte*>(span.data),
                                 divisor};
                stats.h2d_bytes += geometry.bytes;
            }
            stage_piece(staging, staging_cursor, payload.data(),
                        static_cast<std::size_t>(geometry.bytes), std::span(targets, target_count));
            continue;
        }

        std::span<const std::uint8_t> payload_span(
            reinterpret_cast<const std::uint8_t*>(payload.data()), payload.size());
        const Weight full_weight = make_full_weight(payload, geometry, divisor);
        std::vector<WeightShard> shards;
        std::int32_t shard_n = static_cast<std::int32_t>(n);
        std::int32_t shard_k = static_cast<std::int32_t>(k);
        if (kind == WeightSplitKind::GatherRows) {
            const auto* split = spec.find(placement.object);
            if (!split || split->parts.empty()) {
                throw artifact::ArtifactError(
                    "tp gather split is missing its row parts");
            }
            std::span<const RowPart> parts(split->parts.data(), split->parts.size());
            shards = {gather_weight_rows(payload_span, full_weight, parts, 0),
                      gather_weight_rows(payload_span, full_weight, parts, 1)};
            shard_n = shards[0].weight.n;
        } else if (kind == WeightSplitKind::GatherCols) {
            const auto* split = spec.find(placement.object);
            if (!split || split->parts.empty()) {
                throw artifact::ArtifactError(
                    "tp gather split is missing its column parts");
            }
            std::span<const RowPart> parts(split->parts.data(), split->parts.size());
            shards = {gather_weight_cols(payload_span, full_weight, parts, 0),
                      gather_weight_cols(payload_span, full_weight, parts, 1)};
            shard_k = shards[0].weight.k;
        } else {
            shards = split_weight(payload_span, full_weight, kind);
            shard_n = kind == WeightSplitKind::ColumnParallel ? static_cast<std::int32_t>(n / 2)
                                                              : static_cast<std::int32_t>(n);
            shard_k = kind == WeightSplitKind::ColumnParallel ? static_cast<std::int32_t>(k)
                                                              : static_cast<std::int32_t>(k / 2);
        }
        const auto shard_geo = shard_geometry(geometry, shard_n, shard_k);
        for (int shard = 0; shard < 2; ++shard) {
            if (!spec.on_shard(placement.object, shard)) { continue; }
            DeviceArena& arena  = shard == 0 ? a0_arena : a1_arena;
            auto& storage =
                shard == 0 ? a0_objects.at(placement.object.index)
                           : a1_objects.at(placement.object.index);
            artifact::MaterializationStats& stats = shard == 0 ? a0_stats : a1_stats;
            DeviceContext& device                 = shard == 0 ? device0 : device1;
            const std::size_t bytes = shards[shard].payload.size();
            auto span = arena.alloc_bytes(bytes,
                                          static_cast<std::size_t>(shard_geo.alignment));
            // Each shard piece takes its own slot: the two payloads are disjoint host ranges,
            // and staging them through one slot would serialize the ring for no reason.
            const UploadTarget target{static_cast<std::byte*>(span.data), &device, shard,
                                      "upload tp shard"};
            stage_piece(staging, staging_cursor,
                        reinterpret_cast<const std::byte*>(shards[shard].payload.data()), bytes,
                        std::span(&target, 1));
            storage.device = WeightParent{shard_geo,
                                          static_cast<const std::byte*>(span.data),
                                          divisor};
            stats.h2d_bytes += bytes;
        }
    }

    // The weights must be fully resident before the engine takes ownership: drain the ring
    // and both transfer streams. Nothing above synchronized per object — a later object's
    // reads already overlapped this one's transfers.
    for (auto& slot : staging) { slot.wait(); }
    check_cuda(cudaStreamSynchronize(device0.transfer_stream), "complete tp uploads");
    check_cuda(cudaStreamSynchronize(device1.transfer_stream), "complete tp uploads");
    return {std::move(a0), std::move(a1)};
}

} // namespace ninfer::tp
