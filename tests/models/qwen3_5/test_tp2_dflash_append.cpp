// TP-2 DFlash2 masked-draft round (tp2-plan.md section 3.6, stage B4).
//
// The engine gate keeps --spec dflash2 off the TP-2 route until the masked draft has its own verify
// wiring (src/runtime/engine/model_instance.cpp:101-104, deliberately kept), so this test drives the
// production round component the runtime core itself builds
// (src/models/qwen3_5/program/dflash_round.h) instead of opening that gate. The component owns the
// draft's persistent context (prefill target features, positions, pending staging and the local K/V
// ring), its exact-B decode frame and a proposal workspace of its own, and it sequences the three
// steps a request makes against them: the target prefill's feature sink, the append of each captured
// chunk, and the production masked-block proposal. The loader-only path (plan_load(DFlash2) +
// materialize_model_tp2) is unchanged; no Engine option normalization is involved.
//
// The single-card draft-only oracle is not constructible: the loader binds the whole text component
// unconditionally (src/models/qwen3_5/load.cpp:56) and Parameters::Parameters prepares the full text
// stack (src/models/qwen3_5/execution/parameters.cpp:260-270), which does not fit one 16 GiB card
// (the smallest local 27B artifact is 17.4 GB). Accepting the TP-2 route therefore takes its
// evidence on both cards, and this test establishes, on the real artifact:
//
//   1. the sink does not perturb the target: the last-column logits of an identical prefill are
//      bit-identical with and without the sink installed (both shards);
//   2. the append chain completes: the round's consumer runs exactly once and the sink captured
//      every configured feature layer and its positions;
//   3. the draft ring is finite, non-zero and structurally sane: K BF16 / V FP16 with the configured
//      geometry, the addressed prefix [0, chunk) written for all five layers, the untouched capacity
//      remainder exactly zero, and the ring bit-identical to the same features appended directly;
//   4. the append is chunk-width independent: one width=chunk append and two width=chunk/2 appends
//      over the same captured features produce a bit-identical ring;
//   5. the same input, prefilled twice from a zeroed round, reproduces the ring bit for bit;
//   6. the chunk=1024 append fits the shipped 192 MiB workspace (tp2_generation_core.cpp:148), with
//      the peak reported against that budget;
//   7. the assembled round's proposal produces the documented stage-B3 outputs: K drafts, [16,K,B]
//      candidate ids and proposal q, and the [K+1,B] masked query positions; every candidate row is
//      distinct and in the public token domain, every draft is one of its position's candidates,
//      and the greedy selector's q is the exact one-hot distribution that names the draft;
//   8. the same ring proposed twice, and the whole prefill -> append -> proposal chain re-run from a
//      zeroed round, agree bit for bit (hash reported, cross-process compared);
//   9. the round leaves the shard workspace untouched: propose_dflash2_batch resets the arena it is
//      handed (execution/draft.cpp:284), so the component must pass its own; a poisoned resident
//      prefill_hidden survives a proposal bit for bit and the resident arena watermark does not move;
//  10. its cost: the context, the frame, the round's own proposal arena and the transient peak, plus
//      CUDA-event wall time per proposal, against the shipped 192 MiB arena.
//
// What it does not establish: the ring values are not checked against an independent host oracle
// that decodes the stored q8_g32_fp16 feature/context weights with their scales, so a wrong-but-
// deterministic fused result would pass; and there is no acceptance oracle for the drafts
// themselves (that needs the target verify/accept stage, B5). The selector's Engine publication, the
// verify/accept/fold loop and the session/checkpoint state are deliberately not part of this stage.
//
// The artifact is selected with NINFER_TEST_ARTIFACT; two identical sm_120a devices are required.
// Without either the test skips with exit code 77.

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "core/linear_attention_state.h"
#include "core/tp/device_pair.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/dflash_round.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "ninfer/ops/position.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace qwen = ninfer::models::qwen3_5;

// Mirrors the shipped runtime budget (tp2_generation_core.cpp:148). The point of the test is to run
// the chunk=1024 sink prefill inside exactly the arena the product route gives it, not a test-sized
// one.
constexpr std::size_t kWorkspaceBytes = 192ULL << 20;
constexpr std::uint32_t kTestCacheTokens      = 2048;
constexpr std::uint32_t kTestCachePages       = kTestCacheTokens / 64;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::pair<int, int> pick_devices() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { return {-1, -1}; }
    // Only sm_120a devices can run the TP-2 kernels; a foreign device in the enumeration must not
    // mask the pair we can actually use.
    std::vector<int> candidates;
    std::vector<std::string> names(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, index) != cudaSuccess) { return {-1, -1}; }
        if (prop.major != 12) { continue; }
        candidates.push_back(index);
        names[static_cast<std::size_t>(index)] = prop.name;
    }
    for (std::size_t a = 0; a < candidates.size(); ++a) {
        for (std::size_t b = a + 1; b < candidates.size(); ++b) {
            const std::size_t left  = static_cast<std::size_t>(candidates[a]);
            const std::size_t right = static_cast<std::size_t>(candidates[b]);
            if (names[left] == names[right]) { return {candidates[a], candidates[b]}; }
        }
    }
    return {-1, -1};
}

// Minimal per-shard execution state, mirroring the runtime core's build_shard: a paged KV cache
// large enough for the test chunk, a zeroed GDN state pool, and a workspace arena of exactly the
// shipped size.
struct ShardState {
    std::unique_ptr<DeviceArena> kv_arena;
    std::unique_ptr<qwen::DecoderState> decoder;
    std::unique_ptr<DeviceArena> state_arena;
    std::unique_ptr<LinearAttentionStatePool> state;
    DeviceSpan state_backing;
    std::vector<DeviceKVPageLease> kv_pages;
    std::vector<DeviceKVPageHandle> kv_handles;
    KVExecutionRowLease kv_row;
    std::unique_ptr<DeviceArena> workspace;
    qwen::RoundState io;
    Tensor prefill_hidden;
};

ShardState build_shard_state(DeviceContext& device, const qwen::TextConfig& config) {
    ShardState shard;
    LayoutBuilder kv_builder;
    const qwen::DecoderStateLayout kv_layout = qwen::plan_decoder_state(
        kv_builder, qwen::DecoderStateSpec{
                        .full_attention_layers = config.full_attention_layers,
                        .mtp_layers            = 1,
                        .capacity              = kTestCacheTokens,
                        .kv_heads = qwen::execution::dimension(config.attention->num_key_value_heads),
                        .attention_head_dim = qwen::execution::dimension(config.attention->head_dim),
                        .kv_storage         = KvCacheStorage::BFloat16,
                        .enable_mtp         = false,
                        .kv_table_rows      = 1,
                        .text_physical_page_groups = kTestCachePages,
                        .mtp_physical_page_groups  = 0,
                    });
    const std::size_t kv_bytes = kv_builder.finish(256);
    device.bind_to_current_thread();
    shard.kv_arena = std::make_unique<DeviceArena>(kv_bytes);
    shard.decoder  = std::make_unique<qwen::DecoderState>(shard.kv_arena->alloc_bytes(kv_bytes, 256),
                                                          kv_layout);

    const LinearAttentionStatePoolSpec gdn_spec{
        .layers         = config.linear_attention_layers,
        .conv_channels  = (config.gdn ? qwen::execution::dimension(config.gdn->conv_channels()) : 0),
        .conv_width     = (config.gdn ? qwen::execution::dimension(config.gdn->linear_conv_kernel_dim - 1) : 0),
        .value_heads    = (config.gdn ? qwen::execution::dimension(config.gdn->linear_num_value_heads) : 0),
        .value_head_dim = (config.gdn ? qwen::execution::dimension(config.gdn->linear_value_head_dim) : 0),
        .key_head_dim   = (config.gdn ? qwen::execution::dimension(config.gdn->linear_key_head_dim) : 0),
        .slot_count     = 1,
        .conv_dtype     = DType::BF16,
    };
    LayoutBuilder state_builder;
    const LinearAttentionStatePoolLayout state_layout =
        plan_linear_attention_state_pool(state_builder, gdn_spec);
    const std::size_t state_bytes = state_builder.finish(256);
    device.bind_to_current_thread();
    shard.state_arena   = std::make_unique<DeviceArena>(state_bytes);
    shard.state_backing = shard.state_arena->alloc_bytes(state_bytes, 256);
    CUDA_CHECK(cudaMemset(shard.state_backing.data, 0, state_bytes));
    shard.state = std::make_unique<LinearAttentionStatePool>(shard.state_backing, state_layout);

    device.bind_to_current_thread();
    shard.workspace = std::make_unique<DeviceArena>(kWorkspaceBytes);
    shard.prefill_hidden =
        shard.workspace->alloc(DType::BF16, {2 * qwen::execution::dimension(config.hidden_size), 1});
    return shard;
}

// The draft context, the decode frame and the proposal workspace are owned by the production round
// component (models/qwen3_5/program/dflash_round.h); the test constructs it rather than laying
// those buffers out itself, so what it exercises is the assembly the runtime core also uses.
std::vector<std::int32_t> read_i32(DeviceContext& device, const Tensor& tensor) {
    device.bind_to_current_thread();
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    std::vector<std::int32_t> out(static_cast<std::size_t>(tensor.bytes() / sizeof(std::int32_t)));
    CUDA_CHECK(cudaMemcpy(out.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
    return out;
}

std::vector<float> read_fp32(DeviceContext& device, const Tensor& tensor) {
    device.bind_to_current_thread();
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    std::vector<float> out(static_cast<std::size_t>(tensor.bytes() / sizeof(float)));
    CUDA_CHECK(cudaMemcpy(out.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
    return out;
}

std::size_t weight_bytes(const ninfer::Weight& weight) {
    return static_cast<std::size_t>(weight.payload_bytes + weight.high_plane_bytes);
}

std::size_t linear_bytes(const qwen::execution::LinearParameters& parameter) {
    return weight_bytes(parameter.weight);
}

// Resident bytes of the whole draft block that the proposal reads on this shard: the executable
// draft weights, the reduced proposal head it resolves its top-k against, and the selector when this
// shard is the one that materialized it (TP-2 places it on the peer).
std::size_t draft_weight_bytes(const qwen::execution::DraftParameters& draft,
                               const qwen::execution::SelectorParameters* selector) {
    std::size_t total =
        linear_bytes(draft.feature_projection) + draft.context_norm.bytes() + draft.final_norm.bytes();
    for (const auto& layer : draft.layers) {
        total += layer.input_norm.bytes() + layer.post_attention_norm.bytes() +
                 layer.query_norm.bytes() + layer.key_norm.bytes();
        // The fused QKV parent and its context_key/context_value row views are one allocation;
        // count it once through whichever binding the artifact materializes.
        if (layer.query_key_value) {
            total += linear_bytes(*layer.query_key_value);
        } else {
            total += linear_bytes(layer.context_key);
        }
        total += linear_bytes(layer.output);
        total += linear_bytes(layer.mlp.gate_up) + linear_bytes(layer.mlp.down);
        if (layer.attention_conv) {
            total += layer.attention_conv->base_kernel.bytes() +
                     linear_bytes(layer.attention_conv->kernel_projection);
        }
        if (layer.mlp_conv) {
            total += layer.mlp_conv->base_kernel.bytes() +
                     linear_bytes(layer.mlp_conv->kernel_projection);
        }
    }
    if (selector != nullptr) {
        total += linear_bytes(selector->hidden_projection) +
                 selector->predecessor_codebook.bytes() + selector->successor_codebook.bytes();
    }
    total += linear_bytes(draft.output_head);
    return total;
}

const char* qtype_name(QType type) {
    switch (type) {
    case QType::Q4_G64_FP16: return "q4_g64_fp16";
    case QType::Q5_G64_FP16: return "q5_g64_fp16";
    case QType::Q6_G64_FP16: return "q6_g64_fp16";
    case QType::Q8_G32_FP16: return "q8_g32_fp16";
    case QType::BF16: return "bf16";
    case QType::FP32: return "fp32";
    case QType::INT32: return "int32";
    case QType::NVFP4: return "nvfp4";
    case QType::FP8_E4M3FN_ROW_BF16: return "fp8_e4m3fn_row_bf16";
    }
    return "unknown";
}

const char* dtype_name(DType type) {
    switch (type) {
    case DType::BF16: return "bf16";
    case DType::FP32: return "fp32";
    case DType::I32: return "i32";
    case DType::U8: return "u8";
    case DType::I64: return "i64";
    case DType::I8: return "i8";
    case DType::FP16: return "fp16";
    case DType::FP8_E4M3FN: return "fp8_e4m3fn";
    }
    return "unknown";
}

std::vector<std::uint16_t> read_bf16(DeviceContext& device, const Tensor& tensor) {
    device.bind_to_current_thread();
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    std::vector<std::uint16_t> out(static_cast<std::size_t>(tensor.bytes() / sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMemcpy(out.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
    return out;
}

struct RingSnapshot {
    std::vector<std::vector<std::byte>> k;
    std::vector<std::vector<std::byte>> v;
    std::uint64_t hash = 0;
    bool operator==(const RingSnapshot& other) const {
        return k == other.k && v == other.v;
    }
};

std::uint64_t fnv1a(std::span<const std::byte> data) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const std::byte byte : data) {
        hash ^= static_cast<std::uint64_t>(std::to_integer<unsigned char>(byte));
        hash *= 1099511628211ULL;
    }
    return hash;
}

RingSnapshot read_ring(DeviceContext& device, const CyclicKVCache& ring) {
    RingSnapshot out;
    device.bind_to_current_thread();
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    std::uint64_t hash = 1469598103934665603ULL;
    const auto feed      = [&hash](const std::vector<std::byte>& bytes) {
        hash ^= fnv1a(bytes);
        hash *= 1099511628211ULL;
    };
    for (std::uint32_t layer = 0; layer < ring.layer_count(); ++layer) {
        const CyclicKVCacheLayerView view = ring.layer_view(layer);
        std::vector<std::byte> k(view.k.bytes());
        std::vector<std::byte> v(view.v.bytes());
        CUDA_CHECK(cudaMemcpy(k.data(), view.k.data, k.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v.data(), view.v.data, v.size(), cudaMemcpyDeviceToHost));
        feed(k);
        feed(v);
        out.k.push_back(std::move(k));
        out.v.push_back(std::move(v));
    }
    out.hash = hash;
    return out;
}

float bf16_at(const std::vector<std::byte>& bytes, std::size_t element) {
    std::uint16_t raw = 0;
    std::memcpy(&raw, bytes.data() + element * 2, 2);
    const __nv_bfloat16 value = *reinterpret_cast<const __nv_bfloat16*>(&raw);
    return __bfloat162float(value);
}

float fp16_at(const std::vector<std::byte>& bytes, std::size_t element) {
    std::uint16_t raw = 0;
    std::memcpy(&raw, bytes.data() + element * 2, 2);
    const __half value = *reinterpret_cast<const __half*>(&raw);
    return __half2float(value);
}

struct RingLayerStats {
    std::size_t live_elements   = 0;
    std::size_t live_nonzero    = 0;
    std::size_t live_nonfinite  = 0;
    std::size_t dead_nonzero_kv = 0;
    float max_abs               = 0.0F;
    double mean_abs             = 0.0;
};

// The ring stores dim0 = head_dim, dim1 = padded capacity, dim2 = kv heads, dim3 = lane, with dim0
// contiguous. Addresses [0, chunk) are live after a from-scratch append of chunk positions; the
// untouched remainder must still be zero.
RingLayerStats ring_layer_stats(const std::vector<std::byte>& bytes, bool is_fp16,
                                std::int32_t head_dim, std::int32_t padded_capacity,
                                std::int32_t kv_heads, std::int32_t lane_capacity, std::int32_t chunk,
                                bool* dead_ok) {
    RingLayerStats stats;
    for (std::int32_t p = 0; p < padded_capacity; ++p) {
        for (std::int32_t d = 0; d < head_dim; ++d) {
            for (std::int32_t h = 0; h < kv_heads; ++h) {
                for (std::int32_t lane = 0; lane < lane_capacity; ++lane) {
                    // Tensor ne[0] is the contiguous axis, so the element offset is
                    // d + D * (p + P * (h + H * lane)) for ne = [D, P, H, lane].
                    const std::size_t element =
                        static_cast<std::size_t>(d) +
                        static_cast<std::size_t>(head_dim) *
                            (static_cast<std::size_t>(p) +
                             static_cast<std::size_t>(padded_capacity) *
                                 (static_cast<std::size_t>(h) +
                                  static_cast<std::size_t>(kv_heads) * lane));
                    const float value = is_fp16 ? fp16_at(bytes, element) : bf16_at(bytes, element);
                    if (p < chunk) {
                        ++stats.live_elements;
                        if (value != 0.0F) { ++stats.live_nonzero; }
                        if (!std::isfinite(value)) { ++stats.live_nonfinite; }
                        const float magnitude = std::fabs(value);
                        stats.max_abs         = std::max(stats.max_abs, magnitude);
                        stats.mean_abs += static_cast<double>(magnitude);
                    } else if (value != 0.0F) {
                        ++stats.dead_nonzero_kv;
                    }
                }
            }
        }
    }
    if (stats.live_elements != 0) {
        stats.mean_abs /= static_cast<double>(stats.live_elements);
    }
    *dead_ok = *dead_ok && stats.dead_nonzero_kv == 0;
    return stats;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path path;
        std::int32_t chunk = 1024;
        std::int32_t proposal_drafts = 7;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const auto value = [&]() -> std::string {
                if (++i >= argc) { throw std::invalid_argument("missing value for " + arg); }
                return argv[i];
            };
            if (arg == "--artifact") {
                path = value();
            } else if (arg == "--chunk") {
                chunk = std::stoi(value());
            } else if (arg == "--k") {
                proposal_drafts = std::stoi(value());
            } else if (arg == "--help") {
                std::cout << "--artifact PATH [--chunk N] [--k N]\n";
                return 0;
            } else {
                throw std::invalid_argument("unknown argument " + arg);
            }
        }
        if (path.empty()) {
            if (const char* env = std::getenv("NINFER_TEST_ARTIFACT")) { path = env; }
        }
        if (path.empty()) {
            std::cout << "SKIP: supply an explicit --artifact or set NINFER_TEST_ARTIFACT\n";
            return 77;
        }
        if (chunk < 64 || chunk > 1024 || chunk % 64 != 0) {
            throw std::invalid_argument("--chunk must be a multiple of 64 in [64,1024]");
        }
        if (proposal_drafts < 1 || proposal_drafts > 15) {
            throw std::invalid_argument("--k must be in [1,15]");
        }
        const auto [dev0, dev1] = pick_devices();
        if (dev0 < 0) {
            std::cout << "SKIP: need two identical sm_120a devices\n";
            return 77;
        }
        std::unique_ptr<DeviceContext> device0(new DeviceContext(dev0));
        std::unique_ptr<DeviceContext> device1(new DeviceContext(dev1));

        artifact::Reader reader(path);
        LoadOptions options;
        options.vision      = false;
        options.speculative = SpeculativeBackend::DFlash2;
        auto plan = qwen::plan_load(reader, options);
        auto [model0, model1] =
            qwen::materialize_model_tp2(std::move(plan), *device0, *device1, nullptr);
        require(model0->weights().draft.has_value(), "shard 0 did not materialize the DFlash2 draft");
        require(!model1->has_weight(model0->weights().draft->feature_projection),
                "shard 1 materialized the shard-local DFlash2 draft");
        qwen::execution::Parameters parameters0(*model0);
        qwen::execution::Parameters parameters1(*model1);
        require(parameters0.draft.has_value(), "shard 0 DFlash draft parameters are missing");
        require(!parameters1.draft.has_value(),
                "shard 1 built DFlash draft parameters without the weights");

        const auto& target = model0->config().text;
        const auto& draft  = *model0->config().draft;
        require(draft.dflash2.has_value(), "artifact draft is not DFlash2");
        require(draft.local_layer_count() == draft.num_hidden_layers,
                "test assumes an all-sliding DFlash2 draft");
        const std::int32_t hidden = qwen::execution::dimension(target.hidden_size);
        const std::int32_t vocab  = qwen::execution::dimension(target.vocab_size);
        const std::int32_t kv_heads =
            qwen::execution::dimension(draft.attention.num_key_value_heads);
        const std::int32_t head_dim = qwen::execution::dimension(draft.attention.head_dim);
        const std::int32_t layers   = static_cast<std::int32_t>(draft.local_layer_count());
        std::cout << path.filename().string() << ": TP-2 DFlash2 append | devices=" << dev0 << ","
                  << dev1 << " chunk=" << chunk << " target_layers=" << draft.target_layer_ids.size()
                  << " hidden=" << hidden << " ring_layers=" << layers
                  << " kv_heads=" << kv_heads << " head_dim=" << head_dim
                  << " window=" << draft.sliding_window.value_or(0) << "\n";

        // Per-shard config: the mixer head counts are halved, as the runtime core does, so the KV
        // cache and GDN pool are sized for the per-shard geometry while the replicated components
        // keep the full-model config.
        auto shard_cfg = [](const qwen::TextConfig& full) {
            auto cfg = std::make_unique<qwen::TextConfig>(full);
            if (cfg->attention) {
                cfg->attention->num_attention_heads /= 2;
                cfg->attention->num_key_value_heads /= 2;
            }
            if (cfg->gdn) {
                cfg->gdn->linear_num_key_heads /= 2;
                cfg->gdn->linear_num_value_heads /= 2;
            }
            return cfg;
        };
        auto shard_cfg0 = shard_cfg(target);
        auto shard_cfg1 = shard_cfg(target);
        const qwen::TextConfig& cfg0 = *shard_cfg0;
        const qwen::TextConfig& cfg1 = *shard_cfg1;

        ShardState shard0 = build_shard_state(*device0, cfg0);
        ShardState shard1 = build_shard_state(*device1, cfg1);
        tp::DevicePair pair(dev0, dev1);

        qwen::execution::TextContext card0(*device0, parameters0, *shard0.workspace, {}, *shard0.state,
                                           shard0.io, shard0.prefill_hidden, chunk, 0, {},
                                           &shard0.decoder->text_kv);
        qwen::execution::TextContext card1(*device1, parameters1, *shard1.workspace, {}, *shard1.state,
                                           shard1.io, shard1.prefill_hidden, chunk, 0, {},
                                           &shard1.decoder->text_kv);
        card0.set_shard_config(&cfg0, 0);
        card1.set_shard_config(&cfg1, 1);
        // The masked draft proposes on shard 0 alone, but the text token embedding it shares is
        // column-split. Registering the pair is what lets embedding_full_width gather the peer half.
        card0.set_tp_peer(&card1, &pair);
        card1.set_tp_peer(&card0, &pair);

        // Publish one fixed physical KV row per shard, as the runtime core does once at startup.
        auto publish_kv_rows = [](DeviceContext* device, ShardState* shard) {
            device->bind_to_current_thread();
            auto& pool   = shard->decoder->text_kv.page_pool();
            auto& tables = shard->decoder->text_kv.execution_tables();
            auto reserved = pool.reserve(kTestCachePages);
            if (!reserved.has_value()) { throw std::logic_error("KV page reservation failed"); }
            DeviceKVPageReservation reservation = std::move(*reserved);
            shard->kv_pages.reserve(kTestCachePages);
            pool.materialize(reservation, kTestCachePages, shard->kv_pages);
            shard->kv_handles.clear();
            for (const auto& lease : shard->kv_pages) {
                shard->kv_handles.push_back(lease.handle());
            }
            shard->kv_row = tables.acquire(0);
            tables.publish(shard->kv_row.handle(), 0, shard->kv_handles, device->stream);
        };
        publish_kv_rows(device0.get(), &shard0);
        publish_kv_rows(device1.get(), &shard1);

        // The production round component owns the draft context, the exact-B decode frame and the
        // proposal workspace. The proposal arena is sized to the shipped per-shard workspace budget
        // (192 MiB), which is what the runtime core would give it, so the peak below is comparable
        // across stages.
        const qwen::execution::DFlash2RoundSpec round_spec = qwen::execution::plan_dflash2_round(
            draft, target, static_cast<std::uint32_t>(chunk),
            static_cast<std::uint32_t>(proposal_drafts), kWorkspaceBytes);
        device0->bind_to_current_thread();
        qwen::execution::DFlash2Round round(*device0, round_spec);
        const double ring_bytes = static_cast<double>(layers) * head_dim *
                                  static_cast<double>(round.ring().padded_capacity()) * kv_heads *
                                  (2.0 + 2.0);
        std::printf("[mem] shard 0 DFlash2 round: context %.1f MiB (ring %.1f MiB) | frame %.1f MiB "
                    "| proposal arena %.1f MiB\n",
                    static_cast<double>(round.context_bytes()) / 1048576.0,
                    ring_bytes / 1048576.0,
                    static_cast<double>(round.frame_bytes()) / 1048576.0,
                    static_cast<double>(round.proposal_workspace_capacity()) / 1048576.0);

        // One execution core describes the shard's resident state; each round step hands the
        // component that state and the component substitutes its own frame and arena where the round
        // must own them.
        auto execution_core = [&]() {
            return qwen::execution::ExecutionCore{
                .device           = *device0,
                .parameters       = parameters0,
                .work             = *shard0.workspace,
                .linear_attention = *shard0.state,
                .replay_records   = nullptr,
                .io               = shard0.io,
                .prefill_hidden   = shard0.prefill_hidden,
                .prefill_chunk    = static_cast<std::uint32_t>(chunk),
                .proposal_head    = ProposalHead::Full,
            };
        };

        std::vector<int> ids(static_cast<std::size_t>(chunk));
        for (std::int32_t i = 0; i < chunk; ++i) { ids[static_cast<std::size_t>(i)] = 1000 + i % 997; }

        auto reset_state = [&]() {
            device0->bind_to_current_thread();
            CUDA_CHECK(cudaMemsetAsync(shard0.state_backing.data, 0, shard0.state_backing.bytes,
                                       device0->stream));
            device1->bind_to_current_thread();
            CUDA_CHECK(cudaMemsetAsync(shard1.state_backing.data, 0, shard1.state_backing.bytes,
                                       device1->stream));
            round.zero_context();
        };

        struct RunResult {
            std::vector<std::uint16_t> logits0;
            std::vector<std::uint16_t> logits1;
            std::size_t peak0 = 0;
            std::size_t peak1 = 0;
        };
        auto run_prefill = [&](qwen::execution::DFlashFeatureSink* sink) {
            reset_state();
            shard0.workspace->reset_peak();
            shard1.workspace->reset_peak();
            auto scope0 = shard0.workspace->scope();
            auto scope1 = shard1.workspace->scope();
            device0->bind_to_current_thread();
            Tensor logits0 = shard0.workspace->alloc(DType::BF16, {vocab, 1});
            device1->bind_to_current_thread();
            Tensor logits1 = shard1.workspace->alloc(DType::BF16, {vocab, 1});
            card0.forward_tp2_prefill(card1, pair, ids, 0, &logits0, &logits1, nullptr, nullptr,
                                      nullptr, qwen::TextPhase::Prefill, nullptr, sink);
            RunResult result;
            result.logits0 = read_bf16(*device0, logits0);
            result.logits1 = read_bf16(*device1, logits1);
            result.peak0   = shard0.workspace->peak_used();
            result.peak1   = shard1.workspace->peak_used();
            return result;
        };

        // 1. Target prefill without the sink.
        const RunResult baseline = run_prefill(nullptr);
        std::cout << "  no-sink prefill: workspace peak A=" << (baseline.peak0 >> 20)
                  << " MiB B=" << (baseline.peak1 >> 20) << " MiB" << std::endl;

        // 2. Target prefill with the round's own sink; its consumer appends each captured chunk
        // into the round's draft ring. The wrapper only counts the consumer's calls.
        std::uint32_t consumed = 0;
        qwen::execution::DFlashFeatureSink sink = round.make_prefill_sink(execution_core());
        const qwen::execution::DFlashFeatureSink::PrefillConsumer round_consumer =
            sink.consume_prefill;
        sink.consume_prefill = [&consumed, round_consumer](const Tensor& features,
                                                           const Tensor& positions, bool rewrite) {
            ++consumed;
            round_consumer(features, positions, rewrite);
        };
        const RunResult with_sink = run_prefill(&sink);
        std::cout << "  sink prefill: workspace peak A=" << (with_sink.peak0 >> 20)
                  << " MiB B=" << (with_sink.peak1 >> 20) << " MiB | consumer calls=" << consumed
                  << " captured_mask=0x" << std::hex << sink.captured_mask << std::dec
                  << " active_tokens=" << sink.active_tokens << std::endl;

        const std::uint32_t complete_mask =
            draft.target_layer_ids.size() == 32 ? ~0U
                                                : ((1U << draft.target_layer_ids.size()) - 1U);
        const std::uint32_t first_consumed = consumed;
        require(first_consumed == 1, "the sink consumer did not run exactly once");
        require(sink.captured_mask == complete_mask, "the sink did not capture every feature layer");
        require(sink.active_tokens == chunk, "the sink captured the wrong chunk width");
        require(with_sink.peak0 <= kWorkspaceBytes, "shard 0 workspace exceeded its budget");
        require(with_sink.logits0 == baseline.logits0,
                "shard 0 target logits changed when the sink was installed");
        require(with_sink.logits1 == baseline.logits1,
                "shard 1 target logits changed when the sink was installed");
        require(with_sink.logits0 == with_sink.logits1,
                "the two shards' target logits disagree under the sink");
        std::cout << "  target logits bit-identical with and without the sink; both shards agree"
                  << std::endl;

        // The sink captures the chunk's absolute cache positions; those drive the ring slots.
        std::vector<std::int32_t> captured(static_cast<std::size_t>(chunk), -1);
        device0->bind_to_current_thread();
        CUDA_CHECK(cudaMemcpy(captured.data(), round.state().prefill_positions.data,
                              sizeof(std::int32_t) * static_cast<std::size_t>(chunk),
                              cudaMemcpyDeviceToHost));
        for (std::int32_t i = 0; i < chunk; ++i) {
            require(captured[static_cast<std::size_t>(i)] == i,
                    "the sink captured an unexpected absolute position");
        }
        std::cout << "  sink captured positions [0," << chunk << ")" << std::endl;

        // 3. Ring structure.
        const RingSnapshot ring1 = read_ring(*device0, round.ring());
        require(round.ring().layer_count() == static_cast<std::uint32_t>(layers),
                "ring layer count is wrong");
        require(round.ring().capacity() == draft.sliding_window.value_or(0),
                "ring capacity is not the draft window");
        require(round.ring().num_kv_heads() == kv_heads && round.ring().head_dim() == head_dim,
                "ring head geometry is wrong");
        require(round.ring().lane_capacity() == 1, "ring lane capacity is not one");
        bool dead_ok = true;
        for (std::int32_t layer = 0; layer < layers; ++layer) {
            const RingLayerStats k = ring_layer_stats(
                ring1.k[static_cast<std::size_t>(layer)], false, head_dim,
                static_cast<std::int32_t>(round.ring().padded_capacity()), kv_heads, 1, chunk,
                &dead_ok);
            const RingLayerStats v = ring_layer_stats(
                ring1.v[static_cast<std::size_t>(layer)], true, head_dim,
                static_cast<std::int32_t>(round.ring().padded_capacity()), kv_heads, 1, chunk,
                &dead_ok);
            std::printf("  ring layer %d: K live_nonzero=%zu/%zu nonfinite=%zu max_abs=%.4g "
                        "mean_abs=%.4g | V live_nonzero=%zu/%zu nonfinite=%zu max_abs=%.4g\n",
                        layer, k.live_nonzero, k.live_elements, k.live_nonfinite,
                        static_cast<double>(k.max_abs), k.mean_abs, v.live_nonzero, v.live_elements,
                        v.live_nonfinite, static_cast<double>(v.max_abs));
            require(k.live_nonfinite == 0 && v.live_nonfinite == 0,
                    "the ring holds a non-finite K/V value");
            require(k.live_nonzero > 0 && v.live_nonzero > 0, "the ring layer is entirely zero");
        }
        require(dead_ok, "the ring wrote past the addressed prefix");
        std::cout << "  ring: geometry and coverage sane, capacity remainder zero" << std::endl;

        // 4. The sink path equals the same features appended directly (one width=chunk call). The
        // direct driver is a second round component, so this compares the captured path against an
        // explicit call into the same production append.
        qwen::execution::DFlash2Round direct(*device0, round_spec);
        std::unique_ptr<DeviceArena> direct_arena(
            new DeviceArena(static_cast<std::size_t>(target.hidden_size) *
                                draft.target_layer_ids.size() * chunk * 2 +
                            4096));
        device0->bind_to_current_thread();
        Tensor features_copy = direct_arena->alloc(
            DType::BF16,
            {qwen::execution::dimension(target.hidden_size * draft.target_layer_ids.size()), chunk});
        Tensor direct_positions = direct_arena->alloc(DType::I32, {chunk});
        CUDA_CHECK(cudaMemcpyAsync(features_copy.data, round.state().prefill_features.data,
                                   features_copy.bytes(), cudaMemcpyDeviceToDevice, device0->stream));
        ops::fill_i32_positions(direct_positions, 0, 1, device0->stream);
        direct.append(execution_core(), features_copy, direct_positions,
                      static_cast<std::uint32_t>(chunk));
        const RingSnapshot direct1 = read_ring(*device0, direct.ring());
        require(direct1 == ring1, "the direct append differs from the sink-driven append");
        std::cout << "  direct append over the captured features reproduces the sink ring "
                     "bit-for-bit"
                  << std::endl;

        // 5. Chunk-width independence: two width=chunk/2 appends at absolute positions.
        const std::int32_t half = chunk / 2;
        direct.zero_context();
        Tensor half_features = features_copy.slice(1, 0, half);
        Tensor half_positions = direct_positions.slice(0, 0, half);
        direct.append(execution_core(), half_features, half_positions,
                      static_cast<std::uint32_t>(half));
        Tensor tail_features = features_copy.slice(1, half, half);
        Tensor tail_positions = direct_positions.slice(0, half, half);
        direct.append(execution_core(), tail_features, tail_positions,
                      static_cast<std::uint32_t>(half));
        const RingSnapshot direct2 = read_ring(*device0, direct.ring());
        require(direct2 == direct1,
                "two half-width appends differ from one full-width append");
        std::cout << "  " << chunk << " vs 2x" << half
                  << " append: ring bit-identical (absolute positions honoured)" << std::endl;

        // 6. Reproducibility across runs from a zeroed state.
        const RunResult repeat = run_prefill(&sink);
        const RingSnapshot ring2 = read_ring(*device0, round.ring());
        require(repeat.logits0 == with_sink.logits0, "the repeat run changed shard 0 logits");
        require(ring2 == ring1, "the repeat run changed the ring");
        std::printf("  repeat run: ring bit-identical (fnv1a=0x%016llx)\n",
                    static_cast<unsigned long long>(ring1.hash));

        // 7. The assembled round's proposal on shard 0, after a prefill chunk has been consumed into
        // the draft ring. The round owns the frame, the draft state and its own workspace, and runs
        // the production propose path (execution/draft.cpp propose_dflash2_batch) through the B3 seam
        // dflash_propose_batch. The selector inside that function is production code; no
        // selector/verify/session stage is added here.
        const std::int32_t k          = proposal_drafts;
        const std::int32_t width      = k + 1;
        const std::int32_t frontier   = chunk - 1;
        const std::int32_t selector_k = static_cast<std::int32_t>(draft.dflash2->selector_top_k);
        const std::int32_t public_tokens =
            static_cast<std::int32_t>(parameters0.model.resources().public_token_count);

        // One greedy decode round's ingress for the single resident row: the anchor is the last
        // prompt token, the frontier is its cache position, and the draft's own attention uses its
        // logical positions [frontier, frontier + width) (decode.cpp:662-674).
        qwen::DFlashDecodeIngress& host_ingress = round.ingress();
        host_ingress.anchors[0]                 = ids[static_cast<std::size_t>(frontier)];
        host_ingress.execution_frontiers[0]     = frontier;
        host_ingress.context_frontiers[0]       = chunk;
        host_ingress.proposal_extents[0]        = k;
        host_ingress.proposal_valid_columns[0]  = width;
        host_ingress.target_valid_columns[0]    = width;
        host_ingress.active_lanes[0]            = 0;
        host_ingress.state_source_slots[0]      = 0;
        host_ingress.state_destination_slots[0] = 0;
        host_ingress.sampling[0].temperature    = 0.0F;
        const qwen::execution::DFlashEnvelopes envelopes{
            .local  = {0, static_cast<std::uint32_t>(frontier)},
            .full   = {0, static_cast<std::uint32_t>(frontier)},
            .append = {0, static_cast<std::uint32_t>(width)}};

        auto enqueue_proposal = [&]() {
            round.propose(execution_core(), shard0.decoder->text_kv, &card0,
                          static_cast<std::uint32_t>(k), envelopes);
        };

        struct ProposalResult {
            std::vector<std::int32_t> drafts;
            std::vector<std::int32_t> candidates;
            std::vector<float> proposal_q;
            std::size_t peak = 0;
            std::uint64_t hash = 0;
        };
        auto run_proposal = [&]() {
            round.reset_proposal_workspace_peak();
            enqueue_proposal();
            CUDA_CHECK(cudaStreamSynchronize(device0->stream));
            const qwen::execution::DFlash2Proposal& published = round.proposal();
            ProposalResult result;
            result.drafts     = read_i32(*device0, published.drafts);
            result.candidates = read_i32(*device0, published.candidate_ids);
            result.proposal_q = read_fp32(*device0, published.scores);
            result.peak       = round.proposal_workspace_peak();
            std::uint64_t hash = 1469598103934665603ULL;
            const auto feed    = [&hash](const void* data, std::size_t bytes) {
                hash ^= fnv1a(std::span<const std::byte>(
                    static_cast<const std::byte*>(data), bytes));
                hash *= 1099511628211ULL;
            };
            feed(result.drafts.data(), result.drafts.size() * sizeof(std::int32_t));
            feed(result.candidates.data(), result.candidates.size() * sizeof(std::int32_t));
            feed(result.proposal_q.data(), result.proposal_q.size() * sizeof(float));
            result.hash = hash;
            return result;
        };

        const ProposalResult proposal1 = run_proposal();
        require(round.frame().draft_tokens.ne[0] == k && round.frame().draft_tokens.ne[1] == 1,
                "the draft token buffer has an unexpected shape");
        require(round.frame().draft_tokens.dtype == DType::I32 &&
                    round.frame().candidate_ids.dtype == DType::I32 &&
                    round.frame().proposal_q.dtype == DType::FP32,
                "the proposal output buffers have unexpected dtypes");
        require(round.frame().candidate_ids.ne[0] == selector_k &&
                    round.frame().candidate_ids.ne[1] == k && round.frame().candidate_ids.ne[2] == 1,
                "the candidate buffer has an unexpected shape");
        require(round.frame().proposal_q.ne[0] == selector_k &&
                    round.frame().proposal_q.ne[1] == k && round.frame().proposal_q.ne[2] == 1,
                "the proposal-q buffer has an unexpected shape");
        require(round.frame().proposal_ids.ne[0] == width &&
                    round.frame().proposal_positions.ne[0] == width,
                "the proposal query block has an unexpected width");
        require(proposal1.drafts.size() == static_cast<std::size_t>(k) &&
                    proposal1.candidates.size() == static_cast<std::size_t>(selector_k) * k &&
                    proposal1.proposal_q.size() == static_cast<std::size_t>(selector_k) * k,
                "the proposal produced the wrong number of elements");

        // The masked block's query positions must be the anchor's own position first, then one
        // position per draft step, forwarded unchanged to the target verify in a later stage.
        std::vector<std::int32_t> proposal_positions =
            read_i32(*device0, round.frame().proposal_positions);
        for (std::int32_t i = 0; i < width; ++i) {
            require(proposal_positions[static_cast<std::size_t>(i)] == frontier + i,
                    "the masked proposal block has an unexpected query position");
        }

        std::size_t one_hot_rows = 0;
        std::size_t distinct_ok  = 0;
        for (std::int32_t position = 0; position < k; ++position) {
            std::vector<std::int32_t> row(static_cast<std::size_t>(selector_k));
            for (std::int32_t rank = 0; rank < selector_k; ++rank) {
                const std::size_t index =
                    static_cast<std::size_t>(position) * selector_k + rank;
                row[static_cast<std::size_t>(rank)] = proposal1.candidates[index];
                require(proposal1.candidates[index] >= 0 &&
                            proposal1.candidates[index] < public_tokens,
                        "a candidate id left the public token domain");
            }
            std::vector<std::int32_t> sorted = row;
            std::sort(sorted.begin(), sorted.end());
            require(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
                    "a candidate row repeats a token id");
            ++distinct_ok;
            require(std::find(row.begin(), row.end(),
                              proposal1.drafts[static_cast<std::size_t>(position)]) != row.end(),
                    "a draft token is not one of its position's candidates");

            // Greedy selection writes the exact one-hot distribution over the candidate ranks.
            const float* distribution =
                proposal1.proposal_q.data() + static_cast<std::size_t>(position) * selector_k;
            float total = 0.0F;
            std::int32_t selected_rank = -1;
            for (std::int32_t rank = 0; rank < selector_k; ++rank) {
                total += distribution[rank];
                if (distribution[rank] > 0.0F) {
                    require(selected_rank < 0, "the greedy proposal q is not one-hot");
                    selected_rank = rank;
                }
            }
            require(selected_rank >= 0 && std::fabs(total - 1.0F) <= 1.0e-6F,
                    "the proposal q does not sum to one");
            require(row[static_cast<std::size_t>(selected_rank)] ==
                        proposal1.drafts[static_cast<std::size_t>(position)],
                    "the proposal q and the draft token disagree");
            ++one_hot_rows;
        }
        std::printf("  proposal: K=%d width=%d positions=[%d,%d) candidates/row=%d "
                    "rows=%zu distinct=%zu one_hot=%zu\n",
                    k, width, frontier, frontier + width, selector_k, one_hot_rows, distinct_ok,
                    one_hot_rows);

        // 9. Arena ownership: propose_dflash2_batch resets the arena it is handed, so the round must
        // pass its own and leave shard 0's resident workspace alone. Poison the resident
        // prefill_hidden, propose again, and require the pattern and the workspace watermark to
        // survive while the round's own arena reports the transient peak.
        {
            device0->bind_to_current_thread();
            std::vector<std::uint16_t> poison(
                static_cast<std::size_t>(shard0.prefill_hidden.bytes() / sizeof(std::uint16_t)));
            for (std::size_t index = 0; index < poison.size(); ++index) {
                poison[index] = (index % 2U == 0U) ? 0x3F80U : 0xC000U;
            }
            CUDA_CHECK(cudaMemcpyAsync(shard0.prefill_hidden.data, poison.data(),
                                       shard0.prefill_hidden.bytes(), cudaMemcpyHostToDevice,
                                       device0->stream));
            CUDA_CHECK(cudaStreamSynchronize(device0->stream));
            const std::size_t resident_used = shard0.workspace->used();
            const std::size_t resident_peak = shard0.workspace->peak_used();
            round.reset_proposal_workspace_peak();
            enqueue_proposal();
            CUDA_CHECK(cudaStreamSynchronize(device0->stream));
            require(read_bf16(*device0, shard0.prefill_hidden) == poison,
                    "the proposal overwrote the shard's resident prefill_hidden");
            require(shard0.workspace->used() == resident_used,
                    "the proposal moved the shard workspace watermark");
            require(shard0.workspace->peak_used() == resident_peak,
                    "the proposal raised the shard workspace peak");
            const std::size_t round_peak = round.proposal_workspace_peak();
            require(round_peak > 0 && round_peak <= round.proposal_workspace_capacity(),
                    "the proposal did not run on the round's own arena");
            std::printf("  arena ownership: resident prefill_hidden and shard workspace watermark "
                        "unchanged; round arena peak %.1f MiB of %.1f MiB\n",
                        static_cast<double>(round_peak) / 1048576.0,
                        static_cast<double>(round.proposal_workspace_capacity()) / 1048576.0);
        }

        // Determinism: a second proposal on the same ring, then the whole prefill -> append ->
        // proposal chain from a zeroed state, must agree bit for bit.
        const ProposalResult proposal_same_state = run_proposal();
        require(proposal_same_state.hash == proposal1.hash,
                "two proposals on the same ring differ");
        const RunResult chain_prefill = run_prefill(&sink);
        require(chain_prefill.logits0 == with_sink.logits0,
                "the proposal chain's prefill changed shard 0 logits");
        const ProposalResult proposal_chain = run_proposal();
        require(proposal_chain.hash == proposal1.hash,
                "the proposal differs after a fresh prefill and append");
        std::printf("  proposal determinism: same-state + fresh-chain bit-identical "
                    "(fnv1a=0x%016llx)\n",
                    static_cast<unsigned long long>(proposal1.hash));

        // Cost: the round's own proposal arena, sized to the shipped per-shard 192 MiB budget, timed
        // with CUDA events.
        for (int warm = 0; warm < 3; ++warm) { enqueue_proposal(); }
        CUDA_CHECK(cudaStreamSynchronize(device0->stream));
        constexpr int kTimedProposals = 20;
        cudaEvent_t start_event = nullptr, stop_event = nullptr;
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
        CUDA_CHECK(cudaEventRecord(start_event, device0->stream));
        for (int iteration = 0; iteration < kTimedProposals; ++iteration) { enqueue_proposal(); }
        CUDA_CHECK(cudaEventRecord(stop_event, device0->stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));
        const double per_proposal_ms = static_cast<double>(elapsed_ms) / kTimedProposals;

        const auto& draft_weights = *parameters0.draft;
        const qwen::execution::SelectorParameters* selector =
            parameters0.dflash_selector ? &*parameters0.dflash_selector : nullptr;
        const std::size_t draft_bytes = draft_weight_bytes(draft_weights, selector);
        const std::size_t head_bytes  = linear_bytes(draft_weights.output_head);
        const std::size_t codebook_bytes =
            selector != nullptr
                ? selector->predecessor_codebook.bytes() + selector->successor_codebook.bytes()
                : 0;
        const std::size_t selector_projection_bytes =
            selector != nullptr ? linear_bytes(selector->hidden_projection) : 0;
        // The arena the runtime core's own planner asks for at this geometry, for comparison with the
        // shipped-budget arena the test sizes the round with.
        const std::size_t planned_proposal_bytes = qwen::execution::dflash2_proposal_workspace_bytes(
            parameters0, target, kTestCacheTokens, ProposalHead::Full, width, 1);
        std::printf("[mem] shard 0 DFlash2 draft weights %.1f MiB | round frame %.1f MiB | "
                    "context arena %.1f MiB | round proposal arena %.1f MiB | proposal workspace peak "
                    "%.1f MiB (%.0f%% of %zu MiB) | planner %.1f MiB\n",
                    static_cast<double>(draft_bytes) / 1048576.0,
                    static_cast<double>(round.frame_bytes()) / 1048576.0,
                    static_cast<double>(round.context_bytes()) / 1048576.0,
                    static_cast<double>(round.proposal_workspace_capacity()) / 1048576.0,
                    static_cast<double>(proposal1.peak) / 1048576.0,
                    100.0 * static_cast<double>(proposal1.peak) /
                        static_cast<double>(kWorkspaceBytes),
                    kWorkspaceBytes >> 20,
                    static_cast<double>(planned_proposal_bytes) / 1048576.0);
        std::printf(
            "[mem]   draft block: feature_projection %.1f MiB | layers %.1f MiB | selector "
            "codebooks %.1f MiB | selector hidden_projection %.1f MiB | tied output head %.1f MiB "
            "(shared with the target)\n",
            static_cast<double>(linear_bytes(draft_weights.feature_projection)) / 1048576.0,
            static_cast<double>(draft_bytes - linear_bytes(draft_weights.feature_projection) -
                                codebook_bytes - head_bytes - selector_projection_bytes) /
                1048576.0,
            static_cast<double>(codebook_bytes) / 1048576.0,
            static_cast<double>(selector_projection_bytes) / 1048576.0,
            static_cast<double>(head_bytes) / 1048576.0);
        std::printf("[mem]   qtypes: feature_projection=%s output_head=%s layer0.qkv=%s "
                    "layer0.down=%s codebook=%s\n",
                    qtype_name(draft_weights.feature_projection.weight.qtype),
                    qtype_name(draft_weights.output_head.weight.qtype),
                    qtype_name(draft_weights.layers.front().query_key_value
                                   ? draft_weights.layers.front().query_key_value->weight.qtype
                                   : draft_weights.layers.front()
                                         .query_key_value_rows[0]
                                         .weight.qtype),
                    qtype_name(draft_weights.layers.front().mlp.down.weight.qtype),
                    selector == nullptr ? "none"
                    : selector->predecessor_codebook.quantized
                        ? qtype_name(selector->predecessor_codebook.weight.qtype)
                        : dtype_name(selector->predecessor_codebook.dense.dtype));
        std::printf("[mem]   peer selector: %s %.1f MiB\n",
                    parameters1.dflash_selector ? "on shard 1" : "absent",
                    static_cast<double>(
                        parameters1.dflash_selector
                            ? linear_bytes(parameters1.dflash_selector->hidden_projection) +
                                  parameters1.dflash_selector->predecessor_codebook.bytes() +
                                  parameters1.dflash_selector->successor_codebook.bytes()
                            : 0) /
                        1048576.0);
        std::printf("  proposal cost: %.3f ms/proposal (events, %d iterations)\n",
                    per_proposal_ms, kTimedProposals);

        std::printf(
            "TP-2 DFlash2 round passed: chunk=%d sink logits bit-identical, "
            "consumer=%u/sink prefill, ring sane+reproducible, direct/split append bit-identical, "
            "shard workspace untouched by the proposal, K=%d proposal deterministic "
            "(fnv1a=0x%016llx), workspace peak A=%.1f MiB (%.0f%% of %zu MiB) B=%.1f MiB\n",
            chunk, first_consumed, k, static_cast<unsigned long long>(proposal1.hash),
            static_cast<double>(with_sink.peak0) / 1048576.0,
            100.0 * static_cast<double>(with_sink.peak0) / static_cast<double>(kWorkspaceBytes),
            kWorkspaceBytes >> 20, static_cast<double>(with_sink.peak1) / 1048576.0);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
