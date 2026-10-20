#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/gdn.h"
#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/mtp.h"
#include "models/qwen3_5/execution/workspace.h"

#include "core/nvtx.h"
#include "models/qwen3_5/execution/visual_scatter.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/program/vision_control.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/token_mask.h"
#include "ninfer/ops/attn_input_proj.h"
#include "models/qwen3_5/execution/selector.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sparse_moe.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace {

void project(const Tensor& x, const LinearParameters& parameters, Tensor& out,
             WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear(x, parameters.weight, out, parameters.policy, workspace, stream);
}

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

// Materialize a contiguous [rows, columns] block of a fused projection output whose row blocks
// (q|k|gate|v for attention, q|k|v|z for GDN) are concatenated along dim 0. Dim 0 is the innermost
// (contiguous) axis of every activation tensor in this file, so a row sub-block of a wider parent is
// a strided window: at T == 1 the slice degenerates to a single column and stays contiguous, which
// is why the single-token decode path can alias it directly, but a batched prefill must materialize
// the block before handing it to ops that require contiguous operands (rmsnorm, RoPE, the paged
// attention, the split convolution, gated_delta_net).
void copy_row_block(const Tensor& parent, std::int32_t row_begin, std::int32_t rows, Tensor& dst,
                    cudaStream_t stream) {
    const std::int32_t parent_rows = parent.ne[0];
    const std::int32_t columns     = parent.ne[1];
    if (parent.ne[2] != 1 || parent.ne[3] != 1 || !parent.is_contiguous() || parent.data == nullptr) {
        throw std::invalid_argument("copy_row_block: parent must be a contiguous matrix");
    }
    if (rows <= 0 || columns <= 0 || row_begin < 0 || rows > parent_rows - row_begin) {
        throw std::invalid_argument("copy_row_block: invalid row range");
    }
    if (dst.dtype != parent.dtype || dst.ne[0] != rows || dst.ne[1] != columns || dst.ne[2] != 1 ||
        dst.ne[3] != 1 || !dst.is_contiguous() || dst.data == nullptr) {
        throw std::invalid_argument("copy_row_block: destination must be a contiguous [rows, cols]");
    }
    const std::size_t element = dtype_size(parent.dtype);
    CUDA_CHECK(cudaMemcpy2DAsync(dst.data, static_cast<std::size_t>(rows) * element,
                                 static_cast<const unsigned char*>(parent.data) +
                                     static_cast<std::size_t>(row_begin) * element,
                                 static_cast<std::size_t>(parent_rows) * element,
                                 static_cast<std::size_t>(rows) * element,
                                 static_cast<std::size_t>(columns), cudaMemcpyDeviceToDevice,
                                 stream));
}

// Inverse of copy_row_block: materialize a contiguous [rows, columns] source into the row block
// [row_begin, row_begin + rows) of a wider contiguous parent. The vocabulary-parallel output head
// needs it because dim 0 is the innermost axis: a shard projects its own vocabulary rows compactly
// and then stamps them into the full-logits buffer at its vocabulary offset.
void write_row_block(const Tensor& src, Tensor& parent, std::int32_t row_begin, cudaStream_t stream) {
    const std::int32_t parent_rows = parent.ne[0];
    const std::int32_t rows        = src.ne[0];
    const std::int32_t columns     = src.ne[1];
    if (parent.ne[2] != 1 || parent.ne[3] != 1 || !parent.is_contiguous() ||
        parent.data == nullptr) {
        throw std::invalid_argument("write_row_block: parent must be a contiguous matrix");
    }
    if (src.ne[2] != 1 || src.ne[3] != 1 || !src.is_contiguous() || src.data == nullptr ||
        src.dtype != parent.dtype) {
        throw std::invalid_argument("write_row_block: source must be a contiguous matrix");
    }
    if (rows <= 0 || columns <= 0 || row_begin < 0 || rows > parent_rows - row_begin) {
        throw std::invalid_argument("write_row_block: invalid row range");
    }
    const std::size_t element = dtype_size(parent.dtype);
    CUDA_CHECK(cudaMemcpy2DAsync(static_cast<unsigned char*>(parent.data) +
                                     static_cast<std::size_t>(row_begin) * element,
                                 static_cast<std::size_t>(parent_rows) * element, src.data,
                                 static_cast<std::size_t>(rows) * element,
                                 static_cast<std::size_t>(rows) * element,
                                 static_cast<std::size_t>(columns), cudaMemcpyDeviceToDevice,
                                 stream));
}

void require_tensor_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                          const char* label) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    int i = 0;
    for (const std::int32_t dim : shape) {
        if (t.ne[i] != dim) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
        ++i;
    }
    for (; i < 4; ++i) {
        if (t.ne[i] != 1) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_tensor_window(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                           const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] != rows || t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

Tensor matrix_window(Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("matrix_window cols must be positive"); }
    if (t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("matrix_window shape mismatch");
    }
    return t.slice(1, 0, cols);
}

class ScopedPositions {
public:
    ScopedPositions(const Tensor*& slot, const Tensor& positions) : slot_(slot) {
        slot_ = &positions;
    }

    ScopedPositions(const ScopedPositions&)            = delete;
    ScopedPositions& operator=(const ScopedPositions&) = delete;

    ~ScopedPositions() { slot_ = nullptr; }

private:
    const Tensor*& slot_;
};

class ScopedEnvelope {
public:
    ScopedEnvelope(const ops::CausalAttentionExecutionEnvelope*& slot,
                   const ops::CausalAttentionExecutionEnvelope& envelope)
        : slot_(slot) {
        slot_ = &envelope;
    }

    ScopedEnvelope(const ScopedEnvelope&)            = delete;
    ScopedEnvelope& operator=(const ScopedEnvelope&) = delete;

    ~ScopedEnvelope() { slot_ = nullptr; }

private:
    const ops::CausalAttentionExecutionEnvelope*& slot_;
};

template <class T>
class ScopedValue {
public:
    ScopedValue(T& slot, T value) : slot_(slot), previous_(slot) { slot_ = value; }

    ScopedValue(const ScopedValue&)            = delete;
    ScopedValue& operator=(const ScopedValue&) = delete;

    ~ScopedValue() { slot_ = previous_; }

private:
    T& slot_;
    T previous_;
};

// Non-timing events that order the collective stream against the compute stream of one device for
// the overlapped TP-2 prefill. The produce/done slots are reused every layer: cudaStreamWaitEvent
// captures the event state at the host call, so re-recording one for the next layer cannot affect a
// wait that was already enqueued.
class Tp2OverlapEvents {
public:
    explicit Tp2OverlapEvents(std::size_t count)
        : produced_(count, nullptr), done_(count, nullptr) {
        if (cudaGetDevice(&device_) != cudaSuccess) {
            throw std::runtime_error("TP-2 prefill overlap: cudaGetDevice failed");
        }
        for (std::size_t i = 0; i < count; ++i) {
            if (cudaEventCreateWithFlags(&produced_[i], cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&done_[i], cudaEventDisableTiming) != cudaSuccess) {
                release();
                throw std::runtime_error("TP-2 prefill overlap: cudaEventCreate failed");
            }
        }
    }
    ~Tp2OverlapEvents() { release(); }
    Tp2OverlapEvents(const Tp2OverlapEvents&)            = delete;
    Tp2OverlapEvents& operator=(const Tp2OverlapEvents&) = delete;

    [[nodiscard]] cudaEvent_t produced(std::size_t index) const { return produced_[index]; }
    [[nodiscard]] cudaEvent_t done(std::size_t index) const { return done_[index]; }

private:
    void release() noexcept {
        (void)cudaSetDevice(device_);
        for (cudaEvent_t& event : produced_) {
            if (event != nullptr) { (void)cudaEventDestroy(event); event = nullptr; }
        }
        for (cudaEvent_t& event : done_) {
            if (event != nullptr) { (void)cudaEventDestroy(event); event = nullptr; }
        }
    }
    std::vector<cudaEvent_t> produced_;
    std::vector<cudaEvent_t> done_;
    int device_ = 0;
};

} // namespace

void DFlashFeatureSink::begin(const Tensor& value) {
    const bool prefill = features != nullptr && positions != nullptr && batch_features == nullptr;
    const bool batch   = batch_features != nullptr && batch_lanes != nullptr &&
                       batch_valid_columns != nullptr && batch_width > 0 && batch_size > 0;
    if ((!prefill && !batch) || layers.empty()) {
        throw std::logic_error("DFlash feature sink is incomplete");
    }
    captured_mask = 0;
    active_tokens = batch ? batch_width * batch_size : value.ne[1];
    if (value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash batch feature source has an invalid width");
    }
}

void DFlashFeatureSink::capture_layer(int layer, const Tensor& value, cudaStream_t stream) {
    const auto it = std::find(layers.begin(), layers.end(), layer);
    if (it == layers.end()) { return; }
    const std::size_t index = static_cast<std::size_t>(it - layers.begin());
    Tensor* destination     = batch_features != nullptr ? batch_features : features;
    if (layers.size() > 32 || active_tokens <= 0 || value.dtype != DType::BF16 ||
        destination == nullptr ||
        value.ne[0] * static_cast<std::int32_t>(layers.size()) != destination->ne[0] ||
        value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash feature capture shape is invalid");
    }
    if (batch_features != nullptr) {
        Tensor source = value.view({value.ne[0], batch_width, batch_size});
        Tensor target =
            batch_features->slice(0, static_cast<std::int32_t>(index) * value.ne[0], value.ne[0]);
        ops::scatter_bf16_batch(source, *batch_lanes, *batch_valid_columns, target, stream);
        captured_mask |= 1U << index;
        return;
    }
    if (active_tokens > features->ne[1]) {
        throw std::logic_error("DFlash prefill feature capture exceeds its buffer");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width_bytes   = static_cast<std::size_t>(value.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(value.nb[1]);
    const std::size_t target_pitch  = static_cast<std::size_t>(features->nb[1]);
    auto* target                    = static_cast<std::byte*>(features->data) + index * width_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(target, target_pitch, value.data, source_pitch, width_bytes,
                                 static_cast<std::size_t>(active_tokens), cudaMemcpyDeviceToDevice,
                                 stream));
    captured_mask |= 1U << index;
}

void DFlashFeatureSink::capture_positions(const Tensor& source, cudaStream_t stream) {
    const std::uint32_t complete_mask = layers.size() == 32 ? ~0U : ((1U << layers.size()) - 1U);
    if (captured_mask != complete_mask) {
        throw std::logic_error("DFlash target call did not publish every feature layer");
    }
    if (batch_features != nullptr) {
        if (source.dtype != DType::I32 || source.ne[0] != batch_width ||
            source.ne[1] != batch_size) {
            throw std::logic_error("DFlash batch feature positions are invalid");
        }
        return;
    }
    if (active_tokens <= 0 || source.dtype != DType::I32 || source.ne[0] != active_tokens ||
        positions == nullptr || active_tokens > positions->ne[0]) {
        throw std::logic_error("DFlash feature positions are invalid");
    }
    CUDA_CHECK(cudaMemcpyAsync(positions->data, source.data,
                               static_cast<std::size_t>(active_tokens) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void DFlashFeatureSink::consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint) {
    if (!consume_prefill || tokens != active_tokens) {
        throw std::logic_error("DFlash prefill feature consumer is unavailable");
    }
    Tensor feature_window  = features->slice(1, 0, tokens);
    Tensor position_window = positions->slice(0, 0, tokens);
    consume_prefill(feature_window, position_window, rewrite_checkpoint);
}

TextContext::TextContext(DeviceContext& ctx, const execution::Parameters& weights,
                         WorkspaceArena& work, qwen3_5::PagedKVCacheView kv,
                         LinearAttentionStatePool& state, qwen3_5::RoundState& io,
                         Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                         std::uint32_t text_kv_base, qwen3_5::PagedKVCacheView mtp_kv,
                         const qwen3_5::PagedKVCache* batch_text_kv,
                         const qwen3_5::PagedKVCache* batch_mtp_kv, std::uint32_t prefill_overlap)
    : ctx_(ctx), parameters_(weights), config_(weights.model.config().text), work_(work), kv_(kv),
      mtp_kv_(mtp_kv), state_(state), io_(io), prefill_hidden_(prefill_hidden),
      prefill_chunk_(prefill_chunk), prefill_overlap_(prefill_overlap),
      text_kv_base_(text_kv_base), batch_text_kv_(batch_text_kv), batch_mtp_kv_(batch_mtp_kv) {
    if (prefill_chunk_ == 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext effective prefill chunk must fit positive int32");
    }
    // Each decode collective's rendezvous window pulls at most this much of this shard's next weight
    // stream into L2, which is free time: the window waits on the peer, not on its own memory.
    // 32 MiB is worth 36.77 -> 37.43 tok/s on the plain TP-2 decode route (express 8192-token
    // prompts, 128 tokens, interleaved arms; +1.8%), and the MTP route's acceptance noise does not
    // resolve an effect either way. Larger budgets turn around: 64 and 128 MiB are back at 37.3 and
    // 37.2 because the window's own lines start leaving L2, so 32 MiB ships and 0 disables.
    if (const char* prefetch = std::getenv("NINFER_TP2_DECODE_PREFETCH_MIB");
        prefetch != nullptr && *prefetch != '\0') {
        const long long mib = std::strtoll(prefetch, nullptr, 10);
        decode_prefetch_bytes_ = mib > 0 ? static_cast<std::size_t>(mib) << 20 : 0;
    }
    if (mtp_enabled() && !io_.mtp_decode && !io_.mtp) {
        throw std::invalid_argument("MTP TextContext requires MTP round state");
    }
    set_linear_state_slots(0, 0);
    embed_      = &parameters_.text.token_embedding;
    final_norm_ = &parameters_.text.final_norm;
    lm_head_    = &parameters_.text.output_head;
    mtp_        = parameters_.mtp ? &*parameters_.mtp : nullptr;
    if (mtp_enabled() && mtp_ == nullptr) {
        throw std::invalid_argument("MTP state requires selected MTP parameters");
    }
    if (parameters_.proposal) {
        const auto& p = *parameters_.proposal;
        set_proposal_head(
            &p.head, p.token_ids ? static_cast<const std::int32_t*>(p.token_ids->data) : nullptr,
            dimension(p.rows));
    }
    if (parameters_.dflash_selector) { selector_ = &*parameters_.dflash_selector; }
}

TextContext::~TextContext() = default;

void TextContext::set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot) {
    if (source_slot < 0 || source_slot >= state_.slot_count() || destination_slot < 0 ||
        destination_slot >= state_.slot_count()) {
        throw std::invalid_argument("TextContext Linear Attention slots are invalid");
    }
    linear_state_source_slot_      = source_slot;
    linear_state_destination_slot_ = destination_slot;
}

void TextContext::set_gdn_state_action(GdnStateAction action,
                                       const GdnReplayRecords* replay_records) {
    if ((action == GdnStateAction::RecordForReplay) != (replay_records != nullptr)) {
        throw std::invalid_argument("TextContext GDN state action has inconsistent records");
    }
    gdn_state_action_ = action;
    replay_records_   = replay_records;
}

void TextContext::set_tp_peer(TextContext* peer, tp::DevicePair* pair) {
    if (peer == nullptr || pair == nullptr) {
        throw std::invalid_argument("set_tp_peer: peer context and device pair must be set");
    }
    if (peer == this) { throw std::invalid_argument("set_tp_peer: a shard cannot be its own peer"); }
    peer_tp_ = peer;
    pair_tp_ = pair;
}

bool TextContext::embedding_is_split() const noexcept {
    return dimension(embed_->k) != dimension(config_.hidden_size);
}

void TextContext::embedding_full_width(const Tensor& ids, Tensor& out) {
    if (!embedding_is_split()) {
        ops::embedding(ids, *embed_, out, ctx_.stream);
        return;
    }
    if (peer_tp_ == nullptr || pair_tp_ == nullptr) {
        throw std::logic_error(
            "embedding_full_width: a column-split token embedding needs set_tp_peer");
    }
    // The peer ids are broadcast inside embedding_tp2 and a peer scratch destination is allocated
    // there, so only this shard's full-width [hidden, T] result is requested.
    embedding_tp2(*peer_tp_, *pair_tp_, ids, nullptr, out, nullptr);
}

void TextContext::merge_local_row_blocks(TextContext& peer, tp::DevicePair& pair,
                                         const Tensor& local, const Tensor& local_peer,
                                         Tensor& destination, Tensor& destination_peer,
                                         std::int32_t local_rows) {
    const std::int32_t full_rows = destination.ne[0];
    const std::int32_t columns   = destination.ne[1];
    if (local_rows <= 0 || local_rows * 2 != full_rows || columns <= 0 || shard_index_ < 0 ||
        peer.shard_index_ < 0 || peer.shard_index_ == shard_index_) {
        throw std::logic_error(
            "merge_local_row_blocks: a split weight needs exactly half the rows and two distinct "
            "shards");
    }
    const auto matrix = [columns](const Tensor& t, std::int32_t rows, const char* label) {
        if (t.dtype != DType::BF16 || t.ne[0] != rows || t.ne[1] != columns || t.ne[2] != 1 ||
            t.ne[3] != 1 || !t.is_contiguous() || t.data == nullptr) {
            throw std::invalid_argument(std::string("merge_local_row_blocks: ") + label +
                                        " must be a contiguous BF16 [rows, T] matrix");
        }
    };
    matrix(local, local_rows, "local block");
    matrix(local_peer, local_rows, "peer block");
    matrix(destination, full_rows, "destination");
    matrix(destination_peer, full_rows, "peer destination");

    ctx_.bind_to_current_thread();
    CUDA_CHECK(cudaMemsetAsync(destination.data, 0, destination.bytes(), ctx_.stream));
    write_row_block(local, destination, shard_index_ * local_rows, ctx_.stream);
    peer.ctx_.bind_to_current_thread();
    CUDA_CHECK(
        cudaMemsetAsync(destination_peer.data, 0, destination_peer.bytes(), peer.ctx_.stream));
    write_row_block(local_peer, destination_peer, peer.shard_index_ * local_rows, peer.ctx_.stream);
    ctx_.bind_to_current_thread();
    pair.allreduce(destination.data, destination_peer.data, destination.bytes(), ctx_.stream,
                   peer.ctx_.stream);
}

void TextContext::embedding_tp2(TextContext& peer, tp::DevicePair& pair, const Tensor& ids,
                                const Tensor* ids_peer, Tensor& x, Tensor* x_peer) {
    const std::int32_t hidden = dimension(config_.hidden_size);
    const std::int32_t local  = dimension(embed_->k);
    if (ids.dtype != DType::I32 || ids.ne[0] <= 0 || ids.ne[1] != 1 || ids.ne[2] != 1 ||
        ids.ne[3] != 1 || !ids.is_contiguous() || ids.data == nullptr) {
        throw std::invalid_argument("embedding_tp2: ids must be a contiguous I32 [T] tensor");
    }
    if (local == hidden) {
        // Replicated table: both shards hold every row, so each gathers its own copy.
        if (ids_peer == nullptr || x_peer == nullptr) {
            throw std::invalid_argument(
                "embedding_tp2: a replicated embedding needs both shards' ids and destinations");
        }
        ctx_.bind_to_current_thread();
        ops::embedding(ids, *embed_, x, ctx_.stream);
        peer.ctx_.bind_to_current_thread();
        ops::embedding(*ids_peer, *peer.embed_, *x_peer, peer.ctx_.stream);
        return;
    }
    if (local <= 0 || local * 2 != hidden || shard_index_ < 0 || peer.shard_index_ < 0 ||
        peer.shard_index_ == shard_index_ || dimension(peer.embed_->k) != local) {
        throw std::logic_error(
            "embedding_tp2: a column-split embedding needs half the hidden size per shard on two "
            "distinct shards");
    }
    const std::int32_t columns = x.ne[1];
    if (x.dtype != DType::BF16 || x.ne[0] != hidden || columns <= 0 || x.ne[2] != 1 || x.ne[3] != 1 ||
        !x.is_contiguous() || x.data == nullptr) {
        throw std::invalid_argument(
            "embedding_tp2: destination must be a contiguous BF16 [hidden, T] matrix");
    }
    ctx_.bind_to_current_thread();
    auto scope      = work_.scope();
    Tensor partial  = work_.alloc(DType::BF16, {local, columns});
    ops::embedding(ids, *embed_, partial, ctx_.stream);

    peer.ctx_.bind_to_current_thread();
    auto peer_scope          = peer.work_.scope();
    Tensor partial_peer      = peer.work_.alloc(DType::BF16, {local, columns});
    Tensor destination_peer  = x_peer != nullptr ? *x_peer
                                                 : peer.work_.alloc(DType::BF16, {hidden, columns});
    Tensor ids_peer_storage;
    Tensor ids_peer_view;
    if (ids_peer == nullptr) {
        // The peer needs the same token ids. The text path already binds them per shard; the MTP
        // stem only owns them here, so a private copy is mirrored into the peer's buffer.
        // The pair's collectives work in 16-byte units, so the buffer is padded up and the tail
        // zeroed; only the leading ids are read. A byte-exact exchange carries them, never the BF16
        // all-reduce: an integer id whose 16-bit lane spells a signaling NaN is quieted by the add
        // and a -0.0 lane is normalised to +0.0, so the sum would hand the peer a different id (the
        // DFlash2 route already exchanges its ids and scores the same way).
        // Each side passes one buffer as both its send and its receive. A's bytes are consumed by
        // the peer before they are overwritten, and the peer's bytes are discarded, so a broadcast
        // still needs no host round trip.
        const std::size_t id_bytes = static_cast<std::size_t>(ids.ne[0]) * sizeof(std::int32_t);
        const std::size_t broadcast_bytes = ((id_bytes + 15u) / 16u) * 16u;
        const std::int32_t words = static_cast<std::int32_t>(broadcast_bytes / sizeof(std::int32_t));
        ctx_.bind_to_current_thread();
        Tensor ids_staging = work_.alloc(DType::I32, {words});
        CUDA_CHECK(cudaMemsetAsync(ids_staging.data, 0, broadcast_bytes, ctx_.stream));
        CUDA_CHECK(cudaMemcpyAsync(ids_staging.data, ids.data, id_bytes,
                                   cudaMemcpyDeviceToDevice, ctx_.stream));
        peer.ctx_.bind_to_current_thread();
        ids_peer_storage = peer.work_.alloc(DType::I32, {words});
        CUDA_CHECK(cudaMemsetAsync(ids_peer_storage.data, 0, broadcast_bytes, peer.ctx_.stream));
        ctx_.bind_to_current_thread();
        pair.sendrecv(ids_staging.data, ids_staging.data, ids_peer_storage.data,
                      ids_peer_storage.data, broadcast_bytes, ctx_.stream, peer.ctx_.stream);
        // The gather validates ids against the output's token extent, so the peer reads only the
        // leading ids from the padded staging buffer.
        ids_peer_view = ids_peer_storage.slice(0, 0, ids.ne[0]);
        ids_peer      = &ids_peer_view;
    }
    peer.ctx_.bind_to_current_thread();
    ops::embedding(*ids_peer, *peer.embed_, partial_peer, peer.ctx_.stream);
    merge_local_row_blocks(peer, pair, partial, partial_peer, x, destination_peer, local);
}

void TextContext::mtp_forward_stem(const Tensor& ids, const Tensor& hidden,
                                   const Tensor* input_embeddings, Tensor& x, Tensor& ah) {
    cudaStream_t s     = ctx_.stream;
    const int T        = ids.ne[0] * ids.ne[1];
    Tensor flat_ids    = ids.view({T});
    Tensor flat_hidden = hidden.view({dimension(config_.hidden_size), T});

    auto roots = workspace::mtp_stem(work_, config_, T, input_embeddings == nullptr);
    Tensor emb;
    if (input_embeddings != nullptr) {
        if (input_embeddings->dtype != DType::BF16 ||
            input_embeddings->ne[0] != dimension(config_.hidden_size) ||
            input_embeddings->numel() !=
                static_cast<std::int64_t>(dimension(config_.hidden_size)) * T ||
            !input_embeddings->is_contiguous() || input_embeddings->data == nullptr) {
            throw std::invalid_argument("MTP input embeddings shape mismatch");
        }
        emb = input_embeddings->view({dimension(config_.hidden_size), T});
    } else {
        emb = roots.embedding;
        if (embedding_is_split()) {
            // The MTP layer runs on this shard alone, so the column split needs the peer's half:
            // broadcast the token ids and let it gather its columns, then merge.
            if (peer_tp_ == nullptr || pair_tp_ == nullptr) {
                throw std::logic_error(
                    "mtp_forward_stem: a column-split token embedding needs set_tp_peer");
            }
            embedding_tp2(*peer_tp_, *pair_tp_, flat_ids, nullptr, emb, nullptr);
        } else {
            ops::embedding(flat_ids, *embed_, emb, s);
        }
    }

    Tensor e = roots.normalized_embedding;
    Tensor h = roots.normalized_hidden;
    ops::rmsnorm(emb, mtp_->embedding_norm, config_.rms_norm_eps, true, e, s);
    ops::rmsnorm(flat_hidden, mtp_->hidden_norm, config_.rms_norm_eps, true, h, s);

    Tensor fc_in = roots.packed_input;
    ops::mtp_pack_fc_input(e, h, fc_in, s);

    x = roots.residual;
    project(fc_in, mtp_->input_projection, x, work_, s);

    ah = roots.attention_hidden;
    ops::rmsnorm(x, mtp_->input_norm, config_.rms_norm_eps, true, ah, s);
}

void TextContext::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto projection = workspace::mtp_attention_projection(work_, config_, T);
    Tensor q              = projection.query.view({dimension(config_.attention->head_dim),
                                                   dimension(config_.attention->num_attention_heads), T});
    Tensor k              = projection.key.view({dimension(config_.attention->head_dim),
                                                 dimension(config_.attention->num_key_value_heads), T});
    Tensor gate           = projection.gate.view({dimension(config_.attention->head_dim),
                                                  dimension(config_.attention->num_attention_heads), T});
    Tensor v              = projection.value.view({dimension(config_.attention->head_dim),
                                                   dimension(config_.attention->num_key_value_heads), T});
    Tensor q_flat         = q.view({dimension(config_.attention->query_width()), T});
    Tensor gate_flat      = gate.view({dimension(config_.attention->query_width()), T});
    Tensor k_flat         = k.view({dimension(config_.attention->key_width()), T});
    Tensor v_flat         = v.view({dimension(config_.attention->key_width()), T});
    mtp_projection(ah, mtp_->projection, *config_.attention, q_flat, gate_flat, k_flat, v_flat,
                   work_, s);

    const auto results = workspace::mtp_attention_results(work_, config_, T);
    Tensor qn =
        results.normalized_query.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    Tensor kn = results.normalized_key.view({dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_key_value_heads), T});
    ops::rmsnorm(q, mtp_->query_norm, config_.rms_norm_eps, true, qn, s);
    ops::rmsnorm(k, mtp_->key_norm, config_.rms_norm_eps, true, kn, s);
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    text_rope(rope_for_op, *config_.rope_parameters, qn, kn, s);

    Tensor a = results.attention.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T ||
            active_backend_kv_table_rows_ == nullptr || active_valid_columns_ == nullptr) {
            throw std::logic_error("MTP sequence batch binding is incomplete");
        }
        Tensor q_batch        = qn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_attention_heads), width,
                                         active_sequence_batch_});
        Tensor k_batch        = kn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_key_value_heads), width,
                                         active_sequence_batch_});
        Tensor v_batch        = v.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_key_value_heads), width,
                                        active_sequence_batch_});
        Tensor a_batch        = a.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_attention_heads), width,
                                        active_sequence_batch_});
        Tensor position_batch = positions.view({width, active_sequence_batch_});
        ops::causal_softmax_attention(
            q_batch, k_batch, v_batch, position_batch, *active_valid_columns_,
            *active_backend_kv_table_rows_,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_mtp_kv_->batch_layer_view(0), envelope, work_, a_batch, s);
    } else {
        ops::causal_softmax_attention(
            qn, kn, v, positions, Tensor{}, io_.backend_kv_table_row,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_mtp_kv_->batch_layer_view(0), envelope, work_, a, s);
    }
    ops::sigmoid_mul(gate, a, s);

    const auto post = workspace::mtp_post_attention(work_, config_, T);
    Tensor o        = post.output;
    project(a.view({dimension(config_.attention->query_width()), T}), mtp_->output, o, work_, s);
    ops::residual_add(o, x, s);

    Tensor mh = post.post_mixer_hidden;
    ops::rmsnorm(x, mtp_->post_attention_norm, config_.rms_norm_eps, true, mh, s);

    {
        auto post_mixer_scope = work_.scope();
        ffn(mh, mtp_->ffn, x, {}, work_, s, true);
    }

    Tensor flat_mtp_hidden = mtp_hidden.view({dimension(config_.hidden_size), T});
    ops::rmsnorm(x, mtp_->final_norm, config_.rms_norm_eps, true, flat_mtp_hidden, s);
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden, const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    nvtx::ScopedRange forward_range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                                    static_cast<std::uint64_t>(ids.numel()));
    auto scratch_scope = work_.scope();
    Tensor x;
    Tensor ah;
    mtp_forward_stem(ids, hidden, input_embeddings, x, ah);
    mtp_forward_tail(x, ah, positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor* input_embeddings, const Tensor& positions,
                                    const Tensor& rope_positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    bool final_chunk, Tensor* final_hidden, Tensor* logits,
                                    Tensor* draft_token) {
    if (!mtp_kv_.valid()) { throw std::runtime_error("MTP prefill is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "MTP prefill hidden");
    require_tensor_shape(positions, DType::I32, {T}, "MTP prefill positions");
    if (rope_positions.dtype != DType::I32 || rope_positions.ne[0] != T ||
        (rope_positions.ne[1] != 1 && rope_positions.ne[1] != 3) || rope_positions.ne[2] != 1 ||
        rope_positions.ne[3] != 1 || !rope_positions.is_contiguous() ||
        rope_positions.data == nullptr) {
        throw std::invalid_argument("MTP prefill rope positions must be [T] or [T,3]");
    }
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP final prefill outputs are required");
        }
        require_tensor_shape(*final_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                             "MTP final prefill hidden");
        require_tensor_shape(*logits, DType::BF16, {dimension(config_.vocab_size), 1},
                             "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    cudaStream_t s     = ctx_.stream;
    auto scratch_scope = work_.scope();
    Tensor x_last;
    Tensor ah_last;
    if (final_chunk) {
        x_last  = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        ah_last = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
    }

    {
        auto bulk_scope = work_.scope();
        Tensor x;
        Tensor ah;
        mtp_forward_stem(ids, hidden, input_embeddings, x, ah);

        Tensor k_flat = work_.alloc(DType::BF16, {dimension(config_.attention->key_width()), T});
        Tensor v_flat = work_.alloc(DType::BF16, {dimension(config_.attention->key_width()), T});
        mtp_kv_projection(ah, mtp_->projection, *config_.attention, k_flat, v_flat, work_, s);
        Tensor k = k_flat.view({dimension(config_.attention->head_dim),
                                dimension(config_.attention->num_key_value_heads), T});
        Tensor v = v_flat.view({dimension(config_.attention->head_dim),
                                dimension(config_.attention->num_key_value_heads), T});
        Tensor kn =
            work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_key_value_heads), T});
        ops::rmsnorm(k, mtp_->key_norm, config_.rms_norm_eps, true, kn, s);
        text_rope(rope_positions, *config_.rope_parameters, kn, s);
        ops::kv_cache_append(kn, v, positions, mtp_kv_.layer_view(0), s);

        if (final_chunk) {
            const std::size_t column_bytes =
                static_cast<std::size_t>(dimension(config_.hidden_size)) * dtype_size(DType::BF16);
            const auto* x_src = static_cast<const unsigned char*>(x.data) +
                                static_cast<std::size_t>(T - 1) * column_bytes;
            const auto* ah_src = static_cast<const unsigned char*>(ah.data) +
                                 static_cast<std::size_t>(T - 1) * column_bytes;
            CUDA_CHECK(
                cudaMemcpyAsync(x_last.data, x_src, column_bytes, cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(
                cudaMemcpyAsync(ah_last.data, ah_src, column_bytes, cudaMemcpyDeviceToDevice, s));
        }
    }

    if (final_chunk) {
        Tensor q_flat = work_.alloc(DType::BF16, {dimension(config_.attention->query_width()), 1});
        Tensor gate_flat =
            work_.alloc(DType::BF16, {dimension(config_.attention->query_width()), 1});
        mtp_query_gate_projection(ah_last, mtp_->projection, *config_.attention, q_flat, gate_flat,
                                  work_, s);
        Tensor q    = q_flat.view({dimension(config_.attention->head_dim),
                                   dimension(config_.attention->num_attention_heads), 1});
        Tensor gate = gate_flat.view({dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_attention_heads), 1});
        Tensor qn =
            work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_attention_heads), 1});
        ops::rmsnorm(q, mtp_->query_norm, config_.rms_norm_eps, true, qn, s);
        Tensor last_position = positions.slice(0, T - 1, 1);
        Tensor last_rope_position;
        if (rope_positions.ne[1] == 1) {
            last_rope_position = rope_positions.slice(0, T - 1, 1);
        } else {
            last_rope_position = work_.alloc(DType::I32, {1, 3});
            for (int axis = 0; axis < 3; ++axis) {
                const auto* src = static_cast<const std::int32_t*>(rope_positions.data) +
                                  static_cast<std::size_t>(axis) * T + (T - 1);
                auto* dst = static_cast<std::int32_t*>(last_rope_position.data) + axis;
                CUDA_CHECK(
                    cudaMemcpyAsync(dst, src, sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
            }
        }
        text_rope(last_rope_position, *config_.rope_parameters, qn, s);

        Tensor a = work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_attention_heads), 1});
        ops::causal_softmax_attention_cached(
            qn, last_position,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            mtp_kv_.layer_view(0), envelope, work_, a, s);
        ops::sigmoid_mul(gate, a, s);

        Tensor o = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        project(a.view({dimension(config_.attention->query_width()), 1}), mtp_->output, o, work_,
                s);
        ops::residual_add(o, x_last, s);

        Tensor mh = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        ops::rmsnorm(x_last, mtp_->post_attention_norm, config_.rms_norm_eps, true, mh, s);
        {
            auto post_mixer_scope = work_.scope();
            ffn(mh, mtp_->ffn, x_last, {}, work_, s, true);
        }
        ops::rmsnorm(x_last, mtp_->final_norm, config_.rms_norm_eps, true, *final_hidden, s);
        proposal_argmax(*final_hidden, *logits, *draft_token);
    }
}

void TextContext::proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens) {
    auto proposal_scope = work_.scope();
    const int T         = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "proposal hidden");
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, dimension(config_.vocab_size), T, "proposal logits");
    nvtx::ScopedRange proposal_range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                                     static_cast<std::uint64_t>(T));
    if (proposal_head_ != nullptr) {
        const std::int32_t rows = dimension(proposal_head_->weight.n);
        if (rows == proposal_head_n_) {
            Tensor proposal_logits = work_.alloc(DType::BF16, {proposal_head_n_, T});
            project(hidden, *proposal_head_, proposal_logits, work_, ctx_.stream);
            ops::argmax(proposal_logits, proposal_tokens,
                        proposal_head_ids_
                            ? proposal_head_n_
                            : dimension(parameters_.model.resources().public_token_count),
                        ctx_.stream);
            if (proposal_head_ids_ != nullptr) {
                ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_,
                                              proposal_head_n_, ctx_.stream);
            }
            return;
        }
        // Vocabulary-split draft head: each shard projects its own row block of the reduced
        // vocabulary into the draft logits, and the pair merges them back to [rows*2, T] on both
        // shards. Only this shard samples, and only this shard runs the MTP layer, so it hands the
        // peer the pre-head hidden state it needs.
        if (rows <= 0 || rows * 2 != proposal_head_n_ || peer_tp_ == nullptr || pair_tp_ == nullptr ||
            peer_tp_->proposal_head_ == nullptr ||
            dimension(peer_tp_->proposal_head_->weight.n) != rows) {
            throw std::logic_error(
                "proposal_argmax: a split proposal head needs half the rows on a peer that holds "
                "the matching half");
        }
        TextContext& peer                  = *peer_tp_;
        // This branch is the one place a proposal head allocates on the peer's arena, so it needs
        // the peer's scope as well: a captured CUDA Graph bakes the peer's watermark into its
        // kernel arguments, and an unwound-only-locally call site makes that watermark depend on
        // how many draft steps ran before the capture.
        auto peer_scope = peer.work_.scope();
        tp::DevicePair& pair               = *pair_tp_;
        const std::int32_t hidden_size = dimension(config_.hidden_size);
        ctx_.bind_to_current_thread();
        Tensor merged = work_.alloc(DType::BF16, {proposal_head_n_, T});
        Tensor hidden_local = work_.alloc(DType::BF16, {hidden_size, T});
        CUDA_CHECK(cudaMemcpyAsync(hidden_local.data, hidden.data, hidden_local.bytes(),
                                   cudaMemcpyDeviceToDevice, ctx_.stream));
        peer.ctx_.bind_to_current_thread();
        Tensor hidden_peer = peer.work_.alloc(DType::BF16, {hidden_size, T});
        CUDA_CHECK(cudaMemsetAsync(hidden_peer.data, 0, hidden_peer.bytes(), peer.ctx_.stream));
        ctx_.bind_to_current_thread();
        pair.allreduce(hidden_local.data, hidden_peer.data, hidden_local.bytes(), ctx_.stream,
                       peer.ctx_.stream);

        Tensor part = work_.alloc(DType::BF16, {rows, T});
        project(hidden_local, *proposal_head_, part, work_, ctx_.stream);
        peer.ctx_.bind_to_current_thread();
        Tensor part_peer = peer.work_.alloc(DType::BF16, {rows, T});
        project(hidden_peer, *peer.proposal_head_, part_peer, peer.work_, peer.ctx_.stream);
        Tensor merged_peer = peer.work_.alloc(DType::BF16, {proposal_head_n_, T});
        merge_local_row_blocks(peer, pair, part, part_peer, merged, merged_peer, rows);

        ctx_.bind_to_current_thread();
        ops::argmax(merged, proposal_tokens,
                    proposal_head_ids_
                        ? proposal_head_n_
                        : dimension(parameters_.model.resources().public_token_count),
                    ctx_.stream);
        if (proposal_head_ids_ != nullptr) {
            ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                          ctx_.stream);
        }
    } else {
        Tensor output_logits = matrix_window(logits, T);
        project(hidden, mtp_->output_head, output_logits, work_, ctx_.stream);
        ops::argmax(output_logits, proposal_tokens,
                    dimension(parameters_.model.resources().public_token_count), ctx_.stream);
    }
}

// Vocabulary-split reduced proposal head (TP-2 masked draft). The table is halved by row, so each
// shard ranks only its own half: it takes a copy of the draft's hidden state, projects its own row
// block and returns its own sixteen candidates. The pair then unions the thirty-two candidates and
// merges them back to the exact top sixteen of the whole table, which is the list the replicated
// table would return. Only the shard that runs the masked draft keeps the merged candidates, so the
// peer's copies and the union buffers are scratch in the peer's own workspace.
void TextContext::proposal_topk_tp2(const Tensor& hidden, Tensor& candidate_ids,
                                    Tensor& candidate_scores) {
    if (peer_tp_ == nullptr || pair_tp_ == nullptr || proposal_head_ == nullptr ||
        proposal_head_ids_ == nullptr) {
        throw std::logic_error(
            "proposal_topk_tp2: a split proposal head needs a registered peer and its id map");
    }
    TextContext& peer    = *peer_tp_;
    tp::DevicePair& pair = *pair_tp_;
    const std::int32_t rows    = dimension(proposal_head_->weight.n);
    const std::int32_t columns = hidden.ne[1];
    if (rows <= 0 || rows * 2 != proposal_head_n_ || shard_index_ < 0 || peer.shard_index_ < 0 ||
        peer.shard_index_ == shard_index_ || peer.proposal_head_ == nullptr ||
        peer.proposal_head_ids_ == nullptr || dimension(peer.proposal_head_->weight.n) != rows) {
        throw std::logic_error(
            "proposal_topk_tp2: a split proposal head needs half the rows on a peer that holds "
            "the matching half");
    }
    const std::int32_t hidden_size = dimension(config_.hidden_size);
    const std::int32_t candidates  = candidate_ids.ne[0];
    require_tensor_shape(hidden, DType::BF16, {hidden_size, columns}, "proposal hidden");
    require_tensor_shape(candidate_ids, DType::I32, {candidates, columns},
                         "proposal candidate ids");
    require_tensor_shape(candidate_scores, DType::FP32, {candidates, columns},
                         "proposal candidate scores");

    auto local_scope = work_.scope();
    auto peer_scope  = peer.work_.scope();
    ctx_.bind_to_current_thread();
    Tensor hidden_local = work_.alloc(DType::BF16, {hidden_size, columns});
    CUDA_CHECK(cudaMemcpyAsync(hidden_local.data, hidden.data, hidden_local.bytes(),
                               cudaMemcpyDeviceToDevice, ctx_.stream));
    peer.ctx_.bind_to_current_thread();
    Tensor hidden_peer = peer.work_.alloc(DType::BF16, {hidden_size, columns});
    CUDA_CHECK(cudaMemsetAsync(hidden_peer.data, 0, hidden_peer.bytes(), peer.ctx_.stream));
    ctx_.bind_to_current_thread();
    pair.allreduce(hidden_local.data, hidden_peer.data, hidden_local.bytes(), ctx_.stream,
                   peer.ctx_.stream);

    // The rows this shard owns are the ones the split placed at its own offset of the whole table,
    // so each side ranks its half through the matching window of the reduced id map.
    ctx_.bind_to_current_thread();
    Tensor ids_local    = work_.alloc(DType::I32, {candidates, columns});
    Tensor scores_local = work_.alloc(DType::FP32, {candidates, columns});
    Tensor map_local(static_cast<void*>(const_cast<std::int32_t*>(proposal_head_ids_) +
                                        static_cast<std::int64_t>(shard_index_) * rows),
                     DType::I32, {rows});
    ops::linear_topk(hidden_local, proposal_head_->weight, map_local, ids_local, scores_local,
                     work_, ctx_.stream);

    peer.ctx_.bind_to_current_thread();
    Tensor ids_peer    = peer.work_.alloc(DType::I32, {candidates, columns});
    Tensor scores_peer = peer.work_.alloc(DType::FP32, {candidates, columns});
    Tensor map_peer(static_cast<void*>(const_cast<std::int32_t*>(peer.proposal_head_ids_) +
                                       static_cast<std::int64_t>(peer.shard_index_) * rows),
                    DType::I32, {rows});
    ops::linear_topk(hidden_peer, peer.proposal_head_->weight, map_peer, ids_peer, scores_peer,
                     peer.work_, peer.ctx_.stream);

    // Union: this shard writes its candidates into the lower half of its union and the peer writes
    // its own into the upper half, so the pair's sum holds all thirty-two on both shards.
    const std::int64_t half_bytes =
        static_cast<std::int64_t>(candidates) * columns * sizeof(std::int32_t);
    ctx_.bind_to_current_thread();
    Tensor union_ids    = work_.alloc(DType::I32, {2 * candidates, columns});
    Tensor union_scores = work_.alloc(DType::FP32, {2 * candidates, columns});
    CUDA_CHECK(cudaMemsetAsync(union_ids.data, 0, union_ids.bytes(), ctx_.stream));
    CUDA_CHECK(cudaMemsetAsync(union_scores.data, 0, union_scores.bytes(), ctx_.stream));
    // Each shard publishes its own candidate list and receives the peer's through the pair's
    // byte-exact exchange: the payloads are I32 ids and FP32 scores, and the all-reduce would
    // reinterpret them as BF16, quieting any 16-bit lane that spells a signaling NaN. This shard
    // keeps its own list in the union's lower half and the peer's in its upper half, which is the
    // order the merge below reads.
    ctx_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(union_ids.data, ids_local.data, ids_local.bytes(),
                               cudaMemcpyDeviceToDevice, ctx_.stream));
    CUDA_CHECK(cudaMemcpyAsync(union_scores.data, scores_local.data, scores_local.bytes(),
                               cudaMemcpyDeviceToDevice, ctx_.stream));
    Tensor union_ids_upper(static_cast<std::uint8_t*>(union_ids.data) + half_bytes, DType::I32,
                           {candidates, columns});
    Tensor union_scores_upper(
        static_cast<std::uint8_t*>(union_scores.data) +
            static_cast<std::int64_t>(candidates) * columns * sizeof(float),
        DType::FP32, {candidates, columns});
    peer.ctx_.bind_to_current_thread();
    // The peer merges nothing, so its receive slot only has to exist: any [candidates, columns]
    // window serves.
    Tensor ids_peer_scratch = peer.work_.alloc(DType::I32, {candidates, columns});
    Tensor scores_peer_scratch = peer.work_.alloc(DType::FP32, {candidates, columns});
    const std::size_t ids_bytes =
        static_cast<std::size_t>(candidates) * columns * sizeof(std::int32_t);
    const std::size_t score_bytes =
        static_cast<std::size_t>(candidates) * columns * sizeof(float);
    pair.sendrecv(ids_local.data, union_ids_upper.data, ids_peer.data, ids_peer_scratch.data,
                  ids_bytes, ctx_.stream, peer.ctx_.stream);
    pair.sendrecv(scores_local.data, union_scores_upper.data, scores_peer.data,
                  scores_peer_scratch.data, score_bytes, ctx_.stream, peer.ctx_.stream);
    // The collective leaves the peer device current; the merge runs on this shard.
    ctx_.bind_to_current_thread();
    Tensor ids_lower(union_ids.data, DType::I32, {candidates, columns});
    Tensor scores_lower(union_scores.data, DType::FP32, {candidates, columns});
    Tensor ids_upper(static_cast<std::uint8_t*>(union_ids.data) + half_bytes, DType::I32,
                     {candidates, columns});
    Tensor scores_upper(static_cast<std::uint8_t*>(union_scores.data) +
                            static_cast<std::int64_t>(candidates) * columns * sizeof(float),
                        DType::FP32, {candidates, columns});
    ops::merge_topk_candidates(ids_lower, scores_lower, ids_upper, scores_upper, candidate_ids,
                               candidate_scores, ctx_.stream);
}


// TP-2 masked draft with the DFlash2 selector on the peer shard. The masked draft produces the hidden
// state and the candidate list on its own shard, hands both to the peer, and the peer runs the
// selector exactly as the one-device route does - its own projection, its own codebooks, its own
// sampling configs - then returns the drafts and proposal q. Each payload crosses byte for byte, so
// the returned values are the ones the local selector would have produced.
void TextContext::dflash_selector_tp2(const Tensor& hidden, const Tensor& candidate_ids,
                                      const Tensor& candidate_scores, const Tensor& anchors,
                                      const Tensor& frontiers, const ops::SamplingConfig* host_configs,
                                      Tensor& draft_tokens, Tensor& proposal_q) {
    if (peer_tp_ == nullptr || pair_tp_ == nullptr || selector_ != nullptr) {
        throw std::logic_error("dflash_selector_tp2: this shard holds the selector");
    }
    TextContext& peer = *peer_tp_;
    if (peer.selector_ == nullptr) {
        throw std::logic_error("dflash_selector_tp2: the peer shard does not hold the selector");
    }
    if (host_configs == nullptr) {
        throw std::invalid_argument("dflash_selector_tp2: sampling configs are required");
    }
    const std::int32_t top_k  = candidate_ids.ne[0];
    const std::int32_t drafts = candidate_ids.ne[1];
    const std::int32_t batch  = candidate_ids.ne[2];
    const std::int32_t columns = drafts * batch;
    const std::int32_t rank    = dimension(peer.selector_->hidden_projection.weight.n);
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), columns},
                         "selector hidden");
    require_tensor_shape(candidate_ids, DType::I32, {top_k, drafts, batch}, "selector candidates");
    require_tensor_shape(candidate_scores, DType::FP32, {top_k, columns}, "selector scores");
    require_tensor_shape(anchors, DType::I32, {batch}, "selector anchors");
    require_tensor_shape(frontiers, DType::I32, {batch}, "selector frontiers");
    require_tensor_shape(draft_tokens, DType::I32, {drafts, batch}, "selector drafts");
    require_tensor_shape(proposal_q, DType::FP32, {top_k, drafts, batch}, "selector proposal q");

    auto local_scope = work_.scope();
    auto peer_scope  = peer.work_.scope();
    const auto config_words = static_cast<std::int32_t>(
        (static_cast<std::size_t>(batch) * sizeof(ops::SamplingConfig) + 3) / 4);
    peer.ctx_.bind_to_current_thread();
    Tensor hidden_peer     = peer.work_.alloc(DType::BF16, {dimension(config_.hidden_size), columns});
    Tensor candidates_peer = peer.work_.alloc(DType::I32, {top_k, drafts, batch});
    Tensor scores_peer     = peer.work_.alloc(DType::FP32, {top_k, columns});
    Tensor anchors_peer    = peer.work_.alloc(DType::I32, {batch});
    Tensor frontiers_peer  = peer.work_.alloc(DType::I32, {batch});
    Tensor configs_peer    = peer.work_.alloc(DType::I32, {config_words});
    Tensor projected_peer  = peer.work_.alloc(DType::BF16, {rank, columns});
    Tensor drafts_peer     = peer.work_.alloc(DType::I32, {drafts, batch});
    Tensor q_peer          = peer.work_.alloc(DType::FP32, {top_k, drafts, batch});
    ctx_.bind_to_current_thread();

    // The draft's hidden state is bfloat16 on both sides, so the pair's in-kernel all-reduce carries
    // it without a host round trip and without the signaling-NaN hazard a raw I32/FP32 payload would
    // have: this shard's buffer plus the peer's zeroed buffer is exactly the broadcast.
    peer.ctx_.bind_to_current_thread();
    CUDA_CHECK(cudaMemsetAsync(hidden_peer.data, 0, hidden_peer.bytes(), peer.ctx_.stream));
    ctx_.bind_to_current_thread();
    pair_tp_->allreduce(hidden.data, hidden_peer.data, hidden.bytes(), ctx_.stream,
                        peer.ctx_.stream);
    ctx_.bind_to_current_thread();
    // The draft state travels to the selector's shard through the pair's byte-exact exchange, packed
    // into one transfer: the payload is I32 ids and FP32 scores, which the BF16 all-reduce would
    // reinterpret. The selector's outputs need a second exchange (they exist only after this one).
    const auto align16 = [](std::size_t value) { return (value + 15U) & ~std::size_t{15U}; };
    // The packed length is announced to the peer, and the peer unpacks in the declared order, so the
    // packing offsets below are checked against this payload rather than trusted (see the handoff
    // table in text.h).
    const std::size_t forward_payload =
        candidate_ids.bytes() + candidate_scores.bytes() + anchors.bytes() + frontiers.bytes();
    const std::size_t forward_bytes = align16(forward_payload);
    const std::size_t back_bytes    = align16(drafts_peer.bytes() + q_peer.bytes());
    Tensor pack_forward = work_.alloc(DType::I32, {static_cast<std::int32_t>(forward_bytes / 4)});
    Tensor recv_forward = work_.alloc(DType::I32, {static_cast<std::int32_t>(forward_bytes / 4)});
    Tensor recv_back    = work_.alloc(DType::I32, {static_cast<std::int32_t>(back_bytes / 4)});
    Tensor pack_back = peer.work_.alloc(DType::I32, {static_cast<std::int32_t>(back_bytes / 4)});
    Tensor recv_pack_forward =
        peer.work_.alloc(DType::I32, {static_cast<std::int32_t>(forward_bytes / 4)});
    std::size_t packed = 0;
    ctx_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(pack_forward.data, candidate_ids.data, candidate_ids.bytes(),
                               cudaMemcpyDeviceToDevice, ctx_.stream));
    packed = candidate_ids.bytes();
    CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(pack_forward.data) + packed,
                               candidate_scores.data, candidate_scores.bytes(),
                               cudaMemcpyDeviceToDevice, ctx_.stream));
    packed += candidate_scores.bytes();
    CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(pack_forward.data) + packed, anchors.data,
                               anchors.bytes(), cudaMemcpyDeviceToDevice, ctx_.stream));
    packed += anchors.bytes();
    CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(pack_forward.data) + packed, frontiers.data,
                               frontiers.bytes(), cudaMemcpyDeviceToDevice, ctx_.stream));
    packed += frontiers.bytes();
    if (packed != forward_payload) {
        throw std::logic_error("dflash_selector_tp2: forward pack does not match its declared length");
    }
    // This exchange's peer half carries nothing: the peer only has to send readable bytes, so it
    // sends from the buffer it receives into (an in-place exchange is safe).
    peer.ctx_.bind_to_current_thread();
    CUDA_CHECK(cudaMemsetAsync(recv_pack_forward.data, 0, forward_bytes, peer.ctx_.stream));
    ctx_.bind_to_current_thread();
    pair_tp_->sendrecv(pack_forward.data, recv_forward.data, recv_pack_forward.data,
                       recv_pack_forward.data, forward_bytes, ctx_.stream, peer.ctx_.stream);
    peer.ctx_.bind_to_current_thread();
    packed = 0;
    CUDA_CHECK(cudaMemcpyAsync(candidates_peer.data, recv_pack_forward.data,
                               candidate_ids.bytes(), cudaMemcpyDeviceToDevice, peer.ctx_.stream));
    packed = candidate_ids.bytes();
    CUDA_CHECK(cudaMemcpyAsync(scores_peer.data, static_cast<char*>(recv_pack_forward.data) + packed,
                               candidate_scores.bytes(), cudaMemcpyDeviceToDevice,
                               peer.ctx_.stream));
    packed += candidate_scores.bytes();
    CUDA_CHECK(cudaMemcpyAsync(anchors_peer.data,
                               static_cast<char*>(recv_pack_forward.data) + packed, anchors.bytes(),
                               cudaMemcpyDeviceToDevice, peer.ctx_.stream));
    packed += anchors.bytes();
    CUDA_CHECK(cudaMemcpyAsync(frontiers_peer.data,
                               static_cast<char*>(recv_pack_forward.data) + packed,
                               frontiers.bytes(), cudaMemcpyDeviceToDevice, peer.ctx_.stream));

    peer.ctx_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(configs_peer.data, host_configs,
                               static_cast<std::size_t>(batch) * sizeof(ops::SamplingConfig),
                               cudaMemcpyHostToDevice, peer.ctx_.stream));
    project(hidden_peer, peer.selector_->hidden_projection, projected_peer, peer.work_,
            peer.ctx_.stream);
    run_candidate_selector(
        *peer.selector_, candidates_peer, scores_peer.view({top_k, drafts, batch}),
        projected_peer.view({rank, drafts, batch}), anchors_peer, frontiers_peer,
        static_cast<const ops::SamplingConfig*>(configs_peer.data), drafts_peer, q_peer,
        peer.work_, peer.ctx_.stream);
    ctx_.bind_to_current_thread();

    // Second exchange: the selector's own outputs come back packed, and this shard's half of that
    // exchange is a zeroed scratch the peer discards.
    peer.ctx_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(pack_back.data, drafts_peer.data, drafts_peer.bytes(),
                               cudaMemcpyDeviceToDevice, peer.ctx_.stream));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(pack_back.data) + drafts_peer.bytes(),
                               q_peer.data, q_peer.bytes(), cudaMemcpyDeviceToDevice,
                               peer.ctx_.stream));
    Tensor back_dummy = work_.alloc(DType::I32, {static_cast<std::int32_t>(back_bytes / 4)});
    ctx_.bind_to_current_thread();
    CUDA_CHECK(cudaMemsetAsync(back_dummy.data, 0, back_bytes, ctx_.stream));
    pair_tp_->sendrecv(back_dummy.data, recv_back.data, pack_back.data, pack_back.data, back_bytes,
                       ctx_.stream, peer.ctx_.stream);
    ctx_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(draft_tokens.data, recv_back.data, drafts_peer.bytes(),
                               cudaMemcpyDeviceToDevice, ctx_.stream));
    CUDA_CHECK(cudaMemcpyAsync(proposal_q.data,
                               static_cast<char*>(recv_back.data) + drafts_peer.bytes(),
                               q_peer.bytes(), cudaMemcpyDeviceToDevice, ctx_.stream));
}

void TextContext::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                    Tensor* draft_token, const Tensor* explicit_rope_positions,
                                    const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T}, "MTP hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {dimension(config_.vocab_size), 1},
                             "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    auto position_scope = work_.scope();
    Tensor generated_rope_positions;
    const Tensor* rope_positions = explicit_rope_positions;
    if (rope_positions == nullptr) {
        generated_rope_positions = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, generated_rope_positions, ctx_.stream);
        rope_positions = &generated_rope_positions;
    } else if (rope_positions->dtype != DType::I32 || rope_positions->ne[0] != T ||
               (rope_positions->ne[1] != 1 && rope_positions->ne[1] != 3) ||
               rope_positions->ne[2] != 1 || rope_positions->ne[3] != 1 ||
               !rope_positions->is_contiguous() || rope_positions->data == nullptr) {
        throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
    }
    mtp_forward_core(ids, hidden, positions, *rope_positions, envelope, mtp_hidden,
                     input_embeddings);

    if (logits_column >= 0) {
        auto logits_scope = work_.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        proposal_argmax(col, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(position, DType::I32, {1}, "MTP AR position");
    require_tensor_shape(previous_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                         "MTP AR previous hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                         "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), 1}, "MTP AR logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");

    auto position_scope  = work_.scope();
    Tensor rope_position = work_.alloc(DType::I32, {1});
    ops::offset_i32_positions(position, io_.rope_delta, rope_position, ctx_.stream);
    mtp_forward_core(token, previous_hidden, position, rope_position, envelope, mtp_hidden,
                     nullptr);
    auto logits_scope = work_.scope();
    proposal_argmax(mtp_hidden, logits, draft_token);
}

void TextContext::ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                                        const Tensor& rope_positions, const Tensor& kv_table_rows,
                                        const Tensor& linear_state_source_slots,
                                        const Tensor& linear_state_destination_slots,
                                        ops::CausalAttentionExecutionEnvelope envelope,
                                        Tensor& hidden, Tensor& logits) {
    const std::int32_t batch = ids.ne[0];
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("ordinary decode batch size must be in [1,8]");
    }
    require_tensor_shape(ids, DType::I32, {batch}, "ordinary decode ids");
    require_tensor_shape(cache_positions, DType::I32, {batch}, "ordinary decode cache positions");
    require_tensor_shape(rope_positions, DType::I32, {batch}, "ordinary decode RoPE positions");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "ordinary decode KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention source slots");
    require_tensor_shape(linear_state_destination_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention destination slots");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), batch},
                         "ordinary decode hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), batch},
                         "ordinary decode logits");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> source_binding(active_linear_state_source_slots_,
                                                  &linear_state_source_slots);
        ScopedValue<const Tensor*> destination_binding(active_linear_state_destination_slots_,
                                                       &linear_state_destination_slots);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, 1);

        Tensor x = work_.alloc(DType::BF16, {dimension(config_.hidden_size), batch});
        ops::embedding(ids, *embed_, x, stream);
        NullTap tap;
        run_layers(x, Phase::Verify, tap);
        ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, hidden, stream);
        project(hidden, *lm_head_, logits, work_, stream);
    }
    work_.reset();
}

template <class Tap>
void TextContext::target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           const Tensor& linear_state_source_slots,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                           const Tensor* logits_mask, Tap& tap) {
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verify batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    require_tensor_shape(ids, DType::I32, {width, batch}, "target verify batch ids");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "target verify batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "target verify batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "target verify batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "target verify batch KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "target verify batch Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "target verify batch hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), width, batch},
                         "target verify batch logits");
    require_tensor_shape(target_tokens, DType::I32, {width, batch}, "target verify batch tokens");
    // The verify window only records the GDN replay; the fold after acceptance is what advances the
    // live state, which is why the caller binds the source slots only. Under any other action the
    // gdn_mix layer would take its UpdateInPlace arm, which reads the destination slots this entry
    // point does not bind - reject the mismatch here instead of dereferencing a null tensor deeper in
    // the walk.
    if (gdn_state_action_ != GdnStateAction::RecordForReplay) {
        throw std::logic_error("target verify batch requires the RecordForReplay GDN action");
    }

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_source_slots_,
                                                 &linear_state_source_slots);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);

        Tensor x        = work_.alloc(DType::BF16, {dimension(config_.hidden_size), columns});
        Tensor flat_ids = ids.view({columns});
        ops::embedding(flat_ids, *embed_, x, stream);
        if constexpr (Tap::enabled) { tap.begin(x); }
        run_layers(x, Phase::Verify, tap);
        if constexpr (requires { tap.capture_positions(cache_positions, stream); }) {
            tap.capture_positions(cache_positions, stream);
        }
        Tensor flat_hidden = hidden.view({dimension(config_.hidden_size), columns});
        Tensor flat_logits = logits.view({dimension(config_.vocab_size), columns});
        Tensor flat_tokens = target_tokens.view({columns});
        ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, flat_hidden, stream);
        project(flat_hidden, *lm_head_, flat_logits, work_, stream);
        // The declared-name mask enters between the projection and the argmax so the accepted token
        // is drawn from the constrained distribution rather than corrected afterwards.
        if (logits_mask != nullptr) {
            ops::apply_token_mask(flat_logits, *logits_mask, stream);
        }
        ops::argmax(flat_logits, flat_tokens,
                    dimension(parameters_.model.resources().public_token_count), stream);
    }
    work_.reset();
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                      const Tensor* logits_mask) {
    NullTap tap;
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, envelope, hidden, logits, target_tokens,
                             logits_mask, tap);
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                      DFlashFeatureSink& sink, const Tensor* logits_mask) {
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, envelope, hidden, logits, target_tokens,
                             logits_mask, sink);
}

void TextContext::mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                           const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& mtp_hidden) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "MTP decode batch target hidden");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "MTP decode batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "MTP decode batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "MTP decode batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "MTP decode batch KV rows");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "MTP decode batch hidden");

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden, nullptr);
}

void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), batch},
                         "MTP proposal batch hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), batch},
                         "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    proposal_argmax(hidden, logits, draft_tokens);
}

void TextContext::attn_mix(const BlockParameters& w, Tensor& x, int fidx, Phase ph,
                           Tensor* delta, Tensor* activation) {
    // A shard's attention runs over its own query heads, so its post-attention activation is its
    // half of the full query width. A shard mixer assembles no wider activation, so an output
    // projection that gathers its input columns has no arrangement here (see tp_mixer_layer).
    if (activation != nullptr) {
        throw std::logic_error(
            "tensor parallel split: an attention output projection that gathers its input "
            "columns would need the mixer's full-width activation assembled across the pair");
    }
    const auto& p   = std::get<AttentionParameters>(w.mixer);
    const auto& cfg = shard_config();
    cudaStream_t s  = ctx_.stream;
    const int T    = x.ne[1];
    if (active_causal_attention_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }

    auto projection = workspace::text_attention_projection(work_, cfg, T);
    Tensor h        = projection.hidden;
    ops::rmsnorm(x, w.input_norm, cfg.rms_norm_eps, true, h, s);

    Tensor q         = projection.query.view({dimension(cfg.attention->head_dim),
                                              dimension(cfg.attention->num_attention_heads), T});
    Tensor gate      = projection.gate.view({dimension(cfg.attention->head_dim),
                                             dimension(cfg.attention->num_attention_heads), T});
    Tensor k         = projection.key.view({dimension(cfg.attention->head_dim),
                                            dimension(cfg.attention->num_key_value_heads), T});
    Tensor v         = projection.value.view({dimension(cfg.attention->head_dim),
                                              dimension(cfg.attention->num_key_value_heads), T});
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&p.projection)) {
        // A GGUF projection stores one weight per block type, so it cannot run as the single fused
        // GEMM below. The projection op reads them part by part and writes q/gate/k/v directly, in
        // the same row order the fused buffer would have used.
        Tensor q_flat    = q.view({dimension(cfg.attention->query_width()), T});
        Tensor gate_flat = gate.view({dimension(cfg.attention->query_width()), T});
        Tensor k_flat    = k.view({dimension(cfg.attention->key_width()), T});
        Tensor v_flat    = v.view({dimension(cfg.attention->key_width()), T});
        ops::attn_input_proj(h, *gguf, q_flat, gate_flat, k_flat, v_flat, work_, s);
    } else {
        // The fused QKV+gate projection is a single GEMM whose output rows are laid out
        // [q | k | gate | v]. The per-shard fused weight (q/gate rows halved, k/v rows unchanged)
        // is run through the generic linear op, then the output is viewed into q/k/gate/v. This
        // mirrors the FFN-delta decomposition and avoids the shape-specific fused attn GEMM kernels
        // (which are registered for the full-model row counts only).
        const std::int32_t q_rows  = dimension(cfg.attention->query_width());
        const std::int32_t k_rows  = dimension(cfg.attention->key_width());
        const std::int32_t fused_n = 2 * q_rows + 2 * k_rows;
        Tensor fused = work_.alloc(DType::BF16, {fused_n, T});
        const auto& proj_w = std::get<LinearParameters>(p.projection);
        ops::linear(h, proj_w.weight, fused, proj_w.policy, work_, s);
        if (T > 1) {
            // Batched width: the four blocks are strided row windows of the fused output, so they
            // are materialized into the (already sized) workspace projection buffers.
            copy_row_block(fused, 0, q_rows, projection.query, s);
            copy_row_block(fused, q_rows, k_rows, projection.key, s);
            copy_row_block(fused, q_rows + k_rows, q_rows, projection.gate, s);
            copy_row_block(fused, 2 * q_rows + k_rows, k_rows, projection.value, s);
        } else {
            // Single column: the slices are contiguous, so q/k/gate/v alias the fused buffer
            // directly.
            q = fused.slice(0, 0, q_rows).view({dimension(cfg.attention->head_dim),
                                                dimension(cfg.attention->num_attention_heads), T});
            k = fused.slice(0, q_rows, k_rows)
                    .view({dimension(cfg.attention->head_dim),
                           dimension(cfg.attention->num_key_value_heads), T});
            gate = fused.slice(0, q_rows + k_rows, q_rows)
                       .view({dimension(cfg.attention->head_dim),
                              dimension(cfg.attention->num_attention_heads), T});
            v = fused.slice(0, 2 * q_rows + k_rows, k_rows)
                    .view({dimension(cfg.attention->head_dim),
                           dimension(cfg.attention->num_key_value_heads), T});
        }
    }

    const auto results = workspace::text_attention_results(work_, cfg, T);
    Tensor qn =
        results.normalized_query.view({dimension(cfg.attention->head_dim),
                                       dimension(cfg.attention->num_attention_heads), T});
    Tensor kn = results.normalized_key.view({dimension(cfg.attention->head_dim),
                                             dimension(cfg.attention->num_key_value_heads), T});
    ops::rmsnorm(q, p.query_norm, cfg.rms_norm_eps, true, qn, s);
    ops::rmsnorm(k, p.key_norm, cfg.rms_norm_eps, true, kn, s);
    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_positions =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    text_rope(rope_for_op, *cfg.rope_parameters, qn, kn, s);

    Tensor a = results.attention.view({dimension(cfg.attention->head_dim),
                                       dimension(cfg.attention->num_attention_heads), T});
    const Tensor& kv_table_rows =
        active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("Text sequence batch binding does not match aggregate columns");
        }
        Tensor q_batch        = qn.view({dimension(cfg.attention->head_dim),
                                         dimension(cfg.attention->num_attention_heads), width,
                                         active_sequence_batch_});
        Tensor k_batch        = kn.view({dimension(cfg.attention->head_dim),
                                         dimension(cfg.attention->num_key_value_heads), width,
                                         active_sequence_batch_});
        Tensor v_batch        = v.view({dimension(cfg.attention->head_dim),
                                        dimension(cfg.attention->num_key_value_heads), width,
                                        active_sequence_batch_});
        Tensor a_batch        = a.view({dimension(cfg.attention->head_dim),
                                        dimension(cfg.attention->num_attention_heads), width,
                                        active_sequence_batch_});
        Tensor position_batch = cache_positions.view({width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        ops::causal_softmax_attention(
            q_batch, k_batch, v_batch, position_batch, valid, kv_table_rows,
            {dimension(cfg.attention->head_dim),
             dimension(cfg.attention->num_attention_heads),
             dimension(cfg.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(cfg.attention->head_dim))),
            batch_text_kv_->batch_layer_view(fidx), *active_causal_attention_envelope_, work_,
            a_batch, s);
    } else {
        // Same optional column mask as the batched branch: a speculative window whose trailing
        // columns are clamped to the last valid column's position must not have those duplicates
        // append KV (they would race the position's owner inside one launch).
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        ops::causal_softmax_attention(
            qn, kn, v, cache_positions, valid, kv_table_rows,
            {dimension(cfg.attention->head_dim),
             dimension(cfg.attention->num_attention_heads),
             dimension(cfg.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(cfg.attention->head_dim))),
            batch_text_kv_->batch_layer_view(fidx), *active_causal_attention_envelope_, work_, a,
            s);
    }
    ops::sigmoid_mul(gate, a, s);

    const Tensor a_flat = a.view({dimension(cfg.attention->query_width()), T});
    if (delta == nullptr) {
        ops::linear_add(a_flat, p.output.weight, x, p.output.policy, work_, s);
    } else {
        ops::linear(a_flat, p.output.weight, *delta, p.output.policy, work_, s);
    }
}

void TextContext::gdn_mix(const BlockParameters& w, Tensor& x, int gidx, Phase ph,
                          Tensor* delta, Tensor* activation) {
    const auto& p   = std::get<GdnParameters>(w.mixer);
    const auto& cfg = shard_config();
    cudaStream_t s  = ctx_.stream;
    const int T    = x.ne[1];

    const auto control = workspace::gdn_control(work_, cfg, T);
    Tensor h           = control.hidden;
    // The GDN gating projections (a/b) run in full on both shards: a_log/dt_bias are replicated
    // at the full 48 value heads, so g/beta must be allocated at the full head count (the
    // gating op hardcodes 48). The delta net below slices them to this shard's local heads.
    const std::int32_t full_value_heads = dimension(config_.gdn->linear_num_value_heads);
    Tensor g      = work_.alloc(DType::FP32, {full_value_heads, T});
    Tensor beta   = work_.alloc(DType::FP32, {full_value_heads, T});
    gdn_norm_control(x, w.input_norm, cfg.rms_norm_eps, p, h, g, beta, work_,
                     ctx_.execution_view());

    // The GDN projection is phase-dependent:
    // - Batched Verify (speculative decoding): fused shape-specific ops (gdn_projection_snapshot/
    //   record), registered for the full-model row counts. Uses workspace-allocated qc/kc/vc/z.
    // - Prefill (T>1, single-GPU): fused shape-specific op (gdn_projection), registered for the
    //   full-model row counts. Uses workspace-allocated qc/kc/vc/z + separate qkv buffer.
    // - Everything else, i.e. single-token decode (T=1) and every head-split shard width:
    //   decomposed GEMM (generic ops::linear on the shard's fused weight) + conv. The fused GEMM
    //   output [q|k|v|z] is sliced into the conv input (qkv) and the output gate (z), and qc/kc/vc
    //   are separate workspace buffers the conv writes, so the conv input never overlaps its
    //   outputs.
    Tensor qc, kc, vc, z;
    // A head-split shard has no fused record workspace for the width>1 op, so it takes the same
    // decomposed route as its prefill; the phase still selects the decode-equivalent kernels.
    const bool batched_verify =
        (ph == Phase::Verify) && (active_sequence_batch_ > 1 || active_sequence_width_ > 1) &&
        shard_config_ == nullptr;
    // Replay recording. The full model records through the fused width>1 ops above. A head-split
    // shard cannot: those ops are registered for the full 16384-row fused parent only, which is
    // why this route used to skip recording entirely. It records through the recurrent record op
    // instead, which is registered for the shard geometry (8 key heads, 24 value heads) and is the
    // geometry the fold plan is built for.
    const bool recording = gdn_state_action_ == GdnStateAction::RecordForReplay;
    if (recording && replay_records_ == nullptr) {
        throw std::logic_error("Replay-record GDN has no record storage");
    }
    if (recording && T < 2) {
        throw std::logic_error("Replay-record GDN requires at least two window columns");
    }
    const std::int32_t record_rows = active_sequence_batch_ > 0 ? active_sequence_batch_ : 1;
    GdnReplayRecordLayer records{};
    if (recording && shard_config_ != nullptr) {
        // One record row per lane: the record planes are [..., width, record_rows] with the lane as
        // the outer dimension, so a multi-lane verify window records one physical row per lane.
        records = replay_records_->layer(gidx, record_rows);
    }
    if (batched_verify) {
        if (active_sequence_batch_ == 0 || active_linear_state_source_slots_ == nullptr) {
            throw std::logic_error(
                "Verify GDN requires an explicit sequence batch and state slots");
        }
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        if (gdn_state_action_ == GdnStateAction::UpdateInPlace && width != 1) {
            throw std::logic_error("In-place batched GDN update requires width one");
        }
        const auto projection = workspace::gdn_projection(work_, cfg, T);
        z  = projection.output_gate.view({dimension(cfg.gdn->linear_value_head_dim),
                                          dimension(cfg.gdn->linear_num_value_heads), T});
        qc = projection.query;
        kc = projection.key;
        vc = projection.value;
        Tensor projection_input =
            h.view({dimension(cfg.hidden_size), width, active_sequence_batch_});
        Tensor query_output =
            qc.view({dimension(cfg.gdn->key_width()), width, active_sequence_batch_});
        Tensor key_output =
            kc.view({dimension(cfg.gdn->key_width()), width, active_sequence_batch_});
        Tensor value_output =
            vc.view({dimension(cfg.gdn->value_width()), width, active_sequence_batch_});
        Tensor gate_output =
            z.view({dimension(cfg.gdn->value_width()), width, active_sequence_batch_});
        Tensor conv_states = state_.layer_view(static_cast<std::uint32_t>(gidx)).conv;
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            if (replay_records_ == nullptr) {
                throw std::logic_error("Replay-record GDN has no record storage");
            }
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            gdn_projection_record(projection_input, p, *cfg.gdn, conv_states, valid,
                                  *active_linear_state_source_slots_, records.conv, query_output,
                                  key_output, value_output, gate_output, work_, s);
        } else {
            gdn_projection_snapshot(projection_input, p, *cfg.gdn, conv_states, valid,
                                    *active_linear_state_source_slots_,
                                    *active_linear_state_destination_slots_, query_output,
                                    key_output, value_output, gate_output, work_, s);
        }
    } else if (T > 1 && shard_config_ == nullptr) {
        // Prefill (single-GPU, T>1): fused shape-specific GEMM + conv (original behavior). The
        // fused op is registered for the full-model fused row count only, so a head-split shard
        // (half the fused rows) takes the decomposed route below instead.
        const auto projection = workspace::gdn_projection(work_, cfg, T);
        z  = projection.output_gate.view({dimension(cfg.gdn->linear_value_head_dim),
                                          dimension(cfg.gdn->linear_num_value_heads), T});
        qc = projection.query;
        kc = projection.key;
        vc = projection.value;
        Tensor qkv    = workspace::gdn_prefill_conv(work_, cfg, T);
        Tensor z_flat = z.view({dimension(cfg.gdn->value_width()), T});
        gdn_projection(h, p, qkv, z_flat, work_, s);
        Tensor conv_state_in =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor conv_state_out =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::causal_conv1d_silu_split(qkv, p.convolution, conv_state_in, conv_state_out, qc, kc, vc,
                                      s);
    } else {
        // Single-token decode (T=1): decomposed GEMM (generic ops::linear on the per-shard
        // fused weight) + conv. The GEMM writes [q|k|v|z] into a single fused buffer; q/k/v are
        // sliced from it as the conv input, z is sliced as the output gate, and qc/kc/vc are
        // separate workspace buffers (conv outputs) so the conv input does not overlap them.
        const std::int32_t gdn_q_rows = dimension(cfg.gdn->key_width());
        const std::int32_t gdn_v_rows = dimension(cfg.gdn->value_width());
        const std::int32_t fused_n    = 2 * gdn_q_rows + 2 * gdn_v_rows;
        Tensor fused = work_.alloc(DType::BF16, {fused_n, T});
        if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&p.projection)) {
            // A GGUF projection stores one weight per block type, so it is not one fused GEMM. Its
            // q/k/value and z rows land in this buffer's blocks directly, in the order the slices
            // below expect: the destination is the whole [q|k|v|z, T] matrix, which a head-split
            // shard's materialized parent holds contiguously and the GGUF product requires (it
            // writes contiguous destinations only).
            ops::gdn_input_proj(h, *gguf, fused, work_, s);
        } else {
            const auto& proj_w = std::get<LinearParameters>(p.projection);
            ops::linear(h, proj_w.weight, fused, proj_w.policy, work_, s);
        }
        auto projection = workspace::gdn_projection(work_, cfg, T);
        qc = projection.query;
        kc = projection.key;
        vc = projection.value;
        Tensor qkv;
        if (T > 1) {
            // Batched width: the [q|k|v] and [z] blocks are strided row windows of the fused
            // output, so they are materialized into contiguous buffers before the split
            // convolution (which requires a contiguous input) and the output gate.
            qkv = workspace::gdn_prefill_conv(work_, cfg, T);
            copy_row_block(fused, 0, 2 * gdn_q_rows + gdn_v_rows, qkv, s);
            copy_row_block(fused, 2 * gdn_q_rows + gdn_v_rows, gdn_v_rows, projection.output_gate,
                           s);
            z = projection.output_gate.view({dimension(cfg.gdn->linear_value_head_dim),
                                             dimension(cfg.gdn->linear_num_value_heads), T});
        } else {
            // Single column: the slices are contiguous and alias the fused buffer directly.
            qkv = fused.slice(0, 0, 2 * gdn_q_rows + gdn_v_rows);
            z = fused.slice(0, 2 * gdn_q_rows + gdn_v_rows, gdn_v_rows)
                    .view({dimension(cfg.gdn->linear_value_head_dim),
                           dimension(cfg.gdn->linear_num_value_heads), T});
        }
        // A batched verify on a head-split shard needs one convolution window per lane, which the
        // split form cannot express: its state pair is scalar and its accepted row profile is a
        // fixed triplet. The snapshot form takes the per-lane initial and destination slot vectors
        // and writes [C, W, B]; each lane here is one column and advances its own slot in place
        // (its destination interval is its own initial slot, which the op allows). Its output is a
        // single contiguous [q|k|v, B] buffer, so the three regions are copied into the compact
        // buffers the recurrence views, because a row slice of the wider parent is a strided
        // window.
        const bool per_lane_conv = shard_config_ != nullptr &&
                                   (ph == Phase::Verify || recording) &&
                                   active_sequence_batch_ > 1;
        if (per_lane_conv) {
            if (!recording && active_sequence_width_ != 1) {
                throw std::logic_error("the GDN shard window supports one column per lane");
            }
            const std::int32_t conv_channels = 2 * gdn_q_rows + gdn_v_rows;
            const std::int32_t window = recording ? T / active_sequence_batch_ : 1;
            Tensor conv_out    = work_.alloc(DType::BF16, {conv_channels, T});
            Tensor conv_states = state_.layer_view(static_cast<std::uint32_t>(gidx)).conv;
            const Tensor conv_input = qkv.view({conv_channels, window, active_sequence_batch_});
            Tensor conv_window      = conv_out.view({conv_channels, window, active_sequence_batch_});
            if (recording) {
                // A recording window is wider than one column, so its per-lane convolution runs on a
                // private copy of the pool: the fold rebuilds the committed history from the record
                // and the restored pool, so this window must not advance the live convolution state,
                // and the [W] slot reservations a wide window needs cannot overlap live slots. The
                // staging copy keeps the whole pool in front, so the initial channel is still the
                // pool's own slot index and each lane's reservation follows after it.
                const std::int32_t lanes = active_sequence_batch_;
                const std::int32_t slots = static_cast<std::int32_t>(conv_states.ne[2]);
                Tensor staged =
                    work_.alloc(DType::BF16, {conv_channels, 3, slots + lanes * window});
                CUDA_CHECK(cudaMemcpyAsync(staged.data, conv_states.data, conv_states.bytes(),
                                           cudaMemcpyDeviceToDevice, s));
                // Each lane's reservation starts one window after the previous lane's, so the
                // bases are the arithmetic progression `slots + lane * window`. They are built on
                // the device: a host staging pointer would be baked into a captured verify graph
                // (P2.2c) and read a dead stack slot on every replay.
                Tensor base_slots = work_.alloc(DType::I32, {lanes});
                ops::fill_i32_positions(base_slots, slots, window, s);
                ops::causal_conv1d_silu_snapshot(conv_input, p.convolution, staged, Tensor{},
                                                 *active_linear_state_source_slots_, base_slots,
                                                 conv_window, s);
            } else {
                ops::causal_conv1d_silu_snapshot(
                    conv_input, p.convolution, conv_states, Tensor{},
                    *active_linear_state_source_slots_, *active_linear_state_destination_slots_,
                    conv_window, s);
            }
            const Tensor conv_matrix = conv_out.view({conv_channels, T});
            copy_row_block(conv_matrix, 0, gdn_q_rows, qc, s);
            copy_row_block(conv_matrix, gdn_q_rows, gdn_q_rows, kc, s);
            copy_row_block(conv_matrix, 2 * gdn_q_rows, gdn_v_rows, vc, s);
        } else {
            Tensor conv_state_in =
                state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
            Tensor conv_state_out =
                state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
            ops::causal_conv1d_silu_split(qkv, p.convolution, conv_state_in, conv_state_out, qc,
                                          kc, vc, s);
        }
        if (recording && shard_config_ != nullptr) {
            // The fold rebuilds the convolution history as tail_3(old_history || record[0:committed]),
            // so the record is the raw pre-convolution window. This route already materializes that
            // buffer contiguously as [q|k|v], which is the record plane's channel order and layout.
            if (records.conv.data == nullptr || records.conv.bytes() != qkv.bytes()) {
                throw std::logic_error("GDN convolution record does not match the shard window");
            }
            CUDA_CHECK(cudaMemcpyAsync(records.conv.data, qkv.data, qkv.bytes(),
                                       cudaMemcpyDeviceToDevice, s));
        }
    }

    Tensor q_recurrent = qc.view({dimension(cfg.gdn->linear_key_head_dim),
                                  dimension(cfg.gdn->linear_num_key_heads), T});
    Tensor k_recurrent = kc.view({dimension(cfg.gdn->linear_key_head_dim),
                                  dimension(cfg.gdn->linear_num_key_heads), T});

    // The GDN gating projections run in full (48 heads, replicated a_log/dt_bias); slice
    // g/beta to this shard's local value heads (shard 0 = heads [0, H), shard 1 = [H, 2H)).
    const std::int32_t local_heads = dimension(cfg.gdn->linear_num_value_heads);
    const std::int32_t g_head0     = shard_index_ * local_heads;
    Tensor g_local    = g.slice(0, g_head0, local_heads);
    Tensor beta_local = beta.slice(0, g_head0, local_heads);
    if (T > 1) {
        // Batched width: the shard's heads are a strided row window of the full 48-head gating
        // output, and gated_delta_net requires contiguous g/beta.
        Tensor g_block    = work_.alloc(DType::FP32, {local_heads, T});
        Tensor beta_block = work_.alloc(DType::FP32, {local_heads, T});
        copy_row_block(g, g_head0, local_heads, g_block, s);
        copy_row_block(beta, g_head0, local_heads, beta_block, s);
        g_local    = g_block;
        beta_local = beta_block;
    }

    Tensor vv = vc.view({dimension(cfg.gdn->linear_value_head_dim),
                         dimension(cfg.gdn->linear_num_value_heads), T});
    Tensor o  = workspace::gdn_recurrent_output(work_, cfg, T)
                   .view({dimension(cfg.gdn->linear_value_head_dim),
                          dimension(cfg.gdn->linear_num_value_heads), T});
    // A head-split shard runs the decomposed recurrent path for the verify window too: the batched
    // verify views need the fused record workspace the shard does not have, and the recurrence is
    // sequential in both phases, so the shard's numerics do not depend on this branch.
    if (ph == Phase::Verify && shard_config_ == nullptr) {
        Tensor recurrent_states  = state_.layer_view(static_cast<std::uint32_t>(gidx)).recurrent;
        const std::int32_t width = active_sequence_width_;
        Tensor q_batch           = q_recurrent.view({dimension(cfg.gdn->linear_key_head_dim),
                                                     dimension(cfg.gdn->linear_num_key_heads), width,
                                                     active_sequence_batch_});
        Tensor k_batch           = k_recurrent.view({dimension(cfg.gdn->linear_key_head_dim),
                                                     dimension(cfg.gdn->linear_num_key_heads), width,
                                                     active_sequence_batch_});
        Tensor v_batch           = vv.view({dimension(cfg.gdn->linear_value_head_dim),
                                            dimension(cfg.gdn->linear_num_value_heads), width,
                                            active_sequence_batch_});
        Tensor g_batch =
            g_local.view({dimension(cfg.gdn->linear_num_value_heads), width, active_sequence_batch_});
        Tensor beta_batch = beta_local.view(
            {dimension(cfg.gdn->linear_num_value_heads), width, active_sequence_batch_});
        Tensor out_batch =
            o.view({dimension(cfg.gdn->linear_value_head_dim),
                    dimension(cfg.gdn->linear_num_value_heads), width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            ops::gated_delta_net_replay_record(
                q_batch, k_batch, v_batch, g_batch, beta_batch,
                static_cast<float>(
                    1.0 / std::sqrt(static_cast<double>(cfg.gdn->linear_key_head_dim))),
                recurrent_states, valid, *active_linear_state_source_slots_, records.key,
                records.value, records.gate, out_batch, s);
        } else {
            ops::gated_delta_net_batch_update(
                q_batch, k_batch, v_batch, g_batch, beta_batch,
                static_cast<float>(
                    1.0 / std::sqrt(static_cast<double>(cfg.gdn->linear_key_head_dim))),
                /*normalize_qk=*/true, recurrent_states, *active_linear_state_source_slots_,
                *active_linear_state_destination_slots_, out_batch, s);
        }
    } else if (recording && shard_config_ != nullptr) {
        // Head-split shard, replay recording. The record op reads the initial state from an absolute
        // slot and leaves every state slot untouched, so the window's advance is performed only by
        // the fold that replays the committed prefix. Its outputs are defined to be bit-identical to
        // the normalized gated_delta_net below, so the window's own columns keep the same logits.
        Tensor recurrent_states  = state_.layer_view(static_cast<std::uint32_t>(gidx)).recurrent;
        const std::int32_t width = T / record_rows;
        Tensor q_batch           = q_recurrent.view({dimension(cfg.gdn->linear_key_head_dim),
                                                      dimension(cfg.gdn->linear_num_key_heads), width,
                                                      record_rows});
        Tensor k_batch           = k_recurrent.view({dimension(cfg.gdn->linear_key_head_dim),
                                                      dimension(cfg.gdn->linear_num_key_heads), width,
                                                      record_rows});
        Tensor v_batch           = vv.view({dimension(cfg.gdn->linear_value_head_dim),
                                            dimension(cfg.gdn->linear_num_value_heads), width,
                                            record_rows});
        Tensor g_batch =
            g_local.view({dimension(cfg.gdn->linear_num_value_heads), width, record_rows});
        Tensor beta_batch =
            beta_local.view({dimension(cfg.gdn->linear_num_value_heads), width, record_rows});
        Tensor out_batch = o.view({dimension(cfg.gdn->linear_value_head_dim),
                                   dimension(cfg.gdn->linear_num_value_heads), width, record_rows});
        // A one-row record keeps the historic scalar slot; a multi-lane window hands the engine's
        // per-lane source slots straight through as the op's [record_rows] initial channel.
        Tensor initial_slots;
        if (active_linear_state_source_slots_ != nullptr) {
            initial_slots = *active_linear_state_source_slots_;
        } else {
            initial_slots = work_.alloc(DType::I32, {record_rows});
            ops::set_i32_scalar(initial_slots, linear_state_source_slot_, s);
        }
        ops::gated_delta_net_replay_record(
            q_batch, k_batch, v_batch, g_batch, beta_batch,
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(cfg.gdn->linear_key_head_dim))),
            recurrent_states, Tensor{}, initial_slots, records.key, records.value, records.gate,
            out_batch, s);
    } else if (shard_config_ != nullptr && ph == Phase::Verify && active_sequence_batch_ > 1) {
        // Batched verify on a head-split shard: one state slot per lane, one column each. This is
        // the same kernel the full-model batched verify runs; the shard (8 key heads, 24 value
        // heads) pair passes the head-count rule, the pool layer holds one recurrent plane per
        // slot, and each lane advances its own slot in place. The gating rows arrive as a compact
        // [local_heads, T] block because the shard heads are a strided window of the full 48.
        if (active_sequence_width_ != 1) {
            throw std::logic_error("the GDN shard window supports one column per lane");
        }
        const std::int32_t value_heads = dimension(cfg.gdn->linear_num_value_heads);
        Tensor recurrent_states = state_.layer_view(static_cast<std::uint32_t>(gidx)).recurrent;
        Tensor q_batch          = q_recurrent.view({dimension(cfg.gdn->linear_key_head_dim),
                                                    dimension(cfg.gdn->linear_num_key_heads), 1,
                                                    active_sequence_batch_});
        Tensor k_batch          = k_recurrent.view({dimension(cfg.gdn->linear_key_head_dim),
                                                    dimension(cfg.gdn->linear_num_key_heads), 1,
                                                    active_sequence_batch_});
        Tensor v_batch = vv.view({dimension(cfg.gdn->linear_value_head_dim), value_heads, 1,
                                  active_sequence_batch_});
        Tensor g_batch    = g_local.view({value_heads, 1, active_sequence_batch_});
        Tensor beta_batch = beta_local.view({value_heads, 1, active_sequence_batch_});
        Tensor out_batch  = o.view({dimension(cfg.gdn->linear_value_head_dim), value_heads, 1,
                                   active_sequence_batch_});
        ops::gated_delta_net_batch_update(
            q_batch, k_batch, v_batch, g_batch, beta_batch,
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(cfg.gdn->linear_key_head_dim))),
            /*normalize_qk=*/true, recurrent_states, *active_linear_state_source_slots_,
            *active_linear_state_destination_slots_, out_batch, s);
    } else {
        Tensor recurrent_state_in =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor recurrent_state_out =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::gated_delta_net(
            q_recurrent, k_recurrent, vv, g_local, beta_local,
            static_cast<float>(1.0 /
                               std::sqrt(static_cast<double>(cfg.gdn->linear_key_head_dim))),
            /*normalize_qk=*/true, work_, recurrent_state_in, recurrent_state_out, o, s);
    }

    Tensor on = workspace::gdn_normalized_output(work_, cfg, T)
                    .view({dimension(cfg.gdn->linear_value_head_dim),
                           dimension(cfg.gdn->linear_num_value_heads), T});
    ops::gated_rmsnorm(o, p.norm, z, cfg.rms_norm_eps, on, s);

    const Tensor on_flat = on.view({dimension(cfg.gdn->value_width()), T});
    if (activation != nullptr) {
        // The output projection's stored columns run over value heads of both shards (a GGUF block
        // checkpoint stores them tiled), so this shard cannot project its own heads alone. Leave
        // them in this shard's rows of the full-width grouped activation, zeroed elsewhere, and let
        // the tensor-parallel driver merge the pair and project (see run_layers_tp2).
        const std::int32_t local = dimension(cfg.gdn->value_width());
        const std::int32_t full  = dimension(config_.gdn->value_width());
        if (delta == nullptr || shard_index_ < 0 || local * 2 != full ||
            activation->dtype != DType::BF16 || !activation->is_contiguous() ||
            activation->ne[0] != full || activation->ne[1] != T || activation->ne[2] != 1 ||
            activation->ne[3] != 1 || activation->data == nullptr) {
            throw std::logic_error(
                "the GDN activation merge needs a full-width BF16 [value_width, T] buffer");
        }
        CUDA_CHECK(cudaMemsetAsync(activation->data, 0, activation->bytes(), s));
        write_row_block(on_flat, *activation, shard_index_ * local, s);
        return;
    }
    if (delta == nullptr) {
        ops::linear_add(on_flat, p.output.weight, x, p.output.policy, work_, s);
    } else {
        ops::linear(on_flat, p.output.weight, *delta, p.output.policy, work_, s);
    }
}

ops::SparseMoeHints TextContext::next_projection_hints(int layer) const {
    const auto next = static_cast<std::size_t>(layer) + 1;
    return next < parameters_.text.layers.size() ? parameters_.text.layers[next].projection_prefetch
                                                 : ops::SparseMoeHints{};
}

void TextContext::mlp_tail(const BlockParameters& weights, Tensor& x, Phase,
                           const ops::SparseMoeHints& hints) {
    Tensor h = workspace::post_mixer_hidden(work_, config_, x.ne[1]);
    ops::rmsnorm(x, weights.post_attention_norm, config_.rms_norm_eps, true, h, ctx_.stream);
    ffn(h, weights.ffn, x, hints, work_, ctx_.stream);
}

void TextContext::mixer_layer(const BlockParameters& block, Tensor& x, std::size_t layer,
                              Phase ph, Tensor* delta, Tensor* activation) {
    const bool prefill = ph == Phase::Prefill;
    const bool full    = config_.layer_types[layer] == MixerKind::FullAttention;
    const auto compact = dimension(config_.compact_layer_indices[layer]);
    nvtx::ScopedRange layer_range(
        full ? (prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull)
             : (prefill ? nvtx::Name::PrefillLayerGdn : nvtx::Name::VerifyLayerGdn),
        full ? nvtx::Category::Attention : nvtx::Category::Gdn, layer);
    try {
        nvtx::ScopedRange mixer_range(
            full ? (prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention)
                 : (prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn),
            full ? nvtx::Category::Attention : nvtx::Category::Gdn, layer);
        auto scope = work_.scope();
        if (full) {
            attn_mix(block, x, compact, ph, delta, activation);
        } else {
            gdn_mix(block, x, compact, ph, delta, activation);
        }
    } catch (const std::exception& error) {
        throw std::runtime_error("text/layers/" + std::to_string(layer) +
                                 (prefill ? " prefill" : " verify") +
                                 " columns=" + std::to_string(x.ne[1]) + ": " + error.what());
    }
}

void TextContext::mlp_layer(const BlockParameters& block, Tensor& x, std::size_t layer,
                            Phase ph) {
    const bool prefill = ph == Phase::Prefill;
    try {
        nvtx::ScopedRange range(prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                                nvtx::Category::PostMixer, layer);
        auto scope = work_.scope();
        mlp_tail(block, x, ph, next_projection_hints(static_cast<int>(layer)));
    } catch (const std::exception& error) {
        throw std::runtime_error("text/layers/" + std::to_string(layer) +
                                 (prefill ? " prefill" : " verify") +
                                 " columns=" + std::to_string(x.ne[1]) + ": " + error.what());
    }
}

void TextContext::single_layer(const BlockParameters& block, Tensor& x, std::size_t layer,
                               Phase ph) {
    mixer_layer(block, x, layer, ph);
    mlp_layer(block, x, layer, ph);
}

namespace {

// The first stored weight of a projection: where that projection's stream begins. This is the same
// rule the MoE projection hint uses (the prefetch helper in parameters.cpp), kept in one place here
// for the tensor-parallel window.
const Weight* projection_stream_head(const ops::ProjectionWeights& projection) {
    return std::visit(
        [](const auto& part) -> const Weight* {
            using Kind = std::decay_t<decltype(part)>;
            if constexpr (std::is_same_v<Kind, ops::SingleProjectionWeight>) {
                return &part.weight;
            } else if constexpr (std::is_same_v<Kind, ops::PairedProjectionWeights>) {
                return &part.first;
            } else {
                return part.parts.empty() ? nullptr : &part.parts.front().weight;
            }
        },
        projection);
}

} // namespace

const Weight* TextContext::tp_mixer_stream_weight(std::size_t layer) const {
    if (layer >= parameters_.text.layers.size()) { return nullptr; }
    const auto& mixer = parameters_.text.layers[layer].mixer;
    if (const auto* attention = std::get_if<AttentionParameters>(&mixer)) {
        return projection_stream_head(attention->projection);
    }
    if (const auto* gdn = std::get_if<GdnParameters>(&mixer)) {
        return projection_stream_head(gdn->projection);
    }
    return nullptr;
}

const Weight* TextContext::tp_ffn_stream_weight(std::size_t layer) const {
    if (layer >= parameters_.text.layers.size()) { return nullptr; }
    const auto* dense = std::get_if<DenseParameters>(&parameters_.text.layers[layer].ffn);
    return dense == nullptr ? nullptr : &dense->gate_up.weight;
}

tp::DevicePair::ArPrefetchWindow TextContext::tp_prefetch_window(const Weight* mine,
                                                                const Weight* peer) const {
    tp::DevicePair::ArPrefetchWindow window;
    if (decode_prefetch_bytes_ == 0) { return window; }
    const auto span = [this](const Weight* weight) {
        if (weight == nullptr || weight->payload == nullptr) { return std::pair<const void*, std::size_t>{}; }
        return std::pair<const void*, std::size_t>{
            weight->payload, static_cast<std::size_t>(
                                std::min<std::uint64_t>(weight->payload_bytes, decode_prefetch_bytes_))};
    };
    const auto [base_a, bytes_a] = span(mine);
    const auto [base_b, bytes_b] = span(peer);
    window.base_a  = base_a;
    window.bytes_a = bytes_a;
    window.base_b  = base_b;
    window.bytes_b = bytes_b;
    return window;
}

std::int32_t TextContext::tp_mixer_activation_width(const BlockParameters& block) const {
    if (const auto* gdn = std::get_if<GdnParameters>(&block.mixer)) {
        return gdn->output.weight.input_columns != nullptr
                   ? dimension(config_.gdn->value_width())
                   : 0;
    }
    return std::get<AttentionParameters>(block.mixer).output.weight.input_columns != nullptr
               ? dimension(config_.attention->query_width())
               : 0;
}

void TextContext::tp_mixer_layer(const BlockParameters& block, Tensor& x, Tensor& delta,
                                 std::size_t layer, Phase ph, Tensor* activation) {
    mixer_layer(block, x, layer, ph, &delta, activation);
}

void TextContext::tp_mixer_activation_delta(const BlockParameters& block, const Tensor& activation,
                                            Tensor& delta, std::size_t layer, Phase ph) {
    const bool prefill = ph == Phase::Prefill;
    if (!std::holds_alternative<GdnParameters>(block.mixer)) {
        throw std::logic_error("the tensor-parallel activation merge is a GDN mixer stage");
    }
    try {
        nvtx::ScopedRange mixer_range(
            prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn, nvtx::Category::Gdn, layer);
        auto scope   = work_.scope();
        const auto& p = std::get<GdnParameters>(block.mixer);
        const std::int32_t width = dimension(config_.gdn->value_width());
        const std::int32_t columns = activation.ne[1];
        ops::linear(activation.view({width, columns}), p.output.weight, delta, p.output.policy,
                    work_, ctx_.stream);
    } catch (const std::exception& error) {
        throw std::runtime_error("text/layers/" + std::to_string(layer) +
                                 (prefill ? " prefill" : " verify") +
                                 " columns=" + std::to_string(activation.ne[1]) + ": " +
                                 error.what());
    }
}

void TextContext::tp_mlp_layer(const BlockParameters& block, Tensor& x, std::size_t layer,
                               Phase ph) {
    mlp_layer(block, x, layer, ph);
}

void TextContext::tp_mlp_delta(const BlockParameters& block, Tensor& x, Tensor& delta,
                               std::size_t layer, Phase ph) {
    const bool prefill = ph == Phase::Prefill;
    try {
        nvtx::ScopedRange range(prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                                nvtx::Category::PostMixer, layer);
        auto scope = work_.scope();
        Tensor h = workspace::post_mixer_hidden(work_, config_, x.ne[1]);
        ops::rmsnorm(x, block.post_attention_norm, config_.rms_norm_eps, true, h, ctx_.stream);
        ffn_delta(h, block.ffn, delta, next_projection_hints(static_cast<int>(layer)), work_,
                  ctx_.stream);
    } catch (const std::exception& error) {
        throw std::runtime_error("text/layers/" + std::to_string(layer) +
                                 (prefill ? " prefill" : " verify") +
                                 " columns=" + std::to_string(x.ne[1]) + ": " + error.what());
    }
}

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    for (std::size_t layer = 0; layer < parameters_.text.layers.size(); ++layer) {
        const auto& block = parameters_.text.layers[layer];
        single_layer(block, x, layer, ph);
        if constexpr (Tap::enabled) {
            tap.capture_layer(static_cast<int>(layer), x, ctx_.stream);
        }
    }
}

void TextContext::run_layers(Tensor& x, Phase ph) {
    NullTap tap;
    run_layers(x, ph, tap);
}

void TextContext::project_head_tp2(TextContext& peer, tp::DevicePair& pair, const Tensor& hidden,
                                   const Tensor& hidden_peer, Tensor& logits, Tensor& logits_peer) {
    const std::int32_t vocab = dimension(config_.vocab_size);
    const std::int32_t local = dimension(lm_head_->weight.n);
    if (local == vocab) {
        // Replicated head: both shards hold every row and their inputs agree exactly (replicated
        // mixers plus the all-reduced FFN delta), so each computes the full-vocabulary logits on
        // its own and no collective is needed.
        ctx_.bind_to_current_thread();
        project(hidden, *lm_head_, logits, work_, ctx_.stream);
        peer.ctx_.bind_to_current_thread();
        project(hidden_peer, *peer.lm_head_, logits_peer, peer.work_, peer.ctx_.stream);
        return;
    }
    if (local <= 0 || local * 2 != vocab || shard_index_ < 0 || peer.shard_index_ < 0 ||
        peer.shard_index_ == shard_index_) {
        throw std::logic_error(
            "project_head_tp2: a vocabulary-split output head needs exactly half the vocabulary and "
            "two distinct shards");
    }
    // Each shard projects only its own vocabulary rows into a compact block; the merge stamps them
    // into the full-logits buffers at their vocabulary offsets and sums the pair, which yields the
    // complete [V, T] logits on both shards.
    const std::int32_t columns = logits.ne[1];
    ctx_.bind_to_current_thread();
    auto scope      = work_.scope();
    Tensor partial  = work_.alloc(DType::BF16, {local, columns});
    project(hidden, *lm_head_, partial, work_, ctx_.stream);
    peer.ctx_.bind_to_current_thread();
    auto peer_scope     = peer.work_.scope();
    Tensor partial_peer = peer.work_.alloc(DType::BF16, {local, columns});
    project(hidden_peer, *peer.lm_head_, partial_peer, peer.work_, peer.ctx_.stream);
    merge_local_row_blocks(peer, pair, partial, partial_peer, logits, logits_peer, local);
}

void TextContext::forward_tp2(TextContext& peer, tp::DevicePair& pair, std::int32_t token,
                              std::int32_t position, Tensor& logits, Tensor& logits_peer,
                              Tensor* mtp_input_hidden) {
    const std::int32_t hidden = dimension(config_.hidden_size);
    const std::int32_t vocab  = dimension(config_.vocab_size);
    if (vocab % 2 != 0 || logits.ne[0] != vocab || logits.ne[1] != 1 ||
        logits_peer.ne[0] != vocab || logits_peer.ne[1] != 1) {
        throw std::invalid_argument("forward_tp2: logits buffers must be full-vocabulary [V,1]");
    }
    // No work_.reset() here: the caller owns the logits buffers in this arena, and resetting
    // would let the internal allocations overwrite them.
    // Bind the replicated-mixer state on both shards. A single token at position zero attends
    // only to itself, so the paged cache is empty and the GDN state slot is zeroed. The RAII
    // guards live until the end of the forward (a bind lambda would destroy them immediately),
    // and each shard's device is bound before any of its allocations or kernels.
    struct BindState {
        Tensor cache_positions;
        Tensor rope_positions;
        Tensor kv_table_rows;
        Tensor state_source;
        Tensor state_destination;
        ops::CausalAttentionExecutionEnvelope envelope{1, 1};
    };
    auto make_bind = [&](TextContext& c, WorkspaceArena& arena) {
        c.ctx_.bind_to_current_thread();
        BindState b;
        const auto envelope_tokens = static_cast<std::uint32_t>(position + 1);
        b.envelope                 = {envelope_tokens, envelope_tokens};
        b.cache_positions = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(b.cache_positions, position, c.ctx_.stream);
        b.rope_positions = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(b.rope_positions, position, c.ctx_.stream);
        b.kv_table_rows     = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(b.kv_table_rows, 0, c.ctx_.stream);
        b.state_source      = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(b.state_source, 0, c.ctx_.stream);
        b.state_destination = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(b.state_destination, 0, c.ctx_.stream);
        return b;
    };
    const BindState bind0 = make_bind(*this, work_);
    const BindState bind1 = make_bind(peer, peer.work_);
    ScopedPositions cache0(active_cache_positions_, bind0.cache_positions);
    ScopedPositions rope0(active_rope_positions_, bind0.rope_positions);
    ScopedEnvelope envelope0(active_causal_attention_envelope_, bind0.envelope);
    ScopedValue<const Tensor*> kv0(active_kv_table_rows_, &bind0.kv_table_rows);
    ScopedValue<const Tensor*> source0(active_linear_state_source_slots_, &bind0.state_source);
    ScopedValue<const Tensor*> destination0(active_linear_state_destination_slots_,
                                            &bind0.state_destination);
    ScopedValue<std::int32_t> batch0(active_sequence_batch_, 1);
    ScopedValue<std::int32_t> width0(active_sequence_width_, 1);
    ScopedPositions cache1(peer.active_cache_positions_, bind1.cache_positions);
    ScopedPositions rope1(peer.active_rope_positions_, bind1.rope_positions);
    ScopedEnvelope envelope1(peer.active_causal_attention_envelope_, bind1.envelope);
    ScopedValue<const Tensor*> kv1(peer.active_kv_table_rows_, &bind1.kv_table_rows);
    ScopedValue<const Tensor*> source1(peer.active_linear_state_source_slots_,
                                       &bind1.state_source);
    ScopedValue<const Tensor*> destination1(peer.active_linear_state_destination_slots_,
                                            &bind1.state_destination);
    ScopedValue<std::int32_t> batch1(peer.active_sequence_batch_, 1);
    ScopedValue<std::int32_t> width1(peer.active_sequence_width_, 1);

    ctx_.bind_to_current_thread();
    Tensor ids = work_.alloc(DType::I32, {1});
    ops::set_i32_scalar(ids, token, ctx_.stream);
    peer.ctx_.bind_to_current_thread();
    Tensor ids_peer = peer.work_.alloc(DType::I32, {1});
    ops::set_i32_scalar(ids_peer, token, peer.ctx_.stream);

    ctx_.bind_to_current_thread();
    Tensor x      = work_.alloc(DType::BF16, {hidden, 1});
    Tensor x_peer = peer.work_.alloc(DType::BF16, {hidden, 1});
    embedding_tp2(peer, pair, ids, &ids_peer, x, &x_peer);
    NullTap tap;
    run_layers_tp2(peer, pair, x, x_peer, Phase::Verify, tap);

    ctx_.bind_to_current_thread();
    Tensor hidden_out      = work_.alloc(DType::BF16, {hidden, 1});
    peer.ctx_.bind_to_current_thread();
    Tensor hidden_out_peer = peer.work_.alloc(DType::BF16, {hidden, 1});
    ctx_.bind_to_current_thread();
    ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, hidden_out, ctx_.stream);
    if (mtp_input_hidden != nullptr) {
        // The MTP layer consumes the final-norm hidden; hand the caller its own stable copy so this
        // context's workspace scope can end.
        if (mtp_input_hidden->dtype != DType::BF16 || mtp_input_hidden->ne[0] != hidden ||
            mtp_input_hidden->ne[1] != 1) {
            throw std::invalid_argument("forward_tp2: MTP hidden buffer must be [hidden,1] BF16");
        }
        CUDA_CHECK(cudaMemcpyAsync(mtp_input_hidden->data, hidden_out.data, hidden_out.bytes(),
                                   cudaMemcpyDeviceToDevice, ctx_.stream));
    }
    peer.ctx_.bind_to_current_thread();
    ops::rmsnorm(x_peer, *peer.final_norm_, config_.rms_norm_eps, true, hidden_out_peer,
                 peer.ctx_.stream);


    // The head is replicated or vocabulary-split; project_head_tp2 covers both, leaving the full
    // [V, 1] logits on either shard.
    project_head_tp2(peer, pair, hidden_out, hidden_out_peer, logits, logits_peer);
}

void TextContext::forward_tp2_decode_window(TextContext& peer, tp::DevicePair& pair,
                                            const std::int32_t* token, const std::int32_t* position,
                                            ops::CausalAttentionExecutionEnvelope envelope,
                                            Tensor& logits, std::int32_t lane) {
    const std::int32_t hidden = dimension(config_.hidden_size);
    const std::int32_t vocab  = dimension(config_.vocab_size);
    if (token == nullptr || position == nullptr) {
        throw std::invalid_argument(
            "forward_tp2_decode_window requires pinned host token and position");
    }
    if (vocab % 2 != 0 || logits.ne[0] != vocab || logits.ne[1] != 1) {
        throw std::invalid_argument("forward_tp2_decode_window: logits must be [V,1]");
    }
    if (envelope.min_visible_keys == 0 || envelope.max_visible_keys < envelope.min_visible_keys) {
        throw std::invalid_argument("forward_tp2_decode_window: envelope is invalid");
    }
    // Same sequence as forward_tp2, with only its two per-round host values made capturable: the
    // decoded token and its absolute position now arrive through a memcpy node out of pinned host
    // memory, and the attention envelope is a capture parameter instead of the exact extent. The
    // envelope is a bound rather than a decision - the small-T route reads the real window from the
    // device-side positions and derives the active split count and the key partition from it, so a
    // bucket-wide envelope reduces to the same kernels and the same key ranges.
    struct BindState {
        Tensor ids;
        Tensor cache_positions;
        Tensor rope_positions;
        Tensor kv_table_rows;
        Tensor state_source;
        Tensor state_destination;
    };
    auto make_bind = [&](TextContext& card, WorkspaceArena& arena) {
        card.ctx_.bind_to_current_thread();
        BindState bind;
        bind.ids = arena.alloc(DType::I32, {1});
        copy_i32(token, bind.ids, card.ctx_.stream);
        bind.cache_positions = arena.alloc(DType::I32, {1});
        copy_i32(position, bind.cache_positions, card.ctx_.stream);
        bind.rope_positions = arena.alloc(DType::I32, {1});
        copy_i32(position, bind.rope_positions, card.ctx_.stream);
        bind.kv_table_rows = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(bind.kv_table_rows, lane, card.ctx_.stream);
        bind.state_source = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(bind.state_source, lane, card.ctx_.stream);
        bind.state_destination = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(bind.state_destination, lane, card.ctx_.stream);
        return bind;
    };
    const BindState bind0 = make_bind(*this, work_);
    const BindState bind1 = make_bind(peer, peer.work_);
    ScopedPositions cache0(active_cache_positions_, bind0.cache_positions);
    ScopedPositions rope0(active_rope_positions_, bind0.rope_positions);
    ScopedEnvelope envelope0(active_causal_attention_envelope_, envelope);
    ScopedValue<const Tensor*> kv0(active_kv_table_rows_, &bind0.kv_table_rows);
    ScopedValue<const Tensor*> source0(active_linear_state_source_slots_, &bind0.state_source);
    ScopedValue<const Tensor*> destination0(active_linear_state_destination_slots_,
                                            &bind0.state_destination);
    ScopedValue<std::int32_t> batch0(active_sequence_batch_, 1);
    ScopedValue<std::int32_t> width0(active_sequence_width_, 1);
    ScopedPositions cache1(peer.active_cache_positions_, bind1.cache_positions);
    ScopedPositions rope1(peer.active_rope_positions_, bind1.rope_positions);
    ScopedEnvelope envelope1(peer.active_causal_attention_envelope_, envelope);
    ScopedValue<const Tensor*> kv1(peer.active_kv_table_rows_, &bind1.kv_table_rows);
    ScopedValue<const Tensor*> source1(peer.active_linear_state_source_slots_, &bind1.state_source);
    ScopedValue<const Tensor*> destination1(peer.active_linear_state_destination_slots_,
                                            &bind1.state_destination);
    ScopedValue<std::int32_t> batch1(peer.active_sequence_batch_, 1);
    ScopedValue<std::int32_t> width1(peer.active_sequence_width_, 1);

    ctx_.bind_to_current_thread();
    Tensor x      = work_.alloc(DType::BF16, {hidden, 1});
    Tensor x_peer = peer.work_.alloc(DType::BF16, {hidden, 1});
    embedding_tp2(peer, pair, bind0.ids, &bind1.ids, x, &x_peer);
    NullTap tap;
    run_layers_tp2(peer, pair, x, x_peer, Phase::Verify, tap);

    ctx_.bind_to_current_thread();
    Tensor hidden_out      = work_.alloc(DType::BF16, {hidden, 1});
    peer.ctx_.bind_to_current_thread();
    Tensor hidden_out_peer = peer.work_.alloc(DType::BF16, {hidden, 1});
    ctx_.bind_to_current_thread();
    ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, hidden_out, ctx_.stream);
    peer.ctx_.bind_to_current_thread();
    ops::rmsnorm(x_peer, *peer.final_norm_, config_.rms_norm_eps, true, hidden_out_peer,
                 peer.ctx_.stream);
    Tensor logits_peer = peer.work_.alloc(DType::BF16, {vocab, 1});
    project_head_tp2(peer, pair, hidden_out, hidden_out_peer, logits, logits_peer);
}

void TextContext::forward_tp2_decode_window_batch(TextContext& peer, tp::DevicePair& pair,
                                                  const std::int32_t* tokens,
                                                  const std::int32_t* positions,
                                                  const std::int32_t* kv_table_rows,
                                                  const std::int32_t* state_slots,
                                                  std::int32_t batch,
                                                  ops::CausalAttentionExecutionEnvelope envelope,
                                                  Tensor& logits) {
    const std::int32_t hidden = dimension(config_.hidden_size);
    const std::int32_t vocab  = dimension(config_.vocab_size);
    if (tokens == nullptr || positions == nullptr || kv_table_rows == nullptr ||
        state_slots == nullptr) {
        throw std::invalid_argument(
            "forward_tp2_decode_window_batch requires pinned host lane arrays");
    }
    if (batch < 1 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("forward_tp2_decode_window_batch: batch is out of range");
    }
    if (vocab % 2 != 0 || logits.ne[0] != vocab || logits.ne[1] != batch) {
        throw std::invalid_argument("forward_tp2_decode_window_batch: logits must be [V,B]");
    }
    if (envelope.min_visible_keys == 0 || envelope.max_visible_keys < envelope.min_visible_keys) {
        throw std::invalid_argument("forward_tp2_decode_window_batch: envelope is invalid");
    }
    // The batched GDN state kernels start at two lanes, so a one-column batch still runs the scalar
    // path, and that path reads the slot pair published by set_linear_state_slots instead of the
    // bound per-lane vectors. A batch that shrank to one lane therefore has to publish its own slot:
    // without this it keeps the last prefill's slot and advances another lane's recurrence.
    if (batch == 1) {
        const std::int32_t slot = state_slots[0];
        set_linear_state_slots(slot, slot);
        peer.set_linear_state_slots(slot, slot);
    }
    // Same lockstep sequence as forward_tp2_decode_window, widened from one column to batch: the
    // layer loop derives every width from x.ne[1], the mixers read their batch through the scoped
    // bindings below, and the head is column-generic. The only per-lane operands are the four
    // vectors, which arrive as pinned host arrays and become one small device copy per shard.
    struct BindState {
        Tensor ids;
        Tensor cache_positions;
        Tensor rope_positions;
        Tensor kv_table_rows;
        Tensor state_slots;
    };
    auto make_bind = [&](TextContext& card, WorkspaceArena& arena) {
        card.ctx_.bind_to_current_thread();
        BindState bind;
        bind.ids = arena.alloc(DType::I32, {batch});
        copy_i32(tokens, bind.ids, card.ctx_.stream);
        bind.cache_positions = arena.alloc(DType::I32, {batch});
        copy_i32(positions, bind.cache_positions, card.ctx_.stream);
        bind.rope_positions = arena.alloc(DType::I32, {batch});
        copy_i32(positions, bind.rope_positions, card.ctx_.stream);
        bind.kv_table_rows = arena.alloc(DType::I32, {batch});
        copy_i32(kv_table_rows, bind.kv_table_rows, card.ctx_.stream);
        bind.state_slots = arena.alloc(DType::I32, {batch});
        copy_i32(state_slots, bind.state_slots, card.ctx_.stream);
        return bind;
    };
    const BindState bind0 = make_bind(*this, work_);
    const BindState bind1 = make_bind(peer, peer.work_);
    ScopedPositions cache0(active_cache_positions_, bind0.cache_positions);
    ScopedPositions rope0(active_rope_positions_, bind0.rope_positions);
    ScopedEnvelope envelope0(active_causal_attention_envelope_, envelope);
    ScopedValue<const Tensor*> kv0(active_kv_table_rows_, &bind0.kv_table_rows);
    ScopedValue<const Tensor*> source0(active_linear_state_source_slots_, &bind0.state_slots);
    ScopedValue<const Tensor*> destination0(active_linear_state_destination_slots_,
                                            &bind0.state_slots);
    ScopedValue<std::int32_t> batch0(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width0(active_sequence_width_, 1);
    ScopedPositions cache1(peer.active_cache_positions_, bind1.cache_positions);
    ScopedPositions rope1(peer.active_rope_positions_, bind1.rope_positions);
    ScopedEnvelope envelope1(peer.active_causal_attention_envelope_, envelope);
    ScopedValue<const Tensor*> kv1(peer.active_kv_table_rows_, &bind1.kv_table_rows);
    ScopedValue<const Tensor*> source1(peer.active_linear_state_source_slots_, &bind1.state_slots);
    ScopedValue<const Tensor*> destination1(peer.active_linear_state_destination_slots_,
                                            &bind1.state_slots);
    ScopedValue<std::int32_t> batch1(peer.active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width1(peer.active_sequence_width_, 1);

    ctx_.bind_to_current_thread();
    Tensor x      = work_.alloc(DType::BF16, {hidden, batch});
    Tensor x_peer = peer.work_.alloc(DType::BF16, {hidden, batch});
    embedding_tp2(peer, pair, bind0.ids, &bind1.ids, x, &x_peer);
    NullTap tap;
    run_layers_tp2(peer, pair, x, x_peer, Phase::Verify, tap);

    ctx_.bind_to_current_thread();
    Tensor hidden_out      = work_.alloc(DType::BF16, {hidden, batch});
    peer.ctx_.bind_to_current_thread();
    Tensor hidden_out_peer = peer.work_.alloc(DType::BF16, {hidden, batch});
    ctx_.bind_to_current_thread();
    ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, hidden_out, ctx_.stream);
    peer.ctx_.bind_to_current_thread();
    ops::rmsnorm(x_peer, *peer.final_norm_, config_.rms_norm_eps, true, hidden_out_peer,
                 peer.ctx_.stream);
    Tensor logits_peer = peer.work_.alloc(DType::BF16, {vocab, batch});
    project_head_tp2(peer, pair, hidden_out, hidden_out_peer, logits, logits_peer);
}

std::int32_t TextContext::forward_tp2_token(TextContext& peer, tp::DevicePair& pair,
                                            std::int32_t token, std::int32_t position) {
    const std::int32_t vocab = dimension(config_.vocab_size);
    auto scope = work_.scope();
    Tensor logits      = work_.alloc(DType::BF16, {vocab, 1});
    Tensor logits_peer = peer.work_.alloc(DType::BF16, {vocab, 1});
    forward_tp2(peer, pair, token, position, logits, logits_peer);
    // forward_tp2 leaves the current device on the peer card; bind each shard before its
    // argmax so the device pointer resolves in the correct address space. The lm_head is
    // replicated, so the two shards' logits are bit-identical and their argmax must agree;
    // comparing both makes a shard divergence a hard failure at the sampling boundary.
    ctx_.bind_to_current_thread();
    Tensor sampled = work_.alloc(DType::I32, {1});
    ops::argmax(logits, sampled, vocab, ctx_.stream);
    peer.ctx_.bind_to_current_thread();
    Tensor sampled_peer = peer.work_.alloc(DType::I32, {1});
    ops::argmax(logits_peer, sampled_peer, vocab, peer.ctx_.stream);
    std::int32_t token_id      = 0;
    std::int32_t token_id_peer = 0;
    ctx_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(&token_id, sampled.data, sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, ctx_.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx_.stream));
    peer.ctx_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(&token_id_peer, sampled_peer.data, sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, peer.ctx_.stream));
    CUDA_CHECK(cudaStreamSynchronize(peer.ctx_.stream));
    if (token_id != token_id_peer) {
        throw std::logic_error("forward_tp2_token: shard argmax mismatch at position " +
                               std::to_string(position) + " (" + std::to_string(token_id) +
                               " vs " + std::to_string(token_id_peer) + ")");
    }
    return token_id;
}

void TextContext::forward_tp2_prefill(TextContext& peer, tp::DevicePair& pair,
                                      std::span<const int> ids, std::int32_t first_position,
                                      Tensor* logits, Tensor* logits_peer,
                                      Tensor* mtp_input_hidden, Tensor* logits_columns,
                                      Tensor* hidden_columns, Phase phase,
                                      const Tp2VisionChunk* vision, DFlashFeatureSink* sink,
                                      std::int32_t lane) {
    const std::int32_t hidden = dimension(config_.hidden_size);
    const std::int32_t vocab  = dimension(config_.vocab_size);
    const std::int32_t tokens = static_cast<std::int32_t>(ids.size());
    if (tokens <= 0) { throw std::invalid_argument("forward_tp2_prefill requires tokens"); }
    if (first_position < 0) {
        throw std::invalid_argument("forward_tp2_prefill: position must be non-negative");
    }
    if (logits_columns == nullptr) {
        if (logits == nullptr || logits_peer == nullptr) {
            throw std::invalid_argument(
                "forward_tp2_prefill: a prefill chunk must request the last column logits");
        }
        if (logits->ne[0] != vocab || logits->ne[1] != 1 || logits_peer->ne[0] != vocab ||
            logits_peer->ne[1] != 1) {
            throw std::invalid_argument(
                "forward_tp2_prefill: logits buffers must be full-vocabulary [V,1]");
        }
    } else if (logits != nullptr || logits_peer != nullptr) {
        // The per-column buffer carries the last column as well, so a second lm_head read
        // would score exactly the same bytes again.
        throw std::invalid_argument(
            "forward_tp2_prefill: per-column logits replace the [V,1] last-column buffers");
    }
    // The execution envelope is the chunk's inclusive key extent: the chunk appends positions
    // [first_position, first_position + tokens) and its last query reads every one of them. This
    // is exactly the binding the validated single-device prefill chunk uses.
    const std::uint32_t visible_end =
        static_cast<std::uint32_t>(first_position) + static_cast<std::uint32_t>(tokens);
    if (vision != nullptr &&
        (vision->positions == nullptr ||
         vision->prompt_tokens < static_cast<std::size_t>(first_position) +
                                   static_cast<std::size_t>(tokens))) {
        throw std::invalid_argument(
            "forward_tp2_prefill: multimodal chunk has no prompt position table");
    }
    struct BindState {
        Tensor ids;
        Tensor positions;
        Tensor rope_positions;
        Tensor kv_table_rows;
        Tensor state_source;
        Tensor state_destination;
        ops::CausalAttentionExecutionEnvelope envelope{1, 1};
    };
    // Bind the replicated mixer state on both shards. The RAII guards live until the end of the
    // forward (a bind lambda would destroy them immediately), and each shard's device is bound
    // before any of its allocations or kernels. No work_.reset() here: the caller owns the logits
    // buffers in this arena, and resetting would let the internal allocations overwrite them.
    auto make_bind = [&](TextContext& c, WorkspaceArena& arena) {
        c.ctx_.bind_to_current_thread();
        BindState b;
        b.envelope    = {visible_end, visible_end};
        b.ids         = arena.alloc(DType::I32, {tokens});
        b.positions   = arena.alloc(DType::I32, {tokens});
        copy_i32(ids.data(), b.ids, c.ctx_.stream);
        ops::fill_i32_positions(b.positions, first_position, 1, c.ctx_.stream);
        if (vision != nullptr) {
            // A multimodal prompt carries its own 3-axis (temporal, height, width) RoPE table for
            // every token; the cache position stays the plain absolute index, so the two bindings
            // stop coinciding for this request.
            // [tokens, 3]: the token index is contiguous, matching the single-device route's
            // [tokens, axes] binding (neo[1] == 3 is what makes the RoPE op read the MRoPE table).
            b.rope_positions = arena.alloc(DType::I32, {tokens, 3});
            std::vector<std::int32_t> host(static_cast<std::size_t>(3) * tokens);
            for (int axis = 0; axis < 3; ++axis) {
                std::copy_n(vision->positions +
                                static_cast<std::size_t>(axis) * vision->prompt_tokens +
                                static_cast<std::size_t>(first_position),
                            tokens, host.data() + static_cast<std::size_t>(axis) * tokens);
            }
            copy_i32(host.data(), b.rope_positions, c.ctx_.stream);
        } else {
            b.rope_positions = b.positions;
        }
        b.kv_table_rows = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(b.kv_table_rows, lane, c.ctx_.stream);
        b.state_source = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(b.state_source, lane, c.ctx_.stream);
        b.state_destination = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(b.state_destination, lane, c.ctx_.stream);
        return b;
    };
    const BindState bind0 = make_bind(*this, work_);
    const BindState bind1 = make_bind(peer, peer.work_);
    // Cache and RoPE positions coincide for a text prompt: the TP-2 path runs without a RoPE delta,
    // so the cache position of every chunk token is its own absolute position. A multimodal prompt
    // binds the prompt's 3-axis table instead (see make_bind).
    ScopedPositions cache0(active_cache_positions_, bind0.positions);
    ScopedPositions rope0(active_rope_positions_, bind0.rope_positions);
    ScopedEnvelope envelope0(active_causal_attention_envelope_, bind0.envelope);
    ScopedValue<const Tensor*> kv0(active_kv_table_rows_, &bind0.kv_table_rows);
    ScopedValue<const Tensor*> source0(active_linear_state_source_slots_, &bind0.state_source);
    ScopedValue<const Tensor*> destination0(active_linear_state_destination_slots_,
                                            &bind0.state_destination);
    ScopedPositions cache1(peer.active_cache_positions_, bind1.positions);
    ScopedPositions rope1(peer.active_rope_positions_, bind1.rope_positions);
    ScopedEnvelope envelope1(peer.active_causal_attention_envelope_, bind1.envelope);
    ScopedValue<const Tensor*> kv1(peer.active_kv_table_rows_, &bind1.kv_table_rows);
    ScopedValue<const Tensor*> source1(peer.active_linear_state_source_slots_,
                                      &bind1.state_source);
    ScopedValue<const Tensor*> destination1(peer.active_linear_state_destination_slots_,
                                            &bind1.state_destination);

    ctx_.bind_to_current_thread();
    Tensor x      = work_.alloc(DType::BF16, {hidden, tokens});
    Tensor x_peer = peer.work_.alloc(DType::BF16, {hidden, tokens});
    embedding_tp2(peer, pair, bind0.ids, &bind1.ids, x, &x_peer);
    if (vision != nullptr && vision->has_item()) {
        const auto& control = *vision->control;
        const auto scatter  = std::span<const std::int32_t>(control.scatter_indices);
        const auto begin    = std::lower_bound(scatter.begin(), scatter.end(), first_position);
        const auto stop     = std::lower_bound(begin, scatter.end(), first_position + tokens);
        const auto count    = static_cast<std::int32_t>(stop - begin);
        if (count != 0) {
            const std::int32_t merged = static_cast<std::int32_t>(control.merged_count);
            if (vision->embeddings == nullptr || vision->embeddings->dtype != DType::BF16 ||
                vision->embeddings->ne[0] != hidden || vision->embeddings->ne[1] != merged) {
                throw std::invalid_argument(
                    "forward_tp2_prefill: vision embeddings do not match the encoded item");
            }
            const std::int32_t visual_begin = static_cast<std::int32_t>(begin - scatter.begin());
            if (visual_begin + count > merged) {
                throw std::invalid_argument(
                    "forward_tp2_prefill: vision scatter runs past the encoded item");
            }
            std::vector<std::int32_t> local(static_cast<std::size_t>(count));
            for (std::int32_t i = 0; i < count; ++i) {
                local[static_cast<std::size_t>(i)] = begin[i] - first_position;
            }
            const std::size_t row_bytes = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
            const std::size_t bytes     = row_bytes * static_cast<std::size_t>(count);
            // The Vision shard owns the encoded item, so only its side reads the handoff tensor: the
            // pair copies this chunk's contiguous column range into both arenas with one in-place
            // all-reduce over a zeroed text-shard buffer. That is exact (a sum with zero), keeps the
            // resident cost bounded by the chunk rather than by the item, and needs no second tower.
            //
            // The handoff is dead once the scatter below has consumed it, so it is released here
            // rather than held to the end of the forward. Each shard's buffer is touched by that
            // shard's stream alone - ar_exchange reduces through the pair's mapped host staging,
            // never through the peer's arena - and the scatter that reads it is ordered after the
            // collective on the same stream, so rewinding both arenas before the layer loop cannot
            // race the layers that reuse the bytes. Holding it instead adds a full chunk of visual
            // rows on top of the layer peak, which is what overruns the fixed workspace when a
            // chunk is built entirely of vision tokens: 182.3 MiB of layers (the measured all-text
            // peak at a 1024-column chunk) plus this chunk's 10.0 MiB of staging against 192 MiB.
            auto handoff_scope_a = work_.scope();
            auto handoff_scope_b = peer.work_.scope();
            ctx_.bind_to_current_thread();
            Tensor staging = work_.alloc(DType::BF16, {hidden, count});
            CUDA_CHECK(cudaMemsetAsync(staging.data, 0, bytes, ctx_.stream));
            peer.ctx_.bind_to_current_thread();
            Tensor source = peer.work_.alloc(DType::BF16, {hidden, count});
            // The tensor payload is untyped, so the byte offset needs an explicit byte pointer
            // (void arithmetic is a GCC extension).
            const auto* embeddings = static_cast<const std::byte*>(vision->embeddings->data);
            CUDA_CHECK(cudaMemcpyAsync(
                source.data, embeddings + static_cast<std::size_t>(visual_begin) * row_bytes, bytes,
                cudaMemcpyDeviceToDevice, peer.ctx_.stream));
            pair.allreduce(staging.data, source.data, bytes, ctx_.stream, peer.ctx_.stream);
            ctx_.bind_to_current_thread();
            Tensor indices = work_.alloc(DType::I32, {count});
            copy_i32(local.data(), indices, ctx_.stream);
            ops::scatter(staging, indices, x, ctx_.stream);
            peer.ctx_.bind_to_current_thread();
            Tensor indices_peer = peer.work_.alloc(DType::I32, {count});
            copy_i32(local.data(), indices_peer, peer.ctx_.stream);
            ops::scatter(source, indices_peer, x_peer, peer.ctx_.stream);
            ctx_.bind_to_current_thread();
        }
    }
    NullTap tap;
    const std::int32_t overlap_width = static_cast<std::int32_t>(prefill_overlap_);
    bool overlap_ok = overlap_width > 0 && phase == Phase::Prefill && sink == nullptr &&
                      vision == nullptr && pair.in_kernel_allreduce() &&
                      tokens % overlap_width == 0 && tokens / overlap_width >= 2;
    if (overlap_ok) {
        for (const auto& layer : parameters_.text.layers) {
            if (tp_mixer_activation_width(layer) != 0) {
                overlap_ok = false;
                break;
            }
        }
    }
    auto run_layers = [&] {
        if (overlap_ok) {
            run_layers_tp2_overlap(peer, pair, x, x_peer, phase, overlap_width, first_position);
        } else if (sink != nullptr) {
            run_layers_tp2(peer, pair, x, x_peer, phase, *sink);
        } else {
            run_layers_tp2(peer, pair, x, x_peer, phase, tap);
        }
    };
    if (sink != nullptr) { sink->begin(x); }
    if (phase == Phase::Verify) {
        // The verify window runs the phase the single-token decode path runs, so its per-column
        // logits agree with decode on near-ties. That phase needs the explicit sequence bindings the
        // decode path sets (one row, one column per window token, the live state as the source slot),
        // which prefill leaves unset because it is a single column.
        Tensor verify_source_slots = work_.alloc(DType::I32, {1});
        ops::set_i32_scalar(verify_source_slots, 0, ctx_.stream);
        // Batched MTP attention keys off active_sequence_batch_ != 0, so the window also needs the
        // per-row valid-column count and the backend KV row table that decode binds per row.
        Tensor verify_valid_columns = work_.alloc(DType::I32, {1});
        ops::set_i32_scalar(verify_valid_columns, tokens, ctx_.stream);
        Tensor verify_backend_rows = work_.alloc(DType::I32, {1});
        ops::set_i32_scalar(verify_backend_rows, 0, ctx_.stream);
        ScopedValue<std::int32_t> verify_batch(active_sequence_batch_, 1);
        ScopedValue<std::int32_t> verify_width(active_sequence_width_, tokens);
        ScopedValue<const Tensor*> verify_source(active_linear_state_source_slots_,
                                                 &verify_source_slots);
        ScopedValue<const Tensor*> verify_valid(active_valid_columns_, &verify_valid_columns);
        ScopedValue<const Tensor*> verify_backend(active_backend_kv_table_rows_,
                                                  &verify_backend_rows);
        // The verify path reads positions as [width, batch]; prefill binds them flat for its chunk.
        Tensor verify_positions = work_.alloc(DType::I32, {tokens, 1});
        ops::fill_i32_positions(verify_positions, first_position, 1, ctx_.stream);
        ScopedPositions verify_cache(active_cache_positions_, verify_positions);
        ScopedPositions verify_rope(active_rope_positions_, verify_positions);
        ScopedValue<std::int32_t> verify_peer_batch(peer.active_sequence_batch_, 1);
        ScopedValue<std::int32_t> verify_peer_width(peer.active_sequence_width_, tokens);
        ScopedValue<const Tensor*> verify_peer_source(peer.active_linear_state_source_slots_,
                                                      &verify_source_slots);
        ScopedValue<const Tensor*> verify_peer_valid(peer.active_valid_columns_,
                                                     &verify_valid_columns);
        ScopedValue<const Tensor*> verify_peer_backend(peer.active_backend_kv_table_rows_,
                                                       &verify_backend_rows);
        ScopedPositions verify_peer_cache(peer.active_cache_positions_, verify_positions);
        ScopedPositions verify_peer_rope(peer.active_rope_positions_, verify_positions);
        run_layers();
    } else {
        run_layers();
    }
    if (sink != nullptr && phase != Phase::Verify) {
        // Positions are captured after the layers so the sink can check every feature layer published.
        sink->capture_positions(*active_cache_positions_, ctx_.stream);
        sink->consume_prefill_chunk(tokens, false);
    }

    ctx_.bind_to_current_thread();
    Tensor xf      = work_.alloc(DType::BF16, {hidden, tokens});
    peer.ctx_.bind_to_current_thread();
    Tensor xf_peer = peer.work_.alloc(DType::BF16, {hidden, tokens});
    ctx_.bind_to_current_thread();
    ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, xf, ctx_.stream);
    if (mtp_input_hidden != nullptr) {
        // The MTP layer's prefill chunk consumes the whole chunk's final-norm hidden.
        if (mtp_input_hidden->dtype != DType::BF16 || mtp_input_hidden->ne[0] != hidden ||
            mtp_input_hidden->ne[1] != tokens) {
            throw std::invalid_argument(
                "forward_tp2_prefill: MTP hidden buffer must be [hidden,T] BF16");
        }
        CUDA_CHECK(cudaMemcpyAsync(mtp_input_hidden->data, xf.data, xf.bytes(),
                                   cudaMemcpyDeviceToDevice, ctx_.stream));
    }
    peer.ctx_.bind_to_current_thread();
    ops::rmsnorm(x_peer, *peer.final_norm_, config_.rms_norm_eps, true, xf_peer, peer.ctx_.stream);

    // Only the chunk's last column feeds the first sample, exactly like the single-device prefill's
    // final chunk. project_head_tp2 covers both head layouts: replicated (two independent full
    // projections that agree exactly) and vocabulary-split (one half per shard plus an all-reduce).
    ctx_.bind_to_current_thread();
    if (logits_columns != nullptr) {
        // Speculative verification scores every column: column j predicts the token after the one
        // it reads, so it judges the draft placed at j+1. Its last column is exactly the logits
        // the [V,1] projection below would produce, so that read is not repeated. Only the driving
        // shard carries this buffer, so a split head needs a peer scratch window of the same size.
        if (logits_columns->dtype != DType::BF16 || logits_columns->ne[0] != vocab ||
            logits_columns->ne[1] != tokens) {
            throw std::invalid_argument(
                "forward_tp2_prefill: per-column logits buffer must be [V,T] BF16");
        }
        Tensor logits_columns_peer = peer.work_.alloc(DType::BF16, {vocab, tokens});
        project_head_tp2(peer, pair, xf, xf_peer, *logits_columns, logits_columns_peer);
    } else {
        project_head_tp2(peer, pair, xf.slice(1, tokens - 1, 1), xf_peer.slice(1, tokens - 1, 1),
                         *logits, *logits_peer);
    }
    if (hidden_columns != nullptr) {
        // The next round's MTP bridge consumes the final-norm hidden of the accepted column.
        if (hidden_columns->dtype != DType::BF16 || hidden_columns->ne[0] != hidden ||
            hidden_columns->ne[1] != tokens) {
            throw std::invalid_argument(
                "forward_tp2_prefill: per-column hidden buffer must be [hidden,T] BF16");
        }
        ctx_.bind_to_current_thread();
        CUDA_CHECK(cudaMemcpyAsync(hidden_columns->data, xf.data, xf.bytes(),
                                   cudaMemcpyDeviceToDevice, ctx_.stream));
    }
}

void TextContext::run_layers_tp2_overlap(TextContext& peer, tp::DevicePair& pair, Tensor& x,
                                         Tensor& x_peer, Phase ph, int width,
                                         std::int32_t first_position) {
    const std::int32_t hidden = dimension(config_.hidden_size);
    const std::int32_t tokens = x.ne[1];
    // The split must be exact: a remainder block would run at a width the schedule has not been
    // measured at, and 320/384 and 128 both cost more than they hide. A chunk that does not hold two
    // whole sub-blocks keeps the plain lockstep loop instead (forward_tp2_prefill's gate).
    if (width <= 0 || width % 64 != 0 || tokens <= 0 || tokens % width != 0 || tokens / width < 2 ||
        x_peer.ne[1] != tokens) {
        throw std::invalid_argument(
            "run_layers_tp2_overlap: a chunk needs at least two whole 64-aligned sub-blocks");
    }
    const int parts = tokens / width;
    const Tensor* const base_cache = active_cache_positions_;
    const Tensor* const base_rope  = active_rope_positions_;
    const auto* const base_envelope = active_causal_attention_envelope_;
    const Tensor* const base_cache_peer = peer.active_cache_positions_;
    const Tensor* const base_rope_peer  = peer.active_rope_positions_;
    const auto* const base_envelope_peer = peer.active_causal_attention_envelope_;
    if (base_cache == nullptr || base_rope == nullptr || base_envelope == nullptr ||
        base_cache_peer == nullptr || base_rope_peer == nullptr || base_envelope_peer == nullptr) {
        throw std::logic_error("run_layers_tp2_overlap: the chunk bindings are not installed");
    }
    const Tensor base_cache_t      = *base_cache;
    const Tensor base_rope_t       = *base_rope;
    const Tensor base_cache_t_peer = *base_cache_peer;
    const Tensor base_rope_t_peer  = *base_rope_peer;

    // Every sub-block is exactly `width` wide. That width is 64-aligned, which keeps each sub-block
    // on the ops' documented scheduling boundary, so the result stays bit-identical to the
    // single-block loop.
    std::vector<std::int32_t> offset(static_cast<std::size_t>(parts), 0);
    std::vector<std::int32_t> length(static_cast<std::size_t>(parts), 0);
    for (int s = 0; s < parts; ++s) {
        offset[static_cast<std::size_t>(s)] = s * width;
        length[static_cast<std::size_t>(s)] = width;
    }
    std::vector<Tensor> xs, xs_peer, cache_s, rope_s, cache_s_peer, rope_s_peer;
    for (int s = 0; s < parts; ++s) {
        xs.push_back(x.slice(1, offset[s], length[s]));
        xs_peer.push_back(x_peer.slice(1, offset[s], length[s]));
        cache_s.push_back(base_cache_t.slice(0, offset[s], length[s]));
        rope_s.push_back(base_rope_t.slice(0, offset[s], length[s]));
        cache_s_peer.push_back(base_cache_t_peer.slice(0, offset[s], length[s]));
        rope_s_peer.push_back(base_rope_t_peer.slice(0, offset[s], length[s]));
    }
    ctx_.bind_to_current_thread();
    Tp2OverlapEvents ev_local(static_cast<std::size_t>(parts));
    peer.ctx_.bind_to_current_thread();
    Tp2OverlapEvents ev_peer(static_cast<std::size_t>(parts));

    for (std::size_t layer = 0; layer < parameters_.text.layers.size(); ++layer) {
        const auto& block      = parameters_.text.layers[layer];
        const auto& block_peer = peer.parameters_.text.layers[layer];
        if (tp_mixer_activation_width(block) != 0 ||
            peer.tp_mixer_activation_width(block_peer) != 0) {
            throw std::logic_error(
                "run_layers_tp2_overlap: a gather mixer needs the activation merge");
        }
        auto layer_scope      = work_.scope();
        auto layer_scope_peer = peer.work_.scope();
        std::vector<Tensor> mix, mix_peer, ffn, ffn_peer;
        for (int s = 0; s < parts; ++s) {
            mix.push_back(work_.alloc(DType::BF16, {hidden, length[s]}));
            mix_peer.push_back(peer.work_.alloc(DType::BF16, {hidden, length[s]}));
            ffn.push_back(work_.alloc(DType::BF16, {hidden, length[s]}));
            ffn_peer.push_back(peer.work_.alloc(DType::BF16, {hidden, length[s]}));
        }
        // Stage 1: every sub-block's mixer runs first, so the collective of sub-block s can be issued
        // as soon as its own mixer landed and overlap the next sub-block's mixer.
        for (int s = 0; s < parts; ++s) {
            const std::uint32_t visible = static_cast<std::uint32_t>(first_position) +
                                          static_cast<std::uint32_t>(offset[s] + length[s]);
            const ops::CausalAttentionExecutionEnvelope envelope{visible, visible};
            ctx_.bind_to_current_thread();
            {
                ScopedPositions bound_cache(active_cache_positions_, cache_s[s]);
                ScopedPositions bound_rope(active_rope_positions_, rope_s[s]);
                ScopedEnvelope  bound_envelope(active_causal_attention_envelope_, envelope);
                tp_mixer_layer(block, xs[s], mix[s], layer, ph, nullptr);
            }
            CUDA_CHECK(cudaEventRecord(ev_local.produced(s), ctx_.stream));
            peer.ctx_.bind_to_current_thread();
            {
                ScopedPositions bound_cache(peer.active_cache_positions_, cache_s_peer[s]);
                ScopedPositions bound_rope(peer.active_rope_positions_, rope_s_peer[s]);
                ScopedEnvelope  bound_envelope(peer.active_causal_attention_envelope_, envelope);
                peer.tp_mixer_layer(block_peer, xs_peer[s], mix_peer[s], layer, ph, nullptr);
            }
            CUDA_CHECK(cudaEventRecord(ev_peer.produced(s), peer.ctx_.stream));
        }
        for (int s = 0; s < parts; ++s) {
            ctx_.bind_to_current_thread();
            CUDA_CHECK(cudaStreamWaitEvent(ctx_.collective_stream, ev_local.produced(s), 0));
            peer.ctx_.bind_to_current_thread();
            CUDA_CHECK(cudaStreamWaitEvent(peer.ctx_.collective_stream, ev_peer.produced(s), 0));
            pair.allreduce(mix[s].data, mix_peer[s].data, mix[s].bytes(), ctx_.collective_stream,
                           peer.ctx_.collective_stream);
            ctx_.bind_to_current_thread();
            CUDA_CHECK(cudaEventRecord(ev_local.done(s), ctx_.collective_stream));
            peer.ctx_.bind_to_current_thread();
            CUDA_CHECK(cudaEventRecord(ev_peer.done(s), peer.ctx_.collective_stream));
        }
        for (int s = 0; s < parts; ++s) {
            ctx_.bind_to_current_thread();
            CUDA_CHECK(cudaStreamWaitEvent(ctx_.stream, ev_local.done(s), 0));
            ops::residual_add(mix[s], xs[s], ctx_.stream);
            peer.ctx_.bind_to_current_thread();
            CUDA_CHECK(cudaStreamWaitEvent(peer.ctx_.stream, ev_peer.done(s), 0));
            ops::residual_add(mix_peer[s], xs_peer[s], peer.ctx_.stream);
        }
        // Stage 2: the FFN tail follows the same shape, overlapping the next sub-block's FFN.
        for (int s = 0; s < parts; ++s) {
            ctx_.bind_to_current_thread();
            tp_mlp_delta(block, xs[s], ffn[s], layer, ph);
            CUDA_CHECK(cudaEventRecord(ev_local.produced(s), ctx_.stream));
            peer.ctx_.bind_to_current_thread();
            peer.tp_mlp_delta(block_peer, xs_peer[s], ffn_peer[s], layer, ph);
            CUDA_CHECK(cudaEventRecord(ev_peer.produced(s), peer.ctx_.stream));
        }
        for (int s = 0; s < parts; ++s) {
            ctx_.bind_to_current_thread();
            CUDA_CHECK(cudaStreamWaitEvent(ctx_.collective_stream, ev_local.produced(s), 0));
            peer.ctx_.bind_to_current_thread();
            CUDA_CHECK(cudaStreamWaitEvent(peer.ctx_.collective_stream, ev_peer.produced(s), 0));
            pair.allreduce(ffn[s].data, ffn_peer[s].data, ffn[s].bytes(), ctx_.collective_stream,
                           peer.ctx_.collective_stream);
            ctx_.bind_to_current_thread();
            CUDA_CHECK(cudaEventRecord(ev_local.done(s), ctx_.collective_stream));
            peer.ctx_.bind_to_current_thread();
            CUDA_CHECK(cudaEventRecord(ev_peer.done(s), peer.ctx_.collective_stream));
        }
        for (int s = 0; s < parts; ++s) {
            ctx_.bind_to_current_thread();
            CUDA_CHECK(cudaStreamWaitEvent(ctx_.stream, ev_local.done(s), 0));
            ops::residual_add(ffn[s], xs[s], ctx_.stream);
            peer.ctx_.bind_to_current_thread();
            CUDA_CHECK(cudaStreamWaitEvent(peer.ctx_.stream, ev_peer.done(s), 0));
            ops::residual_add(ffn_peer[s], xs_peer[s], peer.ctx_.stream);
        }
    }
    // The collective stream is a different stream from the one the caller's next chunk uses, so
    // order the last collectives forward before any consumer can run, then restore the bindings the
    // outer scopes installed (the Scoped* guards reset their slot to null on the way out).
    ctx_.bind_to_current_thread();
    for (int s = 0; s < parts; ++s) {
        CUDA_CHECK(cudaStreamWaitEvent(ctx_.stream, ev_local.done(s), 0));
    }
    active_cache_positions_           = base_cache;
    active_rope_positions_            = base_rope;
    active_causal_attention_envelope_ = base_envelope;
    peer.ctx_.bind_to_current_thread();
    for (int s = 0; s < parts; ++s) {
        CUDA_CHECK(cudaStreamWaitEvent(peer.ctx_.stream, ev_peer.done(s), 0));
    }
    peer.active_cache_positions_           = base_cache_peer;
    peer.active_rope_positions_            = base_rope_peer;
    peer.active_causal_attention_envelope_ = base_envelope_peer;
}

void TextContext::forward_tp2_window(TextContext& peer, tp::DevicePair& pair,
                                     const std::int32_t* ids, const std::int32_t* positions,
                                     ops::CausalAttentionExecutionEnvelope envelope,
                                     Tensor& logits_columns, Tensor* hidden_columns,
                                     DFlashFeatureSink* sink, const std::int32_t* valid_columns) {
    // The single-lane verify window is the batch form with no per-lane bindings, so it keeps the
    // historic scalar execution row 0 and state slot 0 and its numerics stay untouched.
    const std::int32_t window = logits_columns.ne[1];
    forward_tp2_window_batch(peer, pair, ids, positions, nullptr, nullptr, window, 1, envelope,
                             logits_columns, hidden_columns, sink, valid_columns);
}

void TextContext::forward_tp2_window_batch(TextContext& peer, tp::DevicePair& pair,
                                           const std::int32_t* ids, const std::int32_t* positions,
                                           const std::int32_t* kv_table_rows,
                                           const std::int32_t* state_slots, std::int32_t width,
                                           std::int32_t batch,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& logits_columns, Tensor* hidden_columns,
                                           DFlashFeatureSink* sink,
                                           const std::int32_t* valid_columns) {
    const std::int32_t hidden = dimension(config_.hidden_size);
    const std::int32_t vocab  = dimension(config_.vocab_size);
    if (ids == nullptr || positions == nullptr) {
        throw std::invalid_argument("forward_tp2_window_batch requires pinned host ids and positions");
    }
    if (width <= 0 || batch < 1 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("forward_tp2_window_batch: the window shape is out of range");
    }
    // The per-lane bindings are what select the batched aggregate shape. The single-lane caller
    // passes neither and keeps the scalar bindings; a one-member batch of the multi-lane route still
    // passes both, because its KV execution row and state slot belong to a lane other than 0.
    const bool per_lane_window = kv_table_rows != nullptr || state_slots != nullptr;
    if (per_lane_window && (kv_table_rows == nullptr || state_slots == nullptr)) {
        throw std::invalid_argument("forward_tp2_window_batch requires both per-lane bindings");
    }
    if (batch > 1 && !per_lane_window) {
        throw std::invalid_argument("forward_tp2_window_batch requires the per-lane bindings");
    }
    if (logits_columns.dtype != DType::BF16 || logits_columns.ne[0] != vocab) {
        throw std::invalid_argument("forward_tp2_window_batch: logits columns must be [V,width*B] BF16");
    }
    const std::int32_t tokens = width * batch;
    if (logits_columns.ne[1] != tokens) {
        throw std::invalid_argument("forward_tp2_window_batch: logits columns do not match the window");
    }
    if (envelope.min_visible_keys == 0 || envelope.max_visible_keys < envelope.min_visible_keys) {
        throw std::invalid_argument("forward_tp2_window: envelope does not cover the window");
    }
    if (hidden_columns != nullptr &&
        (hidden_columns->dtype != DType::BF16 || hidden_columns->ne[0] != hidden ||
         hidden_columns->ne[1] != tokens)) {
        throw std::invalid_argument("forward_tp2_window: hidden columns must be [hidden,T] BF16");
    }

    // Execution rows and state slots are per lane, not per column: the paged-KV row and the linear
    // state slot identify a sequence, and the batch ops take one entry per lane while the column
    // axis stays inside the op (cache positions are the per-column binding, laid out lane-major).
    struct BindState {
        Tensor ids;
        Tensor positions;
        Tensor kv_table_rows;
        Tensor state_source;
        Tensor state_destination;
        Tensor valid;
        ops::CausalAttentionExecutionEnvelope envelope{1, 1};
    };
    // Both shards bind their own copies of the window: the pair has no peer access, and a captured
    // sequence needs fixed addresses, so each shard's arena owns its [T] buffers and the only
    // per-round input is the pinned host source the memcpy node reads.
    auto make_bind = [&](TextContext& card, WorkspaceArena& arena) {
        card.ctx_.bind_to_current_thread();
        BindState bind;
        bind.envelope  = envelope;
        bind.ids       = arena.alloc(DType::I32, {tokens});
        bind.positions = arena.alloc(DType::I32, {tokens});
        copy_i32(ids, bind.ids, card.ctx_.stream);
        copy_i32(positions, bind.positions, card.ctx_.stream);
        if (per_lane_window) {
            bind.kv_table_rows     = arena.alloc(DType::I32, {batch});
            bind.state_source      = arena.alloc(DType::I32, {batch});
            bind.state_destination = arena.alloc(DType::I32, {batch});
            copy_i32(kv_table_rows, bind.kv_table_rows, card.ctx_.stream);
            copy_i32(state_slots, bind.state_source, card.ctx_.stream);
            copy_i32(state_slots, bind.state_destination, card.ctx_.stream);
        } else {
            bind.kv_table_rows     = arena.alloc(DType::I32, {1});
            bind.state_source      = arena.alloc(DType::I32, {1});
            bind.state_destination = arena.alloc(DType::I32, {1});
            ops::set_i32_scalar(bind.kv_table_rows, 0, card.ctx_.stream);
            ops::set_i32_scalar(bind.state_source, 0, card.ctx_.stream);
            ops::set_i32_scalar(bind.state_destination, 0, card.ctx_.stream);
        }
        if (valid_columns != nullptr) {
            // valid_columns carries one entry per lane, so the window binding is [batch]: the attention
            // op takes [batch] and a scalar would be an out-of-range read on the two-lane route. The
            // single-lane caller is already one member wide and keeps [1].
            bind.valid = arena.alloc(DType::I32, {per_lane_window ? batch : 1});
            copy_i32(valid_columns, bind.valid, card.ctx_.stream);
        }
        return bind;
    };
    const BindState bind0 = make_bind(*this, work_);
    const BindState bind1 = make_bind(peer, peer.work_);
    // The window is a contiguous append at consecutive absolute positions, so its cache and RoPE
    // positions coincide: this path carries no RoPE delta and no multimodal position table.
    ScopedPositions cache0(active_cache_positions_, bind0.positions);
    ScopedPositions rope0(active_rope_positions_, bind0.positions);
    ScopedValue<const Tensor*> valid0(active_valid_columns_, &bind0.valid);
    ScopedEnvelope envelope0(active_causal_attention_envelope_, bind0.envelope);
    ScopedValue<const Tensor*> kv0(active_kv_table_rows_, &bind0.kv_table_rows);
    ScopedValue<const Tensor*> source0(active_linear_state_source_slots_, &bind0.state_source);
    ScopedValue<const Tensor*> destination0(active_linear_state_destination_slots_,
                                            &bind0.state_destination);
    // A multi-lane window publishes the aggregate shape; a one-lane window leaves the scalar
    // binding in place exactly as the single-lane route always has.
    ScopedValue<std::int32_t> batch0(active_sequence_batch_, per_lane_window ? batch : 0);
    ScopedValue<std::int32_t> width0(active_sequence_width_, per_lane_window ? width : 0);
    ScopedPositions cache1(peer.active_cache_positions_, bind1.positions);
    ScopedPositions rope1(peer.active_rope_positions_, bind1.positions);
    ScopedValue<const Tensor*> valid1(peer.active_valid_columns_, &bind1.valid);
    ScopedEnvelope envelope1(peer.active_causal_attention_envelope_, bind1.envelope);
    ScopedValue<const Tensor*> kv1(peer.active_kv_table_rows_, &bind1.kv_table_rows);
    ScopedValue<const Tensor*> source1(peer.active_linear_state_source_slots_,
                                       &bind1.state_source);
    ScopedValue<const Tensor*> destination1(peer.active_linear_state_destination_slots_,
                                            &bind1.state_destination);
    ScopedValue<std::int32_t> batch1(peer.active_sequence_batch_, per_lane_window ? batch : 0);
    ScopedValue<std::int32_t> width1(peer.active_sequence_width_, per_lane_window ? width : 0);
    // A one-member per-lane window still runs the scalar convolution path (the batched GDN state
    // kernels start at two rows) and that path reads the scalar slot pair rather than the bound
    // vectors, so publish this member's slot - exactly what the plain one-column batch does. A wider
    // window records per lane and never reads the scalar pair.
    if (batch == 1 && per_lane_window) {
        const std::int32_t slot = state_slots[0];
        set_linear_state_slots(slot, slot);
        peer.set_linear_state_slots(slot, slot);
    }

    ctx_.bind_to_current_thread();
    Tensor x      = work_.alloc(DType::BF16, {hidden, tokens});
    peer.ctx_.bind_to_current_thread();
    Tensor x_peer = peer.work_.alloc(DType::BF16, {hidden, tokens});
    ctx_.bind_to_current_thread();
    embedding_tp2(peer, pair, bind0.ids, &bind1.ids, x, &x_peer);

    NullTap tap;
    if (sink != nullptr) {
        sink->begin(x);
        run_layers_tp2(peer, pair, x, x_peer, Phase::Prefill, *sink);
    } else {
        run_layers_tp2(peer, pair, x, x_peer, Phase::Prefill, tap);
    }

    ctx_.bind_to_current_thread();
    Tensor xf      = work_.alloc(DType::BF16, {hidden, tokens});
    peer.ctx_.bind_to_current_thread();
    Tensor xf_peer = peer.work_.alloc(DType::BF16, {hidden, tokens});
    ctx_.bind_to_current_thread();
    ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, xf, ctx_.stream);
    peer.ctx_.bind_to_current_thread();
    ops::rmsnorm(x_peer, *peer.final_norm_, config_.rms_norm_eps, true, xf_peer, peer.ctx_.stream);
    // Every column is scored, because column j judges the draft placed at j+1; the last column is
    // exactly the last column's logits, so no second lm_head read is needed.
    ctx_.bind_to_current_thread();
    Tensor logits_columns_peer = peer.work_.alloc(DType::BF16, {vocab, tokens});
    project_head_tp2(peer, pair, xf, xf_peer, logits_columns, logits_columns_peer);
    if (hidden_columns != nullptr) {
        // The next round's MTP bridge consumes the final-norm hidden of the accepted column.
        ctx_.bind_to_current_thread();
        CUDA_CHECK(cudaMemcpyAsync(hidden_columns->data, xf.data, xf.bytes(),
                                   cudaMemcpyDeviceToDevice, ctx_.stream));
    }
}

template <class Tap>
PrefillChunkResult
TextContext::prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                          const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end) {
    runtime::ExecutionTimingRecorder timing;
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    cudaStream_t s           = ctx_.stream;
    const int T              = static_cast<int>(ids.size());
    const int chunk          = static_cast<int>(prefill_chunk_);
    const std::uint32_t base = text_kv_base_;

    if (text_prefill != nullptr) {
        if (multimodal != nullptr || base != text_prefill->begin ||
            text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("text prefill chunk does not match its full prompt");
        }
    }
    if (multimodal != nullptr) {
        if (base != multimodal->begin ||
            multimodal->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("multimodal prefill suffix does not match its cache base");
        }
        if (multimodal->positions.size() != 3 * multimodal->token_ids.size()) {
            throw std::invalid_argument("multimodal positions must have shape [3,T]");
        }
        if (multimodal->vision == nullptr) {
            throw std::invalid_argument("multimodal prefill requires a Vision session");
        }
        rope_delta_ = multimodal->rope_delta;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    const std::int64_t base64    = static_cast<std::int64_t>(base);
    const std::int64_t split_abs = prefill_split_frontier_;
    const bool has_split = split_abs > base64 && split_abs <= base64 + static_cast<std::int64_t>(T);
    const int split_rel  = has_split ? static_cast<int>(split_abs - base64) : -1;
    const bool prepare_mtp_prompt = mtp_enabled() && io_.mtp.has_value();
    if (prepare_mtp_prompt &&
        mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP proposal extent exceeds the configured draft window");
    }
    int t0 = 0;
    for (; t0 < T;) {
        int len = std::min(chunk, T - t0);
        if (split_rel > 0 && t0 < split_rel && t0 + len > split_rel) { len = split_rel - t0; }
        work_.reset();

        VisionChunk vision_chunk;
        const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);
        if (multimodal != nullptr) {
            if (multimodal->vision == nullptr) {
                throw std::logic_error("multimodal prefill has no Vision session");
            }
            vision_chunk =
                multimodal->vision->prepare_chunk(prompt_t0, static_cast<std::uint32_t>(len));
            len = vision_chunk.length;
        }
        const bool is_last = finalize_at_end && (t0 + len == T);
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(len));

        {
            std::vector<std::int32_t> local_scatter_indices;
            std::int32_t visual_begin = 0;
            if (vision_chunk.control != nullptr) {
                const auto scatter =
                    std::span<const std::int32_t>(vision_chunk.control->scatter_indices);
                const auto begin = std::lower_bound(scatter.begin(), scatter.end(), prompt_t0);
                const auto end   = std::lower_bound(begin, scatter.end(), prompt_t0 + len);
                const auto count = static_cast<std::int32_t>(end - begin);
                visual_begin     = static_cast<std::int32_t>(begin - scatter.begin());
                local_scatter_indices.resize(static_cast<std::size_t>(count));
                for (std::int32_t i = 0; i < count; ++i) {
                    local_scatter_indices[static_cast<std::size_t>(i)] =
                        begin[i] - static_cast<std::int32_t>(prompt_t0);
                }
            }

            const std::int32_t rope_axes = multimodal != nullptr ? 3 : (rope_delta_ != 0 ? 1 : 0);
            const auto roots             = workspace::text_prefill_roots(
                work_, config_, len, rope_axes,
                static_cast<std::int32_t>(local_scatter_indices.size()));
            Tensor ids_device = roots.ids;
            copy_i32(ids.data() + t0, ids_device, s);

            Tensor positions = roots.positions;
            ops::fill_i32_positions(positions, base_i + t0, 1, s);

            Tensor rope_positions = positions;
            std::vector<std::int32_t> rope_positions_host;
            if (multimodal != nullptr) {
                rope_positions = roots.rope_positions;
                rope_positions_host.resize(static_cast<std::size_t>(3) * len);
                const std::size_t prompt_tokens = multimodal->token_ids.size();
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* src = multimodal->positions.data() +
                                      static_cast<std::size_t>(axis) * prompt_tokens + prompt_t0;
                    std::copy_n(src, len,
                                rope_positions_host.data() + static_cast<std::size_t>(axis) * len);
                }
                copy_i32(rope_positions_host.data(), rope_positions, s);
            } else if (rope_delta_ != 0) {
                rope_positions = roots.rope_positions;
                ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
            }
            ScopedPositions scoped_cache(active_cache_positions_, positions);
            ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
            const auto visible = static_cast<std::uint32_t>(base_i + t0 + len);
            const ops::CausalAttentionExecutionEnvelope chunk_envelope{visible, visible};
            ScopedEnvelope scoped_envelope(active_causal_attention_envelope_, chunk_envelope);

            Tensor x = roots.residual;
            ops::embedding(ids_device, *embed_, x, s);
            if (!local_scatter_indices.empty()) {
                Tensor indices_device = roots.scatter_indices;
                copy_i32(local_scatter_indices.data(), indices_device, s);
                Tensor embeddings = vision_chunk.embeddings.slice(
                    1, visual_begin, static_cast<std::int32_t>(local_scatter_indices.size()));
                ops::scatter(embeddings, indices_device, x, s);
            }
            if constexpr (Tap::enabled) { tap.begin(x); }
            run_layers(x, Phase::Prefill, tap);
            if constexpr (requires { tap.capture_positions(positions, s); }) {
                tap.capture_positions(positions, s);
            }

            Tensor xf = prefill_hidden_.data != nullptr
                            ? matrix_window(prefill_hidden_, len)
                            : work_.alloc(DType::BF16, {dimension(config_.hidden_size), len});
            ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, xf, s);

            if (is_last) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(io_.logits, 1);
                project(last_xf, *lm_head_, logits, work_, s);
                if (prefill_token_mask_.data != nullptr) {
                    ops::apply_token_mask(logits, prefill_token_mask_, s);
                }
                // Set io_.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same io_.pos).
                ops::set_i32_scalar(io_.pos, base_i + T, s);
                ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
                if (sampling_config_ != nullptr) {
                    ops::sample(logits, io_.token,
                                dimension(parameters_.model.resources().public_token_count),
                                sampling_config_, io_.pos, ops::kSamplePurposePrefill, work_, s);
                } else {
                    ops::argmax(logits, io_.token,
                                dimension(parameters_.model.resources().public_token_count), s);
                }
            }

            if (prepare_mtp_prompt) {
                const std::uint32_t alignment_tokens =
                    multimodal != nullptr ? static_cast<std::uint32_t>(multimodal->token_ids.size())
                    : text_prefill != nullptr
                        ? static_cast<std::uint32_t>(text_prefill->token_ids.size())
                        : static_cast<std::uint32_t>(T);
                const std::uint32_t alignment_begin =
                    multimodal != nullptr || text_prefill != nullptr
                        ? prompt_t0
                        : static_cast<std::uint32_t>(t0);
                const qwen3_5::MtpAlignmentWindow mtp_window = qwen3_5::plan_mtp_alignment_window(
                    alignment_tokens, alignment_begin, static_cast<std::uint32_t>(len));
                const std::span<const int> alignment_ids =
                    multimodal != nullptr     ? multimodal->token_ids
                    : text_prefill != nullptr ? text_prefill->token_ids
                                              : ids;
                const int prompt_columns =
                    len - static_cast<int>(mtp_window.final_column_uses_generated_token);
                Tensor mtp_ids = work_.alloc(DType::I32, {len});
                if (prompt_columns != 0) {
                    Tensor prompt_mtp_ids = mtp_ids.slice(0, 0, prompt_columns);
                    copy_i32(alignment_ids.data() + mtp_window.shifted_embedding_begin,
                             prompt_mtp_ids, s);
                }
                if (mtp_window.final_column_uses_generated_token) {
                    Tensor generated_mtp_id = mtp_ids.slice(0, len - 1, 1);
                    CUDA_CHECK(cudaMemcpyAsync(generated_mtp_id.data, io_.token.data,
                                               sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
                }

                Tensor mtp_input_embeddings;
                const Tensor* mtp_input_embeddings_ptr = nullptr;
                if (multimodal != nullptr) {
                    mtp_input_embeddings =
                        work_.alloc(DType::BF16, {dimension(config_.hidden_size), len});
                    ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, s);
                    if (vision_chunk.control != nullptr) {
                        const qwen3_5::MtpVisualOverlap overlap = qwen3_5::shifted_visual_overlap(
                            vision_chunk.control->scatter_indices, alignment_tokens, mtp_window);
                        if (!overlap.empty()) {
                            Tensor shifted_indices = workspace::visual_scatter_indices(
                                work_, static_cast<std::int32_t>(overlap.size()));
                            qwen3_5::detail::scatter_shifted_visual_embeddings(
                                mtp_input_embeddings, vision_chunk.embeddings, overlap,
                                shifted_indices, s);
                        }
                    }
                    mtp_input_embeddings_ptr = &mtp_input_embeddings;
                }
                if (is_last && mtp_proposal_extent_ != 0) {
                    Tensor logits = matrix_window(io_.logits, 1);
                    Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, true, &io_.mtp->ar_hidden,
                                      &logits, &draft0);

                    Tensor ar_position = io_.mtp->position.slice(0, 0, 1);
                    ops::set_i32_scalar(ar_position, base_i + T, s);
                    for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                        Tensor prev_token = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                        Tensor next_token = io_.mtp->draft_tokens.slice(0, i, 1);
                        Tensor next_hidden =
                            work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
                        const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
                        const ops::CausalAttentionExecutionEnvelope ar_envelope{ar_visible,
                                                                                ar_visible};
                        mtp_forward_ar_step(prev_token, io_.mtp->ar_hidden, ar_position,
                                            ar_envelope, next_hidden, logits, next_token);
                        CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, next_hidden.data,
                                                   io_.mtp->ar_hidden.bytes(),
                                                   cudaMemcpyDeviceToDevice, s));
                        ops::increment_i32_scalar(ar_position, s);
                    }
                } else {
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, false, nullptr, nullptr,
                                      nullptr);
                }
            }

            if (split_rel > 0 && t0 + len == split_rel &&
                rewrite_checkpoint_hidden_output_ != nullptr) {
                require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16,
                                     {dimension(config_.hidden_size), 1},
                                     "rewrite checkpoint hidden output");
                const Tensor checkpoint_hidden = xf.slice(1, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data,
                                           checkpoint_hidden.data, checkpoint_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice, s));
            }
        }

        if constexpr (requires { tap.consume_prefill_chunk(len, false); }) {
            work_.reset();
            tap.consume_prefill_chunk(len, split_rel > 0 && t0 + len == split_rel);
        }

        t0 += len;
        break;
    }

    prefill_split_frontier_ = -1;

    timing.begin_wait();
    ctx_.synchronize();
    timing.end_wait();
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(t0),
                              .finalized        = finalize_at_end && t0 == T,
                              .timing           = timing.finish()};
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    NullTap tap;
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, sink,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    NullTap tap;
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, sink,
                        finalize_at_end);
}

} // namespace ninfer::models::qwen3_5::execution
