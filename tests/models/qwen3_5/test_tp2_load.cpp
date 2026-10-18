#include "artifact/reader.h"
#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/load.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace qwen = ninfer::models::qwen3_5;

std::pair<int, int> pick_devices() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { return {-1, -1}; }
    std::vector<std::string> names(count);
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, i);
        names[i] = prop.name;
    }
    for (int a = 0; a < count; ++a) {
        for (int b = a + 1; b < count; ++b) {
            if (names[a] == names[b]) { return {a, b}; }
        }
    }
    return {0, 1};
}

void check_shard_shapes(const qwen::execution::Parameters& parameters, const qwen::Model& model,
                        int shard, bool expect_mtp, bool expect_split_head,
                        bool expect_proposal_split, bool expect_dflash2, bool expect_vision) {
    const auto& weights = model.weights();
    if (expect_vision) {
        // The Vision tower is the other shard-local component: the static split keeps it on shard 1,
        // where the encode/handoff workspace is also allocated, and shard 0 must not pay for its
        // bytes. Where it is present it stays replicated whole, because the tower's activations are
        // consumed as a full-width embedding by both shards.
        if (!weights.vision.has_value()) {
            throw std::runtime_error("shard " + std::to_string(shard) +
                                     " has no Vision weights under --vision");
        }
        const auto& vision = *weights.vision;
        if (shard == 1) {
            if (!model.has_weight(vision.patch_embedding)) {
                throw std::runtime_error("shard 1 did not materialize the Vision tower");
            }
            const auto& embedding = model.weight(vision.patch_embedding).view;
            if (embedding.shape[0] != 1152) {
                throw std::runtime_error(
                    "shard 1 Vision patch embedding is not replicated [1152,patch_width]");
            }
            if (!model.has_weight(vision.merger_fc2)) {
                throw std::runtime_error("shard 1 did not materialize the Vision merger");
            }
        } else if (model.has_weight(vision.patch_embedding)) {
            throw std::runtime_error("shard 0 materialized the shard-local Vision tower");
        }
    }
    if (expect_mtp) {
        // The MTP layer is a shard-local component: it drafts on shard 0 alone and no execution path
        // on the other shard reads it, so shard 1 must not pay for its bytes. Where it is present it
        // stays replicated whole - the proposal's input is bit-identical on both shards, so halving
        // it would only add a second lockstep all-reduce.
        if (!weights.mtp.has_value()) {
            throw std::runtime_error("shard " + std::to_string(shard) +
                                     " has no MTP weights under --spec mtp");
        }
        if (shard == 0) {
            const auto& mtp = *weights.mtp;
            if (!model.has_weight(mtp.input_projection)) {
                throw std::runtime_error("shard 0 did not materialize the MTP layer");
            }
            const auto& proj = model.weight(mtp.input_projection).view;
            if (proj.shape[0] != 5120 || proj.shape[1] != 10240) {
                throw std::runtime_error("shard 0 MTP input_projection is not replicated [5120,10240]");
            }
            const auto* mixer = std::get_if<qwen::AttentionWeights>(&mtp.layer.mixer);
            if (mixer == nullptr) { throw std::runtime_error("MTP layer mixer is not attention"); }
            const auto& q = model.weight(mixer->query).view;
            if (q.shape[0] != 6144 || q.shape[1] != 5120) {
                throw std::runtime_error("shard 0 MTP attention query is not replicated [6144,5120]");
            }
        } else if (model.has_weight(weights.mtp->input_projection)) {
            throw std::runtime_error("shard 1 materialized the shard-local MTP layer");
        }
    }
    if (expect_dflash2) {
        // The DFlash2 masked draft is the third shard-local component: it drafts on shard 0 alone,
        // and it stays replicated whole there because the conditioning features are bit-identical
        // on both shards and the selector's top-k spans the whole-vocabulary codebook.
        if (!weights.draft.has_value()) {
            throw std::runtime_error("shard " + std::to_string(shard) +
                                     " has no draft weights under --spec dflash2");
        }
        const auto& draft = *weights.draft;
        if (shard == 0) {
            if (!model.has_weight(draft.feature_projection)) {
                throw std::runtime_error("shard 0 did not materialize the DFlash2 draft");
            }
            const auto& feature = model.weight(draft.feature_projection).view;
            if (feature.shape[0] != 5120 || feature.shape[1] != 25600) {
                throw std::runtime_error("shard 0 DFlash2 feature_projection is not replicated");
            }
            const auto& q = model.weight(draft.layers[0].attention.query).view;
            if (q.shape[0] != 4096 || q.shape[1] != 5120) {
                throw std::runtime_error("shard 0 DFlash2 draft query is not replicated");
            }
            if (!draft.selector.has_value()) {
                throw std::runtime_error("shard 0 DFlash2 draft has no selector weights");
            }
            // The reduced proposal head is a draft-only table that either route halves: MTP merges
            // the pair's row blocks in proposal_argmax, and the masked draft ranks each half
            // through linear_topk and merges both candidate lists back to the whole table's stable
            // sixteen. Shard 0 therefore holds its own row block, not a replicated table.
            if (expect_proposal_split && weights.proposal.has_value()) {
                const auto& head = model.weight(weights.proposal->head).view;
                if (head.shape[0] != 65536 || head.shape[1] != 5120) {
                    throw std::runtime_error("shard 0 DFlash2 proposal head is not [65536,5120]");
                }
            }
            // The selector's whole-vocabulary codebooks are the peer shard's bytes on TP-2: the
            // masked draft hands its state over and shard 1 runs the selector against the whole
            // codebook, so shard 0 must not pay for them.
            if (draft.selector.has_value() &&
                model.has_weight(draft.selector->predecessor_codebook)) {
                throw std::runtime_error("shard 0 holds the DFlash2 selector codebook");
            }
        } else {
            if (model.has_weight(draft.feature_projection)) {
                throw std::runtime_error("shard 1 materialized the shard-local DFlash2 draft");
            }
            if (!draft.selector.has_value() ||
                !model.has_weight(draft.selector->predecessor_codebook)) {
                throw std::runtime_error("shard 1 did not materialize the DFlash2 selector codebook");
            }
            const auto& codebook = model.weight(draft.selector->predecessor_codebook).view;
            if (codebook.shape[0] != 248320 || codebook.shape[1] != 256) {
                throw std::runtime_error("shard 1 DFlash2 predecessor codebook is not [248320,256]");
            }
        }
    }
    const auto layer0   = weights.text.layers[0];
    const auto* dense   = std::get_if<qwen::DenseWeights>(&layer0.ffn);
    if (!dense) { throw std::runtime_error("layer 0 FFN is not dense"); }
    // gate/up: full [17408,5120] -> shard [8704,5120].
    const auto& gate = model.weight(dense->gate).view;
    if (gate.shape[0] != 8704 || gate.shape[1] != 5120) {
        throw std::runtime_error("shard " + std::to_string(shard) +
                                 " gate shape is not [8704,5120]");
    }
    // down: full [5120,17408] -> shard [5120,8704].
    const auto& down = model.weight(dense->down).view;
    if (down.shape[0] != 5120 || down.shape[1] != 8704) {
        throw std::runtime_error("shard " + std::to_string(shard) +
                                 " down shape is not [5120,8704]");
    }
    // output_head: vocabulary-parallel when nothing else consumes the head (no speculative
    // backend, or the optimized proposal head), otherwise replicated in full because the head is
    // weight-tied to the MTP proposal that runs on shard 0 alone. A split shard keeps the rows
    // [shard * V/2, (shard + 1) * V/2) and the forward assembles the full logits from the pair.
    const std::int64_t head_rows = expect_split_head ? 124160 : 248320;
    const auto& head             = model.weight(weights.text.output_head).view;
    if (head.shape[0] != head_rows || head.shape[1] != 5120) {
        throw std::runtime_error("shard " + std::to_string(shard) + " output_head shape is not [" +
                                 std::to_string(head_rows) + ",5120]");
    }
    // token_embedding: column-parallel over the hidden dimension. Every vocabulary row stays on
    // both shards, so no token id can fall outside a shard's table; the forward merges the two
    // halves of the hidden state after the gather.
    const auto& embedding = model.weight(weights.text.token_embedding).view;
    if (embedding.shape[0] != 248320 || embedding.shape[1] != 2560) {
        throw std::runtime_error("shard " + std::to_string(shard) +
                                 " token_embedding shape is not [248320,2560]");
    }
    if (expect_mtp) {
        // The MTP proposal head is the optimized reduced head (both shards hold half of its rows
        // when the split is on) or the weight-tied text head in the unoptimized route.
        const auto& mtp_head = model.weight(weights.mtp->output_head).view;
        const std::int64_t expect_rows = expect_proposal_split ? 65536 : 0;
        if (mtp_head.shape[1] != 5120 ||
            (expect_proposal_split ? mtp_head.shape[0] != expect_rows : mtp_head.shape[0] == 0)) {
            throw std::runtime_error("shard " + std::to_string(shard) +
                                     " MTP proposal head shape is not the expected row block");
        }
    }
    // The head-split mixer weights are per-shard halves (attention q [3072,5120] for a
    // full-attn layer: 12 of 24 q heads; GDN q [1024,5120] for a GDN layer: 8 of 16 k heads).
    const auto& mixer = layer0.mixer;
    if (const auto* attention = std::get_if<qwen::AttentionWeights>(&mixer)) {
        const auto& q = model.weight(attention->query).view;
        if (q.shape[0] != 3072 || q.shape[1] != 5120) {
            throw std::runtime_error("shard " + std::to_string(shard) +
                                     " attention query is not per-shard [3072,5120]");
        }
    } else {
        const auto& gdn = std::get<qwen::GdnWeights>(mixer);
        const auto& q   = model.weight(gdn.query).view;
        if (q.shape[0] != 1024 || q.shape[1] != 5120) {
            throw std::runtime_error("shard " + std::to_string(shard) +
                                     " GDN query is not per-shard [1024,5120]");
        }
    }
    (void)parameters;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path path;
        LoadOptions options;
        options.vision = false;
        options.speculative = SpeculativeBackend::None;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const auto value = [&]() -> std::string {
                if (++i >= argc) { throw std::invalid_argument("missing value for " + arg); }
                return argv[i];
            };
            if (arg == "--artifact") {
                path = value();
            } else if (arg == "--spec") {
                const std::string backend = value();
                if (backend == "mtp") {
                    options.speculative = SpeculativeBackend::Mtp;
                } else if (backend == "dflash2") {
                    options.speculative = SpeculativeBackend::DFlash2;
                } else {
                    throw std::invalid_argument("--spec supports mtp and dflash2");
                }
            } else if (arg == "--lm-head-draft") {
                options.proposal_head = ProposalHead::Optimized;
            } else if (arg == "--vision") {
                options.vision = true;
            } else if (arg == "--embedding-host") {
                options.embedding_host = true;
            } else if (arg == "--help") {
                std::cout << "--artifact PATH [--spec mtp|dflash2] [--lm-head-draft] [--vision]"
                             " [--embedding-host]\n";
                return 0;
            } else {
                throw std::invalid_argument("unknown argument " + arg);
            }
        }
        if (path.empty()) {
            std::cout << "SKIP: supply an explicit --artifact\n";
            return 77;
        }
        const auto [dev0, dev1] = pick_devices();
        if (dev0 < 0) {
            std::cout << "SKIP: need two GPUs\n";
            return 77;
        }
        std::unique_ptr<DeviceContext> device0(new DeviceContext(dev0));
        std::unique_ptr<DeviceContext> device1(new DeviceContext(dev1));
        artifact::Reader reader(path);
        auto plan = qwen::plan_load(reader, options);
        auto [model0, model1] =
            qwen::materialize_model_tp2(std::move(plan), *device0, *device1, nullptr);
        // Construct both shard Parameters: this is the blocker validated this round.
        qwen::execution::Parameters parameters0(*model0);
        qwen::execution::Parameters parameters1(*model1);
        const bool expect_mtp     = options.speculative == SpeculativeBackend::Mtp;
        const bool expect_dflash2 = options.speculative == SpeculativeBackend::DFlash2;
        const bool expect_split   = !options.speculative_enabled() || options.proposal_enabled();
        const bool expect_proposal_split = options.proposal_enabled();
        check_shard_shapes(parameters0, *model0, 0, expect_mtp, expect_split, expect_proposal_split,
                           expect_dflash2, options.vision);
        check_shard_shapes(parameters1, *model1, 1, expect_mtp, expect_split, expect_proposal_split,
                           expect_dflash2, options.vision);
        std::cout << path.filename().string() << ": TP-2 dual-shard load passed "
                  << "devices=" << dev0 << "," << dev1 << " layers="
                  << model0->weights().text.layers.size() << " shard_gate=[8704,5120] "
                  << "shard_down=[5120,8704] head=["
                  << (expect_split ? "124160" : "248320") << ",5120] "
                  << (expect_split ? "vocabulary-parallel" : "replicated") << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
