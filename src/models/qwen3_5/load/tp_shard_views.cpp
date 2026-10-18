#include "models/qwen3_5/load/tp_shard_views.h"

#include "core/tp/weight_splitter.h"

#include <map>
#include <optional>
#include <stdexcept>

namespace ninfer::models::qwen3_5::loading {
namespace {

using artifact::ObjectHandle;
using tp::RowPart;
using tp::TPSplitSpec;
using tp::WeightSplitKind;

struct ObjectShape {
    std::uint64_t n = 0;
    std::uint64_t k = 0;
};

ObjectShape object_shape(const artifact::Directory& directory, ObjectHandle handle) {
    const auto& object = directory.object(handle);
    const auto* tensor = std::get_if<artifact::TensorObject>(&object);
    if (tensor == nullptr || tensor->shape.size() != 2) {
        throw std::invalid_argument("shard view: weight object is not a matrix");
    }
    return {tensor->shape[0], tensor->shape[1]};
}

} // namespace

std::vector<BoundWeight> shard_views(std::span<const PendingWeight> pending,
                                     const artifact::Directory& directory,
                                     const artifact::MaterializedArtifact& shard_backing,
                                     const TPSplitSpec& spec, int shard) {
    if (shard != 0 && shard != 1) { throw std::invalid_argument("shard must be 0 or 1"); }
    // Auxiliary input gathers ('input_columns') index the full-width activation a mixer projects,
    // not a shard's own half of it: a matrix whose stored column order is its own (a GGUF block
    // checkpoint's tiled value heads) pairs its columns with heads from anywhere in the pair. Such
    // a matrix is therefore row-parallel - the shard holds half of the stored columns, and the same
    // half of the gather travels with them, so column c of the half is the parent's column
    // shard * k/2 + c and the gather still indexes the activation the pair assembles. A complete or
    // column-parallel placement instead would pair a shard's columns with an activation that does
    // not cover the gather's domain, or leave the shard a whole product its all-reduce would
    // double.
    //
    // A component placed on one shard alone runs its whole mixer there, so its activation already is
    // the gather's domain and its auxiliary stays whole. One auxiliary shared by the two placements
    // has no single answer and is refused.
    std::map<std::size_t, bool> gather_is_halved;
    for (std::size_t index = 0; index < pending.size(); ++index) {
        const auto& item = pending[index];
        std::optional<WeightId> gathered;
        for (const auto& use : item.uses) {
            if (use.input_columns) {
                gathered = use.input_columns;
                break;
            }
        }
        if (!gathered) { continue; }
        if (gathered->index >= pending.size()) {
            throw std::invalid_argument("shard view: " + item.reference.name +
                                        " names an input gather outside the model");
        }
        const auto& gather = pending[gathered->index];
        if (gather.reference.shape.size() != 1 || item.reference.shape.size() != 2 ||
            gather.reference.shape[0] != item.reference.shape[1]) {
            throw std::invalid_argument("shard view: " + item.reference.name +
                                        " must gather a [K] vector over its input columns");
        }
        bool halved = false;
        for (const auto& part : item.reference.binding.parts) {
            if (!spec.on_shard(part.object, 0) || !spec.on_shard(part.object, 1)) { continue; }
            const auto* object = spec.find(part.object);
            const WeightSplitKind kind = object ? object->kind : WeightSplitKind::Replicated;
            if (kind != WeightSplitKind::RowParallel) {
                throw std::invalid_argument(
                    "tensor parallel split: " + item.reference.name +
                    " gathers its input columns, which only a row-parallel (K) split pairs with "
                    "the full-width activation the pair assembles");
            }
            halved = true;
        }
        const auto known = gather_is_halved.find(gathered->index);
        if (known != gather_is_halved.end() && known->second != halved) {
            throw std::invalid_argument("shard view: " + item.reference.name +
                                        " shares an input gather with a differently placed weight");
        }
        gather_is_halved[gathered->index] = halved;
    }
    std::vector<BoundWeight> out;
    out.reserve(pending.size());
    for (std::size_t index = 0; index < pending.size(); ++index) {
        const auto& item = pending[index];
        if (!artifact::device_readable(item.reference.residency)) {
            throw std::invalid_argument(
                "shard view: a shard weight must be device-readable (device-resident or host-mapped)");
        }
        // Shard-local components (the MTP layer on shard 0, the Vision tower on shard 1) are absent
        // from the other shard. Keep the entry so WeightId indices stay aligned with the parameter
        // array, but leave the view empty: consumers test Model::has_weight before dereferencing it,
        // and a weight that mixes shard-local and shared objects would be a split-spec bug.
        std::size_t present_parts = 0;
        for (const auto& part : item.reference.binding.parts) {
            present_parts += spec.on_shard(part.object, shard) ? 1U : 0U;
        }
        if (present_parts != item.reference.binding.parts.size()) {
            if (present_parts != 0) {
                throw std::invalid_argument(
                    "shard view: weight mixes objects placed on different shards");
            }
            BoundWeight absent;
            absent.name           = item.reference.name;
            absent.source_objects = item.source_objects;
            absent.uses           = item.uses;
            out.push_back(std::move(absent));
            continue;
        }
        BoundWeight bw;
        bw.name           = item.reference.name;
        bw.source_objects = item.source_objects;
        bw.uses           = item.uses;
        WeightView view;
        view.shape = item.reference.shape;
        const auto gather = gather_is_halved.find(index);
        if (gather != gather_is_halved.end() && gather->second) {
            // This shard's half of the gather, in step with the half of the matrix's stored columns
            // it holds. The auxiliary itself stays whole on the device, so the half is a window of
            // the parent's [K] vector.
            if (item.reference.binding.parts.size() != 1 || item.reference.shape[0] % 2 != 0) {
                throw std::invalid_argument(
                    "shard view: " + item.reference.name +
                    " must be an even [K] gather over one object");
            }
            const auto& part         = item.reference.binding.parts.front();
            const std::uint64_t half = item.reference.shape[0] / 2;
            WeightRegion region;
            region.parent = &shard_backing.device_parent(part.object);
            region.begin  = part.begin + static_cast<std::uint64_t>(shard) * half;
            region.end    = region.begin + half;
            view.shape[0] = half;
            view.parts.push_back(region);
            bw.view = std::move(view);
            out.push_back(std::move(bw));
            continue;
        }
        for (const auto& part : item.reference.binding.parts) {
            const auto* split = spec.find(part.object);
            const WeightSplitKind kind = split ? split->kind : WeightSplitKind::Replicated;
            WeightRegion shard_part;
            shard_part.parent = &shard_backing.device_parent(part.object);
            if (kind == WeightSplitKind::Replicated) {
                // Vectors (norms, biases) and full matrices are replicated verbatim; the shard
                // keeps the parent's begin/end and the reference shape, so no matrix shape query
                // is needed (1-D objects would otherwise fail the matrix check).
                shard_part.begin = part.begin;
                shard_part.end   = part.end;
            } else {
                const auto [n, k] = object_shape(directory, part.object);
                if (kind == WeightSplitKind::GatherRows) {
                if (!split || split->parts.empty()) {
                    throw std::invalid_argument("shard view: GatherRows split has no parts");
                }
                // Identify the row block this part addresses: begin == row_begin * k.
                const std::uint64_t r0 = part.begin / k;
                const std::uint64_t r1 = part.end / k;
                if (part.begin % k != 0 || part.end % k != 0 || r0 >= r1) {
                    throw std::invalid_argument("shard view: GatherRows part is not row-aligned");
                }
                std::uint64_t cum = 0;
                bool found = false;
                std::uint64_t block = 0;
                for (const auto& p : split->parts) {
                    const std::uint64_t pb = static_cast<std::uint64_t>(p.row_begin);
                    const std::uint64_t pc = static_cast<std::uint64_t>(p.row_count);
                    if (pb == r0 && pb + pc == r1) {
                        block = pc;
                        found = true;
                        break;
                    }
                    if (pb >= r0) { break; }
                    cum += pc / 2;
                }
                if (!found) {
                    throw std::invalid_argument("shard view: GatherRows part not in split");
                }
                // The shard payload concatenates, per part, this shard's half of the part's rows
                // (the gather already selected the shard's rows), so the block's offset within the
                // shard's row space is just the cumulative half-rows of the prior parts - identical
                // for both shards. Adding shard * (block / 2) would push the later block past the
                // shard's row count.
                const std::uint64_t row_begin = cum;
                const std::uint64_t row_count = block / 2;
                shard_part.begin = row_begin * k;
                shard_part.end   = (row_begin + row_count) * k;
                view.shape[0]    = row_count;
                view.shape[1]    = k;
            } else if (kind == WeightSplitKind::RowParallel) {
                if (part.begin != 0 || part.end != n * k) {
                    throw std::invalid_argument(
                        "shard view: RowParallel requires a whole-object part");
                }
                shard_part.begin = 0;
                shard_part.end   = n * (k / 2);
                view.shape[0]    = n;
                view.shape[1]    = k / 2;
            } else if (kind == WeightSplitKind::GatherCols) {
                // Column gather (GDN causal conv channels): the parent's columns are tiled into
                // channel blocks, each halved and concatenated, so the shard has n rows and
                // k/2 columns. The conv is a single whole-object part, so the shard region is the
                // first n * (k / 2) BF16 elements (row-major).
                if (part.begin != 0 || part.end != n * k) {
                    throw std::invalid_argument(
                        "shard view: GatherCols requires a whole-object part");
                }
                shard_part.begin = 0;
                shard_part.end   = n * (k / 2);
                view.shape[0]    = n;
                view.shape[1]    = k / 2;
            } else {
                // ColumnParallel: whole-object row split into two equal halves. The materialized
                // shard payload already holds exactly this shard's rows, so its view covers that
                // payload from the start - the same shard-local convention as GatherRows/GatherCols
                // (the parent geometry here is the shard's own n/2 x k slice).
                if (part.begin != 0 || part.end != n * k) {
                    throw std::invalid_argument(
                        "shard view: ColumnParallel requires a whole-object part");
                }
                shard_part.begin = 0;
                shard_part.end   = (n / 2) * k;
                view.shape[0]    = n / 2;
                view.shape[1]    = k;
            }
            }
            view.parts.push_back(shard_part);
        }
        bw.view = std::move(view);
        out.push_back(std::move(bw));
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
