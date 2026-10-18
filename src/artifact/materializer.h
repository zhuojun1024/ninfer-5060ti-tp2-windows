#pragma once

#include "artifact/schema.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer/types.h"

#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::tp {
struct TPSplitSpec;
}

namespace ninfer::artifact {

class Reader;

struct DevicePlacement {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 256;
};

struct HostPlacement {
    ObjectHandle object;
    // Already-read resources move into final storage without invalidating their byte views.
    std::vector<std::byte> data;
};

// cudaHostAlloc returns page-aligned memory, so a mapped host weight can satisfy any layout
// alignment up to a page and nothing beyond it.
inline constexpr std::uint64_t kMappedHostAlignmentBytes = 4096;

// A weight whose bytes stay in host memory but remain device-readable (one pinned, UVA-mapped
// allocation per shard). The model's kernels read it over PCIe instead of from a VRAM copy, so it
// costs no device capacity and no H2D upload.
struct MappedPlacement {
    ObjectHandle object;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 256;
};

struct MaterializationPlan {
    const Reader* source                = nullptr;
    std::size_t object_count            = 0;
    std::uint64_t device_capacity_bytes = 0;
    std::uint64_t prior_read_bytes      = 0;
    std::uint64_t owned_value_bytes     = 0;
    std::vector<DevicePlacement> device_objects;
    std::vector<HostPlacement> host_objects;
    std::vector<MappedPlacement> mapped_objects;
};

struct MaterializationStats {
    std::uint64_t file_bytes = 0; // Declared container file set, including framing.
    std::uint64_t read_bytes = 0; // Actual payload reads, including direct-I/O alignment.
    std::uint64_t h2d_bytes  = 0;
    std::uint64_t device_capacity_bytes = 0;
    std::uint64_t retained_host_bytes   = 0;
    std::uint64_t owned_value_bytes     = 0;
    std::uint64_t peak_staging_bytes    = 0;
    // Weight bytes that live in mapped host memory instead of VRAM. They are device-readable, so
    // they are not counted in device_capacity_bytes.
    std::uint64_t host_mapped_bytes     = 0;
    std::size_t device_object_count     = 0;
    std::size_t host_object_count       = 0;
    double upload_seconds               = 0;
};

class MaterializedArtifact {
public:
    struct ObjectStorage {
        std::optional<WeightParent> device;
        std::optional<WeightParent> host;
        std::vector<std::byte> host_data;
    };

    MaterializedArtifact()                                           = default;
    ~MaterializedArtifact()                                          = default;
    MaterializedArtifact(MaterializedArtifact&&) noexcept            = default;
    MaterializedArtifact& operator=(MaterializedArtifact&&) noexcept = default;
    MaterializedArtifact(const MaterializedArtifact&)                = delete;
    MaterializedArtifact& operator=(const MaterializedArtifact&)     = delete;

    [[nodiscard]] const WeightParent& device_parent(ObjectHandle handle) const;
    [[nodiscard]] const WeightParent& host_parent(ObjectHandle handle) const;
    [[nodiscard]] std::span<const std::byte> host_bytes(ObjectHandle handle) const;
    [[nodiscard]] bool has_device(ObjectHandle handle) const noexcept;

    [[nodiscard]] const MaterializationStats& stats() const noexcept { return stats_; }

    // TP-2 construction access (used by ninfer::tp::materialize_tp2).
    [[nodiscard]] std::vector<ObjectStorage>& tp_objects() noexcept { return objects_; }
    [[nodiscard]] DeviceArena& tp_arena() noexcept { return *arena_; }
    [[nodiscard]] MaterializationStats& tp_stats() noexcept { return stats_; }
    // Host-mapped weight bytes: one pinned, device-readable mapping per (object, shard). The weight
    // views point into them, so the artifact owns them for the model's lifetime.
    [[nodiscard]] std::vector<std::unique_ptr<PinnedHostBuffer>>& tp_mapped() noexcept {
        return mapped_;
    }
    void tp_init(std::size_t object_count, std::uint64_t capacity_bytes) {
        objects_.resize(object_count);
        arena_ = std::make_unique<DeviceArena>(static_cast<std::size_t>(capacity_bytes));
        stats_.device_capacity_bytes = capacity_bytes;
    }

private:
    friend MaterializedArtifact materialize(const Reader&, MaterializationPlan&&, DeviceContext&,
                                            const StartupObserver*);

    std::unique_ptr<DeviceArena> arena_;
    std::vector<ObjectStorage> objects_;
    std::vector<std::unique_ptr<PinnedHostBuffer>> mapped_;
    MaterializationStats stats_;
};

[[nodiscard]] MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                               DeviceContext& device,
                                               const StartupObserver* startup_observer = nullptr);

} // namespace ninfer::artifact
