#pragma once

#include "core/dtype.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace ninfer {

struct DeviceSpan {
    void* data        = nullptr;
    std::size_t bytes = 0;
};

// Owning device allocation for long-lived buffers. DeviceArena remains the
// suballocation primitive for workspaces; this type owns exactly one cudaMalloc.
class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;
    explicit DeviceBuffer(std::size_t size_bytes);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    void fill(int byte_value = 0);
    // Completes the upload before returning. Callers must first order any prior device
    // accesses to the destination range.
    void copy_from_host(const void* source, std::size_t count, std::size_t byte_offset = 0);
    void copy_to_host(void* destination, std::size_t count, std::size_t byte_offset = 0) const;

    // Raw access is intentional: Tensor and Weight are non-owning views.
    void* p           = nullptr;
    std::size_t bytes = 0;

private:
    void require_range(std::size_t byte_offset, std::size_t count, const char* operation) const;
};

class DeviceArena {
public:
    class Scope {
    public:
        ~Scope() noexcept;

        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&&) = delete;

    private:
        friend class DeviceArena;

        explicit Scope(DeviceArena& arena) noexcept;

        DeviceArena* arena_       = nullptr;
        std::size_t saved_offset_ = 0;
    };

    explicit DeviceArena(std::size_t capacity_bytes);
    // Non-owning arena over an already allocated device region.
    explicit DeviceArena(DeviceSpan storage);
    ~DeviceArena();

    DeviceArena(const DeviceArena&)            = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    DeviceArena(DeviceArena&& other) noexcept;
    DeviceArena& operator=(DeviceArena&& other) noexcept;

    DeviceSpan alloc_bytes(std::size_t bytes, std::size_t align = 256);
    Tensor alloc(DType dtype, std::initializer_list<std::int32_t> shape, std::size_t align = 256);
    [[nodiscard]] Scope scope() noexcept;
    void reset() noexcept;
    // Rewinds the bump pointer to a watermark recorded earlier with used(). A caller whose own
    // allocations are dead (a speculative proposal chain, say) uses this to place the allocations
    // that follow at offsets that do not depend on how much the dead sequence consumed - which is
    // what a captured CUDA Graph bakes into its kernel arguments. The watermark must not lie past
    // the current pointer.
    void rewind(std::size_t watermark);

    void* base() const noexcept;
    std::size_t used() const noexcept;
    std::size_t capacity() const noexcept;
    std::size_t peak_used() const noexcept;
    void reset_peak() noexcept;

private:
    void* base_       = nullptr;
    std::size_t cap_  = 0;
    std::size_t off_  = 0;
    std::size_t peak_ = 0;
    bool owns_        = true;
};

class PinnedHostBuffer {
public:
    // A portable allocation is visible as pinned memory from every CUDA context, which a captured
    // sequence spanning two devices needs: cudaMallocHost alone pins the memory only for the context
    // that allocated it, and a memcpy node reading it from the other device would not be capturable.
    explicit PinnedHostBuffer(std::size_t size_bytes, bool portable = false);
    ~PinnedHostBuffer();

    PinnedHostBuffer(const PinnedHostBuffer&)            = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

    void* data() const noexcept;
    std::size_t size() const noexcept;
    // Device-visible address of the same bytes, for a weight the device reads over PCIe instead of
    // holding a VRAM copy (PinnedHostBuffer with portable=true, which is the mapped variant). The
    // query is a no-op under UVA - it returns the host pointer - but it is the documented way to
    // obtain that address and stays correct where a driver remaps host memory.
    void* device_data() const;

private:
    void* data_       = nullptr;
    std::size_t size_ = 0;
};

// Which memory the OS may evict under commit pressure decides whether a large host-side KV arena
// can exist at all: a pinned mapping is resident for the process lifetime and cudaMallocHost fails
// when the commit limit is tight, so host KV backing stays pageable by default and pins only on an
// explicit request.
enum class HostPinning { Pageable, PreferPinned };

// Contiguous host storage for state copied to and from the device. The pageable path is ordinary
// committed memory (evictable, no lock budget); the pinned path trades that for faster transfers
// and falls back to pageable when the allocation is refused.
class HostBuffer {
public:
    explicit HostBuffer(std::size_t size_bytes, HostPinning pinning = HostPinning::Pageable);
    ~HostBuffer();

    HostBuffer(const HostBuffer&)            = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    HostBuffer(HostBuffer&& other) noexcept;
    HostBuffer& operator=(HostBuffer&& other) noexcept;

    void* data() const noexcept;
    std::size_t size() const noexcept;
    bool pinned() const noexcept;

private:
    void* data_       = nullptr;
    std::size_t size_ = 0;
    bool pinned_      = false;
};

using WorkspaceArena = DeviceArena;

} // namespace ninfer
