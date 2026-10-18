#pragma once

#include "artifact/materializer.h"
#include "artifact/reader.h"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::artifact {

// How a parameter's bytes reach the kernel. Device spends VRAM, Host keeps them on the host for a
// caller that reads them itself, and MappedHost keeps them in host memory while leaving them
// device-readable, so a weight the device reads rarely (the token embedding table) costs no VRAM.
enum class Residency { Device, Host, MappedHost, Values };

// Whether a residency's bytes are addressed from the device: either they sit in VRAM (Device) or in
// host memory mapped into the device address space (MappedHost). Everything that binds a weight
// view asks this, so a new residency cannot be device-readable in one place and not in another.
[[nodiscard]] constexpr bool device_readable(Residency residency) noexcept {
    return residency == Residency::Device || residency == Residency::MappedHost;
}

struct ParameterReference {
    std::string name;
    Shape shape;
    Binding binding;
    Residency residency = Residency::Device;
};

struct HostValues {
    QType format           = QType::FP32;
    std::uint64_t elements = 0;
    std::vector<std::byte> data;

    [[nodiscard]] float scalar_f32() const;
    [[nodiscard]] std::vector<std::int32_t> integers() const;
};

// Collects selected logical demands; neither physical object IDs nor whole-artifact profiles
// determine parameter shapes or native operation support.
class Binder {
public:
    explicit Binder(const Reader& reader);

    [[nodiscard]] ParameterReference parameter(std::string_view name, Shape shape,
                                               Residency residency = Residency::Device,
                                               std::optional<QType> exact_format = {});
    [[nodiscard]] ParameterReference binding(std::string name, const Binding& binding, Shape shape,
                                             Residency residency,
                                             std::optional<QType> exact_format = {});
    [[nodiscard]] const Use& use(std::string_view parameter, std::string_view input) const;
    [[nodiscard]] bool contains(std::string_view parameter) const;

    [[nodiscard]] const Reader& reader() const noexcept { return reader_; }

    void require_device(ObjectHandle object, std::uint64_t alignment = 256);
    // Demands the object's bytes in mapped host memory: no device capacity, no H2D upload, and a
    // device-visible address the kernels read through. Materialization gives each shard its own
    // mapping, so a split object is mapped per shard.
    void require_mapped_host(ObjectHandle object, std::uint64_t alignment = 256);
    [[nodiscard]] std::span<const std::byte> host_object(ObjectHandle object);
    [[nodiscard]] ObjectHandle resource(std::string_view component, std::string_view role);
    [[nodiscard]] HostValues values(const Binding& binding, std::optional<QType> format = {});
    [[nodiscard]] MaterializationPlan finish() &&;

private:
    struct Demand {
        bool device             = false;
        bool host               = false;
        bool mapped_host        = false;
        std::uint64_t alignment = 256;
        std::vector<std::byte> host_data;
    };

    const Reader& reader_;
    std::vector<Demand> demands_;
    std::uint64_t read_bytes_        = 0;
    std::uint64_t owned_value_bytes_ = 0;
};

} // namespace ninfer::artifact
