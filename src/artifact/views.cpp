#include "artifact/views.h"

namespace ninfer::artifact {

WeightView bind_view(const ParameterReference& reference,
                     const MaterializedArtifact& materialized) {
    if (reference.residency == Residency::Values) {
        throw ArtifactError(reference.name + ": owning values do not have parent backing");
    }
    WeightView out;
    out.shape             = reference.shape;
    std::uint64_t covered = 0;
    for (const auto& part : reference.binding.parts) {
        // A mapped host weight is device-readable, so its view binds like a device-resident one;
        // only the address it points at lives in host memory.
        const auto& parent = device_readable(reference.residency)
                                 ? materialized.device_parent(part.object)
                                 : materialized.host_parent(part.object);
        if (part.begin >= part.end || part.end > parent.geometry.elements) {
            throw ArtifactError(reference.name + ": invalid materialized region");
        }
        covered = checked_add(covered, part.end - part.begin, reference.name);
        out.parts.push_back({&parent, part.begin, part.end});
    }
    if (covered != weight_element_count(out.shape)) {
        throw ArtifactError(reference.name + ": materialized coverage differs from logical shape");
    }
    return out;
}

} // namespace ninfer::artifact
