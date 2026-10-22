#include "runtime/engine/prefix_reuse.h"

#include <algorithm>
#include <limits>

namespace ninfer::runtime {

const Anchor* AnchorTable::floor(std::uint32_t position) const {
    // Anchors are sorted, so the candidate is the last entry at or before `position`.
    const auto next = std::upper_bound(anchors_.begin(), anchors_.end(), position,
                                       [](std::uint32_t value, const Anchor& anchor) {
                                           return value < anchor.position;
                                       });
    if (next == anchors_.begin()) { return nullptr; }
    return &*(next - 1);
}

std::optional<Anchor> AnchorTable::insert(Anchor anchor) {
    const auto at = std::lower_bound(anchors_.begin(), anchors_.end(), anchor.position,
                                     [](const Anchor& candidate, std::uint32_t value) {
                                         return candidate.position < value;
                                     });
    if (at != anchors_.end() && at->position == anchor.position) {
        const Anchor replaced = *at;
        *at                   = anchor;
        return replaced;
    }
    anchors_.insert(at, anchor);
    return std::nullopt;
}

std::optional<Anchor> AnchorTable::erase_at(std::uint32_t position) {
    const auto at = std::lower_bound(anchors_.begin(), anchors_.end(), position,
                                     [](const Anchor& candidate, std::uint32_t value) {
                                         return candidate.position < value;
                                     });
    if (at == anchors_.end() || at->position != position) { return std::nullopt; }
    const Anchor erased = *at;
    anchors_.erase(at);
    return erased;
}

ReuseLedger::ReuseLedger(ReuseBudget budget) : budget_(budget) {
    used_slots_.assign(budget_.capacity, false);
    owned_slots_.assign(budget_.capacity, false);
}

std::uint32_t ReuseLedger::free_slots() const noexcept {
    return static_cast<std::uint32_t>(std::count(used_slots_.begin(), used_slots_.end(), false));
}

ConversationLog& ReuseLedger::conversation(std::uint64_t session) {
    ConversationLog& log = conversations_[session];
    log.session          = session;
    return log;
}

void ReuseLedger::forget(std::uint64_t session) {
    const auto found = conversations_.find(session);
    if (found == conversations_.end()) { return; }
    for (const Anchor& anchor : found->second.anchors.all()) { release_slot(anchor.slot); }
    conversations_.erase(found);
}

std::int32_t ReuseLedger::allocate_slot() {
    for (std::size_t slot = 0; slot < used_slots_.size(); ++slot) {
        if (used_slots_[slot]) { continue; }
        used_slots_[slot]  = true;
        owned_slots_[slot] = true;
        return static_cast<std::int32_t>(slot);
    }
    return kNoSlot;
}

void ReuseLedger::release_slot(std::int32_t slot) {
    if (slot < 0 || static_cast<std::size_t>(slot) >= used_slots_.size()) { return; }
    const std::size_t index = static_cast<std::size_t>(slot);
    // A caller-owned slot is not the ledger's to free: clearing it would hand the same image to a
    // second anchor while the ring still holds it.
    if (!owned_slots_[index]) { return; }
    used_slots_[index] = false;
    owned_slots_[index] = false;
}

bool ReuseLedger::evict_one() {
    // The position farthest from the frontier loses first, i.e. the smallest one. Sampling and
    // eviction run in the same direction, so whatever survives is a contiguous recent window: there
    // is never a hole in the middle of the coverage.
    ConversationLog* victim = nullptr;
    std::uint32_t victim_at = 0;
    bool have_victim        = false;
    for (auto& entry : conversations_) {
        for (const Anchor& anchor : entry.second.anchors.all()) {
            if (anchor.policy != AnchorPolicy::Lru) { continue; }
            if (have_victim && anchor.position >= victim_at) { continue; }
            victim      = &entry.second;
            victim_at   = anchor.position;
            have_victim = true;
        }
    }
    if (!have_victim) { return false; }
    const std::optional<Anchor> erased = victim->anchors.erase_at(victim_at);
    if (erased) { release_slot(erased->slot); }
    return true;
}

ReuseLedger::RecordResult ReuseLedger::record(ConversationLog& conversation,
                                              const AnchorRequest& request) {
    RecordResult result;
    // Only a host anchor the ledger has to allocate needs a slot from the budget. A device anchor
    // costs nothing, and a caller-owned image is already in place.
    const bool needs_allocation = request.storage == AnchorStorage::Host && request.slot == kNoSlot;
    if (needs_allocation && budget_.capacity == 0) { return result; }

    // A PromptEnd is the position the next turn continues from, so it replaces the previous one
    // wherever that sat: two of them would be two claims about the same boundary.
    if (request.provenance == AnchorProvenance::PromptEnd) {
        std::vector<std::uint32_t> stale;
        for (const Anchor& anchor : conversation.anchors.all()) {
            if (anchor.provenance == AnchorProvenance::PromptEnd) { stale.push_back(anchor.position); }
        }
        for (const std::uint32_t position : stale) {
            const std::optional<Anchor> erased = conversation.anchors.erase_at(position);
            if (erased) { release_slot(erased->slot); }
        }
    }

    Anchor anchor;
    anchor.position     = request.position;
    anchor.draft_extent = request.draft_extent;
    anchor.provenance   = request.provenance;
    anchor.policy       = request.policy;
    anchor.storage      = request.storage;
    anchor.kv_source    = request.kv_source;
    anchor.lane         = request.storage == AnchorStorage::Device ? request.lane : kAnyLane;

    if (anchor.storage == AnchorStorage::Host) {
        if (!needs_allocation) {
            // The caller owns this image; the ledger only tracks where it lives.
            anchor.slot = request.slot;
        } else {
            std::int32_t slot = allocate_slot();
            if (slot == kNoSlot) {
                if (!evict_one()) {
                    // Every image is Pinned. Refusing the new anchor is the honest outcome: clearing a
                    // Pinned image to make room would break the guarantee it exists for.
                    result.degraded = true;
                    return result;
                }
                slot = allocate_slot();
                if (slot == kNoSlot) {
                    result.degraded = true;
                    return result;
                }
            }
            anchor.slot = slot;
        }
    }

    const std::optional<Anchor> replaced = conversation.anchors.insert(anchor);
    if (replaced) { release_slot(replaced->slot); }
    result.recorded = true;
    result.slot     = anchor.slot;
    return result;
}

DecideResult ReuseLedger::decide(const ConversationLog& conversation,
                                 std::span<const TokenId> incoming) const {
    DecideResult result;
    std::size_t shared = 0;
    const std::size_t common =
        std::min(conversation.tokens.size(), static_cast<std::size_t>(incoming.size()));
    while (shared < common && conversation.tokens[shared] == incoming[shared]) { ++shared; }
    if (shared == 0) { return result; }

    const Anchor* candidate = conversation.anchors.floor(static_cast<std::uint32_t>(shared));
    if (candidate == nullptr) { return result; }
    result.reuse  = candidate->position;
    result.anchor = candidate;
    result.reason = ReuseReason::Used;
    return result;
}

std::uint32_t ReuseLedger::extent_for(const ConversationLog& conversation,
                                      AnchorKvSource source) const {
    return source == AnchorKvSource::HostSlab ? conversation.host_extent
                                              : conversation.device_extent;
}

ReuseLedger::CommitResult ReuseLedger::commit(ConversationLog& conversation,
                                              std::span<const TokenId> history,
                                              std::uint32_t device_extent,
                                              std::uint32_t host_extent) {
    conversation.tokens.assign(history.begin(), history.end());
    conversation.device_extent = device_extent;
    conversation.host_extent   = host_extent;

    // The one trim rule, applied per store. A divergence shortens the device extent and must not take
    // the host-backed anchors with it: those are the only images deep history can still use.
    std::vector<std::uint32_t> stale;
    for (const Anchor& anchor : conversation.anchors.all()) {
        if (anchor.position > extent_for(conversation, anchor.kv_source)) {
            stale.push_back(anchor.position);
        }
    }
    CommitResult result;
    for (const std::uint32_t position : stale) {
        const std::optional<Anchor> erased = conversation.anchors.erase_at(position);
        if (!erased) { continue; }
        release_slot(erased->slot);
        ++result.trimmed;
    }
    return result;
}

} // namespace ninfer::runtime
