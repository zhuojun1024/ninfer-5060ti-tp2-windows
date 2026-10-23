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
    owner_sessions_.assign(budget_.capacity, 0);
    borrowers_.resize(budget_.capacity);
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
    // Only the images this conversation owns come back; a borrowed one belongs to its owner, which
    // releases it (and this anchor with it).
    for (const Anchor& anchor : found->second.anchors.all()) {
        if (!anchor.borrowed) { release_slot(anchor.slot); }
    }
    conversations_.erase(found);
    // Its borrowed claims on other conversations' images go with it.
    for (std::vector<Borrower>& claims : borrowers_) {
        claims.erase(
            std::remove_if(claims.begin(), claims.end(),
                           [session](const Borrower& claim) { return claim.session == session; }),
            claims.end());
    }
}

std::int32_t ReuseLedger::reserve_image_slot() {
    std::int32_t slot = allocate_slot();
    if (slot != kNoSlot) { return slot; }
    // Every image is held by a Pinned anchor: refusing is the honest outcome, because clearing a
    // pinned image to make room would break the guarantee it exists for.
    if (!evict_one()) { return kNoSlot; }
    return allocate_slot();
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
    used_slots_[index]     = false;
    owned_slots_[index] = false;
    owner_sessions_[index] = 0;
    // The image is about to be overwritten, so every conversation that was reading it through a
    // borrowed anchor loses that anchor here rather than restoring a stale buffer later.
    release_borrowers(slot);
}

void ReuseLedger::release_borrowers(std::int32_t slot) {
    if (slot < 0 || static_cast<std::size_t>(slot) >= borrowers_.size()) { return; }
    std::vector<Borrower>& claims = borrowers_[static_cast<std::size_t>(slot)];
    for (const Borrower& claim : claims) {
        const auto found = conversations_.find(claim.session);
        if (found == conversations_.end()) { continue; }
        found->second.anchors.erase_at(claim.position);
    }
    claims.clear();
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
            // A borrower names somebody else's image, so evicting it frees nothing; the owner is
            // the one that can make room.
            if (anchor.borrowed) { continue; }
            if (have_victim && anchor.position >= victim_at) { continue; }
            victim      = &entry.second;
            victim_at   = anchor.position;
            have_victim = true;
        }
    }
    if (!have_victim) { return false; }
    const std::optional<Anchor> erased = victim->anchors.erase_at(victim_at);
    if (erased && !erased->borrowed) { release_slot(erased->slot); }
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
            if (erased && !erased->borrowed) { release_slot(erased->slot); }
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
            // A slot that came from `reserve_image_slot` is already owned, and staying owned is
            // what returns it to the budget when this anchor is replaced or evicted. A slot the
            // ledger is not holding at all - an image the ring still carries after its anchor was
            // evicted - is claimed here, or a later image could be placed on top of a live anchor's
            // buffer. A fixed slot is held but never owned, so it is left alone.
            anchor.slot = request.slot;
            if (anchor.slot >= 0 && static_cast<std::size_t>(anchor.slot) < used_slots_.size()) {
                const std::size_t index = static_cast<std::size_t>(anchor.slot);
                if (!used_slots_[index]) {
                    used_slots_[index]     = true;
                    owned_slots_[index]    = true;
                    owner_sessions_[index] = conversation.session;
                } else if (owned_slots_[index] && owner_sessions_[index] == 0) {
                    // A slot the caller reserved through `reserve_image_slot` and has not claimed
                    // yet: recording the anchor is the claim.
                    owner_sessions_[index] = conversation.session;
                } else if (!owned_slots_[index] || owner_sessions_[index] != conversation.session) {
                    // Another conversation holds this image (or the ring keeps it for a fixed
                    // role): this anchor reads it and disappears with it. Two owners over one image
                    // would let one of them free a buffer the other is still restoring from.
                    anchor.borrowed = true;
                    borrowers_[index].push_back(Borrower{conversation.session, request.position});
                }
            }
        } else {
            const std::int32_t slot = reserve_image_slot();
            if (slot == kNoSlot) {
                // Every image is Pinned. Refusing the new anchor is the honest outcome: clearing a
                // Pinned image to make room would break the guarantee it exists for.
                result.degraded = true;
                return result;
            }
            anchor.slot = slot;
            owner_sessions_[static_cast<std::size_t>(slot)] = conversation.session;
        }
    }

    const std::optional<Anchor> replaced = conversation.anchors.insert(anchor);
    // Re-recording the slot that serves this very anchor must not free it: the image is still the
    // one the anchor names, and clearing it would hand the same buffer to a second anchor.
    if (replaced && !replaced->borrowed && replaced->slot != anchor.slot) {
        release_slot(replaced->slot);
    }
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
        if (!erased->borrowed) { release_slot(erased->slot); }
        ++result.trimmed;
    }
    return result;
}

} // namespace ninfer::runtime
