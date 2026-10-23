#pragma once

// Prefix-reuse bookkeeping: the anchors a conversation owns, the decision one incoming prompt gets,
// and the single commit rule. No CUDA, no HTTP. This is the layer whose gaps turn into a full
// re-prefill, so it stays small enough to unit-test without loading a model.

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace ninfer::runtime {

// Why an anchor exists. The provenance is a diagnostic label; the policy is what the budget acts on.
enum class AnchorProvenance : std::uint8_t {
    PromptEnd,   // a completed prefill: the exact position the next turn continues from
    Divergence,  // where the incoming prompt parted from the history this request paid for
    Block,       // the end of a stable block
    Grid,        // the fixed-stride sample over committed history
    Generated,   // the fixed-stride sample over tokens this engine generated
};

enum class AnchorPolicy : std::uint8_t {
    Pinned,  // evicted only by an explicit replacement
    Lru,
};

enum class AnchorStorage : std::uint8_t {
    Device,  // a device state image bound to one lane: the zero-copy path
    Host,    // a host state image: restorable onto any lane
};

// Which KV the anchor depends on. The two stores shorten at different moments: a divergence trims
// the device pages, while the host slab still covers the positions they used to hold.
enum class AnchorKvSource : std::uint8_t {
    Device,
    HostSlab,
};

// A host anchor is restorable onto whichever lane runs the request.
inline constexpr std::int32_t kAnyLane = -1;
inline constexpr std::int32_t kNoSlot  = -1;

struct Anchor {
    std::uint32_t position      = 0;  // the state after [0, position)
    std::uint32_t draft_extent  = 0;  // DFlash2 draft frontier; 0 when the route has no draft ring
    AnchorProvenance provenance = AnchorProvenance::Grid;
    AnchorPolicy policy         = AnchorPolicy::Lru;
    AnchorStorage storage       = AnchorStorage::Host;
    AnchorKvSource kv_source    = AnchorKvSource::Device;
    std::int32_t lane           = kAnyLane;
    std::int32_t slot           = kNoSlot;  // host image slot; kNoSlot for a device anchor
};

// Every decision names why it did or did not reuse. A free-text log line cannot be asserted on, so
// the reason is an enumeration and the trace renders it.
//
// There is deliberately no "candidate sits past the end of the prompt" reason: the candidate is the
// deepest anchor at or before the shared prefix, and the shared prefix never exceeds the prompt.
enum class ReuseReason : std::uint8_t {
    Used,                // reuse == anchor->position
    NoAnchorAtOrBefore,  // the conversation holds no image at or before the shared prefix
};

struct DecideResult {
    std::uint32_t reuse  = 0;
    const Anchor* anchor = nullptr;
    ReuseReason reason   = ReuseReason::NoAnchorAtOrBefore;
};

// One conversation's anchors, sorted by position.
class AnchorTable {
public:
    // Largest anchor at or before `position`; null when the conversation has none that deep.
    [[nodiscard]] const Anchor* floor(std::uint32_t position) const;

    // Inserts, replacing whatever sits at the same position, and returns the replaced anchor so the
    // caller can release its slot.
    std::optional<Anchor> insert(Anchor anchor);

    [[nodiscard]] std::size_t size() const noexcept { return anchors_.size(); }
    [[nodiscard]] bool empty() const noexcept { return anchors_.empty(); }
    [[nodiscard]] const std::vector<Anchor>& all() const noexcept { return anchors_; }

    void clear() noexcept { anchors_.clear(); }

    // Removes and returns the anchor at `position`, if any.
    std::optional<Anchor> erase_at(std::uint32_t position);

private:
    std::vector<Anchor> anchors_;
};

struct ReuseBudget {
    // The coverage contract. One stride is the worst-case re-prefill - 8,192 tokens is about 4.9 s at
    // the measured 1,687 tok/s - and it does not follow the image budget.
    std::uint32_t stride = 8192;
    // Host state images this route may hold. 0 disables reuse.
    std::uint32_t capacity = 20;
};

// A conversation's canonical token stream plus the images it owns.
//
// `tokens` is the canonical stream: once a turn has been spliced to the engine's own bytes, the log
// keeps those bytes, so every position below is in one coordinate space.
struct ConversationLog {
    std::uint64_t session = 0;
    std::vector<TokenId> tokens;
    // Identity of the media items in order. Token equality alone cannot tell two images apart.
    std::vector<std::uint64_t> media;
    // Everything an anchor may name. Registration is what bounds this table: an anchor is offered
    // only when its own store really holds that position, so the table can never name a position a
    // store does not cover, and there is no post-hoc trim that could delete one store's anchors
    // because another store is shorter.
    AnchorTable anchors;
};

// Owns the conversations and the host-image budget. One instance per engine core.
class ReuseLedger {
public:
    explicit ReuseLedger(ReuseBudget budget);

    [[nodiscard]] std::uint32_t stride() const noexcept { return budget_.stride; }
    [[nodiscard]] std::uint32_t capacity() const noexcept { return budget_.capacity; }
    [[nodiscard]] std::uint32_t free_slots() const noexcept;

    // Creates the conversation on first use. The reference stays valid across later insertions.
    ConversationLog& conversation(std::uint64_t session);
    // Drops a conversation and releases every slot it owned.
    void forget(std::uint64_t session);
    [[nodiscard]] std::size_t conversation_count() const noexcept { return conversations_.size(); }

    struct AnchorRequest {
        std::uint32_t position     = 0;
        std::uint32_t draft_extent = 0;
        AnchorProvenance provenance = AnchorProvenance::Grid;
        AnchorPolicy policy         = AnchorPolicy::Pinned;
        AnchorStorage storage       = AnchorStorage::Host;
        AnchorKvSource kv_source    = AnchorKvSource::Device;
        std::int32_t lane           = kAnyLane;
        // A host anchor's image slot. kNoSlot lets the ledger allocate one from its budget; a
        // non-negative value names a slot the caller already owns (the host ring today), which the
        // ledger then only tracks. Two allocators over one pool would be one too many.
        std::int32_t slot = kNoSlot;
    };

    struct RecordResult {
        bool recorded = false;
        // The pool held nothing evictable: the caller carries on without this anchor. Degrading is
        // reported, never silent, and never a reason to drop a Pinned anchor.
        bool degraded = false;
        std::int32_t slot = kNoSlot;
    };
    // Records one anchor, replacing the conversation's previous anchor of the same provenance
    // position. A PromptEnd replaces the conversation's previous PromptEnd wherever it sat.
    RecordResult record(ConversationLog& conversation, const AnchorRequest& request);

    // The one decision: the deepest anchor at or before the shared prefix.
    //
    // The returned `anchor` points into the conversation's table and stays valid until that table is
    // modified. `incoming` must already be in the conversation's coordinate space.
    [[nodiscard]] DecideResult decide(const ConversationLog& conversation,
                                      std::span<const TokenId> incoming) const;

    // Reserves a host image slot for a caller that is about to write the image itself, and keeps
    // ownership of it: pass the result back as `AnchorRequest::slot`, and replacing or evicting
    // that anchor returns the slot to the budget. Reserve before the copy is enqueued, because a
    // full pool evicts here — the evicted anchor's image is the one the copy is about to overwrite.
    // kNoSlot means the pool holds nothing evictable; the caller then skips the capture.
    [[nodiscard]] std::int32_t reserve_image_slot();

private:
    [[nodiscard]] std::int32_t allocate_slot();
    // Registers `session` as a holder of the image in `slot`. Idempotent: a conversation that names
    // one image twice holds it once, and re-registering the same image after a scan rebuild is
    // free.
    void hold_slot(std::size_t slot, std::uint64_t session);
    // Removes one holder; the slot itself is only freed when the last one goes.
    void drop_hold(std::size_t slot, std::uint64_t session);
    // Returns the slot to the budget if this ledger owns it and nobody holds it any more. An image
    // another conversation still names is not the caller's to free.
    void release_slot(std::int32_t slot);
    // Frees unconditionally; only called once the holder set is empty.
    void free_slot(std::size_t slot);
    // Evicts the farthest Lru anchor this ledger may take; false when none is evictable.
    bool evict_one();

    ReuseBudget budget_;
    std::vector<bool> used_slots_;
    // Slots this ledger allocated, as opposed to ones a caller owns. Only these come back.
    std::vector<bool> owned_slots_;
    // Per slot, every conversation whose anchor names that image. An image lives while any of them
    // still names it: ownership is a reference count, not a single owner, because a conversation
    // that wrote the image can leave while another one is still restoring from it (the lane that
    // wrote it is not necessarily the lane that reads it next).
    std::vector<std::vector<std::uint64_t>> holders_;
    std::unordered_map<std::uint64_t, ConversationLog> conversations_;
};

} // namespace ninfer::runtime
