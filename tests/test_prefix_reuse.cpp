#include "runtime/engine/prefix_reuse.h"

#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace ninfer::runtime;
using ninfer::TokenId;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::vector<TokenId> make_tokens(std::size_t count, TokenId base = 100) {
    std::vector<TokenId> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        result[index] = base + static_cast<TokenId>(index);
    }
    return result;
}

ReuseLedger::AnchorRequest grid(std::uint32_t position) {
    ReuseLedger::AnchorRequest request;
    request.position   = position;
    request.provenance = AnchorProvenance::Grid;
    request.policy     = AnchorPolicy::Lru;
    return request;
}

ReuseLedger::AnchorRequest pinned(std::uint32_t position, AnchorProvenance provenance) {
    ReuseLedger::AnchorRequest request;
    request.position   = position;
    request.provenance = provenance;
    request.policy     = AnchorPolicy::Pinned;
    return request;
}

} // namespace

int main() {
    int failures = 0;

    // The table keeps one anchor per position and answers with the deepest one at or before a sort
    // key. Without that, every decision would need its own scan and could disagree with the trim.
    {
        AnchorTable table;
        failures += check(table.floor(1000) == nullptr, "an empty table offered an anchor");
        Anchor first;
        first.position = 100;
        Anchor second;
        second.position = 200;
        table.insert(first);
        table.insert(second);
        failures += check(table.floor(150) != nullptr && table.floor(150)->position == 100,
                          "floor did not return the deepest anchor at or before the key");
        failures += check(table.floor(99) == nullptr, "floor returned an anchor past the key");
        Anchor replacement;
        replacement.position = 200;
        replacement.slot     = 7;
        const std::optional<Anchor> replaced = table.insert(replacement);
        failures += check(replaced.has_value(), "re-inserting a position did not report the replaced anchor");
        failures += check(table.size() == 2, "re-inserting a position grew the table");
        failures += check(table.floor(200)->slot == 7, "the replacement anchor did not win");
    }

    // With no anchor the conversation offers nothing: this is the state that costs a full prefill.
    {
        ReuseLedger ledger(ReuseBudget{.stride = 8192, .capacity = 4});
        ConversationLog& conversation = ledger.conversation(1);
        const std::vector<TokenId> history = make_tokens(64);
        conversation.tokens                = history;
        const DecideResult decision        = ledger.decide(conversation, history);
        failures += check(decision.reuse == 0 &&
                              decision.reason == ReuseReason::NoAnchorAtOrBefore,
                          "an empty anchor table reported a reuse");
    }

    // The decision takes the deepest anchor at or before the shared prefix, never one beyond it: the
    // state at a later position describes tokens this prompt no longer agrees with.
    {
        ReuseLedger ledger(ReuseBudget{.stride = 8192, .capacity = 8});
        ConversationLog& conversation = ledger.conversation(1);
        conversation.tokens           = make_tokens(20000);
        conversation.tokens.resize(20000);
        conversation.device_extent = 20000;
        conversation.host_extent   = 20000;
        failures += check(ledger.record(conversation, grid(8192)).recorded, "grid anchor was refused");
        failures += check(ledger.record(conversation, grid(16384)).recorded, "grid anchor was refused");

        std::vector<TokenId> incoming = make_tokens(12000);
        DecideResult decision         = ledger.decide(conversation, incoming);
        failures += check(decision.reuse == 8192 && decision.reason == ReuseReason::Used,
                          "the decision did not take the deepest anchor inside the shared prefix");

        // A prompt that ends before the only candidate cannot use it: the candidate search stops at
        // the shared prefix, so a later anchor is simply not a candidate.
        ReuseLedger short_ledger(ReuseBudget{.stride = 8192, .capacity = 4});
        ConversationLog& short_conversation = short_ledger.conversation(2);
        short_conversation.tokens           = make_tokens(9000);
        short_ledger.record(short_conversation, grid(8192));
        const std::vector<TokenId> short_prompt = make_tokens(4096);
        decision = short_ledger.decide(short_conversation, short_prompt);
        failures += check(decision.reuse == 0 &&
                              decision.reason == ReuseReason::NoAnchorAtOrBefore,
                          "an anchor past the end of the prompt was offered");

        // A prompt the engine has already processed in full is a legal reuse of the whole prompt:
        // the request becomes decode-only. This is the case a "beyond the prompt" guard would break.
        const std::vector<TokenId> whole_prompt = make_tokens(8192);
        decision = short_ledger.decide(short_conversation, whole_prompt);
        failures += check(decision.reuse == 8192 && decision.reason == ReuseReason::Used,
                          "an exactly matching prompt was not reused in full");
    }

    // A PromptEnd replaces the conversation's previous one wherever it sat: two of them would be two
    // claims about the position the next turn continues from.
    {
        ReuseLedger ledger(ReuseBudget{.stride = 8192, .capacity = 4});
        ConversationLog& conversation = ledger.conversation(1);
        ledger.record(conversation, pinned(100, AnchorProvenance::PromptEnd));
        failures += check(ledger.free_slots() == 3, "the first PromptEnd did not take a slot");
        ledger.record(conversation, pinned(500, AnchorProvenance::PromptEnd));
        failures += check(conversation.anchors.size() == 1, "the previous PromptEnd survived");
        failures += check(ledger.free_slots() == 3, "replacing PromptEnd leaked its slot");
        failures += check(conversation.anchors.floor(900)->position == 500,
                          "the new PromptEnd did not replace the old position");
    }

    // The budget evicts the farthest Lru anchor, so whatever survives is a contiguous recent window.
    {
        ReuseLedger ledger(ReuseBudget{.stride = 8192, .capacity = 3});
        ConversationLog& conversation = ledger.conversation(1);
        conversation.device_extent    = 100000;
        for (const std::uint32_t position : {8192u, 16384u, 24576u, 32768u}) {
            failures += check(ledger.record(conversation, grid(position)).recorded,
                              "a grid anchor was refused while the pool had room to evict");
        }
        failures += check(conversation.anchors.size() == 3, "the budget did not bound the table");
        failures += check(conversation.anchors.floor(8192) == nullptr,
                          "the farthest anchor was not the one evicted");
        failures += check(conversation.anchors.floor(32768)->position == 32768,
                          "the newest anchor did not survive");
        failures += check(ledger.free_slots() == 0, "eviction did not stay inside the budget");
    }

    // A Pinned anchor is never sacrificed for a new one: the pool reports the refusal instead.
    {
        ReuseLedger ledger(ReuseBudget{.stride = 8192, .capacity = 1});
        ConversationLog& conversation = ledger.conversation(1);
        failures += check(ledger.record(conversation, pinned(100, AnchorProvenance::PromptEnd)).recorded,
                          "the Pinned anchor was refused in an empty pool");
        const ReuseLedger::RecordResult refused = ledger.record(conversation, grid(200));
        failures += check(!refused.recorded && refused.degraded,
                          "an exhausted pool silently dropped or accepted an anchor");
        failures += check(conversation.anchors.floor(1000)->position == 100,
                          "the Pinned anchor was evicted");
    }

    // The two stores trim independently: a divergence shortens the device pages, and the host-backed
    // anchor beyond that point is exactly the one deep history still needs.
    {
        ReuseLedger ledger(ReuseBudget{.stride = 8192, .capacity = 4});
        ConversationLog& conversation = ledger.conversation(1);

        ReuseLedger::AnchorRequest device_request = grid(1500);
        device_request.storage                     = AnchorStorage::Device;
        device_request.lane                        = 0;
        device_request.kv_source                   = AnchorKvSource::Device;
        ledger.record(conversation, device_request);

        ReuseLedger::AnchorRequest host_request = grid(1800);
        host_request.storage                     = AnchorStorage::Host;
        host_request.kv_source                   = AnchorKvSource::HostSlab;
        ledger.record(conversation, host_request);

        const std::vector<TokenId> history = make_tokens(4096);
        const ReuseLedger::CommitResult commit = ledger.commit(conversation, history, 1000, 2000);
        failures += check(commit.trimmed == 1, "the trim did not drop exactly the uncovered anchor");
        failures += check(conversation.anchors.floor(2000) != nullptr &&
                              conversation.anchors.floor(2000)->position == 1800,
                          "the host-slab anchor was trimmed with the device pages");
        failures += check(conversation.tokens.size() == history.size(),
                          "commit did not publish the canonical stream");

        // A prompt identical to the published history reuses the surviving anchor.
        const DecideResult decision = ledger.decide(conversation, history);
        failures += check(decision.reuse == 1800 && decision.reason == ReuseReason::Used,
                          "the surviving anchor was not offered to an identical prompt");
    }

    // Forgetting a conversation returns its images: a catalog reload must not leak the pool.
    {
        ReuseLedger ledger(ReuseBudget{.stride = 8192, .capacity = 2});
        ledger.record(ledger.conversation(7), grid(8192));
        failures += check(ledger.free_slots() == 1, "the anchor did not take a slot");
        ledger.forget(7);
        failures += check(ledger.free_slots() == 2, "forget did not release the conversation's slots");
        failures += check(ledger.conversation_count() == 0, "forget left the conversation behind");
    }

    // A slot the caller writes itself is reserved before the image exists and stays the ledger's to
    // reclaim: that is what lets the host ring hand its image pool to one allocator instead of two.
    {
        ReuseLedger ledger(ReuseBudget{.stride = 8192, .capacity = 3});
        ConversationLog& conversation = ledger.conversation(1);

        const std::int32_t reserved = ledger.reserve_image_slot();
        failures += check(reserved >= 0, "a free pool refused to reserve an image slot");
        failures += check(ledger.free_slots() == 2, "reserving an image slot did not take it");

        ReuseLedger::AnchorRequest request = grid(8192);
        request.slot                       = reserved;
        failures += check(ledger.record(conversation, request).slot == reserved,
                          "recording a reserved slot did not keep it");

        // Re-recording the same anchor must not free the slot it names.
        failures += check(ledger.record(conversation, request).recorded,
                          "re-recording a reserved slot was refused");
        failures +=
            check(ledger.free_slots() == 2, "re-recording an anchor freed the slot it still names");

        // A fresh anchor takes another slot; replacing the first anchor with a different one must
        // return the slot the first anchor held.
        const ReuseLedger::RecordResult second = ledger.record(conversation, grid(16384));
        failures += check(second.recorded && second.slot != reserved,
                          "the second anchor did not take another slot");
        ReuseLedger::AnchorRequest replacing = grid(8192);
        replacing.slot                       = 2;
        failures += check(ledger.record(conversation, replacing).recorded,
                          "replacing a reserved anchor was refused");
        failures += check(ledger.reserve_image_slot() == reserved,
                          "replacing a reserved anchor did not return its slot");
    }

    // An exhausted pool hands out nothing: the caller skips the capture instead of overwriting an
    // image that a Pinned anchor still needs.
    {
        ReuseLedger ledger(ReuseBudget{.stride = 8192, .capacity = 2});
        ledger.record(ledger.conversation(2), pinned(4096, AnchorProvenance::PromptEnd));
        ledger.record(ledger.conversation(2), pinned(8192, AnchorProvenance::Grid));
        failures += check(ledger.free_slots() == 0, "the pinned anchors did not take the pool");
        failures += check(ledger.reserve_image_slot() == kNoSlot,
                          "an exhausted pool still handed out a slot");
        // Both anchors stay: refusing to evict a Pinned image is the guarantee they exist for.
        failures += check(ledger.conversation(2).anchors.floor(8192) != nullptr,
                          "a pinned anchor was evicted to make room");
    }

    // An image can serve more than one conversation, but only one of them keeps it alive: a
    // borrower goes when the owner's slot is reclaimed, so a restore can never read a buffer that
    // moved on.
    {
        ReuseLedger ledger(ReuseBudget{.stride = 8192, .capacity = 1});
        ConversationLog& owner  = ledger.conversation(1);
        ConversationLog& reader = ledger.conversation(2);

        const ReuseLedger::RecordResult owned = ledger.record(owner, grid(8192));
        failures += check(owned.recorded, "the owning anchor was refused");

        ReuseLedger::AnchorRequest borrowed = grid(8192);
        borrowed.slot                       = owned.slot;
        failures +=
            check(ledger.record(reader, borrowed).recorded, "the borrowing anchor was refused");
        failures +=
            check(reader.anchors.floor(8192) != nullptr, "the borrowing anchor was not recorded");

        // The pool is full; only the owner can make room, because the borrower holds no slot. The
        // borrower must disappear with the image it names.
        failures += check(ledger.record(ledger.conversation(3), grid(16384)).recorded,
                          "the pool refused an anchor that the owner's eviction made room for");
        failures +=
            check(owner.anchors.floor(8192) == nullptr, "the evicted owner kept its anchor");
        failures += check(reader.anchors.floor(8192) == nullptr,
                          "a borrowed anchor survived the eviction of its image");
    }

    std::cout << (failures == 0 ? "ok" : "failures") << '\n';
    return failures == 0 ? 0 : 1;
}
