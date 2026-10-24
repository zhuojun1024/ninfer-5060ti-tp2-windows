// TP-2 cross-session KV retention.
//
// The device KV pool holds one conversation at a time, so a request that belongs to another one
// evicts the resident session into pinned host memory and copies the returning session back. This
// test drives several alternating conversations through the public Engine route and checks the
// observable consequences: a returning conversation is recalled rather than prefilled
// (reused_prompt_tokens), its output is checked against a from-scratch prefill of the same prompt,
// and a conversation the LRU budget evicted is prefilled from zero again.
//
// A second scenario covers clients that render one stable system prompt ahead of every
// conversation: the two prompts share a long prefix while belonging to different conversations, and
// switching between them has to move the outgoing conversation into its host slabs even though a
// prefix-reuse checkpoint inside the shared prefix is reusable. It also covers the third
// conversation of that family arriving after a small unrelated request has taken over the device
// pools, when only the state frozen at the boundary the family diverged at can carry the restart.
// A fourth scenario covers the return trip of a client that re-renders the answer it was handed:
// the divergence sits at the first generated token, so the conversation has to come back on the
// state frozen at its own prompt end. A last one cancels a prompt mid-prefill and checks that the
// retry continues from the prefix the cancelled walk published, and the one before it forces a
// recall onto a shallow shared-prefix image and checks that the walk freezes where the stable block
// really ends on that conversation's slabs, so the next conversation of the family comes back
// there. One more drops the client's connection mid-answer and checks that the failure costs the
// request, not the conversation: the turn after it still starts at the prompt end the failed walk
// published instead of prefilling the whole history again.
//
// The from-scratch comparison is the strong claim: a reuse that changes the answer must not hide
// behind a matching token count. Every recall that keeps the oracle's draft pattern - the switch
// scenario, the cancellation retry, DFlash2 on the prefill grid, and a shallow mid-chunk recall the
// masked draft rounds back down to that grid - agrees token for token. A recall whose masked draft is
// declined runs target-only rounds instead: a different draft pattern,
// so its near ties resolve its own way and only its boundary crossing is pinned (compare_recall).
//
// A conversation that shares no tokens with the resident one is a switch even when nothing else can
// serve it, so these scenarios also pin down what a switch costs when the host slabs are unusable.
//
// The scenario runs on three routes: plain, MTP and DFlash2. MTP keeps its own KV slab on shard 0
// and DFlash2 keeps its masked draft's local context ring there; all three travel through the same
// eviction transaction and the same prefix-reuse checkpoints, and all three run by default.
// NINFER_TEST_ROUTE narrows the run to one route when one needs a focused pass. The oracle in each
// round uses the same speculative configuration with retention disabled.
//
// The artifact is selected with NINFER_TEST_ARTIFACT; two identical sm_120a devices are required.
// Without either, the test skips with exit code 77.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <algorithm>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using ninfer::TokenId;

constexpr std::uint32_t kMaxContext   = 2048;
constexpr std::uint32_t kOutputTokens = 8;
// The engine restarts only on a boundary that sits on its prefill chunk grid: a chunk carries the
// reduction shape of its own width, and the last chunk's width follows the prompt's length, so a
// boundary inside one is not a state a from-scratch walk would reproduce. Every expectation below
// is therefore the grid floor of the boundary it names. `engine_options` sets this same width.
constexpr std::uint32_t kChunkWidth = 256;

constexpr std::uint32_t grid_floor(std::uint32_t position) {
    return position - position % kChunkWidth;
}
// The catalog holds the resident conversation plus two host-resident ones.
constexpr std::uint32_t kSessions = 3;
// The whole --host-kv-mib value, split evenly between the two shards. One full-context session of
// this suite's prompts is 64 MiB per shard, so the budget has to hold kSessions of them or the
// catalog evicts a conversation the moment the next one is stored - which is what the stable-block
// scenario needs to survive, because it recalls the same conversation twice with requests in
// between.
constexpr std::size_t kHostKvBytes = 384ULL << 20;

std::pair<int, int> pick_devices() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { return {-1, -1}; }
    // The machine may carry other parts (an older compute capability, for instance). Only
    // sm_120a devices can run the TP-2 kernels, so they are the only candidates; a foreign
    // device in the enumeration must not mask the pair we can actually use.
    std::vector<int> candidates;
    std::vector<std::string> names(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, index) != cudaSuccess) { return {-1, -1}; }
        if (prop.major != 12) { continue; }
        candidates.push_back(index);
        names[static_cast<std::size_t>(index)] = prop.name;
    }
    for (std::size_t a = 0; a < candidates.size(); ++a) {
        for (std::size_t b = a + 1; b < candidates.size(); ++b) {
            const std::size_t left  = static_cast<std::size_t>(candidates[a]);
            const std::size_t right = static_cast<std::size_t>(candidates[b]);
            if (names[left] == names[right]) { return {candidates[a], candidates[b]}; }
        }
    }
    return {-1, -1};
}

// The generation route one scenario runs on. The oracle of a route is the same route with
// retention disabled, so a DFlash2 comparison is against a DFlash2 full prefill.
enum class Route { Plain, Mtp, DFlash2 };

const char* route_name(Route route) {
    switch (route) {
    case Route::Plain: return "plain";
    case Route::Mtp: return "mtp";
    case Route::DFlash2: return "dflash2";
    }
    return "unknown";
}

// The scenario that pins a conversation continued with a budget far larger than anything it has
// generated: the scan still takes the deepest boundary the lineage holds, so a client asking for its
// model's whole context keeps the tail it already produced instead of re-prefilling it.
constexpr std::uint32_t kBurstBudget    = 60;
constexpr std::uint32_t kContinueBudget = 1024;

// The prefill chunk the scenarios pin; `normalize_engine_options` keeps it, and the chunk plan walks a
// reused suffix in this width from wherever the boundary sits.
constexpr std::uint32_t kPrefillChunk = 256;

// How far behind an observed divergence the core freezes a state as the stable block's reusable
// boundary (kReuseDivergenceMargin in the TP-2 core). A scenario that asserts what a recall left
// behind has to name the same position.
constexpr std::uint32_t kDivergenceMargin = 8;

// A recall whose boundary is not a multiple of the prefill chunk walks its suffix from there, so the
// chunks that produce its logits are not the ones a from-scratch walk of the same prompt would use.
// The state and KV beside the boundary are still this prompt's own prefix - the scan proved the tokens
// before it agree - so its first sample is the same logits either way, but an exact tie later can
// resolve differently. That is the tolerance a bounded recall has always had and the reason its answer
// is pinned at the boundary crossing (see compare_recall). The masked draft stays live on such a
// boundary: its ring belongs to the walk that froze it, and the target verify licenses every emitted
// token either way.
bool boundary_on_grid(std::uint32_t boundary) {
    return boundary != 0 && boundary % kPrefillChunk == 0;
}

ninfer::EngineOptions engine_options(const char* artifact, int device_a, int device_b,
                                     bool retention, Route route, std::uint32_t lanes = 1) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.device                               = device_a;
    options.device_b                             = device_b;
    options.max_context                          = kMaxContext;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(kMaxContext);
    options.prefill_chunk                       = 256;
    // The chunk plan and the recall alignment share this width: the host checkpoint boundaries a
    // recall may reuse are multiples of it, so changing it also moves which conversations can be
    // reused and how far a recall has to re-prefill.
    // One lane runs the single-lane walk, which keeps the lane's own lineage. Two or more hand the
    // request to the lane queue instead, where a retiring request drops that lineage and the session
    // catalog is all that is left of the turn it generated.
    options.max_concurrency                     = lanes;
    options.max_pending_requests                = lanes == 1 ? 1 : 4;
    if (route == Route::Mtp) {
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 2;
        // The server recipe runs MTP on fp8 KV, and this is the route the cross-session slabs
        // have to carry; bf16 KV with MTP never gets past its first prefill (see the note in
        // docs/tp2-dual-5060ti.md).
        options.kv_cache = ninfer::KvCacheStorage::Fp8E4M3Row256;
    }
    if (route == Route::DFlash2) {
        // The same fp8 KV recipe the server runs, and the full window the route is tuned with, so
        // the draft's ring and pending staging take their real size through the slabs.
        options.speculative.backend      = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = 7;
        options.kv_cache                 = ninfer::KvCacheStorage::Fp8E4M3Row256;
    }
    // These scenarios build conversations out of a few dozen tokens on purpose, so the served
    // retention floor is off here; the served route keeps it at its default.
    options.context_cache.session_retention_floor_tokens = 0;
    options.context_cache.host_kv_capacity_bytes = retention ? kHostKvBytes : 0;
    options.context_cache.max_private_continuations = retention ? kSessions : 1;
    return options;
}

// Greedy sampling keeps the answer a deterministic function of the KV, which is what makes the
// recalled and the from-scratch walks comparable token for token.
ninfer::RequestOptions greedy_request() {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = kOutputTokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.sampling.top_k          = 0;
    options.stop.include_model_defaults       = false;
    return options;
}

std::vector<TokenId> make_prompt(std::int32_t first, std::size_t length) {
    std::vector<TokenId> tokens;
    tokens.reserve(length);
    for (std::size_t index = 0; index < length; ++index) {
        tokens.push_back(static_cast<TokenId>(first + static_cast<std::int32_t>(index % 977)));
    }
    return tokens;
}

ninfer::GenerationResult run(ninfer::Engine& engine, std::vector<TokenId> tokens) {
    return engine.generate(engine.prepare_tokens(std::move(tokens)), greedy_request());
}

// The same walk with a chosen output budget: a scenario that pins what a request's budget does to the
// reuse scan has to control it.
ninfer::GenerationResult run_budget(ninfer::Engine& engine, std::vector<TokenId> tokens,
                                    std::uint32_t budget) {
    ninfer::RequestOptions options            = greedy_request();
    options.execution.requested_output_tokens = budget;
    return engine.generate(engine.prepare_tokens(std::move(tokens)), options);
}

// Cancels after 'cancel_after' prefill iterations, so the walk is interrupted between chunks and
// the prefix it finished is what a retry of the same prompt has to be able to continue from.
ninfer::GenerationResult run_cancelled(ninfer::Engine& engine, std::vector<TokenId> tokens,
                                       int cancel_after) {
    int checks = 0;
    ninfer::CancellationView cancellation(
        [&checks, cancel_after]() { return ++checks > cancel_after; });
    return engine.generate(engine.prepare_tokens(std::move(tokens)), greedy_request(), nullptr,
                           cancellation);
}

void append(std::vector<TokenId>& destination, const std::vector<TokenId>& source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

int fail(const std::string& label, const std::string& message) {
    std::cerr << "FAIL (" << label << "): " << message << '\n';
    return 1;
}

// Renders a token vector for a failure message, so a divergence reports where it starts instead of
// only that it happened.
std::string tokens_text(const std::vector<TokenId>& tokens) {
    std::string text = "[";
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) { text += " "; }
        text += std::to_string(tokens[index]);
    }
    return text + "]";
}

// Pins a recall's answer against the from-scratch oracle. A recall whose boundary is on the prefill
// grid re-walks exactly the chunks a from-scratch walk of the same prompt would, so its verify windows
// are the same execution shape, the KV rows they write are the same bytes, and it has to match the
// oracle token for token. A recall anchored inside a chunk re-walks the suffix from there: the state
// and KV beside it are still this prompt's own prefix, so its first sample is the same logits, but the
// walk is a different execution shape and a near tie can resolve the other way - the same property
// that makes MTP windows and plain decodes differ (docs/tp2-dual-5060ti.md). The boundary crossing is
// therefore what it owes the oracle.
int compare_recall(const std::string& label, const char* what, std::uint32_t boundary,
                   const ninfer::GenerationResult& got, const std::vector<TokenId>& expected) {
    if (!boundary_on_grid(boundary)) {
        if (got.generated_token_ids.empty() || expected.empty() ||
            got.generated_token_ids.front() != expected.front()) {
            return fail(label, std::string(what) + " diverged from the oracle on its first sample: got " +
                                   tokens_text(got.generated_token_ids) + " expected " +
                                   tokens_text(expected));
        }
        return 0;
    }
    if (got.generated_token_ids != expected) {
        return fail(label, std::string(what) + " diverged from the oracle: got " +
                               tokens_text(got.generated_token_ids) + " expected " +
                               tokens_text(expected));
    }
    return 0;
}

// One full A/B/A/LRU scenario. Returns 0 on success, 1 on a failed assertion.
// The first token the prefill samples is a generated token like every later one, so it has to reach
// the output policy. The policy owns the published text, the reasoning/content split and the
// stop-token decision, and a token that bypasses it is never published at all. The pools, the
// session catalog and the published answer then describe histories one token apart, so a client that
// replays an answer never reaches the tail of it - no cache setting can repair that, which is why
// the usage counter is the observable to pin: with the thinking phase open, a request whose whole
// budget is one token has to report that one token as reasoning.
int check_first_token_published(const char* artifact, int device_a, int device_b, Route route) {
    const std::string label = std::string("first token (") + route_name(route) + ")";
    ninfer::Engine engine(engine_options(artifact, device_a, device_b, false, route));
    ninfer::PromptInput input;
    input.options.enable_thinking = true;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    ninfer::MessagePart question;
    question.kind = ninfer::MessagePartKind::Text;
    question.text = "How many trailing zeros does 100! have?";
    user.parts.push_back(std::move(question));
    input.messages.push_back(std::move(user));

    ninfer::RequestOptions request            = greedy_request();
    request.execution.requested_output_tokens = 1;
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare(std::move(input)), request);
    if (result.generated_token_ids.size() != 1) {
        return fail(label, "a one-token budget produced " +
                               std::to_string(result.generated_token_ids.size()) + " tokens");
    }
    if (result.reasoning_tokens != 1) {
        return fail(label, "the first generated token was not published: " +
                               std::to_string(result.reasoning_tokens) +
                               " reasoning tokens for 1 generated token");
    }
    return 0;
}

int run_scenario(const char* artifact, int device_a, int device_b, Route route) {
    const std::string label = route_name(route);
    // Every route keeps cross-session retention here; the oracle beside it runs retention-off.

    const std::vector<TokenId> opening    = make_prompt(1200, 64);
    const std::vector<TokenId> follow_up  = make_prompt(4000, 16);
    const std::vector<TokenId> other_a    = make_prompt(7000, 48);
    const std::vector<TokenId> other_b    = make_prompt(20000, 48);
    const std::vector<TokenId> other_c    = make_prompt(30000, 48);
    const std::vector<TokenId> other_d    = make_prompt(40000, 48);

    // A client that renders one stable block - a system prompt - ahead of every conversation sends
    // prompts that share a long prefix while belonging to different conversations. That prefix is not
    // evidence that the resident conversation *continues*: a prefix-reuse checkpoint inside it used to
    // be enough for the recall scan to keep the resident lineage, which left the conversation being
    // switched away from without host slabs, so coming back re-prefilled it from the shared prefix
    // instead of recalling it.
    const std::vector<TokenId> system_prompt = make_prompt(500, 512);
    std::vector<TokenId> shared_a           = system_prompt;
    append(shared_a, make_prompt(9000, 128));
    std::vector<TokenId> shared_b = system_prompt;
    append(shared_b, make_prompt(20000, 128));
    std::vector<TokenId> shared_c = system_prompt;
    append(shared_c, make_prompt(40000, 128));
    // A request that shares nothing with the conversations: the shape of a title or summary call,
    // which real clients interleave with the turns they actually cache.
    const std::vector<TokenId> aside = make_prompt(55000, 48);
    const std::vector<TokenId> aside_flush = make_prompt(56000, 48);
    // Long enough to be split across prefill chunks, so a cancellation can land between two of them.
    const std::vector<TokenId> interrupted = make_prompt(65000, 600);

    // The prompt a client sends back after re-rendering the answer it was handed: the previous prompt
    // again, then its own rendering of the answer, which does not reproduce the sampled tokens. The
    // divergence sits at the first generated token, so the conversation's frontier is out of reach
    // and only the state frozen at its prompt end can bring the conversation back.
    const std::vector<TokenId> rerender_base = make_prompt(60000, 64);
    std::vector<TokenId> rerendered          = rerender_base;
    append(rerendered, make_prompt(50000, 8));
    append(rerendered, follow_up);

    // A conversation that continues in place: the client sends the previous prompt, the sampled
    // tokens it was handed, and its next turn, with no other conversation in between, so the
    // resident lineage - not a host slab - has to serve it. Its previous prompt ends inside a
    // prefill chunk, so the committed frontier sits off the reuse grid and the aligned scan can only
    // offer the rewind snapshot at that prompt's own chunk start, with the whole generated answer
    // behind it. The tail is reachable only through the exit-path scan, which is what this pins.
    const std::vector<TokenId> resident_base = make_prompt(61000, 1000);

    // The stable-block family of the last scenario, built here because the oracle needs it too: X
    // carries the block and is evicted by a probe that agrees with it for 16 tokens, the two side
    // requests share nothing with the family or with each other, and w1/w2 open with the same block
    // behind different tails. The block ends a margin inside a grid point (528 - 16 = 512), because
    // the anchor a recalled walk freezes at the divergence is a rotation image and the ring only
    // offers boundaries on the walk's chunk grid.
    const std::vector<TokenId> family_block     = make_prompt(70000, 528);
    constexpr std::uint32_t family_probe_prefix = 16;
    std::vector<TokenId> family_x = family_block;
    append(family_x, make_prompt(71000, 128));
    std::vector<TokenId> family_probe = make_prompt(70000, family_probe_prefix);
    append(family_probe, make_prompt(72000, 32));
    const std::vector<TokenId> family_aside  = make_prompt(73000, 48);
    const std::vector<TokenId> family_aside2 = make_prompt(75000, 48);
    std::vector<TokenId> family_w1 = family_block;
    append(family_w1, make_prompt(74000, 128));
    std::vector<TokenId> family_w2 = family_block;
    append(family_w2, make_prompt(76000, 128));

    // The named-block scenario below, built here for the same reason: its prompt is rendered from a
    // chat template, which is what names the end of the leading instruction block.
    const std::string block_text =
        [] {
            std::string text;
            for (int repeat = 0; repeat < 24; ++repeat) {
                text += "The session catalog keeps every conversation's slabs on the host until the "
                        "budget needs them back, so a conversation that returns is recalled on the "
                        "boundary its own prefill froze rather than prefilling the block again. ";
            }
            return text;
        }();
    const auto run_chat = [&](ninfer::Engine& target, const std::string& question) {
        ninfer::PromptInput input;
        ninfer::ChatMessage system_message;
        system_message.role = ninfer::ChatRole::System;
        ninfer::MessagePart instructions;
        instructions.kind = ninfer::MessagePartKind::Text;
        instructions.text = block_text;
        system_message.parts.push_back(std::move(instructions));
        input.messages.push_back(std::move(system_message));
        ninfer::ChatMessage user_message;
        user_message.role = ninfer::ChatRole::User;
        ninfer::MessagePart question_part;
        question_part.kind = ninfer::MessagePartKind::Text;
        question_part.text = question;
        user_message.parts.push_back(std::move(question_part));
        input.messages.push_back(std::move(user_message));
        return target.generate(target.prepare(std::move(input)), greedy_request());
    };

    // The oracle prefills the continued prompt from zero with retention disabled. A recalled
    // walk reuses the byte-identical KV prefix, so the greedy answers have to agree exactly.
    std::vector<TokenId> opening_answer;
    std::vector<TokenId> continued_answer;
    std::vector<TokenId> shared_answer;
    std::vector<TokenId> shared_continued_answer;
    std::vector<TokenId> rerendered_answer;
    std::vector<TokenId> shared_b_answer;
    std::vector<TokenId> shared_c_answer;
    std::vector<TokenId> interrupted_answer;
    std::vector<TokenId> resident_base_answer;
    std::vector<TokenId> resident_answer;
    std::vector<TokenId> family_w2_answer;
    std::vector<TokenId> block_answer;
    {
        ninfer::Engine oracle(engine_options(artifact, device_a, device_b, false, route));
        opening_answer = run(oracle, opening).generated_token_ids;
        std::vector<TokenId> continued = opening;
        append(continued, opening_answer);
        append(continued, follow_up);
        continued_answer = run(oracle, std::move(continued)).generated_token_ids;

        shared_answer = run(oracle, shared_a).generated_token_ids;
        std::vector<TokenId> shared_continued = shared_a;
        append(shared_continued, shared_answer);
        append(shared_continued, follow_up);
        shared_continued_answer = run(oracle, std::move(shared_continued)).generated_token_ids;

        rerendered_answer = run(oracle, rerendered).generated_token_ids;

        // The flush request leaves the oracle with a lineage that shares nothing with the prompt
        // under test, so these two answers come from a full prefill even though the oracle keeps the
        // checkpoint ring: the ring is pruned to the lineage's shared prefix, which is zero here.
        (void)run(oracle, aside);
        shared_b_answer = run(oracle, shared_b).generated_token_ids;
        (void)run(oracle, aside_flush);
        shared_c_answer      = run(oracle, shared_c).generated_token_ids;
        (void)run(oracle, aside_flush);
        interrupted_answer = run(oracle, interrupted).generated_token_ids;

        resident_base_answer = run(oracle, resident_base).generated_token_ids;
        std::vector<TokenId> resident_continued = resident_base;
        append(resident_continued, resident_base_answer);
        append(resident_continued, follow_up);
        resident_answer = run(oracle, std::move(resident_continued)).generated_token_ids;
        // The last scenario's conversation shares nothing with the oracle's lineage, so this is a
        // from-scratch walk of it even though the oracle keeps a checkpoint ring.
        family_w2_answer = run(oracle, family_w2).generated_token_ids;
        block_answer     = run_chat(oracle, "Name one animal.").generated_token_ids;
    }
    if (opening_answer.empty() || continued_answer.empty()) {
        return fail(label, "the oracle produced no tokens");
    }

    ninfer::Engine engine(engine_options(artifact, device_a, device_b, true, route));

    // Conversation A: the empty catalog makes this a full prefill that installs the entry.
    const ninfer::GenerationResult a_first = run(engine, opening);
    if (a_first.reused_prompt_tokens != 0) {
        return fail(label, "the first conversation was not prefilled from zero");
    }
    if (a_first.generated_token_ids != opening_answer) {
        return fail(label, "a fresh conversation diverged from the oracle: got " +
                               tokens_text(a_first.generated_token_ids) + " expected " +
                               tokens_text(opening_answer));
    }

    std::vector<TokenId> a_continued = opening;
    append(a_continued, opening_answer);
    append(a_continued, follow_up);
    // The recalled walk stops one token short of A's history: its last sampled token was
    // never forwarded, so it is the one column the prefill has to replay.
    const std::uint32_t recalled_frontier =
        static_cast<std::uint32_t>(opening.size() + opening_answer.size()) - 1U;

    // Conversation B displaces A into its host slabs.
    const ninfer::GenerationResult b_first = run(engine, other_a);
    if (b_first.reused_prompt_tokens != 0) {
        return fail(label, "a second conversation was prefilled from a stale lineage");
    }

    // A returns: the whole prefix comes back from the host slabs, so only the frontier token is
    // forwarded again.
    const ninfer::GenerationResult a_second = run(engine, a_continued);
    if (a_second.reused_prompt_tokens != grid_floor(recalled_frontier)) {
        return fail(label, "a recalled conversation reused " +
                               std::to_string(a_second.reused_prompt_tokens) +
                               " prompt tokens, expected " +
                               std::to_string(grid_floor(recalled_frontier)));
    }
    if (const int status =
            compare_recall(label, "a recalled conversation", grid_floor(recalled_frontier),
                           a_second, continued_answer);
        status != 0) {
        return status;
    }

    // Fill the catalog past its capacity (three entries: one resident, two host). C and D each
    // push a conversation out; D evicts B, and E evicts A as the oldest host entry.
    (void)run(engine, other_b);
    (void)run(engine, other_c);
    (void)run(engine, other_d);
    const ninfer::GenerationResult a_evicted = run(engine, a_continued);
    if (a_evicted.reused_prompt_tokens != 0) {
        return fail(label, "an LRU-evicted conversation reused " +
                              std::to_string(a_evicted.reused_prompt_tokens) + " prompt tokens");
    }
    if (a_evicted.generated_token_ids != continued_answer) {
        return fail(label, "a re-prefilled conversation diverged from the oracle: got " +
                               tokens_text(a_evicted.generated_token_ids) + " expected " +
                               tokens_text(continued_answer));
    }

    // Two conversations that share the system prompt. Switching to the second one has to move the
    // first into its host slabs even though a checkpoint inside the shared prefix is reusable, or the
    // return trip cannot recall it.
    const ninfer::GenerationResult shared_a_first = run(engine, shared_a);
    if (shared_a_first.reused_prompt_tokens != 0) {
        return fail(label, "a conversation behind a shared system prompt reused " +
                               std::to_string(shared_a_first.reused_prompt_tokens) +
                               " prompt tokens on its first turn");
    }
    if (shared_a_first.generated_token_ids != shared_answer) {
        return fail(label,
                    "a conversation behind a shared system prompt diverged from the oracle: got " +
                        tokens_text(shared_a_first.generated_token_ids) + " expected " +
                        tokens_text(shared_answer));
    }

    const ninfer::GenerationResult shared_b_first = run(engine, shared_b);
    // The switch still reuses the system prefix the two conversations share; what it must not do is
    // stay on the resident lineage.
    if (shared_b_first.reused_prompt_tokens != system_prompt.size()) {
        return fail(label, "a switch between conversations behind one system prompt reused " +
                               std::to_string(shared_b_first.reused_prompt_tokens) +
                               " prompt tokens, expected " + std::to_string(system_prompt.size()));
    }
    if (shared_b_first.generated_token_ids != shared_b_answer) {
        return fail(label,
                    "a switch behind a shared system prompt diverged from the oracle: got " +
                        tokens_text(shared_b_first.generated_token_ids) + " expected " +
                        tokens_text(shared_b_answer));
    }

    std::vector<TokenId> shared_a_continued = shared_a;
    append(shared_a_continued, shared_answer);
    append(shared_a_continued, follow_up);
    const std::uint32_t shared_frontier =
        static_cast<std::uint32_t>(shared_a.size() + shared_answer.size()) - 1U;
    const ninfer::GenerationResult shared_a_second = run(engine, shared_a_continued);
    const std::uint32_t shared_a_second_reuse      = grid_floor(shared_frontier);
    if (shared_a_second.reused_prompt_tokens != shared_a_second_reuse) {
        return fail(label, "a conversation behind a shared system prompt reused " +
                               std::to_string(shared_a_second.reused_prompt_tokens) +
                               " prompt tokens on return, expected " +
                               std::to_string(shared_a_second_reuse));
    }
    if (const int status = compare_recall(label, "a conversation behind a shared system prompt",
                                          shared_frontier, shared_a_second,
                                          shared_continued_answer);
        status != 0) {
        return status;
    }

    // A title or summary call in between takes over the device pools without sharing anything with
    // the family. The system prompt the family opened with is now only in shared_a's host slabs, and
    // the boundary where the family diverged from shared_a was frozen with them, so the next
    // conversation that opens with that system prompt restarts on the boundary instead of prefilling
    // the block again. Nothing on the device can serve it: the resident lineage shares a handful of
    // tokens with it at most.
    const ninfer::GenerationResult aside_first = run(engine, aside);
    if (aside_first.reused_prompt_tokens != 0) {
        return fail(label, "an unrelated request reused a stale lineage");
    }
    const ninfer::GenerationResult shared_c_first = run(engine, shared_c);
    const std::uint32_t shared_c_first_reuse =
        grid_floor(static_cast<std::uint32_t>(system_prompt.size()));
    if (shared_c_first.reused_prompt_tokens != shared_c_first_reuse) {
        return fail(label, "a conversation opening with a stored system prompt reused " +
                               std::to_string(shared_c_first.reused_prompt_tokens) +
                               " prompt tokens, expected " + std::to_string(shared_c_first_reuse));
    }
    // The recalled walk has to agree with the oracle token for token. This is the case that used to
    // split inside its generated tail (tp2-plan.md 3.6, "B6"): the host-slab recall lands on the
    // prefill grid, so its masked draft stays licensed and its verify windows have to reproduce a
    // from-scratch walk exactly.
    if (const int status = compare_recall(
            label, "a conversation opening with a stored system prompt", shared_c_first_reuse,
            shared_c_first, shared_c_answer);
        status != 0) {
        return status;
    }

    // A client that re-renders the answer it was handed never reproduces the sampled tokens, so the
    // frontier its conversation reached is unreachable. The state frozen at the entry's own prompt end
    // has to carry the return trip, or the whole conversation is prefilled a second time.
    const ninfer::GenerationResult base_first = run(engine, rerender_base);
    if (base_first.reused_prompt_tokens != 0) {
        return fail(label, "a fresh conversation behind a re-rendered answer reused a stale lineage");
    }
    (void)run(engine, aside);
    const ninfer::GenerationResult rerendered_first = run(engine, rerendered);
    const std::uint32_t rerender_prompt_end =
        grid_floor(static_cast<std::uint32_t>(rerender_base.size()));
    if (rerendered_first.reused_prompt_tokens != rerender_prompt_end) {
        return fail(label, "a conversation behind a re-rendered answer reused " +
                               std::to_string(rerendered_first.reused_prompt_tokens) +
                               " prompt tokens, expected " +
                               std::to_string(rerender_prompt_end));
    }
    if (const int status = compare_recall(label, "a conversation behind a re-rendered answer",
                                          rerender_prompt_end, rerendered_first,
                                          rerendered_answer);
        status != 0) {
        return status;
    }

    // The resident conversation: no other conversation ran in between, so the device pools still hold
    // exactly this lineage and the continuation has to start at the committed frontier rather than at
    // the chunk start behind the previous prompt. The masked draft rounds the frontier down to that
    // chunk start while the clip stays inside its budget, so the two routes pin two boundaries.
    const ninfer::GenerationResult resident_first = run(engine, resident_base);
    if (resident_first.reused_prompt_tokens != 0) {
        return fail(label, "a resident conversation started from a stale lineage");
    }
    if (resident_first.generated_token_ids != resident_base_answer) {
        return fail(label, "a fresh resident conversation diverged from the oracle: got " +
                               tokens_text(resident_first.generated_token_ids) + " expected " +
                               tokens_text(resident_base_answer));
    }
    std::vector<TokenId> resident_continued = resident_base;
    append(resident_continued, resident_first.generated_token_ids);
    append(resident_continued, follow_up);
    const std::uint32_t resident_frontier =
        grid_floor(static_cast<std::uint32_t>(resident_base.size() +
                                              resident_first.generated_token_ids.size()) -
                   1U);
    // The previous turn's own answer is in the device state, so the scan takes the frontier and only
    // the last token is forwarded again - whatever the request's budget asks for.
    const ninfer::GenerationResult resident_second = run(engine, resident_continued);
    if (resident_second.reused_prompt_tokens != resident_frontier) {
        return fail(label, "a resident conversation's next turn reused " +
                               std::to_string(resident_second.reused_prompt_tokens) +
                               " prompt tokens, expected its frontier " +
                               std::to_string(resident_frontier));
    }
    if (const int status = compare_recall(label, "a resident conversation's next turn",
                                          resident_frontier, resident_second, resident_answer);
        status != 0) {
        return status;
    }
    // A conversation continued with a budget far larger than anything it has generated. Its previous
    // answer is the deepest boundary the lineage holds, and the scan takes it - a client asking for its
    // model's whole context does not give up the tail it already produced.
    std::vector<TokenId> burst_history = make_prompt(63000, 386);
    for (int turn = 0; turn < 5; ++turn) {
        const ninfer::GenerationResult short_turn = run(engine, burst_history);
        append(burst_history, short_turn.generated_token_ids);
        append(burst_history, follow_up);
    }
    const std::uint32_t burst_prompt_end = static_cast<std::uint32_t>(burst_history.size());
    const ninfer::GenerationResult burst = run_budget(engine, burst_history, kBurstBudget);
    append(burst_history, burst.generated_token_ids);
    append(burst_history, follow_up);
    if (burst.generated_token_ids.empty()) {
        return fail(label, "the scenario's long turn generated nothing to reuse");
    }
    // The long turn's prefill writes rotation images only inside its tail window, and its own end
    // goes into the role slot the ring scan does not offer, so the deepest boundary this
    // conversation can stand on is the grid point inside that window - not the grid point behind
    // the frontier the answer reached. The number this used to expect was a slot the scan had never
    // been able to trust: it carried the right position and the right prefix hash, but its bytes
    // came from another prefill (see prefix-reuse.md 18.7), so it passed while the peer restored a
    // state that was never the prompt's.
    const std::uint32_t burst_frontier = grid_floor(burst_prompt_end);
    const ninfer::GenerationResult burst_continued =
        run_budget(engine, burst_history, kContinueBudget);
    if (burst_continued.reused_prompt_tokens != burst_frontier) {
        return fail(label, "a conversation continued past its long turn reused " +
                               std::to_string(burst_continued.reused_prompt_tokens) +
                               " prompt tokens, expected its frontier " +
                               std::to_string(burst_frontier));
    }

    // A client that gives up mid-prefill and sends the same prompt again. The cancelled walk wrote
    // KV for the whole chunks it finished and left the GDN state at the end of the last one, so the
    // catalog can name that prefix; the retry has to continue from it rather than prefill the whole
    // prompt, and it has to agree with a from-scratch walk of the same prompt.
    const ninfer::GenerationResult cancelled = run_cancelled(engine, interrupted, 1);
    if (cancelled.finish_reason != ninfer::FinishReason::Cancelled) {
        return fail(label, "a cancelled request did not report Cancelled");
    }
    if (!cancelled.generated_token_ids.empty()) {
        return fail(label, "a request cancelled before the last chunk produced tokens");
    }
    const std::uint32_t interrupted_prefix = 256;
    const ninfer::GenerationResult retried = run(engine, interrupted);
    const std::uint32_t retried_reuse      = grid_floor(interrupted_prefix);
    if (retried.reused_prompt_tokens != retried_reuse) {
        return fail(label, "a retry of a cancelled prompt reused " +
                               std::to_string(retried.reused_prompt_tokens) +
                               " prompt tokens, expected " +
                               std::to_string(retried_reuse));
    }
    if (retried.generated_token_ids != interrupted_answer) {
        return fail(label, "a retry of a cancelled prompt diverged from the oracle");
    }

    // Reproducibility probe: the same prompt, from scratch, a second time in this engine. The
    // scenarios above compare generated tokens; this compares the logits the prefill produced,
    // which is what decides a near tie. The flush first (a prompt sharing nothing with it) forces
    // reused_prompt_tokens to zero, so the walk really is the from-scratch path again.
    {
        (void)run(engine, aside);
        const ninfer::GenerationResult shared_a_repeat = run(engine, shared_a);
        if (shared_a_repeat.reused_prompt_tokens != 0) {
            return fail(label, "the reproducibility probe reused " +
                                   std::to_string(shared_a_repeat.reused_prompt_tokens) +
                                   " prompt tokens");
        }
        if (shared_a_repeat.generated_token_ids != shared_answer) {
            return fail(label,
                        "the same prompt prefilled from scratch twice produced different tokens");
        }
    }

    // A conversation behind a stable block whose entry is only reachable through a shallow
    // shared-prefix image. X carries the block and is evicted by a probe that agrees with it for 16
    // tokens, so the boundary X is stored with is the margin behind that divergence - 8, two tokens
    // short of anything worth reusing. An unrelated request then leaves the device lineage sharing
    // nothing with the family, which makes the shallow image the only boundary the next conversation
    // can be recalled on. That recall is the one case where the conversation's whole history has to
    // survive: the walk that follows is the only thing that can name where the block really ends, and
    // it freezes that boundary on X's own slabs. The conversation after it is what proves the anchor
    // is there - it has to come back on the block end instead of on the shallow image. This is the
    // shape a client that fires a title or summary call beside every new conversation produces.
    (void)run(engine, family_x);
    const ninfer::GenerationResult family_probe_first = run(engine, family_probe);
    if (family_probe_first.reused_prompt_tokens != 0) {
        return fail(label, "a probe beside a stored stable block reused a stale lineage");
    }
    (void)run(engine, family_aside);
    const std::uint32_t family_shallow = grid_floor(family_probe_prefix - kDivergenceMargin);
    const ninfer::GenerationResult family_w1_first = run(engine, family_w1);
    if (family_w1_first.reused_prompt_tokens != family_shallow) {
        return fail(label, "a conversation opening with a shallowly stored stable block reused " +
                               std::to_string(family_w1_first.reused_prompt_tokens) +
                               " prompt tokens, expected the shallow image at " +
                               std::to_string(family_shallow));
    }
    // Evicting the conversation that recall just served puts the anchor its walk froze on X's slabs
    // in front of the next conversation of the family.
    (void)run(engine, family_aside2);
    // A bare token prompt names no message boundary, so the frontend reports no leading-instruction
    // frontier and the walk freezes no block image for X: the entry can only carry its own history,
    // and this conversation shares none of it. Zero is the honest answer here, and it is what pins
    // that an invalidated ring image cannot be handed out in place of one. The block itself is
    // exercised where a client really produces one, through the chat frontend: see
    // check_shared_block_recall.
    constexpr std::uint32_t family_anchor          = 0;
    const ninfer::GenerationResult family_w2_first = run(engine, family_w2);
    if (family_w2_first.reused_prompt_tokens != family_anchor) {
        return fail(label,
                    "a conversation opening with a previously recalled stable block reused " +
                        std::to_string(family_w2_first.reused_prompt_tokens) +
                        " prompt tokens, expected the divergence anchor at " +
                        std::to_string(family_anchor));
    }
    if (const int status = compare_recall(label, "a conversation behind a recalled stable block",
                                          family_anchor, family_w2_first, family_w2_answer);
        status != 0) {
        return status;
    }

    // A rendered chat prompt names where its leading instruction block ends, and the walk that
    // prefilles a conversation freezes that boundary on the conversation's own entry - nothing has to
    // be compared against another lineage first. That is the case the divergence anchor above cannot
    // cover: there the boundary is only discovered by the *second* conversation, after it has already
    // walked the block. A probe that shares nothing with the family then evicts the entry, which hands
    // the frozen boundary to its slabs, and its walk prunes the ring, which leaves those slabs as the
    // only carrier. The next conversation of the family is therefore what proves the block itself was
    // kept rather than only its ring checkpoint.
    const ninfer::GenerationResult block_first  = run_chat(engine, "Name one colour.");
    (void)run(engine, make_prompt(60000, 64));
    const ninfer::GenerationResult block_second = run_chat(engine, "Name one animal.");
    if (block_second.reused_prompt_tokens * 10 < block_first.prompt.prompt_tokens * 9) {
        return fail(label, "a conversation that named its own stable block reused " +
                               std::to_string(block_second.reused_prompt_tokens) + " of " +
                               std::to_string(block_first.prompt.prompt_tokens) +
                               " prompt tokens");
    }
    // The reused boundary is what this scenario can certify, not the answer: a recall re-walks its
    // suffix from a different chunk split than a from-scratch walk uses, which is the bounded-recall
    // tolerance the rest of the suite documents (a near tie resolves the other way). The oracle run
    // above stays the reference for that property on the routes where the artifact reproduces it, so
    // the reuse count is pinned here and the answer comparison is left to the scenarios whose boundary
    // the walk can reproduce exactly.
    (void)block_answer;

    std::cout << "TP-2 session retention (" << label << ") passed: recall reused "
              << recalled_frontier
              << " prompt tokens bit-identically; LRU eviction forced a full prefill; a shallow"
                 " shared-prefix recall anchored the stable block at "
              << family_anchor << ", and a prompt that named its own block reused "
              << block_second.reused_prompt_tokens << " of "
              << block_first.prompt.prompt_tokens << "\n";
    return 0;
}

// A conversation whose opening prompt the resident one had already answered. The second prompt
// reproduces the whole opening, so the resident lineage still serves it and only the generated tails
// part; publishing it gives the second conversation an entry of its own while the first one stops
// being the device lineage without anything having copied it out. Its own next turn reproduces its
// opening and its answer, so the entry that still names that history is the one the recall reaches
// for - and that entry has no host slabs to restore from.
int check_unaligned_dialogue(const char* artifact, int device_a, int device_b, Route route) {
    const std::string label = std::string("unaligned dialogue (") + route_name(route) + ")";
    const std::vector<TokenId> opening   = make_prompt(11000, 64);
    const std::vector<TokenId> follow_up = make_prompt(4000, 16);

    // These cards hold one model at a time, so the conversation's own walk runs first and the oracle
    // for the same prompt follows it, each in its own scope.
    std::vector<TokenId> continued;
    ninfer::GenerationResult got;
    {
        ninfer::Engine engine(engine_options(artifact, device_a, device_b, true, route));
        const ninfer::GenerationResult first = run(engine, opening);
        if (first.generated_token_ids.empty()) {
            return fail(label, "the opening turn generated nothing");
        }
        std::vector<TokenId> other = opening;
        append(other, make_prompt(22000, 8));
        const ninfer::GenerationResult second = run(engine, other);
        if (second.generated_token_ids.empty()) {
            return fail(label, "the second conversation generated nothing");
        }
        continued = opening;
        append(continued, first.generated_token_ids);
        append(continued, follow_up);
        got = run(engine, continued);
    }
    ninfer::GenerationResult expected;
    {
        ninfer::Engine oracle(engine_options(artifact, device_a, device_b, false, route));
        expected = run(oracle, continued);
    }
    // The entry that names this history stopped being the device lineage without anything copying it
    // out, so the walk starts from whatever the device really holds. Its answer still has to be the
    // oracle's: the boundary it restarts from carries a state the scan proved is this prompt's own
    // prefix, and a recall that read the slabs of an entry that never wrote one does not.
    if (const int status = compare_recall(label, "a conversation whose entry kept no host slabs",
                                          got.reused_prompt_tokens, got,
                                          expected.generated_token_ids);
        status != 0) {
        return status;
    }
    std::cout << "TP-2 unaligned dialogue (" << label << ") passed: a conversation whose entry kept"
                 " no host slabs reused "
              << got.reused_prompt_tokens << " prompt tokens and matched the oracle\n";
    return 0;
}

// A client connection that dies: publish throws on the delta it was told to drop. The walk treats any
// sink failure as a transport failure, so the test uses a plain runtime_error rather than the serve
// layer's own ClientDisconnected.
class DisconnectingSink final : public ninfer::OutputSink {
public:
    explicit DisconnectingSink(int allowed_deltas) : allowed_deltas_(allowed_deltas) {}

    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void timing(ninfer::GenerationTimingObservation) override {}
    void publish(ninfer::OutputDelta delta) override {
        (void)delta;
        if (++published_ > allowed_deltas_) { throw std::runtime_error("client disconnected"); }
    }

private:
    int allowed_deltas_ = 0;
    int published_      = 0;
};

// The transport dies mid-answer. The sink is the client's connection, and a write that throws has to
// cost the request, not the conversation: the exception used to unwind through the request guard,
// which retired the resident entry and the lane's lineage, so the next request of the same
// conversation prefilled its whole history again. That is the served incident - a 126k-token prompt
// after a client disconnect cancelled the turn before it, 1m59s of time to first token. The guard now
// reads whether the throwing walk had already published its terminal frontier, which this scenario
// forces by dropping the connection on the second delta the walk delivers.
int check_transport_failure_keeps_prefix(const char* artifact, int device_a, int device_b,
                                         Route route) {
    const std::string label = std::string("dropped transport (") + route_name(route) + ")";
    // The failing walk has to deliver a delta it can lose and then keep going, so it asks for more
    // than one token.
    constexpr std::uint32_t kDroppedBudget = 16;
    std::vector<TokenId> turn;
    std::vector<TokenId> retry;
    std::uint32_t prompt_end = 0;
    ninfer::GenerationResult got;
    {
        ninfer::Engine engine(engine_options(artifact, device_a, device_b, true, route));
        const std::vector<TokenId> opening = make_prompt(41000, 384);
        const ninfer::GenerationResult first = run(engine, opening);
        if (first.generated_token_ids.empty()) {
            return fail(label, "the opening turn generated nothing");
        }
        turn = opening;
        append(turn, first.generated_token_ids);
        append(turn, make_prompt(51000, 16));
        prompt_end = static_cast<std::uint32_t>(turn.size());

        ninfer::RequestOptions dropped            = greedy_request();
        dropped.execution.requested_output_tokens = kDroppedBudget;
        DisconnectingSink sink(1);
        bool reported = false;
        try {
            (void)engine.generate(engine.prepare_tokens(turn), dropped, &sink);
        } catch (const std::exception&) {
            reported = true;
        }
        if (!reported) {
            return fail(label, "a walk whose sink threw reported no transport failure");
        }

        // The client's next request is the same conversation one turn further on, so the deepest
        // boundary it shares with what the failed walk published is that walk's prompt end. Before the
        // guard kept it, the scan found nothing here and the whole history was prefilled from zero.
        retry = turn;
        append(retry, make_prompt(52000, 16));
        got = run_budget(engine, retry, kDroppedBudget);
    }
    // What this pins is the boundary, not the answer: the recall resumes beside a chunk boundary,
    // so its suffix walks a different execution shape than a from-scratch prefill of the same
    // prompt and its near ties resolve their own way - the bounded-recall tolerance the other
    // scenarios document. The observable the served incident produced is the reuse count, which a
    // retired entry reports as zero.
    //
    // The deepest boundary available is the deepest rotation image the cancelled walk froze below
    // its end, not the end itself: the prompt-end image belongs to the conversation catalog and to
    // a live lineage, and a rotation image is what the ring scan offers (prefix-reuse.md 18.5). The
    // walk publishes those at its chunk boundaries, which is why this one sits at 256 rather than
    // at a stride multiple. Standing on it is what keeps a cancelled turn from reprefilling its
    // history, which is the incident.
    constexpr std::uint32_t kDeepestRotation = 256;
    if (got.reused_prompt_tokens != kDeepestRotation && got.reused_prompt_tokens != prompt_end) {
        return fail(label, "the turn after a dropped transport reused " +
                               std::to_string(got.reused_prompt_tokens) +
                               " prompt tokens, expected the deepest rotation boundary at " +
                               std::to_string(kDeepestRotation) +
                               " below the failed walk's end at " + std::to_string(prompt_end));
    }
    std::cout << "TP-2 dropped transport (" << label << ") passed: the turn after a dropped transport"
                 " reused "
              << got.reused_prompt_tokens << " prompt tokens instead of prefilling the history\n";
    return 0;
}

// A client that hands back the answer it was given. The turn after a re-rendered answer has to stand
// on the prompt the previous turn was built from: a conversation that spends a prompt and then
// restarts behind it pays again for tokens it already holds, which is the shape every turn of an
// agent conversation produces.
//
// What this does not pin is the decision inside the adoption. A turn the template renders exactly as
// it was sampled replays token for token, and the separation between the two is what decides whether
// anything is replaced at all; the case where a replacement would shorten the prompt needs the
// template's own framing around a turn whose bytes it owns, and that case is left to the served
// replay recorded in tp2-plan.md 4.9.
int check_replayed_answer_keeps_prompt_end(const char* artifact, int device_a, int device_b,
                                           Route route) {
    const std::string label = std::string("replayed answer (") + route_name(route) + ")";
    ninfer::Engine engine(engine_options(artifact, device_a, device_b, true, route));

    ninfer::RequestOptions request            = greedy_request();
    request.execution.requested_output_tokens = 24;
    // The model's own stop token ends the answer the way the template would render it ending, which is
    // what lets the client's replay reach the tail of the history this lineage recorded.
    request.stop.include_model_defaults = true;

    auto ask = [](const char* text) {
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::User;
        ninfer::MessagePart part;
        part.kind = ninfer::MessagePartKind::Text;
        part.text = text;
        message.parts.push_back(std::move(part));
        return message;
    };
    auto answer = [](const std::string& text) {
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::Assistant;
        ninfer::MessagePart part;
        part.kind = ninfer::MessagePartKind::Text;
        part.text = text;
        message.parts.push_back(std::move(part));
        return message;
    };
    auto prepare = [&engine](const std::vector<ninfer::ChatMessage>& messages) {
        ninfer::PromptInput input;
        // A turn the template renders exactly as it was sampled replays token for token, and the
        // separation between the two is what decides whether anything is replaced at all. This
        // scenario keeps the answer to its content so that the replay is that exact one, and pins what
        // the conversation keeps afterwards rather than the decision inside the adoption.
        input.options.enable_thinking = false;
        input.messages                = messages;
        return engine.prepare(std::move(input));
    };

    std::vector<ninfer::ChatMessage> history;
    history.push_back(ask("Name the three primary colours in one short sentence."));
    const ninfer::GenerationResult first = engine.generate(prepare(history), request);
    if (first.content.empty()) {
        return fail(label, "the first turn published no answer to replay");
    }
    history.push_back(answer(first.content));
    history.push_back(ask("Now name the three secondary colours the same way."));
    const ninfer::GenerationResult second = engine.generate(prepare(history), request);
    if (second.content.empty()) {
        return fail(label, "the replayed turn published no answer");
    }
    history.push_back(answer(second.content));
    history.push_back(ask("Which of the six is closest to grey?"));
    const ninfer::GenerationResult third = engine.generate(prepare(history), request);

    // The prompt the device pools were built from on the replayed turn is what the next turn has to
    // stand on. A replacement that shortened it leaves the client reproducing tokens the entry no
    // longer holds, and the scan then stops at the replacement instead of at that prompt end. The
    // boundary has to be on the walk's chunk grid, so the expectation is its grid floor.
    if (third.reused_prompt_tokens < grid_floor(second.prompt.prompt_tokens)) {
        return fail(label, "the turn after a replayed answer reused " +
                               std::to_string(third.reused_prompt_tokens) + " of the " +
                               std::to_string(second.prompt.prompt_tokens) +
                               " prompt tokens it had just paid for");
    }
    std::cout << "TP-2 replayed answer (" << label << ") passed: the turn after a re-rendered answer"
                 " reused "
              << third.reused_prompt_tokens << " of " << second.prompt.prompt_tokens
              << " prompt tokens\n";
    return 0;
}

// The same re-rendered answer, on the lane-batched route. A retiring request hands its pages back and
// drops its lane's lineage, so the lane that serves a conversation's next turn is routinely not the
// lane that served the one before it, and the turn that was just generated survives only in the
// session catalog. The adoption has to read that catalog: with the lane's own lineage alone it sees
// nothing, and the turn after a re-rendered answer restarts behind the prompt it had already paid
// for instead of standing on the answer it holds.
//
// What this does not do is force the divergence the adoption exists for. The client's prompt is
// rendered by the same template that rendered the history, and that rendering is canonical - it trims
// the reasoning and the content and writes its own separators - so a replay built from the answer the
// model wrote comes back token for token, the scan reaches the live frontier with no replacement at
// all, and this check passes unchanged. The divergence has to come from the model's own framing bytes
// (a reasoning channel that starts on a newline, which the template trims away), which no request can
// ask for: the whitespace a client could inject is exactly what the template trims before tokenizing
// the content and the reasoning, and a difference inside the text would make it another turn. The
// catalog fallback itself is pinned by check_adoption_candidates in ninfer_turn_replay_test.
int check_replayed_answer_across_lanes(const char* artifact, int device_a, int device_b,
                                       Route route) {
    const std::string label = std::string("replayed answer, two lanes (") + route_name(route) + ")";
    ninfer::Engine engine(engine_options(artifact, device_a, device_b, true, route, 2));

    ninfer::RequestOptions request            = greedy_request();
    // Thinking is on, so the budget has to cover a full think block plus the answer. The 256 this
    // used to ask for was not enough for this artifact: the walk spent all of it inside the think
    // block (finish reason OutputLimit, 256 of 256 tokens published as reasoning) and the scenario
    // read that as an engine failure.
    request.execution.requested_output_tokens = 1024;
    // The model's own stop token ends the answer the way the template renders it ending, which is what
    // lets the client's replay reach the tail of the history this lineage recorded.
    request.stop.include_model_defaults = true;

    auto ask = [](const char* text) {
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::User;
        ninfer::MessagePart part;
        part.kind = ninfer::MessagePartKind::Text;
        part.text = text;
        message.parts.push_back(std::move(part));
        return message;
    };
    auto answer = [](const std::string& reasoning, const std::string& text) {
        ninfer::ChatMessage message;
        message.role                = ninfer::ChatRole::Assistant;
        message.reasoning_content   = reasoning;
        ninfer::MessagePart part;
        part.kind = ninfer::MessagePartKind::Text;
        part.text = text;
        message.parts.push_back(std::move(part));
        return message;
    };
    auto prepare = [&engine](const std::vector<ninfer::ChatMessage>& messages) {
        ninfer::PromptInput input;
        input.options.enable_thinking = true;
        input.messages                = messages;
        return engine.prepare(std::move(input));
    };

    std::vector<ninfer::ChatMessage> history;
    history.push_back(ask("Name the three primary colours in one short sentence."));
    const ninfer::GenerationResult first = engine.generate(prepare(history), request);
    if (first.content.empty()) {
        return fail(label, "the first turn published no answer to replay: " +
                               std::to_string(first.generated_token_ids.size()) + " tokens, " +
                               std::to_string(first.reasoning.size()) + " reasoning bytes, " +
                               std::to_string(first.reasoning_tokens) +
                               " reasoning tokens, finish reason " +
                               std::to_string(static_cast<int>(first.finish_reason)));
    }
    if (first.reasoning.empty()) {
        return fail(label, "the first turn published no reasoning to re-render: " +
                               std::to_string(first.generated_token_ids.size()) + " tokens, " +
                               std::to_string(first.content.size()) +
                               " answer bytes, finish reason " +
                               std::to_string(static_cast<int>(first.finish_reason)));
    }
    history.push_back(answer(first.reasoning, first.content));
    history.push_back(ask("Now name the three secondary colours the same way."));
    const ninfer::GenerationResult second = engine.generate(prepare(history), request);
    if (second.content.empty() || second.reasoning.empty()) {
        return fail(label, "the replayed turn published no answer");
    }
    history.push_back(answer(second.reasoning, second.content));
    history.push_back(ask("Which of the six is closest to grey?"));
    const ninfer::GenerationResult third = engine.generate(prepare(history), request);

    // Reach into the turn the catalog still held, not merely to the prompt the replayed turn was
    // built from. The boundary has to sit on the walk's chunk grid and
    // `second.prompt.prompt_tokens` is not on it, so the expectation is the grid floor: standing on
    // the turn's own grid image is what proves the scan read the catalog, because this lane's
    // lineage offers nothing at all (the previous turn ran on the other lane).
    if (third.reused_prompt_tokens < grid_floor(second.prompt.prompt_tokens)) {
        return fail(label, "the turn after a re-rendered answer reused " +
                               std::to_string(third.reused_prompt_tokens) + " of the " +
                               std::to_string(second.prompt.prompt_tokens) +
                               " prompt tokens it had just paid for, and none of the answer");
    }
    std::cout << "TP-2 replayed answer (" << label << ") passed: the turn after a re-rendered answer"
                 " reused "
              << third.reused_prompt_tokens << " tokens, past the " << second.prompt.prompt_tokens
              << "-token prompt it was built from\n";
    return 0;
}

// A reused boundary may only stand on rows this walk would have computed itself. The strongest form
// the engine can be held to: two walks whose chunk sequence is the same up to the boundary have to
// produce the same answer, whether the rows below it are recomputed or loaded from an image.
//
// The writer cuts the block in whole chunks and stops a little past it; the reader shares exactly
// the block, so the deepest boundary the grid allows is the block's end, and the image under it was
// written by the same chunks a cold walk of the reader's own prompt would cut. A difference can
// then only come from the image, not from the shape of the walk that wrote it - which is what
// separates "the model layer's tiles are length-dependent" from "the reuse path is not exact".
int check_image_equals_a_cold_prefix(const char* artifact, int device_a, int device_b,
                                     Route route) {
    const std::string label = std::string("image equality (") + route_name(route) + ")";

    // How many whole chunks of shared block the writer and the reader agree on. A larger block puts
    // the image deeper into the walk; the knob exists so one build can measure several depths.
    const std::uint32_t block_chunks =
        std::getenv("NINFER_IMAGE_BLOCK") != nullptr
            ? static_cast<std::uint32_t>(std::atoi(std::getenv("NINFER_IMAGE_BLOCK")))
            : 4U;
    const std::uint32_t kBlock       = block_chunks * kChunkWidth;
    const std::vector<TokenId> block = make_prompt(81000, kBlock);
    std::vector<TokenId> writer      = block;
    append(writer, make_prompt(82000, 152));
    std::vector<TokenId> reader = block;
    append(reader, make_prompt(83000, 160));

    // The cold answer comes first and its engine goes away before the second one exists: two
    // engines do not fit on the pair at once, which is the same reason run_scenario computes its
    // oracles in their own scope.
    std::vector<TokenId> cold_answer;
    {
        ninfer::Engine oracle(engine_options(artifact, device_a, device_b, false, route));
        cold_answer = run(oracle, reader).generated_token_ids;
    }

    ninfer::Engine engine(engine_options(artifact, device_a, device_b, true, route));
    const ninfer::GenerationResult writer_turn = run(engine, writer);
    if (writer_turn.reused_prompt_tokens != 0) {
        return fail(label, "the writer prefilled the block from a stale lineage: reused " +
                               std::to_string(writer_turn.reused_prompt_tokens) + " prompt tokens");
    }
    const ninfer::GenerationResult reused_turn = run(engine, reader);
    if (reused_turn.reused_prompt_tokens != kBlock) {
        return fail(label, "the reader reused " + std::to_string(reused_turn.reused_prompt_tokens) +
                               " prompt tokens, expected the block at " + std::to_string(kBlock));
    }
    if (reused_turn.generated_token_ids != cold_answer) {
        return fail(
            label,
            "a walk standing on the image diverged from the cold walk of the same prompt: got " +
                tokens_text(reused_turn.generated_token_ids) + " expected " +
                tokens_text(cold_answer));
    }
    std::cout << "TP-2 image equality (" << label << ") passed: the walk that stood on the "
              << reused_turn.reused_prompt_tokens
              << "-token image produced the cold walk's own answer\n";
    return 0;
}

// A stable block that outlives the conversation that wrote it, in the shape a client really sends:
// every request of the family opens with the same system message, and nothing else about them
// agrees. The position where that block ends is the frontend's leading-instruction frontier, which
// is what the walk freezes a state image at and what the catalog then carries - the path a raw
// token prompt cannot reach, because a bare token list names no message boundary.
//
// X writes the block, an unrelated probe and an aside take the pools away from it, and then two new
// conversations open with the same block. Both are recalled on the block, and both have to produce
// the cold walk's own answer: standing on a boundary the conversation did not walk is the failure
// this whole file exists to keep out.
int check_shared_block_recall(const char* artifact, int device_a, int device_b, Route route) {
    const std::string label = std::string("shared block (") + route_name(route) + ")";

    auto block_text = [] {
        std::string text;
        for (int line = 0; line < 40; ++line) {
            text += "Ledger rule " + std::to_string(line) +
                    ": every entry is dated, numbered, and cross-referenced to the shelf it came "
                    "from; an entry without a shelf is returned to the clerk who wrote it. ";
        }
        return text;
    };
    auto system = [](const std::string& text) {
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::System;
        ninfer::MessagePart part;
        part.kind = ninfer::MessagePartKind::Text;
        part.text = text;
        message.parts.push_back(std::move(part));
        return message;
    };
    auto ask = [](const std::string& text) {
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::User;
        ninfer::MessagePart part;
        part.kind = ninfer::MessagePartKind::Text;
        part.text = text;
        message.parts.push_back(std::move(part));
        return message;
    };
    auto prepare_for = [](ninfer::Engine& engine, std::vector<ninfer::ChatMessage> messages) {
        ninfer::PromptInput input;
        input.options.enable_thinking = false;
        input.messages                = std::move(messages);
        return engine.prepare(std::move(input));
    };
    ninfer::RequestOptions request            = greedy_request();
    request.execution.requested_output_tokens = 8;

    const std::string block                          = block_text();
    const std::vector<ninfer::ChatMessage> opening   = {system(block), ask("Open a shelf for me.")};
    std::vector<ninfer::ChatMessage> probe_history   = {system("Answer in one word."),
                                                        ask("What is the weather in Reykjavik?")};
    std::vector<ninfer::ChatMessage> aside_history   = {system("Count silently."),
                                                        ask("Count to five.")};
    std::vector<ninfer::ChatMessage> first_of_family = {
        system(block), ask("Open a shelf for the second archive.")};
    std::vector<ninfer::ChatMessage> second_of_family = {
        system(block), ask("Open a shelf for the third archive.")};

    // The cold answers first, in their own engine: the pair cannot hold two models at once, and the
    // comparison has to be against a walk that never saw this block.
    std::vector<TokenId> cold_first;
    std::vector<TokenId> cold_second;
    {
        ninfer::Engine oracle(engine_options(artifact, device_a, device_b, false, route));
        cold_first =
            oracle.generate(prepare_for(oracle, first_of_family), request).generated_token_ids;
        (void)oracle.generate(prepare_for(oracle, aside_history), request);
        cold_second =
            oracle.generate(prepare_for(oracle, second_of_family), request).generated_token_ids;
    }

    ninfer::Engine engine(engine_options(artifact, device_a, device_b, true, route));
    const ninfer::GenerationResult opening_turn =
        engine.generate(prepare_for(engine, opening), request);
    if (opening_turn.reused_prompt_tokens != 0) {
        return fail(label, "the conversation that carries the block reused " +
                               std::to_string(opening_turn.reused_prompt_tokens) +
                               " prompt tokens");
    }
    // Two unrelated requests: the first pushes the block's conversation into its host slabs, the
    // second proves the block had to survive as a stored boundary rather than as a resident
    // lineage.
    (void)engine.generate(prepare_for(engine, probe_history), request);
    (void)engine.generate(prepare_for(engine, aside_history), request);

    const ninfer::GenerationResult first_reuse =
        engine.generate(prepare_for(engine, first_of_family), request);
    if (first_reuse.reused_prompt_tokens == 0) {
        return fail(label, "a new conversation opening with the stored block re-prefilled it");
    }
    if (first_reuse.reused_prompt_tokens % kChunkWidth != 0) {
        return fail(label, "the first conversation of the family reused " +
                               std::to_string(first_reuse.reused_prompt_tokens) +
                               " prompt tokens, which is not on the walk's chunk grid");
    }
    if (first_reuse.generated_token_ids != cold_first) {
        return fail(label,
                    "a conversation recalled on the shared block diverged from the oracle: got " +
                        tokens_text(first_reuse.generated_token_ids) + " expected " +
                        tokens_text(cold_first));
    }
    const ninfer::GenerationResult second_reuse =
        engine.generate(prepare_for(engine, second_of_family), request);
    if (second_reuse.reused_prompt_tokens < first_reuse.reused_prompt_tokens) {
        return fail(label, "the second conversation of the family reused " +
                               std::to_string(second_reuse.reused_prompt_tokens) +
                               " prompt tokens, fewer than the " +
                               std::to_string(first_reuse.reused_prompt_tokens) +
                               " the first one found");
    }
    if (second_reuse.generated_token_ids != cold_second) {
        return fail(
            label,
            "a second conversation recalled on the shared block diverged from the oracle: got " +
                tokens_text(second_reuse.generated_token_ids) + " expected " +
                tokens_text(cold_second));
    }
    std::cout << "TP-2 shared block (" << label << ") passed: two new conversations reused "
              << first_reuse.reused_prompt_tokens << " and " << second_reuse.reused_prompt_tokens
              << " tokens of the stored block and matched the cold walk\n";
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    const auto devices = pick_devices();
    if (devices.first < 0) {
        std::cout << "skip: two identical sm_120a devices are not available\n";
        return 77;
    }

    // NINFER_TEST_ROUTE=plain|mtp|dflash2 narrows the run to one route, which is what a failing
    // route needs when each of the others costs a full model load. The default set is all three.
    const char* selected_env   = std::getenv("NINFER_TEST_ROUTE");
    const std::string selected = selected_env == nullptr ? "" : selected_env;
    std::vector<Route> routes;
    if (selected.empty()) {
        routes = {Route::Plain, Route::Mtp, Route::DFlash2};
    } else if (selected == "plain") {
        routes = {Route::Plain};
    } else if (selected == "mtp") {
        routes = {Route::Mtp};
    } else if (selected == "dflash2") {
        routes = {Route::DFlash2};
    } else {
        std::cerr << "FAIL: NINFER_TEST_ROUTE must be plain, mtp or dflash2\n";
        return 1;
    }

    try {
        for (const Route route : routes) {
            if (const int status = check_first_token_published(artifact, devices.first,
                                                               devices.second, route);
                status != 0) {
                return status;
            }
            if (const int status =
                    check_unaligned_dialogue(artifact, devices.first, devices.second, route);
                status != 0) {
                return status;
            }
            if (const int status = check_transport_failure_keeps_prefix(
                    artifact, devices.first, devices.second, route);
                status != 0) {
                return status;
            }
            if (const int status = check_replayed_answer_keeps_prompt_end(
                    artifact, devices.first, devices.second, route);
                status != 0) {
                return status;
            }
            if (const int status = check_replayed_answer_across_lanes(artifact, devices.first,
                                                                     devices.second, route);
                status != 0) {
                return status;
            }
            // The image path's own regression: a walk standing on an image has to reproduce the
            // cold walk of the same prompt, at every depth the ring can hand out.
            // `NINFER_IMAGE_BLOCK` sweeps the depth by hand; the gate runs the default one.
            if (const int status = check_image_equals_a_cold_prefix(artifact, devices.first,
                                                                    devices.second, route);
                status != 0) {
                return status;
            }
            if (const int status =
                    check_shared_block_recall(artifact, devices.first, devices.second, route);
                status != 0) {
                return status;
            }
            if (const int status = run_scenario(artifact, devices.first, devices.second, route);
                status != 0) {
                return status;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
