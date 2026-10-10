// Single-instance determinism acceptance for the TP-2 DFlash2 route (tp2-plan.md section 3.6).
//
// The route's clamped verify windows once let their duplicated columns race the position owner's KV
// slot and flipped a near tie about one run in four. This case builds ONE Engine, replays a request
// sequence on it (each continuation rendered from this engine's own answer, which is what a client
// does) and prints every walk's generated tokens plus a digest of them all. Independent processes
// must print one identical digest and identical walk bodies. Inside a run, the recalled walk's
// first sample has to match the evicted (from-scratch) walk of the same prompt - the boundary
// crossing is the property the retention carries. The tails are only compared across runs: a recall
// anchored inside a prefill chunk re-walks its suffix from there, a different execution shape whose
// near ties resolve their own way against a full-window walk (docs/tp2-dual-5060ti.md).

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using TokenId = std::int32_t;

constexpr std::uint32_t kMaxContext   = 2048;
constexpr std::uint32_t kOutputTokens = 8;
constexpr std::size_t kHostKvBytes    = 128ULL << 20;
constexpr std::uint32_t kSessions     = 3;

std::pair<int, int> pick_devices() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { return {-1, -1}; }
    std::vector<std::string> names(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, index);
        names[static_cast<std::size_t>(index)] = prop.name;
    }
    for (int a = 0; a < count; ++a) {
        for (int b = a + 1; b < count; ++b) {
            if (names[static_cast<std::size_t>(a)] == names[static_cast<std::size_t>(b)]) {
                return {a, b};
            }
        }
    }
    return {-1, -1};
}

ninfer::EngineOptions engine_options(const char* artifact, int device_a, int device_b) {
    ninfer::EngineOptions options;
    options.artifact_path                          = artifact;
    options.device                                 = device_a;
    options.device_b                               = device_b;
    options.max_context                            = kMaxContext;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(kMaxContext);
    options.prefill_chunk                          = 256;
    options.max_concurrency                        = 1;
    options.max_pending_requests                   = 1;
    options.speculative.backend                    = ninfer::SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens               = 7;
    options.kv_cache                               = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.context_cache.host_kv_capacity_bytes   = kHostKvBytes;
    options.context_cache.max_private_continuations = kSessions;
    return options;
}

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

void append(std::vector<TokenId>& destination, const std::vector<TokenId>& source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

std::string tokens_text(const std::vector<TokenId>& tokens) {
    std::string text = "[";
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) { text += " "; }
        text += std::to_string(tokens[index]);
    }
    return text + "]";
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = argc > 1 ? argv[1] : std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr) {
        std::cout << "skip: set NINFER_TEST_ARTIFACT or pass the artifact path\n";
        return 77;
    }
    const auto [device_a, device_b] = pick_devices();
    if (device_a < 0) {
        std::cout << "skip: two identical sm_120a devices are not available\n";
        return 77;
    }

    try {
        ninfer::Engine engine(engine_options(artifact, device_a, device_b));

        std::uint64_t digest = 1469598103934665603ULL;
        const auto digest_walk = [&digest](const ninfer::GenerationResult& result) {
            for (const TokenId token : result.generated_token_ids) {
                digest = (digest ^ static_cast<std::uint32_t>(token)) * 1099511628211ULL;
            }
            return digest;
        };
        const auto walk = [&engine, &digest_walk](const char* name,
                                                  std::vector<TokenId> prompt) {
            const ninfer::GenerationResult result =
                engine.generate(engine.prepare_tokens(std::move(prompt)), greedy_request());
            std::cout << "[solo] " << name << " reused=" << result.reused_prompt_tokens
                      << " tokens=" << tokens_text(result.generated_token_ids) << '\n';
            digest_walk(result);
            return result;
        };

        const std::vector<TokenId> opening   = make_prompt(1200, 64);
        const std::vector<TokenId> follow_up = make_prompt(4000, 16);
        const std::vector<TokenId> other_a   = make_prompt(7000, 48);
        const std::vector<TokenId> other_b   = make_prompt(20000, 48);
        const std::vector<TokenId> other_c   = make_prompt(30000, 48);
        const std::vector<TokenId> other_d   = make_prompt(40000, 48);
        const std::vector<TokenId> system_prompt = make_prompt(500, 512);
        std::vector<TokenId> shared_a        = system_prompt;
        append(shared_a, make_prompt(9000, 128));
        std::vector<TokenId> shared_b        = system_prompt;
        append(shared_b, make_prompt(20000, 128));

        std::vector<TokenId> a_continued = opening;
        append(a_continued, walk("opening", opening).generated_token_ids);
        append(a_continued, follow_up);
        walk("other_a", other_a);
        const ninfer::GenerationResult recalled = walk("a_continued", a_continued);
        walk("other_b", other_b);
        walk("other_c", other_c);
        walk("other_d", other_d);
        const ninfer::GenerationResult evicted = walk("a_continued_evicted", a_continued);
        if (recalled.generated_token_ids.empty() || evicted.generated_token_ids.empty() ||
            recalled.generated_token_ids.front() != evicted.generated_token_ids.front()) {
            throw std::runtime_error(
                "the recalled walk diverged from the from-scratch walk on its first sample: " +
                tokens_text(recalled.generated_token_ids) + " vs " +
                tokens_text(evicted.generated_token_ids));
        }
        walk("shared_a", shared_a);
        walk("shared_b", shared_b);

        std::cout << "[solo] digest=0x" << std::hex << digest << std::dec << '\n';
        std::cout << "[solo] PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
