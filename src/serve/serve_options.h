#pragma once

#include "ninfer/tp2_capacity.h"
#include "ninfer/types.h"
#include "product/logging/logging.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// Protocol default when the client omits max_tokens. Engine independently
// clamps the request to its effective context capacity.
inline constexpr int kDefaultMaxTokens                    = 8192;
inline constexpr std::size_t kDefaultMaxRequestBytes      = 384ULL << 20;
inline constexpr std::size_t kDefaultResponseStoreRecords = 1024;
inline constexpr std::size_t kDefaultResponseStoreBytes   = 256ULL << 20;

struct ServeOptions {
    bool help_requested    = false;
    bool version_requested = false;
    std::string artifact_path;
    std::filesystem::path chat_template_path;
    std::string host = "127.0.0.1";
    int port         = 8080;
    std::string api_key;                          // empty => no auth
    std::optional<std::string> model_id_override; // unset => artifact metadata.name
    std::string request_log_jsonl;                // empty => structured request logging disabled
    std::uint32_t max_context          = 8192;
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(8192);
    std::uint32_t max_concurrency      = 1;
    // TP-2 multi-lane admission ceiling in tokens; 0 lets a lane reserve the whole KV pool. See
    // EngineOptions::lane_context. A single-lane route ignores it.
    std::uint32_t lane_context         = 0;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 1024;
    // TP-2 prefill all-reduce overlap sub-block width; see EngineOptions::prefill_overlap.
    std::uint32_t prefill_overlap      = 256;
    std::filesystem::path context_cost_presets;
    std::uint32_t log_stats_interval_ms    = 5000; // 0 disables periodic Engine throughput logs
    std::size_t max_request_bytes          = kDefaultMaxRequestBytes;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
    std::size_t response_store_max_records = kDefaultResponseStoreRecords;
    std::size_t response_store_max_bytes   = kDefaultResponseStoreBytes;
    int device                             = 0;
    // Second device for tensor-parallel (TP-2) generation. -1 (default) selects the single-GPU
    // path; set via --devices <a>,<b> to run the dedicated TP-2 core across two GPUs.
    int device_b                           = -1;
    KvCacheStorage kv_cache                = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    ContextCacheOptions context_cache;
    bool enable_vision      = false;
    // Per-item merged Vision token ceiling; see EngineOptions::vision_item_tokens.
    std::uint32_t vision_item_tokens = kMaximumVisionItemTokens;
    bool use_cuda_graph     = true;
    // Keep the token embedding table in mapped host memory; see EngineOptions::embedding_host.
    bool embedding_host     = false;
    bool allow_prefix_reuse = true;
    std::optional<bool> enable_thinking;
    std::optional<bool> preserve_thinking;
    // Process-level default reasoning effort for thinking-enabled requests that do not select one.
    // Unset keeps the loaded chat template's own default; --reasoning-effort names one explicitly.
    std::optional<ninfer::ReasoningEffort> default_reasoning_effort;
    std::optional<std::uint32_t> default_thinking_budget;
    int default_max_tokens = kDefaultMaxTokens;
    bool enable_cors       = false; // send permissive CORS headers for browser UIs
    // Process-level explicit overrides layered between registered model/mode defaults and request
    // fields. An omitted seed is replaced per request with a fresh random seed.
    SamplingOverrides sampling_overrides;
    bool greedy                 = false; // --greedy: force temperature 0 (exact argmax)
    product::LogLevel log_level = product::LogLevel::Info;

    // Exact process argv for the server-start record. Secret-bearing option values are redacted
    // while parsing; this is provenance only and never affects execution.
    std::vector<std::string> startup_argv;
};

// The Engine normalizes a TP-2 generation route to `tp2_generation_concurrency(max_concurrency)`
// active lanes (one to four) plus `max_pending_requests` queued requests
// (normalize_engine_options). The service layer and the HTTP thread pool must size themselves from
// the same numbers: otherwise a request the service accepted and counted is rejected by the Engine
// as overloaded instead of waiting in its FIFO, and the thread pool admits work the FIFO cannot.
struct EffectiveRequestCapacity {
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 1;
};
[[nodiscard]] EffectiveRequestCapacity effective_request_capacity(const ServeOptions& options);

ServeOptions parse_serve_options(int argc, char** argv);
std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_name);
std::string serve_usage_text(const char* argv0);

} // namespace ninfer::serve
