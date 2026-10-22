#include "serve/serve_options.h"
#include "serve/translate.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int main() {
    int failures = 0;

    const ServeOptions defaults = parse({"ninfer-serve", "model.ninfer"});
    failures += check(defaults.allow_prefix_reuse, "prefix reuse is not enabled by default");
    failures +=
        check(!defaults.preserve_thinking, "thinking history is unexpectedly preserved by default");
    failures += check(!defaults.enable_vision, "Vision is not disabled by default");
    failures += check(defaults.vision_item_tokens == ninfer::kMaximumVisionItemTokens,
                      "Vision item token ceiling default mismatch");
    failures += check(defaults.request_log_jsonl.empty(),
                      "request JSONL logging is not disabled by default");
    failures += check(defaults.capture_requests_dir.empty(),
                      "request-body capture is enabled by default");
    failures += check(defaults.context_cost_presets.empty(),
                      "external context-cost presets are unexpectedly configured by default");
    failures += check(defaults.log_stats_interval_ms == 5000,
                      "periodic throughput interval default mismatch");
    failures += check(defaults.media_cache_bytes == ninfer::kDefaultMediaCacheBytes &&
                          defaults.media_live_bytes == ninfer::kDefaultMediaLiveBytes &&
                          defaults.media_preprocess_threads == 0,
                      "media preparation resource defaults mismatch");
    failures += check(defaults.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          defaults.kv_capacity.explicit_tokens == defaults.max_context,
                      "default KV capacity does not follow max context");
    failures += check(defaults.context_cache.host_state_slots == ninfer::kDefaultHostStateSlots &&
                          defaults.context_cache.host_kv_capacity_bytes ==
                              ninfer::kDefaultHostKvCapacityBytes,
                      "Host context-cache defaults mismatch");
    failures += check(defaults.speculative.backend == ninfer::SpeculativeBackend::None,
                      "speculative decoding is not disabled by default");
    failures += check(defaults.response_store_max_records == kDefaultResponseStoreRecords &&
                          defaults.response_store_max_bytes == kDefaultResponseStoreBytes,
                      "Responses store defaults mismatch");
    failures += check(!defaults.model_id_override.has_value(),
                      "model id override is unexpectedly configured by default");
    failures += check(!defaults.default_thinking_budget,
                      "thinking budget is unexpectedly limited by default");
    failures += check(
        !defaults.sampling_overrides.temperature && !defaults.sampling_overrides.top_p &&
            !defaults.sampling_overrides.top_k && !defaults.sampling_overrides.presence_penalty &&
            !defaults.sampling_overrides.frequency_penalty,
        "server defaults unexpectedly override registered model sampling");
    failures += check(resolve_public_model_id(defaults, "artifact-model") == "artifact-model",
                      "artifact model id was not selected by default");

    const ServeOptions fp8 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "fp8"});
    failures += check(fp8.kv_cache == ninfer::KvCacheStorage::Fp8E4M3Row256,
                      "--kv-dtype fp8 did not select row-scaled E4M3 KV");
    const ServeOptions nvfp4 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ServeOptions k8v4 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    const std::string kv_help = serve_usage_text("ninfer-serve");
    failures += check(kv_help.find("nvfp4") != std::string::npos &&
                          kv_help.find("k8v4") != std::string::npos,
                      "serve help omits a production KV storage mode");

    const ServeOptions model_alias =
        parse({"ninfer-serve", "model.ninfer", "--model-id", "deployment-alias"});
    failures +=
        check(model_alias.model_id_override == "deployment-alias" &&
                  resolve_public_model_id(model_alias, "artifact-model") == "deployment-alias",
              "explicit model id did not override the artifact identity");

    const ServeOptions context_cost =
        parse({"ninfer-serve", "model.ninfer", "--context-cost-presets", "local-costs.json"});
    failures += check(context_cost.context_cost_presets == "local-costs.json",
                      "--context-cost-presets did not preserve its path");

    const ServeOptions thinking_budget =
        parse({"ninfer-serve", "model.ninfer", "--default-thinking-budget", "37"});
    failures += check(thinking_budget.default_thinking_budget == 37,
                      "--default-thinking-budget did not preserve its positive value");
    bool zero_thinking_budget_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--default-thinking-budget", "0"});
    } catch (const std::invalid_argument&) { zero_thinking_budget_rejected = true; }
    failures += check(zero_thinking_budget_rejected, "zero --default-thinking-budget was accepted");

    bool empty_model_id_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--model-id", ""});
    } catch (const std::invalid_argument&) { empty_model_id_rejected = true; }
    failures += check(empty_model_id_rejected, "empty --model-id was accepted");

    const ServeOptions dflash = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash",
                                       "--draft-tokens", "15", "--lm-head-draft"});
    failures += check(dflash.speculative.backend == ninfer::SpeculativeBackend::DFlash,
                      "--spec dflash did not select DFlash");
    failures += check(dflash.speculative.draft_tokens == 15,
                      "--draft-tokens did not preserve the DFlash window");
    failures += check(dflash.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                      "--lm-head-draft did not select the optimized proposal head");

    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto options = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash2",
                                    "--draft-tokens", std::to_string(k), "--lm-head-draft"});
        failures += check(options.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                              options.speculative.draft_tokens == k &&
                              options.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                          "serve options did not preserve DFlash2 configuration");
    }

    const ServeOptions dflash_vision = parse(
        {"ninfer-serve", "model.ninfer", "--spec", "dflash", "--draft-tokens", "15", "--vision"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                          dflash_vision.speculative.draft_tokens == 15,
                      "serve options did not preserve combined DFlash and Vision features");

    const ServeOptions item_ceiling =
        parse({"ninfer-serve", "model.ninfer", "--vision", "--vision-item-tokens", "4096"});
    failures += check(item_ceiling.vision_item_tokens == 4096,
                      "--vision-item-tokens did not reach serving options");

    bool item_ceiling_floor_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--vision-item-tokens", "1024"});
    } catch (const std::invalid_argument&) { item_ceiling_floor_rejected = true; }
    failures +=
        check(item_ceiling_floor_rejected, "--vision-item-tokens accepted a value under the floor");

    bool item_ceiling_ceiling_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--vision-item-tokens", "16385"});
    } catch (const std::invalid_argument&) { item_ceiling_ceiling_rejected = true; }
    failures += check(item_ceiling_ceiling_rejected,
                      "--vision-item-tokens accepted a value above the ceiling");

    bool implicit_backend_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--draft-tokens", "3"});
    } catch (const std::invalid_argument&) { implicit_backend_rejected = true; }
    failures += check(implicit_backend_rejected, "--draft-tokens selected a backend implicitly");

    bool lane_context_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--max-context", "8192",
                     "--lane-context", "16384"});
    } catch (const std::invalid_argument&) { lane_context_rejected = true; }
    failures += check(lane_context_rejected,
                      "--lane-context accepted a ceiling wider than --max-context");

    const ServeOptions lane_scoped = parse({"ninfer-serve", "model.ninfer", "--max-context",
                                            "8192", "--max-concurrency", "4",
                                            "--lane-context", "4096"});
    failures += check(lane_scoped.lane_context == 4096,
                      "--lane-context did not reach serving options");

    const ServeOptions configured = parse({"ninfer-serve",
                                           "model.ninfer",
                                           "--no-prefix-reuse",
                                           "--vision",
                                           "--max-concurrency",
                                           "4",
                                           "--max-pending-requests",
                                           "12",
                                           "--pending-timeout-ms",
                                           "2500",
                                           "--max-context",
                                           "4096",
                                           "--kv-capacity",
                                           "8192",
                                           "--log-stats-interval-ms",
                                           "0",
                                           "--preserve-thinking",
                                           "--media-cache-mib",
                                           "256",
                                           "--media-live-mib",
                                           "512",
                                           "--media-preprocess-threads",
                                           "6"});
    failures += check(!configured.allow_prefix_reuse,
                      "--no-prefix-reuse did not disable server prefix reuse");
    failures += check(configured.context_cache.host_state_slots == 0 &&
                          configured.context_cache.host_kv_capacity_bytes == 0,
                      "root-only server mode retained default Host capacities");
    failures += check(configured.enable_vision, "--vision did not enable Vision");
    failures += check(configured.preserve_thinking == true,
                      "--preserve-thinking did not reach serving options");
    failures +=
        check(configured.max_concurrency == 4, "--max-concurrency did not reach serving options");
    failures += check(configured.max_context == 4096 &&
                          configured.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          configured.kv_capacity.explicit_tokens == 8192,
                      "context and KV capacity options were not kept distinct");
    failures += check(configured.max_pending_requests == 12,
                      "--max-pending-requests did not reach serving options");
    failures += check(configured.pending_timeout_ms == 2500,
                      "--pending-timeout-ms did not reach serving options");
    failures += check(configured.log_stats_interval_ms == 0,
                      "--log-stats-interval-ms did not disable periodic reporting");
    failures += check(configured.media_cache_bytes == (256ULL << 20) &&
                          configured.media_live_bytes == (512ULL << 20) &&
                          configured.media_preprocess_threads == 6,
                      "media preparation limits did not reach serving options");

    const ServeOptions logging = parse({"ninfer-serve", "model.ninfer", "--log-level", "debug"});
    failures += check(logging.log_level == ninfer::product::LogLevel::Debug,
                      "log level did not reach serving options");

    const ServeOptions context_cache =
        parse({"ninfer-serve", "model.ninfer", "--device-state-slots", "3", "--host-state-slots",
               "5", "--host-kv-mib", "64", "--max-private-continuations", "9",
               "--max-shared-prefixes", "4", "--max-long-anchors-per-continuation", "2"});
    failures += check(context_cache.context_cache.enabled &&
                          context_cache.context_cache.device_state_slots == 3 &&
                          context_cache.context_cache.host_state_slots == 5 &&
                          context_cache.context_cache.host_kv_capacity_bytes == (64ULL << 20) &&
                          context_cache.context_cache.max_private_continuations == 9 &&
                          context_cache.context_cache.max_shared_prefixes == 4 &&
                          context_cache.context_cache.max_long_anchors_per_continuation == 2,
                      "context-cache capacities did not reach serving options");
    bool disabled_cache_capacity_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--no-prefix-reuse", "--host-kv-mib", "64"});
    } catch (const std::invalid_argument&) { disabled_cache_capacity_rejected = true; }
    failures += check(disabled_cache_capacity_rejected,
                      "root-only server mode accepted context-cache capacity options");

    const ServeOptions response_store =
        parse({"ninfer-serve", "model.ninfer", "--response-store-max-records", "42",
               "--response-store-max-mib", "8"});
    failures += check(response_store.response_store_max_records == 42 &&
                          response_store.response_store_max_bytes == (8ULL << 20),
                      "Responses store limits did not reach serving options");

    const ServeOptions capture =
        parse({"ninfer-serve", "model.ninfer", "--capture-requests", "captures/run-1"});
    failures += check(capture.capture_requests_dir == std::filesystem::path("captures/run-1"),
                      "--capture-requests did not reach serving options");
    bool empty_capture_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--capture-requests", ""});
    } catch (const std::invalid_argument&) { empty_capture_rejected = true; }
    failures += check(empty_capture_rejected, "--capture-requests accepted an empty directory");

    const ServeOptions sampling =
        parse({"ninfer-serve", "model.ninfer", "--temperature", "0", "--top-p", "0.9", "--top-k",
               "20", "--min-p", "0.1", "--presence-penalty", "1.25", "--frequency-penalty", "-0.5",
               "--seed", "0"});
    failures += check(sampling.sampling_overrides.temperature == 0.0F &&
                          sampling.sampling_overrides.top_p == 0.9F &&
                          sampling.sampling_overrides.top_k == 20 &&
                          sampling.sampling_overrides.min_p == 0.1F &&
                          sampling.sampling_overrides.presence_penalty == 1.25F &&
                          sampling.sampling_overrides.frequency_penalty == -0.5F &&
                          sampling.sampling_overrides.seed == 0,
                      "server sampling flags did not preserve explicit values and zeros");
    bool oversized_top_k_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--top-k", "21"});
    } catch (const std::invalid_argument&) { oversized_top_k_rejected = true; }
    failures += check(oversized_top_k_rejected,
                      "server accepted top_k beyond the executable candidate domain");

    GenerationRequest request;
    request.max_tokens   = 1;
    ninfer::PromptCapabilities prompt_capabilities;
    prompt_capabilities.enable_thinking                 = true;
    prompt_capabilities.reasoning_effort.low            = true;
    prompt_capabilities.reasoning_effort.xhigh          = true;
    prompt_capabilities.reasoning_effort.default_effort = ninfer::ReasoningEffort::XHigh;
    const auto semantics = resolve_prompt_semantics(request, defaults, prompt_capabilities);
    failures += check(!semantics.reasoning_effort && !semantics.enable_thinking &&
                          !semantics.reasoning_effort,
                      "omitted reasoning effort did not resolve to the template default");

    // --reasoning-effort is the process default for requests that omit an effort, an explicit
    // request field still wins, and a level the loaded template cannot render is rejected at parse
    // time instead of silently degrading.
    const ServeOptions effort_default =
        parse({"ninfer-serve", "model.ninfer", "--reasoning-effort", "medium"});
    failures += check(effort_default.default_reasoning_effort == ninfer::ReasoningEffort::Medium,
                      "--reasoning-effort did not reach serving options");
    ninfer::PromptCapabilities medium_capabilities = prompt_capabilities;
    medium_capabilities.reasoning_effort.medium    = true;
    // The effective effort is only defined for a request that thinks: with thinking off there is no
    // budget for an effort level to describe.
    GenerationRequest thinking_request = request;
    thinking_request.enable_thinking   = true;
    const auto medium_semantics =
        resolve_prompt_semantics(thinking_request, effort_default, medium_capabilities);
    failures += check(medium_semantics.effective_reasoning_effort == ninfer::ReasoningEffort::Medium,
                      "server reasoning-effort default was not resolved");
    bool unsupported_effort_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--reasoning-effort", "high"});
    } catch (const std::invalid_argument&) { unsupported_effort_rejected = true; }
    failures += check(unsupported_effort_rejected,
                      "--reasoning-effort accepted a level outside low|medium|xhigh");
    failures +=
        check(to_request_options(request, defaults, semantics, true).execution.allow_prefix_reuse,
              "resolved read-write cache policy did not reach Engine options");
    failures +=
        check(!to_request_options(request, defaults, semantics, false).execution.allow_prefix_reuse,
              "resolved disabled cache policy inherited external enablement");
    const ninfer::RequestOptions inherited_sampling =
        to_request_options(request, sampling, semantics, sampling.allow_prefix_reuse);
    failures += check(inherited_sampling.execution.sampling.temperature == 0.0F &&
                          inherited_sampling.execution.sampling.top_p == 0.9F &&
                          inherited_sampling.execution.sampling.seed == 0,
                      "server sampling overrides did not reach Engine options");
    request.sampling.temperature = 1.1;
    failures += check(to_request_options(request, sampling, semantics, sampling.allow_prefix_reuse)
                              .execution.sampling.temperature == 1.1F,
                      "request sampling override did not win over the server override");
    failures += check(
        to_request_options(request, thinking_budget, semantics, thinking_budget.allow_prefix_reuse)
                .execution.thinking.budget == 37,
        "thinking-enabled request did not inherit the server budget");
    request.enable_thinking = false;
    const auto non_thinking = resolve_prompt_semantics(request, thinking_budget, prompt_capabilities);
    failures += check(!non_thinking.reasoning_effort,
                      "disabled thinking retained an effective reasoning effort");
    failures += check(!to_request_options(request, thinking_budget, non_thinking,
                                          thinking_budget.allow_prefix_reuse)
                           .execution.thinking.budget,
                      "non-thinking request inherited the server thinking budget");
    request.enable_thinking.reset();
    request.reasoning_effort   = RequestedReasoningEffort::Low;
    const auto explicit_effort = resolve_prompt_semantics(request, defaults, prompt_capabilities);
    failures +=
        check(explicit_effort.reasoning_effort == ninfer::ReasoningEffort::Low &&
                  explicit_effort.effective_reasoning_effort == ninfer::ReasoningEffort::Low,
              "explicit reasoning effort did not remain the effective effort");
    failures += check(resolve_prompt_semantics(request, effort_default, medium_capabilities)
                              .effective_reasoning_effort == ninfer::ReasoningEffort::Low,
                      "request reasoning effort did not win over the server default");
    request.reasoning_effort.reset();
    failures += check(
        resolve_prompt_semantics(request, configured, prompt_capabilities).preserve_thinking == true,
        "server preserve-thinking default was not resolved");
    request.preserve_thinking = false;
    failures += check(
        resolve_prompt_semantics(request, configured, prompt_capabilities).preserve_thinking == false,
        "request preserve-thinking override did not win");

    failures +=
        check(serve_usage_text("ninfer-serve").find("--no-prefix-reuse") != std::string::npos,
              "serve help omits --no-prefix-reuse");
    failures += check(serve_usage_text("ninfer-serve").find("--host-kv-mib") != std::string::npos,
                      "serve help omits context-cache capacities");
    failures += check(serve_usage_text("ninfer-serve").find("device-state=max-concurrency") !=
                          std::string::npos,
                      "serve help omits context-cache defaults");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--preserve-thinking") != std::string::npos,
              "serve help omits --preserve-thinking");
    failures += check(serve_usage_text("ninfer-serve").find("--default-thinking-budget") !=
                          std::string::npos,
                      "serve help omits --default-thinking-budget");
    failures += check(serve_usage_text("ninfer-serve").find("--reasoning-effort low|medium|xhigh") !=
                          std::string::npos,
                      "serve help omits --reasoning-effort");
    failures += check(serve_usage_text("ninfer-serve").find("--vision") != std::string::npos,
                      "serve help omits --vision");
    failures += check(serve_usage_text("ninfer-serve").find("--vision-item-tokens") !=
                          std::string::npos,
                      "serve help omits --vision-item-tokens");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--log-stats-interval-ms") != std::string::npos,
              "serve help omits --log-stats-interval-ms");
    failures += check(serve_usage_text("ninfer-serve").find("--log-level") != std::string::npos,
                      "serve help omits the log-level control");
    failures += check(serve_usage_text("ninfer-serve").find("--media-preprocess-threads") !=
                          std::string::npos,
                      "serve help omits media preparation controls");
    failures += check(serve_usage_text("ninfer-serve").find("--kv-capacity") != std::string::npos,
                      "serve help omits --kv-capacity");
    failures += check(serve_usage_text("ninfer-serve").find("--response-store-max-mib") !=
                          std::string::npos,
                      "serve help omits Responses store limits");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--context-cost-presets") != std::string::npos,
              "serve help omits external context-cost presets");
    failures += check(serve_usage_text("ninfer-serve").find("metadata.name") != std::string::npos,
                      "serve help omits the artifact-derived model id default");

    const ServeOptions inherited =
        parse({"ninfer-serve", "model.ninfer", "--max-context", "16384"});
    failures += check(inherited.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          inherited.kv_capacity.explicit_tokens == 16384,
                      "omitted --kv-capacity did not follow --max-context");

    const ServeOptions automatic = parse({"ninfer-serve", "model.ninfer", "--kv-capacity", "auto"});
    failures += check(automatic.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          automatic.kv_capacity.explicit_tokens == 0 &&
                          automatic.kv_capacity.automatic_headroom_bytes ==
                              ninfer::kDefaultKvCapacityHeadroomBytes,
                      "--kv-capacity auto did not select automatic sizing");

    const ServeOptions headroom =
        parse({"ninfer-serve", "model.ninfer", "--kv-headroom-mib", "512", "--kv-capacity", "auto"});
    failures += check(headroom.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          headroom.kv_capacity.automatic_headroom_bytes ==
                              (static_cast<std::size_t>(512) << 20),
                      "--kv-headroom-mib did not override the automatic headroom");

    const ServeOptions maximizing =
        parse({"ninfer-serve", "model.ninfer", "--kv-capacity", "auto", "--kv-headroom-mib", "0"});
    failures += check(maximizing.kv_capacity.automatic_headroom_bytes == 0,
                      "--kv-headroom-mib 0 did not clear the automatic headroom");

    bool headroom_without_auto_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--kv-headroom-mib", "512"});
    } catch (const std::invalid_argument&) { headroom_without_auto_rejected = true; }
    failures += check(headroom_without_auto_rejected,
                      "--kv-headroom-mib without --kv-capacity auto was accepted");

    bool headroom_with_explicit_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--kv-capacity", "8192",
                     "--kv-headroom-mib", "512"});
    } catch (const std::invalid_argument&) { headroom_with_explicit_rejected = true; }
    failures += check(headroom_with_explicit_rejected,
                      "--kv-headroom-mib with an explicit --kv-capacity was accepted");

    failures +=
        check(serve_usage_text("ninfer-serve").find("--kv-headroom-mib") != std::string::npos,
              "serve help omits --kv-headroom-mib");

    failures += check(!automatic.embedding_host,
                      "--embedding-host is on without being requested");
    const ServeOptions mapped =
        parse({"ninfer-serve", "model.ninfer", "--embedding-host"});
    failures += check(mapped.embedding_host, "--embedding-host did not select mapped host weights");
    failures += check(serve_usage_text("ninfer-serve").find("--embedding-host") != std::string::npos,
                      "serve help omits --embedding-host");

    const ServeOptions logged = parse({"ninfer-serve", "model.ninfer", "--request-log-jsonl",
                                       "requests.jsonl", "--api-key", "do-not-log"});
    failures += check(logged.request_log_jsonl == "requests.jsonl",
                      "--request-log-jsonl did not preserve its path");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--request-log-jsonl") != std::string::npos,
              "serve help omits --request-log-jsonl");
    bool secret_present    = false;
    bool redaction_present = false;
    for (const std::string& argument : logged.startup_argv) {
        secret_present    = secret_present || argument == "do-not-log";
        redaction_present = redaction_present || argument == "<redacted>";
    }
    failures += check(!secret_present, "startup argv retained the API key");
    failures += check(redaction_present, "startup argv omitted the API-key redaction marker");

    // The Engine clamps a TP-2 generation route to what its core can batch into one decode round
    // (kTp2GenerationMaxConcurrency) while keeping the configured pending-request depth. The service
    // layer and the HTTP thread pool must size themselves from the same numbers, or a request the
    // service accepted is rejected by the Engine as overloaded instead of waiting in its FIFO.
    const ServeOptions tp2 = parse({"ninfer-serve", "model.ninfer", "--devices", "0,1",
                                    "--max-concurrency",
                                    std::to_string(ninfer::kTp2GenerationMaxConcurrency + 1),
                                    "--max-pending-requests", "5"});
    const EffectiveRequestCapacity tp2_capacity = effective_request_capacity(tp2);
    failures += check(tp2_capacity.max_concurrency == ninfer::kTp2GenerationMaxConcurrency,
                      "TP-2 route did not clamp the service concurrency to the core capability");
    failures += check(tp2_capacity.max_pending_requests == 5,
                      "TP-2 route dropped the service pending-request depth");
    // The MTP and DFlash2 rounds both carry a batch (P2.1b), so both routes advertise the lanes the
    // core executes; a route without one advertises a single lane, or the service would accept a
    // concurrency the core can never reach.
    const ServeOptions tp2_spec = parse({"ninfer-serve", "model.ninfer", "--devices", "0,1",
                                         "--max-concurrency", "2", "--spec", "dflash2",
                                         "--draft-tokens", "5"});
    const EffectiveRequestCapacity tp2_spec_capacity = effective_request_capacity(tp2_spec);
    failures += check(tp2_spec_capacity.max_concurrency == 2,
                      "DFlash2 TP-2 route did not advertise the lanes its core executes");
    const ServeOptions tp2_mtp = parse({"ninfer-serve", "model.ninfer", "--devices", "0,1",
                                        "--max-concurrency", "4", "--spec", "mtp",
                                        "--draft-tokens", "5"});
    failures += check(effective_request_capacity(tp2_mtp).max_concurrency == 4,
                      "MTP TP-2 route did not advertise the lanes its core executes");
    const EffectiveRequestCapacity single_capacity = effective_request_capacity(defaults);
    failures += check(single_capacity.max_concurrency == defaults.max_concurrency &&
                          single_capacity.max_pending_requests == defaults.max_pending_requests,
                      "single-GPU route did not keep the configured request capacity");

    // The TP-2 prefill overlap splits a chunk at the ops' 64-token schedule boundary, so its width
    // is either 0 (off) or a multiple of 64.
    failures += check(defaults.prefill_overlap == 256, "the prefill overlap default changed");
    failures += check(parse({"ninfer-serve", "model.ninfer", "--prefill-overlap", "128"})
                          .prefill_overlap == 128,
                      "the prefill overlap width was not parsed");
    failures += check(parse({"ninfer-serve", "model.ninfer", "--prefill-overlap", "0"})
                          .prefill_overlap == 0,
                      "the prefill overlap could not be disabled");
    bool overlap_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--prefill-overlap", "100"});
    } catch (const std::invalid_argument&) { overlap_rejected = true; }
    failures += check(overlap_rejected, "a misaligned prefill overlap width was accepted");

    // --version must report the release without an artifact path, exactly as --help does.
    const ServeOptions version = parse({"ninfer-serve", "--version"});
    failures += check(version.version_requested && !version.help_requested,
                      "--version did not request the release version");
    failures += check(serve_usage_text("ninfer-serve").find("--version") != std::string::npos,
                      "serve help omits --version");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
