// TP-2 multi-lane admission and scheduling.
//
// The TP-2 route runs a shared KV pool behind its own FIFO lane queue: a driver thread takes the
// oldest up to --max-concurrency requests as one batch, each executor keeps taking newly arrived
// requests into a later round of that batch (P2.2), and a request the pool cannot cover goes back to
// the queue front until another lane retires (S1). This test drives that queue through the public
// Engine route with four lanes and pins the scheduling contract the single-request TP-2 tests cannot
// reach: several requests in one batch each keep their own lane state, more requests than the pool
// can hold are requeued rather than dropped, an expired queue deadline is a QueueTimeout, a queued
// cancellation is a Cancelled completion, a request that arrives while a long one runs starts in a
// later round of that batch, and the driver stops without hanging when the engine is destroyed with
// work in flight.
//
// The artifact is selected with NINFER_TEST_ARTIFACT and two identical sm_120a devices are required;
// without either, the test skips with exit code 77.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using ninfer::TokenId;

constexpr std::uint32_t kMaxContext   = 2048;
constexpr std::uint32_t kLanes        = 4;
constexpr std::uint32_t kOutputTokens = 8;
// 600 prompt tokens round up to ten 64-token KV pages, so the 2048-token pool holds three lanes and
// the fourth request of a four-wide burst has to wait for a lane to retire.
constexpr std::uint32_t kLongTokens = 600;
constexpr std::uint32_t kLongOutput = 64;
constexpr std::uint32_t kPendingQueue = 16;

std::pair<int, int> pick_devices() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { return {-1, -1}; }
    // The machine may carry other parts (an older compute capability, for instance). Only sm_120a
    // devices can run the TP-2 kernels, so they are the only candidates; a foreign device in the
    // enumeration must not mask the pair we can actually use.
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

ninfer::EngineOptions engine_options(const char* artifact, int device_a, int device_b) {
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.device               = device_a;
    options.device_b             = device_b;
    options.max_context          = kMaxContext;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(kMaxContext);
    options.prefill_chunk        = 256;
    options.max_concurrency      = kLanes;
    options.max_pending_requests = kPendingQueue;
    return options;
}

// Ordinary vocabulary ids only: a special token inside the prompt could end the walk on its own and
// make a full-budget assertion meaningless.
std::vector<TokenId> make_prompt(std::uint32_t first, std::size_t length) {
    std::vector<TokenId> tokens;
    tokens.reserve(length);
    for (std::size_t index = 0; index < length; ++index) {
        const std::uint32_t value = first + static_cast<std::uint32_t>(index) * 977U;
        tokens.push_back(static_cast<TokenId>(1000U + value % 60000U));
    }
    return tokens;
}

ninfer::RequestOptions request_options(std::uint32_t budget, bool allow_reuse = true) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = budget;
    options.execution.allow_prefix_reuse      = allow_reuse;
    // Greedy sampling plus no model-default stop token makes the walk a deterministic function of the
    // KV: it always spends the whole budget and finishes on OutputLimit.
    options.execution.sampling.temperature = 0.0F;
    options.execution.sampling.top_k       = 0;
    options.stop.include_model_defaults    = false;
    return options;
}

struct Job {
    std::vector<TokenId> prompt;
    std::uint32_t budget = kOutputTokens;
    std::chrono::steady_clock::time_point deadline{};
    bool cancel      = false;
    bool allow_reuse = true;
};

struct Outcome {
    std::vector<TokenId> tokens;
    ninfer::FinishReason finish    = ninfer::FinishReason::None;
    ninfer::RequestErrorKind kind  = ninfer::RequestErrorKind::Unavailable;
    std::string error;
    bool threw         = false;
    bool request_error = false;
    double seconds     = 0.0;
};

Outcome run_job(ninfer::Engine& engine, Job job) {
    Outcome outcome;
    const auto started = std::chrono::steady_clock::now();
    ninfer::CancellationView cancellation;
    if (job.cancel) { cancellation = ninfer::CancellationView([] { return true; }); }
    try {
        auto handle = engine.submit(engine.prepare_tokens(std::move(job.prompt)),
                                    request_options(job.budget, job.allow_reuse),
                                    ninfer::OutputConsumerMode::Aggregate, {}, job.deadline);
        ninfer::GenerationResult result = handle.wait(nullptr, cancellation);
        outcome.tokens = std::move(result.generated_token_ids);
        outcome.finish = result.finish_reason;
    } catch (const ninfer::RequestError& error) {
        outcome.threw         = true;
        outcome.request_error = true;
        outcome.kind          = error.kind();
        outcome.error         = error.what();
    } catch (const std::exception& error) {
        outcome.threw = true;
        outcome.error = error.what();
    }
    outcome.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return outcome;
}

// One thread per request: the lane queue only forms a batch when several submitters wait at the same
// time, which is what the protocol adapters do with one worker per request.
std::vector<Outcome> run_jobs(ninfer::Engine& engine, std::vector<Job> jobs) {
    std::vector<Outcome> outcomes(jobs.size());
    std::vector<std::thread> threads;
    threads.reserve(jobs.size());
    for (std::size_t index = 0; index < jobs.size(); ++index) {
        threads.emplace_back([&engine, &jobs, &outcomes, index] {
            outcomes[index] = run_job(engine, std::move(jobs[index]));
        });
    }
    for (std::thread& thread : threads) { thread.join(); }
    return outcomes;
}

std::string token_list(const std::vector<TokenId>& tokens) {
    std::string text;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) { text += ' '; }
        text += std::to_string(tokens[index]);
    }
    return text;
}

bool full_budget(const Outcome& outcome, std::uint32_t budget) {
    return !outcome.threw && outcome.finish == ninfer::FinishReason::OutputLimit &&
           outcome.tokens.size() == budget;
}

int check_batch_lanes(ninfer::Engine& engine) {
    int failures = 0;
    // Four distinct prompts in one burst: every lane keeps its own KV row and state slot, so all four
    // spend their own budget.
    std::vector<Job> jobs;
    for (std::uint32_t lane = 0; lane < kLanes; ++lane) {
        jobs.push_back(Job{make_prompt(1000U + lane * 4000U, 64), kOutputTokens, {}, false, true});
    }
    const std::vector<Outcome> distinct = run_jobs(engine, std::move(jobs));
    for (std::size_t lane = 0; lane < distinct.size(); ++lane) {
        if (full_budget(distinct[lane], kOutputTokens)) { continue; }
        std::cerr << "FAIL: batched lane " << lane << " finished on "
                  << static_cast<unsigned>(distinct[lane].finish) << " with "
                  << distinct[lane].tokens.size() << " tokens, expected " << kOutputTokens
                  << " on OutputLimit";
        if (distinct[lane].threw) { std::cerr << " (threw: " << distinct[lane].error << ')'; }
        std::cerr << '\n';
        ++failures;
    }

    // The same prompt four times with reuse off. The lane that arrives first starts decoding while
    // the others are still being admitted, so the four never share a batch shape: the round grows
    // from one column to four as they join, and the column count is part of a decode's numerical
    // path. Bitwise equality between them is therefore not something this engine promises - the
    // verify-window check in test_tp2_forward.cpp states the same thing and tests the property that
    // does hold, which is that two lanes inside one batched call agree exactly.
    //
    // What this case can still assert, and what actually catches a lane reading another lane's KV
    // row or state slot, is the first token: every prefill here is a solo prefill of the same
    // 64-token prompt, so all four first samples come from the same shape and must agree. A lane
    // that mixed its columns would sample the other lane's prompt and fail this immediately.
    std::vector<TokenId> shared = make_prompt(4242U, 64);
    jobs.clear();
    for (std::uint32_t lane = 0; lane < kLanes; ++lane) {
        jobs.push_back(Job{shared, kOutputTokens, {}, false, false});
    }
    const std::vector<Outcome> identical = run_jobs(engine, std::move(jobs));
    for (std::size_t lane = 0; lane < identical.size(); ++lane) {
        if (!full_budget(identical[lane], kOutputTokens)) {
            std::cerr << "FAIL: identical lane " << lane << " finished on "
                      << static_cast<unsigned>(identical[lane].finish) << " with "
                      << identical[lane].tokens.size() << " tokens, expected " << kOutputTokens
                      << " on OutputLimit";
            if (identical[lane].threw) { std::cerr << " (threw: " << identical[lane].error << ')'; }
            std::cerr << '\n';
            ++failures;
        } else if (identical[lane].tokens.empty() ||
                   identical[lane].tokens.front() != identical[0].tokens.front()) {
            std::cerr << "FAIL: identical lane " << lane << " sampled "
                      << identical[lane].tokens.front() << " first, lane 0 sampled "
                      << identical[0].tokens.front() << '\n';
            ++failures;
        }
    }
    if (failures == 0) {
        std::cout << "TP-2 lanes: four batched lanes kept their own state, and four identical "
                     "requests sampled the same first token\n";
    }
    return failures;
}

int check_requeue(ninfer::Engine& engine) {
    // The pool holds three of these lanes at ten pages each, so the requests that do not fit go back
    // to the queue front and are admitted as earlier lanes retire. A lost or dropped member would
    // hang here, and a member that took another lane's pages would fail its budget.
    std::vector<Job> jobs;
    for (std::uint32_t index = 0; index < 6; ++index) {
        jobs.push_back(
            Job{make_prompt(2000U + index * 3000U, kLongTokens), kOutputTokens, {}, false, true});
    }
    const std::vector<Outcome> outcomes = run_jobs(engine, std::move(jobs));
    int failures = 0;
    for (std::size_t index = 0; index < outcomes.size(); ++index) {
        if (full_budget(outcomes[index], kOutputTokens)) { continue; }
        std::cerr << "FAIL: requeued request " << index << " finished on "
                  << static_cast<unsigned>(outcomes[index].finish) << " with "
                  << outcomes[index].tokens.size() << " tokens, expected " << kOutputTokens
                  << " on OutputLimit";
        if (outcomes[index].threw) { std::cerr << " (threw: " << outcomes[index].error << ')'; }
        std::cerr << '\n';
        ++failures;
    }
    if (failures == 0) {
        std::cout << "TP-2 lanes: six requests shared a pool that holds three lanes and all "
                     "completed\n";
    }
    return failures;
}

int check_queue_retirement(ninfer::Engine& engine) {
    int failures = 0;
    // Four long requests hold the lanes while two more arrive: one whose deadline has already passed
    // and one the client cancelled. Neither may take a lane, and each has to be retired with its own
    // outcome - a QueueTimeout for the deadline, a Cancelled completion for the client.
    std::vector<Job> holding;
    for (std::uint32_t lane = 0; lane < kLanes; ++lane) {
        holding.push_back(
            Job{make_prompt(700U + lane * 1000U, kLongTokens), kLongOutput, {}, false, true});
    }
    std::vector<Outcome> held(kLanes);
    std::vector<std::thread> threads;
    threads.reserve(kLanes);
    for (std::size_t lane = 0; lane < holding.size(); ++lane) {
        threads.emplace_back([&engine, &holding, &held, lane] {
            held[lane] = run_job(engine, std::move(holding[lane]));
        });
    }
    // Let the four reach the driver first, so the two below are retired from the queue by their own
    // submitter. The driver retires them the same way when it pops a batch, so both paths satisfy
    // these assertions; this is the one the queue itself owns.
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    std::vector<Job> queued;
    queued.push_back(Job{make_prompt(9001U, 64), kOutputTokens,
                         std::chrono::steady_clock::now() - std::chrono::milliseconds(1), false,
                         true});
    queued.push_back(Job{make_prompt(9101U, 64), kOutputTokens, {}, true, true});
    const std::vector<Outcome> retired = run_jobs(engine, std::move(queued));
    for (std::thread& thread : threads) { thread.join(); }

    for (std::size_t lane = 0; lane < held.size(); ++lane) {
        if (full_budget(held[lane], kLongOutput)) { continue; }
        std::cerr << "FAIL: holding lane " << lane << " finished on "
                  << static_cast<unsigned>(held[lane].finish) << " with " << held[lane].tokens.size()
                  << " tokens, expected " << kLongOutput << " on OutputLimit";
        if (held[lane].threw) { std::cerr << " (threw: " << held[lane].error << ')'; }
        std::cerr << '\n';
        ++failures;
    }

    const Outcome& expired = retired[0];
    if (!expired.threw || !expired.request_error ||
        expired.kind != ninfer::RequestErrorKind::QueueTimeout) {
        std::cerr << "FAIL: the expired request was not retired as a QueueTimeout (threw="
                  << expired.threw << ", kind=" << static_cast<unsigned>(expired.kind) << ", error=\""
                  << expired.error << "\")\n";
        ++failures;
    } else if (expired.error.find("expired while waiting for admission") == std::string::npos) {
        std::cerr << "FAIL: the QueueTimeout did not carry the admission message: \"" << expired.error
                  << "\"\n";
        ++failures;
    }

    const Outcome& cancelled = retired[1];
    if (cancelled.threw || cancelled.finish != ninfer::FinishReason::Cancelled ||
        !cancelled.tokens.empty()) {
        std::cerr << "FAIL: the queued cancellation was not retired as Cancelled (threw="
                  << cancelled.threw << ", finish=" << static_cast<unsigned>(cancelled.finish)
                  << ", tokens=" << cancelled.tokens.size() << ", error=\"" << cancelled.error
                  << "\")\n";
        ++failures;
    }

    // A retirement must not cost the queue its health: the pool the four lanes held is free again.
    const Outcome after = run_job(engine, Job{make_prompt(333U, 32), kOutputTokens, {}, false, true});
    if (!full_budget(after, kOutputTokens)) {
        std::cerr << "FAIL: the request after the retirements finished on "
                  << static_cast<unsigned>(after.finish) << " with " << after.tokens.size()
                  << " tokens, expected " << kOutputTokens << " on OutputLimit";
        if (after.threw) { std::cerr << " (threw: " << after.error << ')'; }
        std::cerr << '\n';
        ++failures;
    }

    if (failures == 0) {
        std::cout << "TP-2 lanes: an expired deadline and a queued cancellation retired without a "
                     "lane, and the queue stayed healthy\n";
    }
    return failures;
}

int check_mid_batch_admission(ninfer::Engine& engine) {
    // A short request that arrives while a long one runs must start in a later round of that batch
    // (P2.2), not after the long walk ends. The long walk is the yardstick: the short one has to
    // finish in well under half of it.
    Job long_job{make_prompt(555U, kLongTokens), 96, {}, false, true};
    Outcome long_outcome;
    std::thread long_thread([&engine, &long_job, &long_outcome] {
        long_outcome = run_job(engine, std::move(long_job));
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const Outcome short_outcome =
        run_job(engine, Job{make_prompt(777U, 32), 1, {}, false, true});
    long_thread.join();

    int failures = 0;
    if (!full_budget(long_outcome, 96)) {
        std::cerr << "FAIL: the running request finished on " << static_cast<unsigned>(long_outcome.finish)
                  << " with " << long_outcome.tokens.size() << " tokens, expected 96 on OutputLimit";
        if (long_outcome.threw) { std::cerr << " (threw: " << long_outcome.error << ')'; }
        std::cerr << '\n';
        ++failures;
    }
    if (!full_budget(short_outcome, 1)) {
        std::cerr << "FAIL: the mid-batch arrival finished on "
                  << static_cast<unsigned>(short_outcome.finish) << " with "
                  << short_outcome.tokens.size() << " tokens, expected 1 on OutputLimit";
        if (short_outcome.threw) { std::cerr << " (threw: " << short_outcome.error << ')'; }
        std::cerr << '\n';
        ++failures;
    }
    if (short_outcome.seconds * 2.0 >= long_outcome.seconds) {
        std::cerr << "FAIL: the request that arrived mid-batch took " << short_outcome.seconds
                  << "s of the running request's " << long_outcome.seconds
                  << "s; it waited for the long walk\n";
        ++failures;
    }
    if (failures == 0) {
        std::cout << "TP-2 lanes: the mid-batch arrival finished in " << short_outcome.seconds
                  << "s of the running request's " << long_outcome.seconds << "s\n";
    }
    return failures;
}

// The driver thread is started by the constructor and stopped and joined by the destructor.
// Destroying the engine with a request still in flight must not hang: the handle is dropped (which
// cancels the request) and the engine is destroyed while the driver owns it.
int check_driver_stop(std::unique_ptr<ninfer::Engine>& engine) {
    {
        auto abandoned =
            engine->submit(engine->prepare_tokens(make_prompt(1234U, kLongTokens)),
                           request_options(kLongOutput), ninfer::OutputConsumerMode::Aggregate);
    }
    engine.reset();
    std::cout << "TP-2 lanes: the driver stopped and joined with a request in flight\n";
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

    try {
        auto engine =
            std::make_unique<ninfer::Engine>(engine_options(artifact, devices.first, devices.second));
        int failures = 0;
        failures += check_batch_lanes(*engine);
        failures += check_requeue(*engine);
        failures += check_queue_retirement(*engine);
        failures += check_mid_batch_admission(*engine);
        if (failures != 0) { return 1; }
        failures += check_driver_stop(engine);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
