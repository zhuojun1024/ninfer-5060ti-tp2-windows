#pragma once

// Diagnostic capture of the requests a server actually received (--capture-requests).
//
// The serving log records outcomes (token counts, reuse decisions); it never records the request
// itself, so a session cannot be replayed from it. Replaying the prefix-reuse layer needs the exact
// prompts a client sent, including the conversation identity that arrives in request headers.
//
// One file per request: "<seq>-<session>.json" holding the path, the non-credential headers and the
// raw body text. The body is stored verbatim rather than re-serialized: the chat template renders
// from those bytes, and a parse/re-serialize round trip changes key order and number formatting,
// which is exactly the surface a byte-exact render comparison has to preserve.
//
// This is a diagnostic tool, not a serving feature: it writes whole conversation prompts to disk,
// so it is off by default and meant to be enabled for one diagnostic run.

#include <httplib.h>

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::serve {

class RequestCapture {
public:
    // Header name/value pairs in request order; credentials are already removed.
    using Headers = std::vector<std::pair<std::string, std::string>>;

    // An empty directory disables capture, which is the default. A directory that cannot be created,
    // or a write that fails later, disables capture and reports once on stderr: a diagnostic must
    // never be able to fail a serving request.
    explicit RequestCapture(std::filesystem::path directory);
    ~RequestCapture() = default;

    RequestCapture(const RequestCapture&)            = delete;
    RequestCapture& operator=(const RequestCapture&) = delete;

    // Copies the headers a replay needs, dropping credentials (authorization, x-api-key, cookie).
    [[nodiscard]] static Headers from_request(const httplib::Request& request);

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    // Writes one request. Empty bodies are skipped: they carry no prompt to replay.
    void write(std::string_view path, const Headers& headers, std::string_view body);

private:
    // Called with mutex_ held.
    void disable(const char* what);

    std::filesystem::path directory_;
    mutable std::mutex mutex_;
    std::uint64_t seq_ = 0;
    bool enabled_      = false;
};

} // namespace ninfer::serve
