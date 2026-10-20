#include "serve/session_identity.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

// Ordered by how directly the header names the harness's own session. The optional marker is the
// header that proves a local plugin chose the name.
struct IdentityHeader {
    const char* name;
    const char* marker;
};

constexpr std::array<IdentityHeader, 5> kIdentityHeaders{{
    {"x-bili-session-id", nullptr},
    {"x-session-id", nullptr},
    {"x-claude-code-session-id", nullptr},
    {"x-deepseek-harness-session-id", nullptr},
    {"x-bili-plugin-conversation", "x-bili-plugin"},
}};

constexpr std::array<const char*, 2> kIdentityBodyFields{{"session_id", "prompt_cache_key"}};

bool session_trace_enabled() {
    const char* env = std::getenv("NINFER_SERVE_SESSION_TRACE");
    return env != nullptr && env[0] == '1';
}

// A session id reaches a log line, so control bytes become '.': a header value is client input and
// must not be able to break the line it is traced in.
std::string printable(std::string_view value) {
    std::string text;
    text.reserve(value.size());
    for (const char character : value) {
        const unsigned char byte = static_cast<unsigned char>(character);
        text.push_back(byte >= 0x20U && byte < 0x7FU ? character : '.');
    }
    return text;
}

// Empty, oversized, or (for a plugin header) unaccompanied names record why they were refused: a
// client that meant to name a conversation has to be able to see that it did not.
bool acceptable(std::string_view value, std::string_view name, std::vector<std::string>& notes) {
    if (value.empty()) {
        notes.emplace_back(std::string(name) + "=empty");
        return false;
    }
    if (value.size() > kMaximumSessionIdentityBytes) {
        notes.emplace_back(std::string(name) + "=oversized");
        return false;
    }
    return true;
}

void trace_identity(std::string_view endpoint, const SessionIdentity& identity,
                    const std::vector<std::string>& present,
                    const std::vector<std::string>& notes) {
    if (!session_trace_enabled()) { return; }
    std::fprintf(stderr, "[serve-session] endpoint=%.*s key=%s source=%s",
                 static_cast<int>(endpoint.size()), endpoint.data(),
                 identity.key ? printable(*identity.key).c_str() : "-",
                 identity.source.empty() ? "-" : identity.source.c_str());
    for (const std::string& entry : present) { std::fprintf(stderr, " %s", entry.c_str()); }
    for (const std::string& note : notes) { std::fprintf(stderr, " refused(%s)", note.c_str()); }
    std::fprintf(stderr, "\n");
}

} // namespace

SessionIdentity resolve_session_identity(const httplib::Request& request, const RequestJson& body,
                                         std::string_view endpoint) {
    SessionIdentity identity;
    std::vector<std::string> present;
    std::vector<std::string> notes;
    for (const IdentityHeader& candidate : kIdentityHeaders) {
        if (!request.has_header(candidate.name)) { continue; }
        const std::string value = request.get_header_value(candidate.name);
        present.emplace_back("hdr:" + std::string(candidate.name) + "=" + printable(value));
        if (identity.key) { continue; }
        if (candidate.marker != nullptr && !request.has_header(candidate.marker)) {
            notes.emplace_back(std::string(candidate.name) + "=no-" + candidate.marker);
            continue;
        }
        if (!acceptable(value, candidate.name, notes)) { continue; }
        identity.key    = value;
        identity.source = candidate.name;
    }
    if (body.is_object()) {
        for (const char* field : kIdentityBodyFields) {
            const auto it = body.find(field);
            if (it == body.end() || !it->is_string()) { continue; }
            const std::string value = it->get<std::string>();
            present.emplace_back("body:" + std::string(field) + "=" + printable(value));
            if (identity.key) { continue; }
            if (!acceptable(value, field, notes)) { continue; }
            identity.key    = value;
            identity.source = field;
        }
    }
    trace_identity(endpoint, identity, present, notes);
    return identity;
}

void apply_session_identity(ContextCacheHints& hints, const SessionIdentity& identity) {
    if (!identity.key) { return; }
    hints.session_key = *identity.key;
    // Naming the conversation is the client's statement that it will return to it: the entry outlives
    // the anonymous continuations around it, and the next publish replaces it rather than appending a
    // variant beside it.
    hints.retention            = CacheRetentionHint::LiveSession;
    hints.update_session_index = true;
}

} // namespace ninfer::serve
