#include "serve/session_identity.h"

#include <initializer_list>
#include <iostream>
#include <string>
#include <utility>

namespace {

using ninfer::CacheRetentionHint;
using ninfer::serve::RequestJson;
using ninfer::serve::SessionIdentity;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAILED: " << message << '\n';
    return 1;
}

httplib::Request named_request(std::initializer_list<std::pair<const char*, std::string>> headers) {
    httplib::Request request;
    for (const auto& [name, value] : headers) { request.set_header(name, value); }
    return request;
}

SessionIdentity resolve(const httplib::Request& request, const RequestJson& body = RequestJson()) {
    return ninfer::serve::resolve_session_identity(request, body, "anthropic-messages");
}

} // namespace

int main() {
    int failures = 0;

    // The harness that forwards its own session id names the conversation directly.
    const SessionIdentity harness =
        resolve(named_request({{"x-deepseek-harness-session-id", "session-1f2e3d4c"}}));
    failures += check(harness.key && *harness.key == "session-1f2e3d4c" &&
                          harness.source == "x-deepseek-harness-session-id",
                      "DeepSeek Harness session header did not name the conversation");

    // A generic or another harness's name still outranks it: the earlier header is the one that
    // proves who is speaking, and the harness header must not override it.
    const SessionIdentity generic =
        resolve(named_request({{"x-deepseek-harness-session-id", "harness"},
                               {"x-session-id", "generic"}}));
    failures += check(generic.key && *generic.key == "generic" && generic.source == "x-session-id",
                      "x-session-id did not outrank the DeepSeek Harness session header");

    const SessionIdentity claude =
        resolve(named_request({{"x-deepseek-harness-session-id", "harness"},
                               {"x-claude-code-session-id", "claude"}}));
    failures += check(claude.key && *claude.key == "claude" &&
                          claude.source == "x-claude-code-session-id",
                      "x-claude-code-session-id did not outrank the harness header");

    // The plugin conversation header stays behind it even when its marker is present.
    const SessionIdentity both =
        resolve(named_request({{"x-deepseek-harness-session-id", "harness"},
                               {"x-bili-plugin-conversation", "plugin"},
                               {"x-bili-plugin", "1"}}));
    failures += check(both.key && *both.key == "harness" &&
                          both.source == "x-deepseek-harness-session-id",
                      "the plugin conversation header outranked the harness header");

    // An unaccompanied plugin header proves nothing.
    const SessionIdentity unmarked =
        resolve(named_request({{"x-bili-plugin-conversation", "plugin"}}));
    failures += check(!unmarked.key && unmarked.source.empty(),
                      "an unmarked plugin conversation header named the conversation");

    // Refused values (empty, oversized) leave the request anonymous rather than naming it wrongly.
    const SessionIdentity empty = resolve(named_request({{"x-deepseek-harness-session-id", ""}}));
    failures += check(!empty.key, "an empty session header named the conversation");

    const SessionIdentity oversized =
        resolve(named_request({{"x-deepseek-harness-session-id", std::string(257, 's')}}));
    failures += check(!oversized.key, "an oversized session header named the conversation");

    const SessionIdentity longest =
        resolve(named_request({{"x-deepseek-harness-session-id", std::string(256, 's')}}));
    failures += check(longest.key && longest.key->size() == 256 &&
                          longest.source == "x-deepseek-harness-session-id",
                      "the 256-byte session header was refused");

    // A refused header does not discard the body fallback the same request still carries.
    RequestJson fallback_body = RequestJson::object();
    fallback_body["prompt_cache_key"] = "cache-key";
    const SessionIdentity fallback = resolve(
        named_request({{"x-deepseek-harness-session-id", std::string(257, 's')}}), fallback_body);
    failures += check(fallback.key && *fallback.key == "cache-key" &&
                          fallback.source == "prompt_cache_key",
                      "an oversized header suppressed the body conversation identity");

    // Body field precedence is fixed by the resolver, not by insertion order in the request.
    RequestJson both_fields = RequestJson::object();
    both_fields["prompt_cache_key"] = "cache-key";
    both_fields["session_id"]       = "session-id";
    const SessionIdentity body = resolve(httplib::Request(), both_fields);
    failures += check(body.key && *body.key == "session-id" && body.source == "session_id",
                      "session_id did not outrank prompt_cache_key in the request body");

    // Naming a conversation makes it live, so the entry outlives anonymous continuations.
    ninfer::ContextCacheHints hints;
    ninfer::serve::apply_session_identity(hints, harness);
    failures += check(hints.session_key && *hints.session_key == "session-1f2e3d4c" &&
                          hints.retention == CacheRetentionHint::LiveSession &&
                          hints.update_session_index,
                      "a named conversation did not become a live session key");

    // An anonymous request keeps the protocol route's own hints untouched.
    ninfer::ContextCacheHints anonymous;
    ninfer::serve::apply_session_identity(anonymous, SessionIdentity{});
    const bool untouched = !anonymous.session_key &&
                           anonymous.retention == CacheRetentionHint::Default &&
                           anonymous.markers.empty() &&
                           anonymous.allow_engine_automatic_shared_prefixes;
    failures += check(untouched, "an anonymous request had its cache hints rewritten");

    if (failures != 0) {
        std::cerr << failures << " session identity check(s) failed\n";
        return 1;
    }
    std::cout << "session identity checks passed\n";
    return 0;
}
