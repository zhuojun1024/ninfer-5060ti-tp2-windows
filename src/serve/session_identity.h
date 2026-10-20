#pragma once

// Conversation identity for the context-cache session catalog.
//
// A conversation reaches this server as a stream of requests whose shared history grows: a Responses
// session, or an Anthropic messages session. The engine's private-continuation catalog otherwise
// identifies a conversation only by the token prefix it shares with the resident entry, so a folded
// context, a re-rendered tail, or a context that lost one byte at the end becomes a second catalog
// entry beside the first - which is how one conversation used to fill every slot and evict the
// conversation it displaced. The identity below names the conversation instead.

#include "ninfer/types.h"
#include "serve/request_json.h"

#include <httplib.h>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::serve {

// Capacity of the engine's PreparedSessionKey. A longer name cannot be stored and is dropped.
inline constexpr std::size_t kMaximumSessionIdentityBytes = 256;

struct SessionIdentity {
    // Empty when the client named no conversation: the request stays anonymous.
    std::optional<std::string> key;
    // Header or body field that supplied the key, for the trace line.
    std::string source;
};

// Resolves the conversation identity of one request. Header precedence: x-bili-session-id,
// x-session-id, x-claude-code-session-id, x-deepseek-harness-session-id, then
// x-bili-plugin-conversation when the request also carries the x-bili-plugin marker (a plugin's
// conversation header names the harness's session only on the requests that plugin actually
// forwarded). Body fallbacks: session_id, prompt_cache_key.
// 'endpoint' only labels the trace line, which NINFER_SERVE_SESSION_TRACE=1 enables.
[[nodiscard]] SessionIdentity resolve_session_identity(const httplib::Request& request,
                                                       const RequestJson& body,
                                                       std::string_view endpoint);

// Names the request's cache hints after the client's conversation, so the engine keeps one catalog
// entry for it. Without an identity the hints are left exactly as the protocol route resolved them.
void apply_session_identity(ContextCacheHints& hints, const SessionIdentity& identity);

} // namespace ninfer::serve
