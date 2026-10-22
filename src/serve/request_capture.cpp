#include "serve/request_capture.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <system_error>
#include <utility>

namespace ninfer::serve {
namespace {

// Headers that identify the conversation, in the precedence order session_identity.h resolves them.
// Only the file name depends on this list: the captured headers stay authoritative, so a new
// identity header is replayable without touching this file.
constexpr std::string_view kSessionHeaders[] = {
    "x-bili-session-id",
    "x-session-id",
    "x-claude-code-session-id",
    "x-deepseek-harness-session-id",
    "x-bili-plugin-conversation",
};

std::string lower_ascii(std::string_view text) {
    std::string lowered(text);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char byte) {
        return static_cast<char>(std::tolower(byte));
    });
    return lowered;
}

bool is_credential(std::string_view name) {
    const std::string lowered = lower_ascii(name);
    return lowered == "authorization" || lowered == "x-api-key" || lowered == "cookie";
}

std::string session_tag(const RequestCapture::Headers& headers) {
    for (const std::string_view wanted : kSessionHeaders) {
        for (const auto& header : headers) {
            if (header.second.empty() || lower_ascii(header.first) != wanted) { continue; }
            std::string tag;
            for (const char character : header.second) {
                if (tag.size() == 8) { break; }
                const auto byte = static_cast<unsigned char>(character);
                tag.push_back(std::isalnum(byte) != 0 || character == '-' ? character : '_');
            }
            if (!tag.empty()) { return tag; }
        }
    }
    return "nosession";
}

std::string file_name(std::uint64_t seq, const RequestCapture::Headers& headers) {
    char prefix[16];
    std::snprintf(prefix, sizeof prefix, "%06llu", static_cast<unsigned long long>(seq));
    return std::string(prefix) + "-" + session_tag(headers) + ".json";
}

} // namespace

RequestCapture::RequestCapture(std::filesystem::path directory) : directory_(std::move(directory)) {
    if (directory_.empty()) { return; }
    std::error_code error;
    std::filesystem::create_directories(directory_, error);
    if (!std::filesystem::is_directory(directory_)) {
        std::fprintf(stderr, "[capture] disabled: %s is not a directory (%s)\n",
                     directory_.string().c_str(), error.message().c_str());
        return;
    }
    enabled_ = true;
    std::fprintf(stderr, "[capture] writing request bodies to %s\n", directory_.string().c_str());
}

RequestCapture::Headers RequestCapture::from_request(const httplib::Request& request) {
    Headers headers;
    headers.reserve(request.headers.size());
    for (const auto& header : request.headers) {
        if (is_credential(header.first)) { continue; }
        headers.emplace_back(header.first, header.second);
    }
    return headers;
}

void RequestCapture::write(std::string_view path, const Headers& headers, std::string_view body) {
    if (!enabled_ || body.empty()) { return; }
    const std::lock_guard lock(mutex_);
    if (!enabled_) { return; }
    const std::uint64_t seq = seq_++;

    nlohmann::ordered_json record;
    record["seq"]  = seq;
    record["path"] = std::string(path);
    // An object, not an array: the identity headers a replay reads are single-valued, and an object
    // stays readable for the person running the diagnostic.
    auto captured  = nlohmann::ordered_json::object();
    for (const auto& header : headers) {
        if (is_credential(header.first)) { continue; }
        captured[header.first] = header.second;
    }
    record["headers"] = std::move(captured);
    record["body"]    = std::string(body);

    std::string text;
    try {
        text = record.dump(2);
    } catch (const nlohmann::json::exception& error) {
        disable(error.what());
        return;
    }

    std::ofstream file(directory_ / file_name(seq, headers), std::ios::binary | std::ios::trunc);
    file << text;
    file.flush();
    if (!file) { disable("cannot write the capture file"); }
}

void RequestCapture::disable(const char* what) {
    enabled_ = false;
    std::fprintf(stderr, "[capture] disabled: %s\n", what);
}

} // namespace ninfer::serve
