#include "serve/request_capture.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>

namespace {

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::size_t file_count(const std::filesystem::path& directory) {
    std::error_code error;
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory, error)) {
        if (entry.is_regular_file()) { ++count; }
    }
    return count;
}

} // namespace

int main() {
    using ninfer::serve::RequestCapture;

    int failures = 0;
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "ninfer_request_capture_test";
    std::error_code error;
    std::filesystem::remove_all(directory, error);

    // Disabled unless a directory is configured: the default must not touch the disk.
    {
        const RequestCapture disabled{std::filesystem::path{}};
        failures += check(!disabled.enabled(), "capture is enabled without a directory");
    }

    RequestCapture capture(directory);
    failures += check(capture.enabled(), "capture did not create its directory");

    httplib::Request request;
    request.path = "/v1/messages";
    request.headers.insert({"content-type", "application/json"});
    request.headers.insert({"x-session-id", "repro-session-1"});
    request.headers.insert({"Authorization", "Bearer sk-secret"});
    request.headers.insert({"x-api-key", "sk-secret"});
    const std::string body =
        R"({"model":"m","messages":[{"role":"user","content":"line\n\ttab"}]})";
    capture.write(request.path, RequestCapture::from_request(request), body);

    // A bodyless request carries nothing to replay.
    capture.write("/v1/messages", RequestCapture::Headers{}, "");

    const std::filesystem::path first = directory / "000000-repro-se.json";
    failures += check(std::filesystem::exists(first), "capture did not write the first request");
    failures += check(file_count(directory) == 1, "capture wrote a file for a bodyless request");
    if (std::filesystem::exists(first)) {
        std::ifstream input(first, std::ios::binary);
        const nlohmann::json record = nlohmann::json::parse(input);
        failures += check(record.at("seq") == 0, "first capture record does not report seq 0");
        failures += check(record.at("path") == "/v1/messages", "capture lost the request path");
        failures += check(record.at("body") == body,
                          "capture did not preserve the body byte for byte");
        const nlohmann::json& headers = record.at("headers");
        failures += check(headers.at("x-session-id") == "repro-session-1",
                          "capture lost the session identity header");
        failures += check(!headers.contains("Authorization"),
                          "capture stored the Authorization credential");
        failures += check(!headers.contains("x-api-key"), "capture stored the x-api-key credential");
    }

    // A request without a session header still gets a file, and the sequence stays monotonic.
    httplib::Request anonymous = request;
    anonymous.headers.erase("x-session-id");
    capture.write(anonymous.path, RequestCapture::from_request(anonymous), R"({"x":1})");
    failures += check(std::filesystem::exists(directory / "000001-nosession.json"),
                      "capture did not fall back to an anonymous session tag");
    failures += check(file_count(directory) == 2, "capture did not write exactly two files");

    std::filesystem::remove_all(directory, error);
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
