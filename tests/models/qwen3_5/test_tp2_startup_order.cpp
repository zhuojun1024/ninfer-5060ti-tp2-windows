// The product's TP-2 startup order: both devices, then one Engine load.
//
// Nothing covered this until a heap overrun made it crash: Engine::Impl sizes the TP2GenerationCore
// block in its own translation unit while the constructor fills it in another, so a stale object of
// either side silently corrupts the heap and the first visible symptom is an access violation
// inside the CUDA driver. Nine separate hypotheses (peer access, mapped staging, the checkpoint
// ring, the codec, the watchdog, the device pair, the context ceiling, the card order, a restart
// race) were eliminated before the guard-page heap below pointed at the constructor's first store.
//
// NINFER_GUARD_HEAP=1 replaces the global allocation operators so every block ends flush against a
// PAGE_NOACCESS page; an out-of-bounds write then faults at the instruction that performs it, and
// the vectored handler prints the module-relative addresses to resolve against the linker map.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#if defined(_WIN32)
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace guard_heap {

constexpr std::size_t kPageBytes = 4096;
constexpr std::uint64_t kMagic   = 0x9E3779B97F4A7C15ULL;

bool enabled() {
    static const bool value = [] {
        const char* env = std::getenv("NINFER_GUARD_HEAP");
        return env != nullptr && env[0] != '0';
    }();
    return value;
}

void* allocate(std::size_t bytes) {
    if (bytes == 0) { bytes = 1; }
    if (!enabled()) {
        void* plain = std::malloc(bytes);
        if (plain == nullptr) { throw std::bad_alloc(); }
        return plain;
    }
    // The payload's last byte sits immediately before the guard page, so any write past the end
    // faults. Eight bytes in front of the payload carry the marker the free path looks for, and the
    // page count leaves room for them.
    const std::size_t data_pages = (bytes + 8 + kPageBytes - 1) / kPageBytes;
    const std::size_t total      = (data_pages + 1) * kPageBytes;
    auto* base = static_cast<unsigned char*>(
        VirtualAlloc(nullptr, total, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (base == nullptr) { throw std::bad_alloc(); }
    DWORD previous = 0;
    VirtualProtect(base + total - kPageBytes, kPageBytes, PAGE_NOACCESS, &previous);
    unsigned char* block = base + total - kPageBytes - bytes;
    *reinterpret_cast<std::uint64_t*>(block - 8) = kMagic;
    return block;
}

void release(void* pointer) noexcept {
    if (pointer == nullptr) { return; }
    if (!enabled()) {
        std::free(pointer);
        return;
    }
    auto* block = static_cast<unsigned char*>(pointer);
    if (*reinterpret_cast<std::uint64_t*>(block - 8) != kMagic) {
        std::free(pointer);
        return;
    }
    *reinterpret_cast<std::uint64_t*>(block - 8) = 0;
    MEMORY_BASIC_INFORMATION region{};
    if (VirtualQuery(block, &region, sizeof(region)) == 0) { return; }
    VirtualFree(region.AllocationBase, 0, MEM_RELEASE);
}

} // namespace guard_heap

void* operator new(std::size_t bytes) { return guard_heap::allocate(bytes); }
void* operator new[](std::size_t bytes) { return guard_heap::allocate(bytes); }
void* operator new(std::size_t bytes, const std::nothrow_t&) noexcept {
    try {
        return guard_heap::allocate(bytes);
    } catch (...) {
        return nullptr;
    }
}
void* operator new[](std::size_t bytes, const std::nothrow_t&) noexcept {
    try {
        return guard_heap::allocate(bytes);
    } catch (...) {
        return nullptr;
    }
}
void operator delete(void* pointer) noexcept { guard_heap::release(pointer); }
void operator delete[](void* pointer) noexcept { guard_heap::release(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { guard_heap::release(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { guard_heap::release(pointer); }
void operator delete(void* pointer, const std::nothrow_t&) noexcept { guard_heap::release(pointer); }
void operator delete[](void* pointer, const std::nothrow_t&) noexcept {
    guard_heap::release(pointer);
}

namespace {

using namespace ninfer;

void describe_address(const char* label, const void* address) {
    MEMORY_BASIC_INFORMATION region{};
    char module[MAX_PATH] = {};
    if (VirtualQuery(address, &region, sizeof(region)) != 0 && region.AllocationBase != nullptr &&
        GetModuleFileNameA(static_cast<HMODULE>(region.AllocationBase), module, MAX_PATH) > 0) {
        const auto base = reinterpret_cast<std::uintptr_t>(region.AllocationBase);
        std::fprintf(stderr, "[fault] %s %s+0x%llx\n", label, module,
                     static_cast<unsigned long long>(
                         reinterpret_cast<std::uintptr_t>(address) - base));
    } else {
        std::fprintf(stderr, "[fault] %s unmapped(%p)\n", label, address);
    }
}

// A vectored handler runs before CUDA's own first-chance filter and sees the fault as it happens;
// the stack comes from the faulting context, not from this frame.
LONG WINAPI trace_first_chance(EXCEPTION_POINTERS* info) {
    const DWORD code = info->ExceptionRecord->ExceptionCode;
    if (code != EXCEPTION_ACCESS_VIOLATION && code != EXCEPTION_ILLEGAL_INSTRUCTION) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    std::fprintf(stderr, "[fault] code=0x%08lx address=%p info0=%llx data=%llx thread=%lu\n", code,
                 info->ExceptionRecord->ExceptionAddress,
                 static_cast<unsigned long long>(info->ExceptionRecord->ExceptionInformation[0]),
                 static_cast<unsigned long long>(info->ExceptionRecord->ExceptionInformation[1]),
                 GetCurrentThreadId());
    describe_address("fault-ip", info->ExceptionRecord->ExceptionAddress);
    const HANDLE process = GetCurrentProcess();
    const HANDLE thread  = GetCurrentThread();
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_NO_PROMPTS);
    SymInitialize(process, nullptr, TRUE);
    CONTEXT context = *info->ContextRecord;
    STACKFRAME64 frame{};
    frame.AddrPC.Offset    = context.Rip;
    frame.AddrPC.Mode      = AddrModeFlat;
    frame.AddrFrame.Offset = context.Rbp;
    frame.AddrFrame.Mode   = AddrModeFlat;
    frame.AddrStack.Offset = context.Rsp;
    frame.AddrStack.Mode   = AddrModeFlat;
    for (unsigned index = 0; index < 40; ++index) {
        if (StackWalk64(IMAGE_FILE_MACHINE_AMD64, process, thread, &frame, &context, nullptr,
                        SymFunctionTableAccess64, SymGetModuleBase64, nullptr) == FALSE) {
            break;
        }
        if (frame.AddrPC.Offset == 0) { break; }
        char label[16] = {};
        std::snprintf(label, sizeof(label), "#%u", index);
        describe_address(label, reinterpret_cast<const void*>(frame.AddrPC.Offset));
    }
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}

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

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path artifact;
        std::uint32_t max_context = 16384;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const auto value = [&]() -> std::string {
                if (++i >= argc) { throw std::invalid_argument("missing value for " + arg); }
                return argv[i];
            };
            if (arg == "--artifact") {
                artifact = value();
            } else if (arg == "--max-context") {
                max_context = static_cast<std::uint32_t>(std::stoul(value()));
            } else if (arg == "--help") {
                std::cout << "--artifact PATH [--max-context N]\n"
                             "NINFER_GUARD_HEAP=1 puts every heap block against a guard page.\n";
                return 0;
            } else {
                throw std::invalid_argument("unknown argument " + arg);
            }
        }
        if (artifact.empty()) {
            std::cout << "SKIP: supply an explicit --artifact\n";
            return 77;
        }
        if (guard_heap::enabled()) { AddVectoredExceptionHandler(1, trace_first_chance); }
        const auto [device_a, device_b] = pick_devices();
        if (device_a < 0) {
            std::cout << "SKIP: need two identical GPUs\n";
            return 77;
        }
        EngineOptions options;
        options.artifact_path = artifact;
        options.device        = device_a;
        options.device_b      = device_b;
        options.max_context   = max_context;
        options.kv_capacity   = KvCapacityPolicy::explicit_capacity(max_context);
        Engine engine(options);
        const LoadSummary summary = engine.load_summary();
        std::cout << artifact.filename().string() << ": TP-2 engine startup passed devices="
                  << device_a << "," << device_b << " arch=" << summary.architecture
                  << " formats=" << summary.weight_formats.size()
                  << " device_objects=" << summary.device_object_count << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
