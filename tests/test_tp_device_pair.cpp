#include "core/tp/device_pair.h"

#include "core/arena.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

// Find two distinct devices. Prefers two devices with the same name (the TP
// pair); falls back to any two devices for the communication contract.
std::pair<int, int> pick_devices() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { return {-1, -1}; }
    std::vector<std::string> names(count);
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, i);
        names[i] = prop.name;
    }
    for (int a = 0; a < count; ++a) {
        for (int b = a + 1; b < count; ++b) {
            if (names[a] == names[b]) { return {a, b}; }
        }
    }
    return {0, 1};
}

std::uint16_t bits_of(__nv_bfloat16 value) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// One payload size, repeated with fresh operands. The repetition is what makes the
// transport's ordering observable: a peer read that resolved to an earlier call's
// staging slot agrees with the current operands only by accident, while a read that
// resolved to the previous cache line of the same slot cannot agree at all. Every
// path (in-kernel, peer-copy and host staging) must produce the identical
// elementwise BF16 rounding add on both shards, so the check is bit for bit.
int check_allreduce(ninfer::tp::DevicePair& pair, std::size_t count_bytes, int iterations) {
    const std::size_t elements = count_bytes / 2;
    pair.a().bind_to_current_thread();
    ninfer::DeviceBuffer buf_a(count_bytes);
    pair.b().bind_to_current_thread();
    ninfer::DeviceBuffer buf_b(count_bytes);

    // The allreduce enqueues on the caller's compute streams; mirror production by
    // driving it on per-device streams and settling them before the read-back.
    cudaStream_t stream_a = nullptr, stream_b = nullptr;
    pair.a().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking);
    pair.b().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking);

    std::mt19937 rng(0x5EEDu + static_cast<unsigned>(count_bytes % 65521u));
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<__nv_bfloat16> bf_a(elements), bf_b(elements);
    std::vector<__nv_bfloat16> got_a(elements), got_b(elements);

    int failures = 0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (std::size_t i = 0; i < elements; ++i) {
            bf_a[i] = __float2bfloat16(dist(rng));
            bf_b[i] = __float2bfloat16(dist(rng));
        }
        pair.a().bind_to_current_thread();
        buf_a.copy_from_host(bf_a.data(), count_bytes);
        pair.b().bind_to_current_thread();
        buf_b.copy_from_host(bf_b.data(), count_bytes);

        pair.allreduce(buf_a.p, buf_b.p, count_bytes, stream_a, stream_b);
        pair.a().bind_to_current_thread();
        cudaStreamSynchronize(stream_a);
        pair.b().bind_to_current_thread();
        cudaStreamSynchronize(stream_b);

        pair.a().bind_to_current_thread();
        buf_a.copy_to_host(got_a.data(), count_bytes);
        pair.b().bind_to_current_thread();
        buf_b.copy_to_host(got_b.data(), count_bytes);

        for (std::size_t i = 0; i < elements; ++i) {
            const __nv_bfloat16 expected =
                __float2bfloat16(__bfloat162float(bf_a[i]) + __bfloat162float(bf_b[i]));
            if (bits_of(got_a[i]) != bits_of(expected) || bits_of(got_b[i]) != bits_of(expected)) {
                if (failures < 4) {
                    std::cerr << "allreduce[" << count_bytes << "] iteration " << iteration
                              << " element " << i << ": expected " << __bfloat162float(expected)
                              << ", shard a " << __bfloat162float(got_a[i]) << ", shard b "
                              << __bfloat162float(got_b[i]) << '\n';
                }
                ++failures;
                break;
            }
        }
    }
    pair.a().bind_to_current_thread();
    cudaStreamDestroy(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamDestroy(stream_b);
    return failures;
}

// The fp8 wire format: both shards consume the same staged sum, so they must land on the identical
// Q(a)+Q(b) for every element, and each element has to stay inside the e4m3 block-relative bound.
// The format is a prefill-only setting, so the sizes are the per-layer delta of a 256-token stage and
// of a short tail chunk.
int check_allreduce_fp8(ninfer::tp::DevicePair& pair, std::size_t count_bytes, int iterations) {
    if (!pair.in_kernel_allreduce() || count_bytes > pair.in_kernel_allreduce_bytes()) {
        std::cout << "SKIP fp8 allreduce[" << count_bytes
                  << "]: the in-kernel transport is unavailable\n";
        return 0;
    }
    const std::size_t elements = count_bytes / 2;
    pair.a().bind_to_current_thread();
    ninfer::DeviceBuffer buf_a(count_bytes);
    pair.b().bind_to_current_thread();
    ninfer::DeviceBuffer buf_b(count_bytes);
    cudaStream_t stream_a = nullptr, stream_b = nullptr;
    pair.a().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking);
    pair.b().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking);

    std::mt19937 rng(0xF8E4u + static_cast<unsigned>(count_bytes % 65521u));
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<__nv_bfloat16> bf_a(elements), bf_b(elements);
    std::vector<__nv_bfloat16> got_a(elements), got_b(elements);

    int failures = 0;
    for (int iteration = 0; iteration < iterations && failures == 0; ++iteration) {
        for (std::size_t i = 0; i < elements; ++i) {
            bf_a[i] = __float2bfloat16(dist(rng));
            bf_b[i] = __float2bfloat16(dist(rng));
        }
        pair.a().bind_to_current_thread();
        buf_a.copy_from_host(bf_a.data(), count_bytes);
        pair.b().bind_to_current_thread();
        buf_b.copy_from_host(bf_b.data(), count_bytes);
        pair.allreduce(buf_a.p, buf_b.p, count_bytes, stream_a, stream_b, {},
                       ninfer::tp::ArCodec::Fp8E4m3);
        pair.a().bind_to_current_thread();
        cudaStreamSynchronize(stream_a);
        pair.b().bind_to_current_thread();
        cudaStreamSynchronize(stream_b);
        pair.a().bind_to_current_thread();
        buf_a.copy_to_host(got_a.data(), count_bytes);
        pair.b().bind_to_current_thread();
        buf_b.copy_to_host(got_b.data(), count_bytes);

        for (std::size_t i = 0; i < elements; ++i) {
            const float a   = __bfloat162float(bf_a[i]);
            const float b   = __bfloat162float(bf_b[i]);
            const float got = __bfloat162float(got_a[i]);
            // Two e4m3 half-ulps (2^-4 relative each) plus the BF16 rounding of the sum, with slack.
            const float bound = 0.125f * (std::fabs(a) + std::fabs(b)) + 1.0e-4f;
            const bool same_shards = bits_of(got_a[i]) == bits_of(got_b[i]);
            if (!same_shards || !(std::fabs(got - (a + b)) <= bound)) {
                if (failures < 4) {
                    std::cerr << "fp8 allreduce[" << count_bytes << "] iteration " << iteration
                              << " element " << i << ": a=" << a << " b=" << b
                              << " got a=" << __bfloat162float(got_a[i])
                              << " got b=" << __bfloat162float(got_b[i])
                              << " bound=" << bound << '\n';
                }
                ++failures;
                break;
            }
        }
    }
    pair.a().bind_to_current_thread();
    cudaStreamDestroy(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamDestroy(stream_b);
    return failures;
}

// The codec's contract: a payload of whole 128-element blocks is required, and a pair without the
// in-kernel transport must refuse instead of silently taking the bf16 host-staging path.
int check_allreduce_fp8_contract(ninfer::tp::DevicePair& pair) {
    const bool in_kernel = pair.in_kernel_allreduce();
    const std::size_t bytes = in_kernel ? 208 : 2048;
    ninfer::DeviceBuffer buf_a(bytes), buf_b(bytes);
    cudaStream_t stream_a = nullptr, stream_b = nullptr;
    pair.a().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking);
    pair.b().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking);
    int failures = 0;
    try {
        pair.allreduce(buf_a.p, buf_b.p, bytes, stream_a, stream_b, {},
                       ninfer::tp::ArCodec::Fp8E4m3);
        std::cerr << "fp8 allreduce[" << bytes << "] was accepted by a pair that cannot run it\n";
        ++failures;
    } catch (const std::exception& expected) {
        std::cout << "fp8 contract[" << bytes << "]: refused as expected: " << expected.what()
                  << '\n';
    }
    pair.a().bind_to_current_thread();
    cudaStreamDestroy(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamDestroy(stream_b);
    return failures;
}

// The production pattern: many allreduces queued back to back on each shard's own
// stream, with no host synchronization between them. Each device then runs ahead of
// the other by as much as its own queue allows, which is the only condition under
// which the two parity slots of the mapped-host staging can be reused while the peer
// still reads them. Every call owns its own device buffers so the queued calls cannot
// overwrite each other's operands, and the whole queue is verified bit for bit.
int check_allreduce_queue(ninfer::tp::DevicePair& pair, std::size_t count_bytes, int calls) {
    const std::size_t elements = count_bytes / 2;
    cudaStream_t stream_a = nullptr, stream_b = nullptr;
    pair.a().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking);
    pair.b().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking);

    std::mt19937 rng(0xA11C0u + static_cast<unsigned>(count_bytes % 65521u));
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<__nv_bfloat16> bf_a(elements), bf_b(elements);
    std::vector<std::unique_ptr<ninfer::DeviceBuffer>> buf_a, buf_b;
    for (int call = 0; call < calls; ++call) {
        for (std::size_t i = 0; i < elements; ++i) {
            bf_a[i] = __float2bfloat16(dist(rng));
            bf_b[i] = __float2bfloat16(dist(rng));
        }
        pair.a().bind_to_current_thread();
        buf_a.push_back(std::make_unique<ninfer::DeviceBuffer>(count_bytes));
        buf_a.back()->copy_from_host(bf_a.data(), count_bytes);
        pair.b().bind_to_current_thread();
        buf_b.push_back(std::make_unique<ninfer::DeviceBuffer>(count_bytes));
        buf_b.back()->copy_from_host(bf_b.data(), count_bytes);
        pair.allreduce(buf_a.back()->p, buf_b.back()->p, count_bytes, stream_a, stream_b);
    }
    pair.a().bind_to_current_thread();
    cudaStreamSynchronize(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamSynchronize(stream_b);

    // Re-derive the operands with the same generator so the check sees each call's
    // own inputs without holding them all on the host at once.
    std::mt19937 verify_rng(0xA11C0u + static_cast<unsigned>(count_bytes % 65521u));
    std::vector<__nv_bfloat16> expect_b(elements), got(elements);
    int failures = 0;
    for (int call = 0; call < calls; ++call) {
        for (std::size_t i = 0; i < elements; ++i) {
            bf_a[i] = __float2bfloat16(dist(verify_rng));
            bf_b[i] = __float2bfloat16(dist(verify_rng));
        }
        expect_b = bf_b;
        for (int which = 0; which < 2; ++which) {
            if (which == 0) {
                pair.a().bind_to_current_thread();
                buf_a[call]->copy_to_host(got.data(), count_bytes);
            } else {
                pair.b().bind_to_current_thread();
                buf_b[call]->copy_to_host(got.data(), count_bytes);
            }
            for (std::size_t i = 0; i < elements; ++i) {
                const __nv_bfloat16 expected =
                    __float2bfloat16(__bfloat162float(bf_a[i]) + __bfloat162float(expect_b[i]));
                if (bits_of(got[i]) != bits_of(expected)) {
                    if (failures < 4) {
                        std::cerr << "allreduce-queue[" << count_bytes << "] call " << call << " shard "
                                  << which << " element " << i << ": expected "
                                  << __bfloat162float(expected) << ", got "
                                  << __bfloat162float(got[i]) << '\n';
                    }
                    ++failures;
                    break;
                }
            }
        }
    }
    pair.a().bind_to_current_thread();
    cudaStreamDestroy(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamDestroy(stream_b);
    return failures;
}

// Byte-exact exchange: each shard must receive the peer's send buffer bit for bit, including lanes
// that spell a signaling NaN, which a BF16 add would quiet. The payloads mix adversarial 16-bit
// lanes with the I32 id / FP32 score bit patterns the split proposal head and the peer selector
// transport carry. Both the single-block and the sliced in-kernel paths are covered by the sizes.
int check_sendrecv(ninfer::tp::DevicePair& pair, std::size_t count_bytes, int iterations) {
    const std::size_t words = count_bytes / 4;
    constexpr std::uint16_t kAdversarial[] = {0x7C01, 0x7C3F, 0xFC01, 0xFC3F,
                                              0xFFFF, 0x0001, 0x7F80, 0x0000};
    constexpr std::size_t kLanes = sizeof(kAdversarial) / sizeof(kAdversarial[0]);
    int failures = 0;
    std::vector<std::uint32_t> host_a(words), host_b(words), got(words);

    pair.a().bind_to_current_thread();
    ninfer::DeviceBuffer send_a(count_bytes), recv_a(count_bytes);
    cudaStream_t stream_a = nullptr;
    cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking);
    pair.b().bind_to_current_thread();
    ninfer::DeviceBuffer send_b(count_bytes), recv_b(count_bytes);
    cudaStream_t stream_b = nullptr;
    cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking);

    for (int iteration = 0; iteration < iterations && failures == 0; ++iteration) {
        for (std::size_t i = 0; i < words; ++i) {
            const std::uint16_t low = kAdversarial[(i + static_cast<std::size_t>(iteration)) % kLanes];
            const std::uint16_t high =
                static_cast<std::uint16_t>((i * 2654435761u + static_cast<std::size_t>(iteration) * 40503u) & 0xFFFFu);
            host_a[i] = static_cast<std::uint32_t>(low) | (static_cast<std::uint32_t>(high) << 16);
            host_b[i] = ~(host_a[i] + 0x9E3779B9u + static_cast<std::uint32_t>(iteration));
        }
        pair.a().bind_to_current_thread();
        cudaMemcpyAsync(send_a.p, host_a.data(), count_bytes, cudaMemcpyHostToDevice, stream_a);
        cudaMemsetAsync(recv_a.p, 0xA5, count_bytes, stream_a);
        pair.b().bind_to_current_thread();
        cudaMemcpyAsync(send_b.p, host_b.data(), count_bytes, cudaMemcpyHostToDevice, stream_b);
        cudaMemsetAsync(recv_b.p, 0x5A, count_bytes, stream_b);
        pair.sendrecv(send_a.p, recv_a.p, send_b.p, recv_b.p, count_bytes, stream_a, stream_b);
        pair.a().bind_to_current_thread();
        cudaMemcpyAsync(got.data(), recv_a.p, count_bytes, cudaMemcpyDeviceToHost, stream_a);
        cudaStreamSynchronize(stream_a);
        if (std::memcmp(got.data(), host_b.data(), count_bytes) != 0) {
            std::cerr << "sendrecv[" << count_bytes << "] iteration " << iteration
                      << ": shard a did not receive shard b's bytes\n";
            ++failures;
            break;
        }
        pair.b().bind_to_current_thread();
        cudaMemcpyAsync(got.data(), recv_b.p, count_bytes, cudaMemcpyDeviceToHost, stream_b);
        cudaStreamSynchronize(stream_b);
        if (std::memcmp(got.data(), host_a.data(), count_bytes) != 0) {
            std::cerr << "sendrecv[" << count_bytes << "] iteration " << iteration
                      << ": shard b did not receive shard a's bytes\n";
            ++failures;
            break;
        }
    }
    cudaStreamDestroy(stream_a);
    cudaStreamDestroy(stream_b);
    return failures;
}

// The MTP stem's id broadcast: one shard owns the token ids and the peer needs them, and each side
// passes its single buffer as both its send and its receive. The payload is I32 token ids, so it
// must travel byte exact - the BF16 all-reduce this route used to take quieted a signaling-NaN lane
// and handed the peer a different token. The ids below are the real trigger set for that failure:
// a low 16-bit lane of +Inf, of a signaling NaN (0x7F81..0x7FBF), and of -0.0, next to ordinary
// vocabulary ids. The aliasing is the property the route relies on and no other case here covers
// it: a transport that read its receive buffer before staging its own send, or that added instead
// of copied, passes every distinct-buffer case above and fails this one.
int check_id_broadcast(ninfer::tp::DevicePair& pair, std::size_t count_bytes, int iterations) {
    const std::size_t words = count_bytes / 4;
    constexpr std::uint32_t kIds[] = {0x00000000u, 0x00000001u, 0x00007F80u, 0x00007F81u,
                                      0x00007FA0u, 0x00007FBFu, 0x00008000u, 0x0000FFFFu,
                                      0x00010000u, 0x00018000u, 0x0001FBF0u, 0x00024E00u};
    constexpr std::size_t kIdCount = sizeof(kIds) / sizeof(kIds[0]);
    int failures = 0;
    std::vector<std::uint32_t> host_a(words), got_b(words);

    pair.a().bind_to_current_thread();
    ninfer::DeviceBuffer buf_a(count_bytes);
    cudaStream_t stream_a = nullptr;
    cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking);
    pair.b().bind_to_current_thread();
    ninfer::DeviceBuffer buf_b(count_bytes);
    cudaStream_t stream_b = nullptr;
    cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking);

    for (int iteration = 0; iteration < iterations && failures == 0; ++iteration) {
        for (std::size_t i = 0; i < words; ++i) {
            host_a[i] = kIds[(i + static_cast<std::size_t>(iteration)) % kIdCount];
        }
        pair.a().bind_to_current_thread();
        cudaMemcpyAsync(buf_a.p, host_a.data(), count_bytes, cudaMemcpyHostToDevice, stream_a);
        pair.b().bind_to_current_thread();
        cudaMemsetAsync(buf_b.p, 0x5A, count_bytes, stream_b);
        pair.sendrecv(buf_a.p, buf_a.p, buf_b.p, buf_b.p, count_bytes, stream_a, stream_b);
        pair.b().bind_to_current_thread();
        cudaMemcpyAsync(got_b.data(), buf_b.p, count_bytes, cudaMemcpyDeviceToHost, stream_b);
        cudaStreamSynchronize(stream_b);
        if (std::memcmp(got_b.data(), host_a.data(), count_bytes) != 0) {
            std::size_t first = 0;
            while (first < words && got_b[first] == host_a[first]) { ++first; }
            std::cerr << "id broadcast[" << count_bytes << "] iteration " << iteration << ": id "
                      << first << " arrived as 0x" << std::hex << got_b[first] << " not 0x"
                      << host_a[first] << std::dec << '\n';
            ++failures;
            break;
        }
    }
    cudaStreamDestroy(stream_a);
    cudaStreamDestroy(stream_b);
    return failures;
}

// One captured send-receive queue replayed many times, with an eager all-reduce queued between
// replays (the production interleave of graphed verify windows and eager propose collectives).
// The graph body's receives are byte exact and identical every replay, so the final readback is
// checkable; a transport whose two sides ever disagree on the call sequence hangs the sync below,
// which is the regression this guards.
int check_graph_queue(ninfer::tp::DevicePair& pair, std::size_t graph_bytes, std::size_t eager_bytes,
                      int calls, int replays) {
    pair.a().bind_to_current_thread();
    ninfer::DeviceBuffer send_a(graph_bytes), recv_a(graph_bytes);
    ninfer::DeviceBuffer add_a(eager_bytes);
    pair.b().bind_to_current_thread();
    ninfer::DeviceBuffer send_b(graph_bytes), recv_b(graph_bytes);
    ninfer::DeviceBuffer add_b(eager_bytes);
    cudaStream_t stream_a = nullptr, stream_b = nullptr;
    pair.a().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking);
    pair.b().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking);

    std::mt19937 rng(0xC0FFEEu + static_cast<unsigned>(graph_bytes % 65521u));
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    const std::size_t graph_elements = graph_bytes / 2;
    const std::size_t eager_elements = eager_bytes / 2;
    std::vector<__nv_bfloat16> host_a(graph_elements), host_b(graph_elements);
    std::vector<__nv_bfloat16> eager_a(eager_elements), eager_b(eager_elements);
    std::vector<__nv_bfloat16> got_a(graph_elements);
    for (std::size_t i = 0; i < graph_elements; ++i) {
        host_a[i] = __float2bfloat16(dist(rng));
        host_b[i] = __float2bfloat16(dist(rng));
    }
    for (std::size_t i = 0; i < eager_elements; ++i) {
        eager_a[i] = __float2bfloat16(dist(rng));
        eager_b[i] = __float2bfloat16(dist(rng));
    }
    pair.a().bind_to_current_thread();
    send_a.copy_from_host(host_a.data(), graph_bytes);
    add_a.copy_from_host(eager_a.data(), eager_bytes);
    pair.b().bind_to_current_thread();
    send_b.copy_from_host(host_b.data(), graph_bytes);
    add_b.copy_from_host(eager_b.data(), eager_bytes);

    // One rendezvous id channel for this graph. The host publishes a fresh id block before every
    // replay, which is what lets the transport tell two replays apart; a reused id would let the
    // peer's stale staging satisfy the spin.
    const auto channel = pair.create_ar_channel();
    cudaGraph_t graph_a = nullptr, graph_b = nullptr;
    cudaGraphExec_t exec_a = nullptr, exec_b = nullptr;
    pair.begin_capture(channel);
    pair.a().bind_to_current_thread();
    cudaStreamBeginCapture(stream_a, cudaStreamCaptureModeThreadLocal);
    pair.b().bind_to_current_thread();
    cudaStreamBeginCapture(stream_b, cudaStreamCaptureModeThreadLocal);
    pair.a().bind_to_current_thread();
    for (int call = 0; call < calls; ++call) {
        pair.sendrecv(send_a.p, recv_a.p, send_b.p, recv_b.p, graph_bytes, stream_a, stream_b);
    }
    cudaStreamEndCapture(stream_a, &graph_a);
    pair.b().bind_to_current_thread();
    cudaStreamEndCapture(stream_b, &graph_b);
    cudaGraphInstantiate(&exec_a, graph_a, 0);
    cudaGraphInstantiate(&exec_b, graph_b, 0);
    pair.end_capture();

    int failures = 0;
    for (int replay = 0; replay < replays; ++replay) {
        // Fresh operands every replay. This is what makes a stale rendezvous id observable: with the
        // same bytes every time, a replay that skipped the peer's write would still read a buffer
        // holding the expected values and the check would pass.
        for (std::size_t i = 0; i < graph_elements; ++i) {
            host_a[i] = __float2bfloat16(dist(rng));
            host_b[i] = __float2bfloat16(dist(rng));
        }
        pair.a().bind_to_current_thread();
        send_a.copy_from_host(host_a.data(), graph_bytes);
        pair.b().bind_to_current_thread();
        send_b.copy_from_host(host_b.data(), graph_bytes);
        // An eager collective interleaved with the replay: the two must draw ids from disjoint ranges.
        pair.allreduce(add_a.p, add_b.p, eager_bytes, stream_a, stream_b);
        pair.arm_round(channel);
        pair.a().bind_to_current_thread();
        cudaGraphLaunch(exec_a, stream_a);
        pair.b().bind_to_current_thread();
        cudaGraphLaunch(exec_b, stream_b);
        pair.a().bind_to_current_thread();
        cudaStreamSynchronize(stream_a);
        pair.b().bind_to_current_thread();
        cudaStreamSynchronize(stream_b);

        pair.a().bind_to_current_thread();
        recv_a.copy_to_host(got_a.data(), graph_bytes);
        for (std::size_t i = 0; i < graph_elements; ++i) {
            if (bits_of(got_a[i]) != bits_of(host_b[i])) {
                if (failures < 4) {
                    std::cerr << "graph queue[" << graph_bytes << "] replay " << replay << " element "
                              << i << ": expected " << __bfloat162float(host_b[i]) << ", got "
                              << __bfloat162float(got_a[i]) << '\n';
                }
                ++failures;
                break;
            }
        }
    }

    cudaGraphExecDestroy(exec_a);
    cudaGraphExecDestroy(exec_b);
    cudaGraphDestroy(graph_a);
    cudaGraphDestroy(graph_b);
    pair.a().bind_to_current_thread();
    cudaStreamDestroy(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamDestroy(stream_b);
    return failures;
}

// Measures what the copy engines can do with the production large-payload shape, as the floor for
// the event-based large-payload path (PLAN Phase 3). That candidate moves exactly these bytes - a
// D2H of the local delta and an H2D of the peer's - so its cost cannot come in under this, and the
// sliced in-kernel path it would replace is already within 2% of the link's raw bound. Print-only,
// enabled by NINFER_TP2_AR_COPY_BENCH=1, because it measures a design decision rather than behavior.
int check_copy_engine_ceiling(ninfer::tp::DevicePair& pair, std::size_t count_bytes, int iterations) {
    // Chunking and stream layout follow the candidate: the plan's chunk size, and one stream per
    // direction per device so both directions are in flight at once. That is what the candidate's
    // event handoff allows (an H2D starts as soon as the peer's chunk lands), and PCIe runs both
    // directions at once, so a probe that serializes them measures something the candidate need not do.
    const std::size_t chunk = std::min<std::size_t>(
        std::max<std::size_t>(count_bytes / 4, 512ULL << 10), 2ULL << 20);
    const int chunks = static_cast<int>((count_bytes + chunk - 1) / chunk);

    pair.a().bind_to_current_thread();
    ninfer::DeviceBuffer src_a(count_bytes);
    ninfer::DeviceBuffer scratch_a(count_bytes);
    void* host_a = nullptr;
    void* peer_a = nullptr;
    cudaHostAlloc(&host_a, count_bytes, cudaHostAllocPortable);
    cudaHostAlloc(&peer_a, count_bytes, cudaHostAllocPortable);
    cudaStream_t d2h_a = nullptr, h2d_a = nullptr;
    cudaStreamCreateWithFlags(&d2h_a, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&h2d_a, cudaStreamNonBlocking);
    pair.b().bind_to_current_thread();
    ninfer::DeviceBuffer src_b(count_bytes);
    ninfer::DeviceBuffer scratch_b(count_bytes);
    void* host_b = nullptr;
    void* peer_b = nullptr;
    cudaHostAlloc(&host_b, count_bytes, cudaHostAllocPortable);
    cudaHostAlloc(&peer_b, count_bytes, cudaHostAllocPortable);
    cudaStream_t d2h_b = nullptr, h2d_b = nullptr;
    cudaStreamCreateWithFlags(&d2h_b, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&h2d_b, cudaStreamNonBlocking);

    const auto issue = [&] {
        pair.a().bind_to_current_thread();
        for (int c = 0; c < chunks; ++c) {
            const std::size_t offset = static_cast<std::size_t>(c) * chunk;
            const std::size_t bytes  = std::min(chunk, count_bytes - offset);
            cudaMemcpyAsync(static_cast<char*>(host_a) + offset,
                            static_cast<const char*>(src_a.p) + offset, bytes,
                            cudaMemcpyDeviceToHost, d2h_a);
            cudaMemcpyAsync(static_cast<char*>(scratch_a.p) + offset,
                            static_cast<const char*>(peer_a) + offset, bytes,
                            cudaMemcpyHostToDevice, h2d_a);
        }
        pair.b().bind_to_current_thread();
        for (int c = 0; c < chunks; ++c) {
            const std::size_t offset = static_cast<std::size_t>(c) * chunk;
            const std::size_t bytes  = std::min(chunk, count_bytes - offset);
            cudaMemcpyAsync(static_cast<char*>(host_b) + offset,
                            static_cast<const char*>(src_b.p) + offset, bytes,
                            cudaMemcpyDeviceToHost, d2h_b);
            cudaMemcpyAsync(static_cast<char*>(scratch_b.p) + offset,
                            static_cast<const char*>(peer_b) + offset, bytes,
                            cudaMemcpyHostToDevice, h2d_b);
        }
    };
    for (int i = 0; i < 5; ++i) { issue(); }

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) { issue(); }
    pair.a().bind_to_current_thread();
    cudaStreamSynchronize(d2h_a);
    cudaStreamSynchronize(h2d_a);
    pair.b().bind_to_current_thread();
    cudaStreamSynchronize(d2h_b);
    cudaStreamSynchronize(h2d_b);
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() /
        iterations;
    const double gbps = 2.0 * static_cast<double>(count_bytes) / (ms * 1e6);
    std::cout << "copy engine ceiling[" << (count_bytes >> 20) << " MiB, " << chunks << " chunks]: "
              << ms << " ms per call (both directions), " << gbps
              << " GB/s counting both directions\n";

    pair.a().bind_to_current_thread();
    cudaFreeHost(host_a);
    cudaFreeHost(peer_a);
    cudaStreamDestroy(d2h_a);
    cudaStreamDestroy(h2d_a);
    pair.b().bind_to_current_thread();
    cudaFreeHost(host_b);
    cudaFreeHost(peer_b);
    cudaStreamDestroy(d2h_b);
    cudaStreamDestroy(h2d_b);
    return 0;
}

// A collective whose peer side never launches must give up on its deadline instead of spinning
// forever: the stall is reported, both streams still drain, and after clear_ar_stall() the same
// transport produces the elementwise sum again. The divergence is injected - the peer launch of one
// collective is skipped - which is the failure class the bound exists for.
int check_ar_timeout(ninfer::tp::DevicePair& pair, std::size_t count_bytes) {
    if (!pair.in_kernel_allreduce()) {
        std::cout << "SKIP ar timeout: the in-kernel transport is unavailable\n";
        return 0;
    }
    const std::size_t elements = count_bytes / 2;
    pair.a().bind_to_current_thread();
    ninfer::DeviceBuffer buf_a(count_bytes);
    pair.b().bind_to_current_thread();
    ninfer::DeviceBuffer buf_b(count_bytes);
    cudaStream_t stream_a = nullptr, stream_b = nullptr;
    pair.a().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking);
    pair.b().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking);

    std::mt19937 rng(0x7E50u + static_cast<unsigned>(count_bytes % 65521u));
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<__nv_bfloat16> bf_a(elements), bf_b(elements);
    std::vector<__nv_bfloat16> got_a(elements), got_b(elements);
    for (std::size_t i = 0; i < elements; ++i) {
        bf_a[i] = __float2bfloat16(dist(rng));
        bf_b[i] = __float2bfloat16(dist(rng));
    }
    pair.a().bind_to_current_thread();
    buf_a.copy_from_host(bf_a.data(), count_bytes);
    pair.b().bind_to_current_thread();
    buf_b.copy_from_host(bf_b.data(), count_bytes);

    int failures = 0;
    pair.set_ar_fault_skip_peer_call(1);
    const auto before = std::chrono::steady_clock::now();
    pair.allreduce(buf_a.p, buf_b.p, count_bytes, stream_a, stream_b);
    pair.a().bind_to_current_thread();
    cudaStreamSynchronize(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamSynchronize(stream_b);
    const auto waited_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                             before)
            .count();
    const std::uint64_t stalled_id = pair.ar_last_id();
    std::cout << "ar timeout[" << count_bytes << "]: gave up after " << waited_ms
              << " ms at rendezvous id " << stalled_id << '\n';
    if (waited_ms < 100) {
        std::cerr << "ar timeout[" << count_bytes << "]: returned without waiting on the deadline\n";
        ++failures;
    }

    if (!pair.ar_stalled()) {
        std::cerr << "ar timeout[" << count_bytes << "]: the bounded spin did not report a stall\n";
        ++failures;
    }
    // While the pair is tripped, every further collective must bail at entry instead of waiting out
    // another deadline, or a desynchronized round would pay one timeout per allreduce.
    pair.set_ar_fault_skip_peer_call(0);
    const auto tripped_before = std::chrono::steady_clock::now();
    pair.allreduce(buf_a.p, buf_b.p, count_bytes, stream_a, stream_b);
    pair.a().bind_to_current_thread();
    cudaStreamSynchronize(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamSynchronize(stream_b);
    const auto tripped_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                             tripped_before)
            .count();
    std::cout << "ar timeout[" << count_bytes << "]: tripped pair bailed after " << tripped_ms
              << " ms\n";
    if (tripped_ms > 500) {
        std::cerr << "ar timeout[" << count_bytes
                  << "]: a tripped pair waited again instead of bailing at entry\n";
        ++failures;
    }
    pair.clear_ar_stall();
    if (pair.ar_stalled()) {
        std::cerr << "ar timeout[" << count_bytes << "]: clear_ar_stall did not clear the trip\n";
        ++failures;
    }
    // The ids are what makes a stalled round recoverable in place: the next collective must not
    // reuse the id the stalled one ran with, or its spin could be satisfied by a stale arrival slot.
    if (pair.ar_last_id() == stalled_id) {
        std::cerr << "ar timeout[" << count_bytes << "]: the rendezvous id was reused after a stall\n";
        ++failures;
    }

    pair.set_ar_fault_skip_peer_call(0);
    pair.a().bind_to_current_thread();
    buf_a.copy_from_host(bf_a.data(), count_bytes);
    pair.b().bind_to_current_thread();
    buf_b.copy_from_host(bf_b.data(), count_bytes);
    pair.allreduce(buf_a.p, buf_b.p, count_bytes, stream_a, stream_b);
    pair.a().bind_to_current_thread();
    cudaStreamSynchronize(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamSynchronize(stream_b);
    pair.a().bind_to_current_thread();
    buf_a.copy_to_host(got_a.data(), count_bytes);
    pair.b().bind_to_current_thread();
    buf_b.copy_to_host(got_b.data(), count_bytes);
    for (std::size_t i = 0; i < elements; ++i) {
        const __nv_bfloat16 expected =
            __float2bfloat16(__bfloat162float(bf_a[i]) + __bfloat162float(bf_b[i]));
        if (bits_of(got_a[i]) != bits_of(expected) || bits_of(got_b[i]) != bits_of(expected)) {
            std::cerr << "ar timeout[" << count_bytes << "]: element " << i
                      << " differs after re-arm\n";
            ++failures;
            break;
        }
    }

    if (failures == 0) {
        std::cout << "ar timeout[" << count_bytes << "]: stalled, re-armed, sum matches\n";
    }
    pair.a().bind_to_current_thread();
    cudaStreamDestroy(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamDestroy(stream_b);
    return failures;
}


// Sets or clears `name` for the pair constructed next: the host-staging hook is read once, at
// construction, so the process-wide value is restored before anything else is built.
void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}


// The skew the rotating arrival banks exist for: one side falls a collective behind and must still
// find the id it is waiting for. The engine produces that skew on its own - only shard A is drained
// where a round reads its sample, so the mirror shard is routinely a call or two behind - and this
// injection makes it deterministic: hold one side's arrival poll of the second queued collective for
// 30 ms, so the other side completes that collective and publishes the next one before the held side
// ever looks. With one arrival bank (the geometry the transport shipped with) that has to give up on
// the deadline; with the rotating banks the whole queue must complete, and every call stays bit exact
// on both shards, because the recovering side reads the staging of the collective it fell behind on.
int check_ar_skew_case(int dev_a, int dev_b, int banks, bool hold_peer, bool expect_stall) {
    constexpr std::size_t kCountBytes = 20480; // the 27B hidden allreduce: one slice, one block
    constexpr int kCalls              = 4;
    constexpr int kHeldCall           = 2;
    constexpr unsigned long long kHoldNs = 30ULL * 1000000ULL;

    const std::string banks_setting = std::to_string(banks);
    set_env("NINFER_TP2_AR_BANKS", banks_setting.c_str());
    // The deadline only has to be longer than the hold: a short one keeps a regression (the held side
    // really missing the id) bounded instead of paying the production 10 s per case.
    set_env("NINFER_TP2_AR_TIMEOUT_MS", "1000");
    std::unique_ptr<ninfer::tp::DevicePair> pair =
        std::make_unique<ninfer::tp::DevicePair>(dev_a, dev_b);
    set_env("NINFER_TP2_AR_BANKS", nullptr);
    set_env("NINFER_TP2_AR_TIMEOUT_MS", nullptr);
    if (!pair->in_kernel_allreduce()) {
        std::cout << "SKIP ar skew: the in-kernel transport is unavailable\n";
        return 0;
    }

    const std::size_t elements = kCountBytes / 2;
    cudaStream_t stream_a = nullptr, stream_b = nullptr;
    pair->a().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking);
    pair->b().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking);

    std::mt19937 rng(0x5C3E1u + static_cast<unsigned>(banks) * 2u + (hold_peer ? 0u : 1u));
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<__nv_bfloat16> host_a(elements), host_b(elements);
    std::vector<std::vector<__nv_bfloat16>> operands_a, operands_b;
    std::vector<std::unique_ptr<ninfer::DeviceBuffer>> buf_a, buf_b;
    for (int call = 0; call < kCalls; ++call) {
        for (std::size_t i = 0; i < elements; ++i) {
            host_a[i] = __float2bfloat16(dist(rng));
            host_b[i] = __float2bfloat16(dist(rng));
        }
        operands_a.push_back(host_a);
        operands_b.push_back(host_b);
        pair->a().bind_to_current_thread();
        buf_a.push_back(std::make_unique<ninfer::DeviceBuffer>(kCountBytes));
        buf_a.back()->copy_from_host(host_a.data(), kCountBytes);
        pair->b().bind_to_current_thread();
        buf_b.push_back(std::make_unique<ninfer::DeviceBuffer>(kCountBytes));
        buf_b.back()->copy_from_host(host_b.data(), kCountBytes);
    }

    pair->set_ar_fault_hold_poll(kHeldCall, kHoldNs, hold_peer);
    const auto start = std::chrono::steady_clock::now();
    for (int call = 0; call < kCalls; ++call) {
        pair->allreduce(buf_a[call]->p, buf_b[call]->p, kCountBytes, stream_a, stream_b);
    }
    pair->a().bind_to_current_thread();
    cudaStreamSynchronize(stream_a);
    pair->b().bind_to_current_thread();
    cudaStreamSynchronize(stream_b);
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count();
    const bool stalled = pair->ar_stalled();

    int failures = 0;
    const char* const side = hold_peer ? "b" : "a";
    if (stalled != expect_stall) {
        std::cerr << "ar skew[" << banks << " bank, " << side << " held]: the pair "
                  << (stalled ? "gave up on the deadline" : "completed") << ", but "
                  << (expect_stall ? "the pre-bank geometry had to give up" : "the banks had to carry it")
                  << '\n';
        ++failures;
    }
    if (expect_stall) {
        if (elapsed_ms < 900) {
            std::cerr << "ar skew[" << banks << " bank, " << side
                      << " held]: gave up after only " << elapsed_ms << " ms\n";
            ++failures;
        }
        std::cout << "ar skew[" << kCountBytes << " bytes, " << banks << " bank, " << side
                  << " held]: gave up after " << elapsed_ms
                  << " ms - the slot this side polls held the\n"
                  << "  next collective's id, which is the miss one arrival bank cannot carry\n";
        pair->clear_ar_stall();
    } else {
        if (elapsed_ms > 900) {
            std::cerr << "ar skew[" << banks << " bank, " << side << " held]: completed but took "
                      << elapsed_ms << " ms\n";
            ++failures;
        }
        std::vector<__nv_bfloat16> got(elements);
        for (int call = 0; call < kCalls; ++call) {
            for (int which = 0; which < 2; ++which) {
                if (which == 0) {
                    pair->a().bind_to_current_thread();
                    buf_a[call]->copy_to_host(got.data(), kCountBytes);
                } else {
                    pair->b().bind_to_current_thread();
                    buf_b[call]->copy_to_host(got.data(), kCountBytes);
                }
                for (std::size_t i = 0; i < elements; ++i) {
                    const __nv_bfloat16 expected =
                        __float2bfloat16(__bfloat162float(operands_a[call][i]) +
                                         __bfloat162float(operands_b[call][i]));
                    if (bits_of(got[i]) != bits_of(expected)) {
                        std::cerr << "ar skew[" << banks << " bank, " << side << " held]: call " << call
                                  << " shard " << which << " element " << i
                                  << " differs after the recovery\n";
                        ++failures;
                        break;
                    }
                }
            }
        }
        std::cout << "ar skew[" << kCountBytes << " bytes, " << banks << " bank, " << side
                  << " held]: " << kCalls << " queued collectives completed bit exact in " << elapsed_ms
                  << " ms\n";
    }

    pair->set_ar_fault_hold_poll(0, 0, true);
    pair->a().bind_to_current_thread();
    cudaStreamDestroy(stream_a);
    pair->b().bind_to_current_thread();
    cudaStreamDestroy(stream_b);
    return failures;
}

// One arrival bank is the pre-bank geometry, so the injection must still produce the give-up there:
// that is the control which shows the skewed case below is the failure the banks remove, not an
// injection that never created the skew. Both sides are held in turn, because either one may be the
// side that falls behind.
int check_ar_skew(int dev_a, int dev_b) {
    int failures = 0;
    failures += check_ar_skew_case(dev_a, dev_b, 1, true, true);
    failures += check_ar_skew_case(dev_a, dev_b, 4, true, false);
    failures += check_ar_skew_case(dev_a, dev_b, 4, false, false);
    return failures;
}

// The host-staging reduction must reproduce the device transport's BF16 add bit for bit, including
// the cases its rounding is *defined* on rather than merely close to: signed zeros, denormals, exact
// ties in both directions, sums that round up into infinity, and NaN payloads, which
// __float2bfloat16 canonicalizes. Every pattern pair is pushed through the real transport as one
// payload, so the check covers the vector loop and its chunking rather than a private helper.
int check_host_add_patterns(ninfer::tp::DevicePair& pair) {
    const std::uint16_t patterns[] = {
        0x0000, 0x8000,                         // +-0
        0x0001, 0x8001, 0x007F, 0x807F,         // smallest denormals, both signs
        0x3F80, 0xBF80,                         // +-1
        0x3F81, 0xBF81,                         // +-(1 + 2^-8)
        0x4000, 0xC000, 0x3F00, 0xBF00,         // +-2, +-0.5
        0x3780, 0xB780,                         // +-2^-16: the exact tie neighbour of 1.0
        0x3380, 0xB380,                         // +-2^-24
        0x7F7F, 0xFF7F,                         // largest finite, both signs
        0x7F80, 0xFF80,                         // +-infinity
        0x7FC0, 0xFFC0, 0x7FA0, 0xFFA0, 0x7FFF, // quiet and signaling NaNs, both signs
    };
    constexpr std::size_t kCount = sizeof(patterns) / sizeof(patterns[0]);

    std::vector<std::uint16_t> host_a, host_b;
    host_a.reserve(kCount * kCount + 8);
    host_b.reserve(kCount * kCount + 8);
    for (std::size_t i = 0; i < kCount; ++i) {
        for (std::size_t j = 0; j < kCount; ++j) {
            host_a.push_back(patterns[i]);
            host_b.push_back(patterns[j]);
        }
    }
    // Pad to the 16-byte group the transport requires; +0 leaves the expected sum unchanged.
    while (host_a.size() % 8 != 0) {
        host_a.push_back(0);
        host_b.push_back(0);
    }
    const std::size_t count_bytes = host_a.size() * sizeof(std::uint16_t);
    const std::size_t elements    = host_a.size();

    // Count the cases the rounding is defined on, so a corpus that stopped covering them shows up
    // instead of passing quietly.
    std::size_t ties = 0, infinities = 0, nans = 0;
    for (std::size_t i = 0; i < elements; ++i) {
        const auto* a = reinterpret_cast<const __nv_bfloat16*>(&host_a[i]);
        const auto* b = reinterpret_cast<const __nv_bfloat16*>(&host_b[i]);
        const float sum = __bfloat162float(*a) + __bfloat162float(*b);
        if (std::isnan(sum)) {
            ++nans;
            continue;
        }
        if (std::isinf(sum)) {
            ++infinities;
            continue;
        }
        std::uint32_t bits = 0;
        std::memcpy(&bits, &sum, sizeof(bits));
        if ((bits & 0xFFFFu) == 0x8000u) { ++ties; }
    }
    if (ties == 0 || infinities == 0 || nans == 0) {
        std::cerr << "host add patterns: corpus lost its edge cases (ties " << ties << ", inf "
                  << infinities << ", nan " << nans << ")\n";
        return 1;
    }

    pair.a().bind_to_current_thread();
    ninfer::DeviceBuffer buf_a(count_bytes);
    pair.b().bind_to_current_thread();
    ninfer::DeviceBuffer buf_b(count_bytes);
    cudaStream_t stream_a = nullptr, stream_b = nullptr;
    pair.a().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking);
    pair.b().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking);
    pair.a().bind_to_current_thread();
    buf_a.copy_from_host(host_a.data(), count_bytes);
    pair.b().bind_to_current_thread();
    buf_b.copy_from_host(host_b.data(), count_bytes);

    pair.allreduce(buf_a.p, buf_b.p, count_bytes, stream_a, stream_b);
    pair.a().bind_to_current_thread();
    cudaStreamSynchronize(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamSynchronize(stream_b);

    std::vector<__nv_bfloat16> got_a(elements), got_b(elements);
    pair.a().bind_to_current_thread();
    buf_a.copy_to_host(got_a.data(), count_bytes);
    pair.b().bind_to_current_thread();
    buf_b.copy_to_host(got_b.data(), count_bytes);

    int failures = 0;
    for (std::size_t i = 0; i < elements; ++i) {
        const __nv_bfloat16 expected = __float2bfloat16(
            __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&host_a[i])) +
            __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&host_b[i])));
        if (bits_of(got_a[i]) == bits_of(expected) && bits_of(got_b[i]) == bits_of(expected)) {
            continue;
        }
        if (failures < 4) {
            std::cerr << "host add patterns: element " << i << " a=0x" << std::hex << host_a[i]
                      << " b=0x" << host_b[i] << " expected=0x" << bits_of(expected) << " shard a=0x"
                      << bits_of(got_a[i]) << " shard b=0x" << bits_of(got_b[i]) << std::dec << '\n';
        }
        ++failures;
    }

    pair.a().bind_to_current_thread();
    cudaStreamDestroy(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamDestroy(stream_b);
    if (failures == 0) {
        std::cout << "host add patterns[" << elements << "]: bit-exact (ties " << ties << ", inf "
                  << infinities << ", nan " << nans << ")\n";
    }
    return failures;
}

// Print-only (NINFER_TP2_AR_STAGING_BENCH=1): what one host-staging collective costs end to end next
// to the scalar BF16 add the reduction used to run, so the change is attributable on the host it
// targets rather than argued from instruction counts.
int bench_host_staging(ninfer::tp::DevicePair& pair, std::size_t count_bytes, int iterations) {
    const std::size_t elements = count_bytes / 2;
    pair.a().bind_to_current_thread();
    ninfer::DeviceBuffer buf_a(count_bytes);
    pair.b().bind_to_current_thread();
    ninfer::DeviceBuffer buf_b(count_bytes);
    std::vector<__nv_bfloat16> host_a(elements), host_b(elements);
    std::mt19937 rng(0xB0A7u);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    for (std::size_t i = 0; i < elements; ++i) {
        host_a[i] = __float2bfloat16(dist(rng));
        host_b[i] = __float2bfloat16(dist(rng));
    }
    pair.a().bind_to_current_thread();
    buf_a.copy_from_host(host_a.data(), count_bytes);
    pair.b().bind_to_current_thread();
    buf_b.copy_from_host(host_b.data(), count_bytes);

    cudaStream_t stream_a = nullptr, stream_b = nullptr;
    pair.a().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking);
    pair.b().bind_to_current_thread();
    cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking);

    // Warm both directions before timing: this host's copy-engine path costs several times its
    // steady-state rate on the first large transfers of a process.
    for (int i = 0; i < 5; ++i) {
        pair.allreduce(buf_a.p, buf_b.p, count_bytes, stream_a, stream_b);
        pair.a().bind_to_current_thread();
        cudaStreamSynchronize(stream_a);
        pair.b().bind_to_current_thread();
        cudaStreamSynchronize(stream_b);
    }

    // Per-call times, reported as the minimum next to the mean: the minimum is the transport's own
    // cost, while the mean carries whatever else the machine did during the run.
    std::vector<double> per_call;
    per_call.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        pair.allreduce(buf_a.p, buf_b.p, count_bytes, stream_a, stream_b);
        // The transport leaves its uploads in the caller's streams; a real round would run the next
        // layer on them, so settle them here or this measures enqueue cost.
        pair.a().bind_to_current_thread();
        cudaStreamSynchronize(stream_a);
        pair.b().bind_to_current_thread();
        cudaStreamSynchronize(stream_b);
        per_call.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count());
    }
    const double transport_min = *std::min_element(per_call.begin(), per_call.end());
    const double transport_ms =
        std::accumulate(per_call.begin(), per_call.end(), 0.0) / static_cast<double>(iterations);

    // The same payload through sendrecv's fallback: the identical D2H / barrier / H2D with no
    // reduction at all, which is what separates the transfer cost from the add's residual cost.
    for (int i = 0; i < 5; ++i) {
        pair.sendrecv(buf_a.p, buf_a.p, buf_b.p, buf_b.p, count_bytes, stream_a, stream_b);
        pair.a().bind_to_current_thread();
        cudaStreamSynchronize(stream_a);
        pair.b().bind_to_current_thread();
        cudaStreamSynchronize(stream_b);
    }
    std::vector<double> transfer_call;
    transfer_call.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        pair.sendrecv(buf_a.p, buf_a.p, buf_b.p, buf_b.p, count_bytes, stream_a, stream_b);
        pair.a().bind_to_current_thread();
        cudaStreamSynchronize(stream_a);
        pair.b().bind_to_current_thread();
        cudaStreamSynchronize(stream_b);
        transfer_call.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count());
    }
    const double transfer_min = *std::min_element(transfer_call.begin(), transfer_call.end());

    // The retired reduction: one scalar __hadd pass over the same element count.
    std::vector<__nv_bfloat16> scalar(elements);
    const auto scalar_start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        for (std::size_t e = 0; e < elements; ++e) { scalar[e] = __hadd(host_a[e], host_b[e]); }
    }
    const double scalar_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - scalar_start)
            .count() /
        iterations;

    const double mb = static_cast<double>(count_bytes) / (1024.0 * 1024.0);
    std::cout << "host staging[" << (count_bytes >> 20) << " MiB, cores "
              << std::thread::hardware_concurrency() << "]: allreduce min " << transport_min
              << " ms, mean " << transport_ms << " ms (" << (2.0 * mb / transport_min)
              << " GB/s both directions at the minimum) | transfers only (sendrecv) min "
              << transfer_min << " ms | retired scalar add " << scalar_ms << " ms ("
              << (mb / scalar_ms) << " GB/s)\n";

    pair.a().bind_to_current_thread();
    cudaStreamDestroy(stream_a);
    pair.b().bind_to_current_thread();
    cudaStreamDestroy(stream_b);
    return 0;
}

} // namespace

// The rendezvous id space is sized from the owning core's graph bucket count, not a fixed block:
// the TP-2 core reserves verify windows + MTP draft chain + plain decode step buckets, which is
// 2 x 9 or more for MTP K >= 3 and already at the old fixed 16 for K = 2. Reserve explicitly here
// and check that the space is exactly what was asked for and refuses one channel past it.
int check_reserve_channels(ninfer::tp::DevicePair& pair) {
    if (!pair.in_kernel_allreduce()) {
        std::cout << "reserve channels: no in-kernel transport, skipped\n";
        return 0;
    }
    int failures = 0;
    constexpr std::size_t kReserved = 24;
    pair.reserve_ar_channels(kReserved);
    for (std::size_t i = 0; i < kReserved; ++i) {
        try {
            (void)pair.create_ar_channel();
        } catch (const std::exception& error) {
            std::cerr << "reserve channels: channel " << i << " refused: " << error.what() << '\n';
            ++failures;
            break;
        }
    }
    bool refused = false;
    try {
        (void)pair.create_ar_channel();
    } catch (const std::exception& error) {
        refused = true;
        std::cout << "reserve channels: overflow refused as expected: " << error.what() << '\n';
    }
    if (!refused) {
        std::cerr << "reserve channels: channel " << kReserved << " past the reservation was not"
                  << " refused\n";
        ++failures;
    }
    return failures;
}

int run_all_checks() {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver || count < 2) {
        std::cout << "SKIP: fewer than two CUDA devices\n";
        return 77;
    }
    const auto [dev_a, dev_b] = pick_devices();
    if (dev_a < 0) {
        std::cout << "SKIP: no usable device pair\n";
        return 77;
    }
    std::cout << "DevicePair(" << dev_a << ", " << dev_b << ")\n";

    int failures = 0;
    {
        ninfer::tp::DevicePair pair(dev_a, dev_b);
        std::cout << "p2p_available=" << pair.p2p_available()
                  << " in_kernel=" << pair.in_kernel_allreduce()
                  << " staging_bytes=" << pair.in_kernel_allreduce_bytes() << '\n';
        // 20480 bytes = 10240 BF16 elements, the 27B hidden allreduce size.
        failures += check_allreduce(pair, 20480, 64);
        // Small and larger shapes.
        failures += check_allreduce(pair, 16, 64);
        failures += check_allreduce(pair, 1 << 20, 64);
        // The decode window shapes the DFlash2 verify allreduces: one hidden
        // column per window position, then the full [vocab, width] logits merge
        // (152064 x 8 BF16 = 2433024 bytes, which the size-keyed transport splits
        // into five slices).
        failures += check_allreduce(pair, 81920, 64);
        failures += check_allreduce(pair, 2433024, 64);
        // The prefill wire format: the 256-token stage delta (5120 x 256 BF16), a short tail chunk,
        // and the codec's refusals.
        failures += check_allreduce_fp8(pair, 2621440, 4);
        failures += check_allreduce_fp8(pair, 8253440, 2);
        failures += check_allreduce_fp8_contract(pair);
        // Past the retired 24 MiB staging bound and inside the current 48 MiB one: the payload has to
        // stay on the in-kernel transport, which is what the wider buffer buys.
        failures += check_allreduce(pair, 25 << 20, 8);
        // Past the staging buffer entirely: the size-triggered host-staging fallback.
        failures += check_allreduce(pair, 50 << 20, 2);
        // The directed rounding corpus through the in-kernel transport: the two transports have to
        // agree on every case the BF16 rounding is defined on, not only on finite random payloads.
        failures += check_host_add_patterns(pair);
        // The same shapes as a deep queue on both shards, which is how a layer
        // stack issues them.
        failures += check_allreduce_queue(pair, 20480, 200);
        failures += check_allreduce_queue(pair, 81920, 200);
        // A prefill-sized payload back to back: each call's upload is still in flight when the next
        // call's download starts, which is what the parity slot has to cover.
        failures += check_allreduce_queue(pair, 4 << 20, 16);
        // Byte-exact exchange at the shapes the split proposal head's candidate union and the peer
        // selector transport use, then at the small and sliced staging classes.
        failures += check_sendrecv(pair, 960, 64);
        failures += check_sendrecv(pair, 64, 64);
        failures += check_sendrecv(pair, 1 << 20, 16);
        // The MTP stem's id broadcast: each side's one buffer used as both send and receive, at the
        // padded sizes a draft window (K <= 15 ids) and a single id produce.
        failures += check_id_broadcast(pair, 16, 64);
        failures += check_id_broadcast(pair, 64, 64);
        // Captured queues replayed between eager calls: small fuse exchanges, sliced large
        // exchanges (write-order chain, separate token bump), and the two classes interleaved.
        failures += check_graph_queue(pair, 960, 20480, 32, 20);
        failures += check_graph_queue(pair, 2 << 20, 2 << 20, 8, 20);
        failures += check_graph_queue(pair, 960, 2 << 20, 16, 20);
        // A skipped peer launch must give up on the deadline and re-arm, not hang the process.
        failures += check_ar_timeout(pair, 20480);
        if (std::getenv("NINFER_TP2_AR_COPY_BENCH") != nullptr) {
            failures += check_copy_engine_ceiling(pair, 10 << 20, 40);
            failures += check_copy_engine_ceiling(pair, 24 << 20, 20);
        }
        // Move semantics: a moved-from pair must not retain p2p state.
        ninfer::tp::DevicePair moved(std::move(pair));
        if (pair.p2p_available()) {
            std::cerr << "moved-from pair retained p2p state\n";
            ++failures;
        }
        failures += check_allreduce(moved, 20480, 8);
    }
    // The host-staging fallback on a pair of its own, forced through its construction hook: this
    // host's mapped-pinned transport takes every call that is not past its staging buffer, so the
    // forced path is the only way to qualify the fallback's reduction at every payload class and
    // its behaviour past the staging bound.
    {
        set_env("NINFER_TP2_AR_FORCE_HOST_STAGING", "1");
        ninfer::tp::DevicePair fallback(dev_a, dev_b);
        set_env("NINFER_TP2_AR_FORCE_HOST_STAGING", nullptr);
        if (fallback.in_kernel_allreduce()) {
            std::cerr << "forced pair still reports an in-kernel transport\n";
            ++failures;
        }
        std::cout << "host-staging pair: in_kernel=" << fallback.in_kernel_allreduce()
                  << " staging_bytes=" << fallback.in_kernel_allreduce_bytes() << '\n';
        failures += check_allreduce(fallback, 20480, 32);
        failures += check_allreduce(fallback, 1 << 20, 16);
        // Several chunks, at a payload that is not a whole number of chunk widths.
        failures += check_allreduce(fallback, (3 << 20) + 16, 8);
        // Past the staging buffer: the fallback carries it, and the reduction splits into strips.
        failures += check_allreduce(fallback, 50 << 20, 2);
        // Without the in-kernel transport the fp8 codec has nowhere to run: it must refuse, not fall
        // back to a bf16 host staging that would silently change the wire contract.
        failures += check_allreduce_fp8_contract(fallback);
        // A deep queue reuses the pinned staging: every call's D2H is ordered after the previous
        // call's uploads on the same streams.
        failures += check_allreduce_queue(fallback, 20480, 32);
        // The same queue at a prefill payload. A single shared staging buffer fails here: the two
        // devices' streams are independent, so the next call's D2H overwrites the half the previous
        // call's H2D is still uploading from, and this queue's own results prove it.
        failures += check_allreduce_queue(fallback, 4 << 20, 8);
        failures += check_host_add_patterns(fallback);
        failures += check_sendrecv(fallback, 960, 16);
        if (std::getenv("NINFER_TP2_AR_STAGING_BENCH") != nullptr) {
            failures += bench_host_staging(fallback, 10 << 20, 20);
            failures += bench_host_staging(fallback, 50 << 20, 12);
            // The same payloads on a pair pinned to one reduction thread, in the same process: the
            // split's own A/B, so the automatic thread count is a measured choice.
            set_env("NINFER_TP2_AR_FORCE_HOST_STAGING", "1");
            set_env("NINFER_TP2_AR_ADD_THREADS", "1");
            ninfer::tp::DevicePair single_thread(dev_a, dev_b);
            set_env("NINFER_TP2_AR_FORCE_HOST_STAGING", nullptr);
            set_env("NINFER_TP2_AR_ADD_THREADS", nullptr);
            failures += bench_host_staging(single_thread, 10 << 20, 20);
            failures += bench_host_staging(single_thread, 50 << 20, 12);
        }
    }
    // A host whose mapped window refuses the full staging keeps the in-kernel transport at the
    // largest rung it can map (the ladder in the pair's constructor), and only the payloads that no
    // longer fit take the fallback. Capping the ladder reproduces such a host here, and - unlike the
    // forced pair above - the size-keyed boundary is real: one pair carries both transports, so a
    // 12 MiB rung still takes every payload the TP-2 core produces at the default chunk clamp.
    {
        set_env("NINFER_TP2_AR_STAGING_MIB", "24");
        ninfer::tp::DevicePair middle(dev_a, dev_b);
        set_env("NINFER_TP2_AR_STAGING_MIB", nullptr);
        std::cout << "24 MiB rung: in_kernel=" << middle.in_kernel_allreduce()
                  << " staging_bytes=" << middle.in_kernel_allreduce_bytes() << '\n';
        if (!middle.in_kernel_allreduce() || middle.in_kernel_allreduce_bytes() != (24U << 20)) {
            std::cerr << "24 MiB rung was not taken\n";
            ++failures;
        }
        failures += check_allreduce(middle, 20 << 20, 8);
        // Past the rung, in the same pair and process: the size-keyed fallback.
        failures += check_allreduce(middle, 25 << 20, 2);
    }
    {
        set_env("NINFER_TP2_AR_STAGING_MIB", "12");
        ninfer::tp::DevicePair floor_rung(dev_a, dev_b);
        set_env("NINFER_TP2_AR_STAGING_MIB", nullptr);
        std::cout << "12 MiB rung: in_kernel=" << floor_rung.in_kernel_allreduce()
                  << " staging_bytes=" << floor_rung.in_kernel_allreduce_bytes() << '\n';
        if (!floor_rung.in_kernel_allreduce() || floor_rung.in_kernel_allreduce_bytes() != (12U << 20)) {
            std::cerr << "12 MiB rung was not taken\n";
            ++failures;
        }
        // A payload at the rung writes to the last byte of each parity slot: a slot stride still
        // tied to the full staging size (the retired constant) would publish past the reservation.
        // 12 MiB is also the widest payload the TP-2 prefill issues at the widest activation.
        failures += check_allreduce(floor_rung, 12U << 20, 8);
        failures += check_sendrecv(floor_rung, 12U << 20, 4);
        failures += check_allreduce_queue(floor_rung, 4 << 20, 16);
        // Past the floor rung.
        failures += check_allreduce(floor_rung, (12U << 20) + 16, 4);
    }
    {
        // A mapped window below the ladder's floor: no rung maps, so the pair has no in-kernel
        // transport and every payload takes the fallback - the state a host without mapped pinned
        // memory (or one whose window cannot hold the smallest rung) is in.
        set_env("NINFER_TP2_AR_STAGING_MIB", "8");
        ninfer::tp::DevicePair below_floor(dev_a, dev_b);
        set_env("NINFER_TP2_AR_STAGING_MIB", nullptr);
        std::cout << "below the floor: in_kernel=" << below_floor.in_kernel_allreduce()
                  << " staging_bytes=" << below_floor.in_kernel_allreduce_bytes() << '\n';
        if (below_floor.in_kernel_allreduce() || below_floor.in_kernel_allreduce_bytes() != 0) {
            std::cerr << "a cap below the ladder floor still reported an in-kernel transport\n";
            ++failures;
        }
        failures += check_allreduce(below_floor, 20480, 8);
    }
    {
        ninfer::tp::DevicePair reserved(dev_a, dev_b);
        failures += check_reserve_channels(reserved);
    }
    // The one-collective skew the rotating arrival banks exist for, with the pre-bank geometry as its
    // positive control.
    failures += check_ar_skew(dev_a, dev_b);
    if (failures == 0) { std::cout << "PASS\n"; return 0; }
    std::cerr << failures << " failures\n";
    return 1;
}

int main() {
    try {
        return run_all_checks();
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 3;
    }
}
