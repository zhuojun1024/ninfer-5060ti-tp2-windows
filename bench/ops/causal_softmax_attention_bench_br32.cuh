#pragma once

// Benchmark-only probe: the production row-scaled E4M3 prompt kernel with Br 64 -> 32.
//
// This exists to answer one question before anyone rewrites the execution envelope in production:
// what does halving the query-row block cost? The 2-CTA/SM shape that would fill the kernel's 26%
// barrier stall is forced to Br=32 *and* Bc=32 *and* 8 warps (see docs/maintainer/
// tp2-adaptation-worklog.md section 7 for the register and smem walls), so both block dimensions
// halve and every per-tile cost is amortized over a quarter of the work. This file isolates the
// Br half of that doubling.
//
// Nothing but the shape constants changes relative to src/ops/softmax_attention/dense/causal_cache/
// prompt_fp8.cuh: the per-row online-softmax recurrence, the KV tile order, the operand of every
// output element and its accumulation order are identical, and only which warp owns which row
// differs. The result is therefore expected to be bit-identical to the production kernel, which
// the bench verifies with --digest.
//
// Do not wire this into the product. If the probe passes its gate, the production change is a
// separate decision (see the plan document under _temp/).

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ops/common/cuda_smem.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_fp8.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::br32_probe {

inline constexpr int kBr32Warps         = 16;
inline constexpr int kBr32Threads       = kBr32Warps * 32;
inline constexpr int kBr32Br            = 32;
inline constexpr int kBr32Bc            = kCausalPromptFp8Bc;
inline constexpr int kBr32DB16          = kCausalPromptFp8DB16;
inline constexpr int kBr32RowTiles      = kBr32Br / 16;
inline constexpr int kBr32ProducerWarps = 2 * kBr32RowTiles;
inline constexpr int kBr32DConsumers    = kBr32Warps / kBr32RowTiles;

inline constexpr int kBr32QBytes      = kBr32Br * kCausalPromptHeadDim;
inline constexpr int kBr32QScaleBytes = kBr32Br * static_cast<int>(sizeof(float));
inline constexpr int kBr32KBytes      = kBr32Bc * kCausalPromptHeadDim;
inline constexpr int kBr32VBytes      = kBr32Bc * kCausalPromptHeadDim;
inline constexpr int kBr32VStageBytes =
    kBr32Bc * kCausalPromptHeadDim * static_cast<int>(sizeof(__half));
inline constexpr int kBr32PBytes = kBr32Br * kBr32Bc * static_cast<int>(sizeof(__half));
inline constexpr int kBr32ScaleBytes =
    2 * kBr32Bc * static_cast<int>(sizeof(__half));
inline constexpr int kBr32StatsBytes = 7 * kBr32Br * static_cast<int>(sizeof(float));
inline constexpr int kBr32SmemBytes =
    kBr32QBytes + kBr32QScaleBytes + kBr32KBytes + kBr32VBytes + kBr32VStageBytes +
    kBr32PBytes + kBr32ScaleBytes + kBr32StatsBytes;

static_assert(kBr32DConsumers == 8);
static_assert(kBr32SmemBytes == 79104);

template <typename Geometry, typename Metadata>
__global__ __maxnreg__(120) void causal_attention_prompt_fp8_br32_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
    const __half* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width) {
    constexpr int D             = kCausalPromptHeadDim;
    constexpr int Br            = kBr32Br;
    constexpr int Bc            = kBr32Bc;
    constexpr int DB16          = kBr32DB16;
    constexpr int QKKs          = D / 32;
    constexpr int QKNt          = (Bc / 2) / 8;
    constexpr int PVNtPerWarp   = D / (kBr32DConsumers * 8);
    constexpr int PVKs          = Bc / 16;
    constexpr int ProducerWarps = kBr32ProducerWarps;
    constexpr int VWorkerWarps  = kBr32Warps - ProducerWarps;
    constexpr int WorkerThreads = VWorkerWarps * 32;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffU;
    static_assert(QKKs == 8);
    static_assert(PVNtPerWarp == 4);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::uint8_t* q_fp8 = reinterpret_cast<std::uint8_t*>(smem_raw);
    float* q_scale      = reinterpret_cast<float*>(q_fp8 + kBr32QBytes);
    std::uint8_t* k_fp8 = reinterpret_cast<std::uint8_t*>(
        reinterpret_cast<unsigned char*>(q_scale) + kBr32QScaleBytes);
    std::uint8_t* v_fp8 = k_fp8 + kBr32KBytes;
    __half* v_f16       = reinterpret_cast<__half*>(v_fp8 + kBr32VBytes);
    __half* p_s         = reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(v_f16) +
                                                    kBr32VStageBytes);
    __half* k_scale_s =
        reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(p_s) + kBr32PBytes);
    __half* v_scale_s    = k_scale_s + Bc;
    float* running_m_s   = reinterpret_cast<float*>(v_scale_s + Bc);
    float* running_l_s   = running_m_s + Br;
    float* partial_m_s   = running_l_s + Br;
    float* partial_l_s   = partial_m_s + 2 * Br;
    float* alpha_s       = partial_l_s + 2 * Br;
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_fp8);
    __nv_bfloat16* k_b16 = reinterpret_cast<__nv_bfloat16*>(k_fp8);

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) return;
    if (q0 >= tokens) {
        causal_prompt_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                                 kBr32Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int tile_rows             = min(Br, tokens - q0);
    const int max_query_abs         = base_pos + q0 + tile_rows - 1;
    const int key_blocks            = max_query_abs / Bc + 1;

    for (int row = warp; row < Br; row += kBr32Warps) {
        float values[8];
        float local_absmax = 0.0F;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            values[r] =
                row < tile_rows
                    ? __bfloat162float(q[causal_prompt_q_index<Geometry>(q_head, d, q0 + row)])
                    : 0.0F;
        }
        normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) local_absmax = fmaxf(local_absmax, fabsf(values[r]));
        const float absmax = warp_max(local_absmax, FullMask);
        const float qs     = absmax > 0.0F ? absmax / kKVCacheFp8MaxFinite : 0.0F;
        const float inv    = qs > 0.0F ? 1.0F / qs : 0.0F;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            causal_prompt_store_byte_swizzled(q_fp8, row, d,
                                              kv_cache_fp8_quant_code(values[r], inv));
        }
        if (lane == 0) q_scale[row] = qs;
    }
    if (tid < Br) {
        running_m_s[tid] = -CUDART_INF_F;
        running_l_s[tid] = 0.0F;
    }
    __syncthreads();

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    float q_scale_r0 = 0.0F;
    float q_scale_r1 = 0.0F;
    if (warp < ProducerWarps) {
        const int row0 = (warp >> 1) * 16 + gid;
        const int row1 = row0 + 8;
        q_scale_r0     = __shfl_sync(FullMask, lid == 0 ? q_scale[row0] : 0.0F, gid * 4);
        q_scale_r1     = __shfl_sync(FullMask, lid == 0 ? q_scale[row1] : 0.0F, gid * 4);
    }

    auto issue_kv_scales = [&](int tile_k0, int cooperative_tid, int cooperative_threads) {
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        const int page_offset0  = tile_k0 & (kPagedKVPageSize - 1);
        for (int key_l = cooperative_tid; key_l < Bc; key_l += cooperative_threads) {
            const int key = tile_k0 + key_l;
            if (key <= max_query_abs) {
                const std::int64_t off = kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head,
                                                                            page_offset0 + key_l);
                k_scale_s[key_l]       = cache_k_scale[off];
                v_scale_s[key_l]       = cache_v_scale[off];
            } else {
                k_scale_s[key_l] = __float2half_rn(0.0F);
                v_scale_s[key_l] = __float2half_rn(0.0F);
            }
        }
    };

    auto issue_kv_codes = [&](int tile_k0, int cooperative_tid, int cooperative_threads) {
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        const int page_offset0  = tile_k0 & (kPagedKVPageSize - 1);
#pragma unroll 1
        for (int chunk = cooperative_tid; chunk < Bc * (D / 16); chunk += cooperative_threads) {
            const int key_l  = chunk / (D / 16);
            const int dc     = chunk - key_l * (D / 16);
            const int d      = dc * 16;
            const int key    = tile_k0 + key_l;
            std::uint8_t* kd = &k_fp8[(key_l * DB16 + causal_prompt_swz(key_l, dc * 8)) * 2];
            std::uint8_t* vd = &v_fp8[key_l * D + d];
            if (key <= max_query_abs) {
                const std::int64_t off = kv_cache_fp8_code_index<Geometry>(physical_page, kv_head,
                                                                           d, page_offset0 + key_l);
                cp_async<16, Cache::cg>(kd, &cache_k[off]);
                cp_async<16, Cache::cg>(vd, &cache_v[off]);
            } else {
                store_vec(kd, make_int4(0, 0, 0, 0));
                store_vec(vd, make_int4(0, 0, 0, 0));
            }
        }
        ninfer::ops::cp_commit();
    };

    auto issue_kv_tile = [&](int tile_k0, int cooperative_tid, int cooperative_threads) {
        issue_kv_scales(tile_k0, cooperative_tid, cooperative_threads);
        issue_kv_codes(tile_k0, cooperative_tid, cooperative_threads);
    };

    issue_kv_tile(0, tid, kBr32Threads);
    ninfer::ops::cp_wait<0>();
    __syncthreads();

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[n][i] = 0.0F;
    }
    const float scale_l2 = scale * Log2E;
    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = kb * Bc;
        if (warp < ProducerWarps) {
            const int row_base = (warp >> 1) * 16;
            const int col_half = warp & 1;
            const int col_base = col_half * (Bc / 2);
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt)
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0F;

#pragma unroll
            for (int kk = 0; kk < QKKs; ++kk) {
                const int acol = kk * 16 + a_coloff;
                unsigned af[4];
                ldmatrix_x4(af[0], af[1], af[2], af[3],
                            smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                             causal_prompt_swz(row_base + a_rowoff, acol)]));
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    const int brow = col_base + nt * 8 + b_rin;
                    const int bcol = kk * 16 + b_koff;
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&k_b16[brow * DB16 + causal_prompt_swz(brow, bcol)]));
                    mma_fp8_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[0],
                                 af[1], af[2], af[3], bf[0], bf[1]);
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int keya = col_base + nt * 8 + 2 * lid;
                const int keyb = keya + 1;
                float ks0      = gid == 0 ? __half2float(k_scale_s[keya]) : 0.0F;
                float ks1      = gid == 0 ? __half2float(k_scale_s[keyb]) : 0.0F;
                ks0            = __shfl_sync(FullMask, ks0, lid);
                ks1            = __shfl_sync(FullMask, ks1, lid);
                score[nt][0] *= q_scale_r0 * ks0;
                score[nt][1] *= q_scale_r0 * ks1;
                score[nt][2] *= q_scale_r1 * ks0;
                score[nt][3] *= q_scale_r1 * ks1;
            }

            const int row0             = row_base + gid;
            const int row1             = row0 + 8;
            const int qabs0            = row0 < tile_rows ? base_pos + q0 + row0 : -1;
            const int qabs1            = row1 < tile_rows ? base_pos + q0 + row1 : -1;
            const bool full_score_tile = q0 + Br <= tokens && k0 + Bc - 1 <= base_pos + q0;
            float bm0                  = -CUDART_INF_F;
            float bm1                  = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + col_base + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                if (!full_score_tile) {
                    score[nt][0] = key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                    score[nt][1] = key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                    score[nt][2] = key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                    score[nt][3] = key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
                }
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);
            if (lid == 0) {
                partial_m_s[col_half * Br + row0] = bm0;
                partial_m_s[col_half * Br + row1] = bm1;
            }
            asm volatile("bar.sync 1, 128;" ::: "memory");

            bm0                     = fmaxf(partial_m_s[row0], partial_m_s[Br + row0]);
            bm1                     = fmaxf(partial_m_s[row1], partial_m_s[Br + row1]);
            const float previous_m0 = running_m_s[row0];
            const float previous_m1 = running_m_s[row1];
            const float nm0         = fmaxf(previous_m0, bm0);
            const float nm1         = fmaxf(previous_m1, bm1);
            const float nm0_scaled  = nm0 * scale_l2;
            const float nm1_scaled  = nm1 * scale_l2;
            const float alpha0      = previous_m0 == -CUDART_INF_F
                                          ? 0.0F
                                          : exp2_approx(__fmaf_rn(previous_m0, scale_l2, -nm0_scaled));
            const float alpha1      = previous_m1 == -CUDART_INF_F
                                          ? 0.0F
                                          : exp2_approx(__fmaf_rn(previous_m1, scale_l2, -nm1_scaled));
            float bl0               = 0.0F;
            float bl1               = 0.0F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = col_base + nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
                const float p00 = score[nt][0] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                      : 0.0F;
                const float p01 = score[nt][1] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0F;
                const float p10 = score[nt][2] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                      : 0.0F;
                const float p11 = score[nt][3] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0F;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                p_s[row0 * Bc + causal_prompt_p_swz<Bc>(row0, col0)] = __float2half_rn(p00);
                p_s[row0 * Bc + causal_prompt_p_swz<Bc>(row0, col1)] = __float2half_rn(p01);
                p_s[row1 * Bc + causal_prompt_p_swz<Bc>(row1, col0)] = __float2half_rn(p10);
                p_s[row1 * Bc + causal_prompt_p_swz<Bc>(row1, col1)] = __float2half_rn(p11);
            }
            bl0 = warp_sum<4>(bl0, FullMask);
            bl1 = warp_sum<4>(bl1, FullMask);
            if (lid == 0) {
                partial_l_s[col_half * Br + row0] = bl0;
                partial_l_s[col_half * Br + row1] = bl1;
            }
            asm volatile("bar.sync 1, 128;" ::: "memory");
            if (col_half == 0 && lid == 0) {
                const float tile_l0 = partial_l_s[row0] + partial_l_s[Br + row0];
                const float tile_l1 = partial_l_s[row1] + partial_l_s[Br + row1];
                running_l_s[row0]   = __fmaf_rn(running_l_s[row0], alpha0, tile_l0);
                running_l_s[row1]   = __fmaf_rn(running_l_s[row1], alpha1, tile_l1);
                running_m_s[row0]   = nm0;
                running_m_s[row1]   = nm1;
                alpha_s[row0]       = alpha0;
                alpha_s[row1]       = alpha1;
            }
        } else if (warp < ProducerWarps + VWorkerWarps) {
            const int worker_tid = tid - ProducerWarps * 32;
            for (int chunk = worker_tid; chunk < Bc * (D / 8); chunk += WorkerThreads) {
                const int key_l = chunk / (D / 8);
                const int dc    = chunk - key_l * (D / 8);
                const int d     = dc * 8;
                const int key   = k0 + key_l;
                __half* dst     = &v_f16[key_l * D + causal_prompt_swz(key_l, d)];
                if (key <= max_query_abs) {
                    store_vec(dst, causal_prompt_fp8_dequant_f16x8(&v_fp8[key_l * D + d],
                                                                   v_scale_s[key_l]));
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) issue_kv_tile((kb + 1) * Bc, tid, kBr32Threads);

        const int row_tile = warp % kBr32RowTiles;
        const int d_slice  = warp / kBr32RowTiles;
        const int row_base = row_tile * 16;
        const float alpha0 = alpha_s[row_base + gid];
        const float alpha1 = alpha_s[row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            unsigned pf[4];
            const int pcol = k * 16 + a_coloff;
            ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                        smem_addr(&p_s[(row_base + a_rowoff) * Bc +
                                       causal_prompt_p_swz<Bc>(row_base + a_rowoff, pcol)]));
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int global_n = d_slice * PVNtPerWarp + n;
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_f16[vrow * D + causal_prompt_swz(vrow, vcol)]));
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                        vf[0], vf[1]);
            }
        }
        if (has_next) ninfer::ops::cp_wait<0>();
        __syncthreads();
    }

    const int row_tile = warp % kBr32RowTiles;
    const int d_slice  = warp / kBr32RowTiles;
    const int row_base = row_tile * 16;
    const int row0     = row_base + gid;
    const int row1     = row0 + 8;
    const float inv_l0 = running_l_s[row0] > 0.0F ? __frcp_rn(running_l_s[row0]) : 0.0F;
    const float inv_l1 = running_l_s[row1] > 0.0F ? __frcp_rn(running_l_s[row1]) : 0.0F;
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int d0 = (d_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        if (row0 < tile_rows) {
            *reinterpret_cast<unsigned*>(
                &out[causal_prompt_q_index<Geometry>(q_head, d0, q0 + row0)]) =
                pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
        }
        if (row1 < tile_rows) {
            *reinterpret_cast<unsigned*>(
                &out[causal_prompt_q_index<Geometry>(q_head, d0, q0 + row1)]) =
                pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
        }
    }

    causal_prompt_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid,
                                             kBr32Threads);
}

template <typename Geometry, typename Metadata>
void br32_launch(const Tensor& q, const Tensor& positions, float scale,
                 const PagedKVLayerView& cache, Metadata metadata, Tensor& out,
                 cudaStream_t stream) {
    ops::detail::ensure_max_dynamic_shared_memory(
        reinterpret_cast<const void*>(causal_attention_prompt_fp8_br32_kernel<Geometry, Metadata>),
        kBr32SmemBytes);

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const dim3 grid(static_cast<unsigned>((tokens + kBr32Br - 1) / kBr32Br),
                    static_cast<unsigned>(Geometry::QHeads), 1u);
    causal_attention_prompt_fp8_br32_kernel<Geometry, Metadata>
        <<<grid, kBr32Threads, kBr32SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::uint8_t*>(cache.k_pages.data),
            static_cast<const std::uint8_t*>(cache.v_pages.data),
            static_cast<const __half*>(cache.k_scale_pages.data),
            static_cast<const __half*>(cache.v_scale_pages.data),
            metadata, static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

// Cached-entry launch: the read-only paged route the deep-context measurement uses. The variant
// covers only this entry; the append entry additionally owns the KV write, which is already
// measured by the production kernel and is not part of the question.
inline void br32_causal_attention_cached(const Tensor& q, const Tensor& positions, float scale,
                                         const PagedKVLayerView& cache, Tensor& out,
                                         cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        br32_launch<CausalD256H24Kv4>(q, positions, scale, cache, metadata, out, stream);
        return;
    }
    if (q.ne[1] == CausalD256H12Kv2::QHeads) {
        br32_launch<CausalD256H12Kv2>(q, positions, scale, cache, metadata, out, stream);
        return;
    }
    br32_launch<CausalD256H16Kv2>(q, positions, scale, cache, metadata, out, stream);
}

} // namespace ninfer::ops::br32_probe
