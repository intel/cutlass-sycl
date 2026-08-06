/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Shared config-driven MoE grouped-GEMM runner (source of truth).

    The reusable runner for the hand-written MoE grouped GEMM across all data
    types (BF16 + low precision). It is SOURCE-ONLY: it consumes the kernel API
    under applications/moe_grouped_gemm/ directly and depends on nothing from
    benchmarks/ or examples/ (no Google Benchmark, no oneMKL, no CLI parsing).
    This mirrors applications/gdn_attention/gdn_runner.hpp (PR #702): a single
    runner under applications/ that both a benchmark and (later) an example can
    consume as thin drivers.

    Consumers include it as "moe_grouped_gemm/runner/moe_gemm_runner.hpp" (with
    ${CUTLASS_DIR}/applications on the include path). Everything lives in
    namespace cutlass::moe. It provides:
      - ScaleKind + the per-dtype Config structs / workgroup tiles,
      - MoETileShape<Config> + choose_tiled_mma<Config> (XE_DPAS_TT for BF16,
        XE_BDPAS_TT for block/tensor scaled),
      - moe_launch_timed<Config>() — the verification-free, single timed launch
        core (one N x K GEMM; returns elapsed device ms),
      - fill_scale<Element>() for the scaled paths, and
      - VerificationHelper (host-reference BF16 verify + host-dequant
        verify_scaled for the low-precision paths).

    The device kernel (MoE::MoEGEMM) is still only instantiated by the consumer's
    lean translation unit (e.g. benchmarks/.../moe_kernel_launch.cpp), keeping the
    AOT device codegen localized there.
*/

#pragma once

#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/initialize_block.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/sycl_tensor_fill.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/relatively_equal.h"
#include "cutlass/util/sycl_event_manager.hpp"

#include "moe_grouped_gemm/kernel/xe_moe_grouped_gemm.hpp"
#include "moe_grouped_gemm/kernel/xe_moe_tile_scheduler.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace cutlass::moe {

using namespace cute;
using namespace MoE;

using ElementAccumulator = float; // <- data type of accumulator

// Block-scale load surface alignment (Intel Xe cache line = 64 bytes).
// Per-expert scale buffers are padded to this so the hardware 2D block-scale
// load surface is aligned.
constexpr int kBlockScaleAlign = 64;

// Scale layout kind for a config.
//   Plain  : no scales — plain BF16 (the kernel takes the non-scaled path).
//   Block  : per-row, K-blocked (and N-blocked) scales (MX style).
//   Tensor : a single global scale per operand tensor (broadcast).
enum class ScaleKind { Plain, Block, Tensor };

///////////////////////////////////////////////////////////////////////////////////////////////////

struct VerificationHelper {

  int m = 0, n = 0, k = 0, groups;
  int *num_rows_per_expert = nullptr;
  std::vector<typename MoE::ProblemShape::UnderlyingProblemShape>
      problem_sizes_host;

  VerificationHelper() = default;

  void parse(const int num_experts, const int *num_tokens_per_expert_host,
             int moe_n, int moe_k,
             const int *num_tokens_per_expert_device = nullptr) {
    m = 0; // reset so a reused helper doesn't accumulate a stale row total
    n = moe_n;
    k = moe_k;
    groups = num_experts;
    num_rows_per_expert = const_cast<int *>(num_tokens_per_expert_device);
    assert(groups > 0);
    problem_sizes_host.clear();
    problem_sizes_host.reserve(groups);
    for (int i = 0; i < groups; i++) {
      problem_sizes_host.push_back({num_tokens_per_expert_host[i], n, k});
      m += num_tokens_per_expert_host[i];
    }
  }

  // BF16 verification via a host reference GEMM (no scaling). Like verify_scaled,
  // the reference runs on the REAL CPU (cutlass::reference::host::compute_gemm),
  // NOT the device path: "device" here is the cycle-accurate AubLoad simulator,
  // which runs on the CPU and is orders of magnitude slower than a native host
  // GEMM. A/B/D are copied to host, A/B upcast to FP32, and each expert is
  // compared with a relative tolerance (bf16/half rounding needs tolerance, not
  // an exact match against an FP32 reference).
  template <class ElementA, class ElementB, class ElementD,
            class = std::enable_if_t<
                is_any_of_v<ElementA, cute::bfloat16_t, cute::half_t> &&
                is_any_of_v<ElementB, cute::bfloat16_t, cute::half_t> &&
                is_any_of_v<ElementD, cute::bfloat16_t, cute::half_t>>>
  bool verify(const ElementA *activations, const ElementB *weights,
              ElementD *outputs) {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    sycl::queue Q = compat::get_default_queue();

    // Copy inputs + kernel output to host (no scaling for the Plain path).
    const int64_t A_elems = int64_t(m) * k;
    const int64_t B_elems = int64_t(groups) * n * k;
    const int64_t D_elems = int64_t(m) * n;
    std::vector<ElementA> h_A(A_elems);
    std::vector<ElementB> h_B(B_elems);
    std::vector<ElementD> h_D(D_elems);
    Q.memcpy(h_A.data(), activations, A_elems * sizeof(ElementA)).wait();
    Q.memcpy(h_B.data(), weights, B_elems * sizeof(ElementB)).wait();
    Q.memcpy(h_D.data(), outputs, D_elems * sizeof(ElementD)).wait();

    // Upcast A/B to FP32 host buffers for the host reference GEMM.
    std::vector<float> h_A_f(A_elems);
    std::vector<float> h_B_f(B_elems);
    for (int64_t i = 0; i < A_elems; i++)
      h_A_f[i] = float(h_A[i]);
    for (int64_t i = 0; i < B_elems; i++)
      h_B_f[i] = float(h_B[i]);

    const float rtol = 1e-2f;
    const float nonzero_floor = 1e-4f;

    std::vector<float> h_C(D_elems, 0.0f); // beta=0, unused C
    std::vector<float> h_ref_D;            // per-expert reference output
    bool passed = true;
    int cumM = 0;
    for (int g = 0; g < groups; g++) {
      int Mg = cute::get<0>(problem_sizes_host[g]);
      const int64_t a_off = int64_t(cumM) * k;   // A packed by tokens
      const int64_t b_off = int64_t(g) * n * k;  // B per expert
      const int64_t d_off = int64_t(cumM) * n;   // D packed by tokens
      h_ref_D.assign(int64_t(Mg) * n, 0.0f);

      cutlass::TensorRef<float, LayoutA> ref_A(h_A_f.data() + a_off, LayoutA::packed({Mg, k}));
      cutlass::TensorRef<float, LayoutB> ref_B(h_B_f.data() + b_off, LayoutB::packed({k, n}));
      cutlass::TensorRef<float, LayoutD> ref_C(h_C.data() + d_off,   LayoutD::packed({Mg, n}));
      cutlass::TensorRef<float, LayoutD> ref_Dt(h_ref_D.data(),      LayoutD::packed({Mg, n}));

      cutlass::reference::host::compute_gemm<
          float, LayoutA, float, LayoutB, float, LayoutD, float, float>(
          {Mg, n, k}, 1.0f, ref_A, ref_B, 0.0f, ref_C, ref_Dt, 0.0f);

      for (int64_t idx = 0; idx < int64_t(Mg) * n; idx++) {
        float got = float(h_D[d_off + idx]);
        if (!cutlass::relatively_equal(h_ref_D[idx], got, rtol, nonzero_floor)) {
          passed = false;
          break;
        }
      }
      if (!passed)
        break;
      cumM += Mg;
    }
    return passed;
  }

  // Verification for block / tensor scaled low-precision via host dequantization:
  // dequantize A and B to FP32 on the host (applying the per-block scales),
  // copy them back to the device, run the trusted device GemmComplex reference
  // per expert, and compare with BlockCompareRelativelyEqual. The device
  // reference scales to any problem size (no host triple-loop GEMM), so there
  // is no size cap.
  //
  // The dequant uses the same scale/data indexing as the (separately verified)
  // MoE block-scale mainloop in
  // applications/moe_grouped_gemm/collective/xe_moe_gemm.hpp:
  //   scaleA: per-expert column-major (M, scale_k) -> cumM*scale_k + row +
  //   kb*Mg scaleB: per-expert row-major   (scale_n, scale_k) -> g*sn*sk +
  //   nb*sk + kb A:  row-major (total_M, K). B:  per-expert KxN row-major: B[g,
  //   kk, col] at g*N*K + kk*N + col.
  // The scale element type is generic (FP32 for tensor scale, E8M0 for MX),
  // read through float().
  template <bool IsTensor, class ElementA, class ElementScale, class ElementD>
  bool verify_scaled(sycl::queue &Q, const ElementA *d_A, const ElementA *d_B,
                     const ElementScale *d_sA, const ElementScale *d_sB,
                     const ElementD *d_D, int GroupN, int GroupK) {
    const int scale_k = (k + GroupK - 1) / GroupK;
    const int scale_n = (n + GroupN - 1) / GroupN;
    // Block (MX) scale storage is padded so the hardware 2D scale load surface
    // is aligned (must match the launcher). Tensor scale is contiguous.
    auto round_up_align = [](int v) {
      return ((v + kBlockScaleAlign - 1) / kBlockScaleAlign) * kBlockScaleAlign;
    };

    constexpr int kBitsPerA = cute::sizeof_bits_v<ElementA>;
    constexpr bool kSubbyte = (kBitsPerA < 8);

    int64_t A_elems = int64_t(m) * k;
    int64_t B_elems = int64_t(groups) * n * k;

    // Copy quantized inputs to host. Sub-byte types are copied as raw packed
    // bytes and read via subbyte_iterator.
    std::vector<uint8_t> h_A_raw;
    std::vector<ElementA> h_A_full;
    std::vector<uint8_t> h_B_raw;
    std::vector<ElementA> h_B_full;
    if constexpr (kSubbyte) {
      constexpr int kElemsPerByte = 8 / kBitsPerA;
      // Ceiling division: a partial final byte still needs to be copied.
      h_A_raw.resize((A_elems + kElemsPerByte - 1) / kElemsPerByte);
      h_B_raw.resize((B_elems + kElemsPerByte - 1) / kElemsPerByte);
      Q.memcpy(h_A_raw.data(), (const uint8_t *)d_A, h_A_raw.size()).wait();
      Q.memcpy(h_B_raw.data(), (const uint8_t *)d_B, h_B_raw.size()).wait();
    } else {
      h_A_full.resize(A_elems);
      h_B_full.resize(B_elems);
      Q.memcpy(h_A_full.data(), d_A, A_elems * sizeof(ElementA)).wait();
      Q.memcpy(h_B_full.data(), d_B, B_elems * sizeof(ElementA)).wait();
    }

    // Per-expert padded M prefix sum and padded scale_n (Block path only).
    std::vector<int64_t> padded_offsetA(groups, 0);
    int64_t padded_M_total = 0;
    for (int g = 0; g < groups; g++) {
      padded_offsetA[g] = padded_M_total;
      int Mg = cute::get<0>(problem_sizes_host[g]);
      padded_M_total += IsTensor ? Mg : round_up_align(Mg);
    }
    const int padded_scale_n = IsTensor ? scale_n : round_up_align(scale_n);

    const int64_t sA_size =
        IsTensor ? int64_t(m) * scale_k : padded_M_total * scale_k;
    const int64_t sB_size = int64_t(groups) * padded_scale_n * scale_k;
    std::vector<ElementScale> h_sA(sA_size);
    std::vector<ElementScale> h_sB(sB_size);
    Q.memcpy(h_sA.data(), d_sA, sA_size * sizeof(ElementScale)).wait();
    Q.memcpy(h_sB.data(), d_sB, sB_size * sizeof(ElementScale)).wait();

    auto get_A = [&](int64_t idx) -> float {
      if constexpr (kSubbyte) {
        cute::subbyte_iterator<const ElementA> it(h_A_raw.data());
        auto ref = it[idx];
        return float(ElementA(ref));
      } else {
        return float(h_A_full[idx]);
      }
    };
    auto get_B = [&](int64_t idx) -> float {
      if constexpr (kSubbyte) {
        cute::subbyte_iterator<const ElementA> it(h_B_raw.data());
        auto ref = it[idx];
        return float(ElementA(ref));
      } else {
        return float(h_B_full[idx]);
      }
    };

    // Dequantize into FP32 host buffers (same layout as the quantized inputs).
    std::vector<float> h_A_dq(A_elems);
    std::vector<float> h_B_dq(B_elems);
    int cumM = 0;
    for (int g = 0; g < groups; g++) {
      int Mg = cute::get<0>(problem_sizes_host[g]);
      const int rowsA = IsTensor ? Mg : round_up_align(Mg); // A kb stride
      // A_dq[(cumM+row), kk] = A * scaleA[row, kk/GroupK]
      // scaleA index:
      //   Tensor: cumM*scale_k + row + kb*Mg          (contiguous (M,scale_k))
      //   Block : padded_offsetA[g]*scale_k + row + kb*round_up(Mg,64)
      //           (MN-major padded (M,scale_k,1) stride (1, round_up_M, ...))
      const int64_t baseA =
          IsTensor ? int64_t(cumM) * scale_k : padded_offsetA[g] * scale_k;
      for (int row = 0; row < Mg; row++) {
        for (int kk = 0; kk < k; kk++) {
          int kb = kk / GroupK;
          float sa = float(h_sA[baseA + row + int64_t(kb) * rowsA]);
          h_A_dq[int64_t(cumM + row) * k + kk] =
              get_A(int64_t(cumM + row) * k + kk) * sa;
        }
      }
      // B_dq[g, kk, col] = B * scaleB[col/GroupN, kk/GroupK]
      // scaleB index:
      //   Tensor: g*scale_n*scale_k + nb*scale_k + kb  (row-major
      //   (scale_n,scale_k)) Block : g*padded_scale_n*scale_k + col +
      //   kb*padded_scale_n
      //           (group_n==1 so nb==col; MN-major padded (scale_n,scale_k,1))
      for (int kk = 0; kk < k; kk++) {
        int kb = kk / GroupK;
        for (int col = 0; col < n; col++) {
          int nb = col / GroupN;
          float sb;
          if constexpr (IsTensor) {
            sb = float(h_sB[int64_t(g) * scale_n * scale_k +
                            int64_t(nb) * scale_k + kb]);
          } else {
            sb = float(h_sB[int64_t(g) * padded_scale_n * scale_k + col +
                            int64_t(kb) * padded_scale_n]);
          }
          h_B_dq[int64_t(g) * n * k + int64_t(kk) * n + col] =
              get_B(int64_t(g) * n * k + int64_t(kk) * n + col) * sb;
        }
      }
      cumM += Mg;
    }

    // Reference on the REAL CPU via cutlass::reference::host::compute_gemm (NOT
    // the device path: "device" here is the cycle-accurate AubLoad simulator,
    // which runs on the CPU and is orders of magnitude slower than a native host
    // GEMM). A/B are already dequantized into FP32 host buffers (h_A_dq/h_B_dq);
    // we copy the kernel output D to host and compare per expert with a relative
    // tolerance (dequant + low-precision accumulation needs tolerance, not exact).
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    std::vector<ElementD> h_D_raw(int64_t(m) * n);
    Q.memcpy(h_D_raw.data(), d_D, int64_t(m) * n * sizeof(ElementD)).wait();

    // relative tolerance: looser for 4-bit (mxfp4), like grouped_gemm.
    const float rtol = kSubbyte ? 2e-2f : 1e-2f;
    const float nonzero_floor = 1e-4f;

    std::vector<float> h_C(int64_t(m) * n, 0.0f); // beta=0, unused C
    std::vector<float> h_ref_D;                   // per-expert reference output
    bool passed = true;
    cumM = 0;
    for (int g = 0; g < groups; g++) {
      int Mg = cute::get<0>(problem_sizes_host[g]);
      const int64_t dq_off = int64_t(cumM) * k;     // A_dq packed by tokens
      const int64_t b_off  = int64_t(g) * n * k;    // B_dq per expert
      const int64_t d_off  = int64_t(cumM) * n;     // D packed by tokens
      h_ref_D.assign(int64_t(Mg) * n, 0.0f);

      cutlass::TensorRef<float, LayoutA> ref_A(h_A_dq.data() + dq_off, LayoutA::packed({Mg, k}));
      cutlass::TensorRef<float, LayoutB> ref_B(h_B_dq.data() + b_off,  LayoutB::packed({k, n}));
      cutlass::TensorRef<float, LayoutD> ref_C(h_C.data() + d_off,     LayoutD::packed({Mg, n}));
      cutlass::TensorRef<float, LayoutD> ref_Dt(h_ref_D.data(),        LayoutD::packed({Mg, n}));

      cutlass::reference::host::compute_gemm<
          float, LayoutA, float, LayoutB, float, LayoutD, float, float>(
          {Mg, n, k}, 1.0f, ref_A, ref_B, 0.0f, ref_C, ref_Dt, 0.0f);

      for (int64_t idx = 0; idx < int64_t(Mg) * n; idx++) {
        float got = float(h_D_raw[d_off + idx]);
        if (!cutlass::relatively_equal(h_ref_D[idx], got, rtol, nonzero_floor)) {
          passed = false; break;
        }
      }
      if (!passed) break;
      cumM += Mg;
    }
    return passed;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Selects the per-target workgroup tile from a Config (defined below): BMG
// keeps its original tiles while CRI/xe35 gets the tuned wide/deep ones. The
// preprocessor resolves this before parsing, so the alias is exactly one of the
// two member tiles. Config::TileShape{Cri,Bmg} are dependent names, so this may
// precede the config definitions.
template <class Config>
using MoETileShape =
#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
    typename Config::TileShapeCri;
#else
    typename Config::TileShapeBmg;
#endif

// Select the MMA atom for the given config. BF16 uses XE_DPAS_TT with a fixed
// K=32 tile; block-scaled low precision uses XE_BDPAS_TT with the K tile
// matched to the DPAS atom K (32 for 8-bit, 64 for 4-bit).
namespace detail {
// SGLayout selection: use Config::SGLayout if the config declares one, else Def.
template <class C, class Def, class = void> struct ConfigSGLayoutT { using type = Def; };
template <class C, class Def>
struct ConfigSGLayoutT<C, Def, cute::void_t<typename C::SGLayout>> {
  using type = typename C::SGLayout;
};
template <class C, class Def> using ConfigSGLayout = typename ConfigSGLayoutT<C, Def>::type;
} // namespace detail

template <class Config, class TA, class TB>
auto choose_tiled_mma(TA *A, TB *B) {
  using TA_non_CV = cutlass::platform::remove_cv_t<TA>;
  using TB_non_CV = cutlass::platform::remove_cv_t<TB>;

  // Subgroup tiling, n-major, chosen by hardware target. These are tuned for
  // the hardcoded input shapes used in this benchmark, which correspond to a
  // prefill step of the gpt-oss-20b model (the recorded per-expert token (M)
  // distribution is highly skewed). Not an endorsement for all input shapes.
  //   CRI / xe35 : 8x4 SGs (32 SGs / threads per WG, matching examples 50/51);
  //                per-SG 32(M)x32(N) over the 256x128 WG tile.
  //   xe20 / BMG : 8x2 SGs (16 SGs); per-SG 32(M)x64(N).
  // Note: the N subgroup count must keep per-SG N (BLK_N / SG_N) >= the GroupN
  // scale-broadcast width in the hand-written block-scale mainloop, so N is not
  // over-subdivided (e.g. a 4x8 layout -> per-SG N=16 breaks the broadcast).
#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
  using DefaultSGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
#else
  using DefaultSGLayout = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
#endif
  // Per-config SGLayout override: a (dtype,tile) config may declare its own
  // SGLayout (e.g. wide-N tiles need a different subgroup split, and the MX
  // block-scale path constrains per-SG N >= GroupN). Falls back to the
  // hardware default when the config does not specify one.
  using SGLayout = detail::ConfigSGLayout<Config, DefaultSGLayout>;

  // Workgroup tile comes from the config, selected per hardware target (see
  // MoETileShape), so the MMA tiling and the tile scheduler stay in lockstep
  // and the tile is tunable per (dtype, target) in one place.
  using WGTile = MoETileShape<Config>;
  if constexpr (Config::scale_kind == ScaleKind::Plain) {
    auto op = XE_DPAS_TT<8, float, TA_non_CV, TB_non_CV>{};
    using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>,
                                        SGLayout>::TiledMMA;
    return MMA{};
  } else {
    // Block-scaled DPAS: fp8/fp4 inputs, scale factors applied per element
    // via 2-element zip tensors inside moe_gemm_scaled(). WGTile K must be a
    // multiple of the DPAS atom K (32 for 8-bit, 64 for 4-bit).
    auto op = XE_BDPAS_TT<8, float, TA_non_CV>{};
    using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>,
                                        SGLayout>::TiledMMA;
    return MMA{};
  }
}

// type tag to define a unique sycl kernel name. MUST encode enough to be unique
// per compiled kernel body: ElementA/B/D + layouts + TileShape are NOT sufficient
// alone — two configs can share A/B/D+tile yet differ in scaling (e.g. fp8-tensor
// vs mxfp8-e4m3 are both e4m3->bf16 at 256x256x64 but take different scale paths).
// The trailing Config makes every (dtype, scale-kind, tile) a distinct name,
// required once one binary compiles more than one config (the merged build).
template <typename, typename, typename, char, char, typename, typename>
class GemmCuteName;

// Verification-free, single-launch CORE of the MoE GEMM launch. This is the
// SINGLE SOURCE OF TRUTH for the kernel-launch geometry (tile shape, scheduler
// params, MMA, grid/nd_range, kernel props) and the timed `parallel_for` body.
// The Google-Benchmark harness (benchmarks/applications/04_moe_gemm) times it in a loop.
// Submits ONE N x K kernel (no up-gate 2x / down-proj expansion), waits, and
// returns the elapsed device time in milliseconds.
template <char layoutA, char layoutB, class Config, typename ElementA,
          typename ElementB, typename ElementS, typename ElementD>
double moe_launch_timed(const ElementA *activations, const ElementB *weights,
                        const ElementS *scalesA, const ElementS *scalesB,
                        ElementD *outputs, const int gemm_n, const int gemm_k,
                        const int *num_rows_per_expert_device,
                        const int num_experts, const int group_n = 0,
                        const int group_k = 0) {
  // Change device_id to another value if you are running on a machine with
  // multiple GPUs and wish to use a GPU other than that with device ID 0.
  // For example, in a framework, you could query device ID.
  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  cutlass::KernelHardwareInfo hw_info{0, sm_count};
  auto dummy_problem_shape = cute::Shape<int, int, int>{1, gemm_k, gemm_n};
  // The GroupedGEMM API requires creation of  a vector of ProblemShape objects
  // for each GEMM problem, which is used in the GroupedGEMM tile-scheduler. If
  // there are 32 groups, then a vector of 32 `ProblemShape` objects is created.
  // Since these would not be known at compile time for a framework, they would
  // have to be created at run-time instead. However, for MoEGEMM, I just
  // provide one dummy shape, and then the custom code in tile scheduler can
  // derive the shape of each GEMM problem.
  auto dummy_group_problem_shape =
      cutlass::gemm::GroupProblemShape<Shape<int, int, int>>{
          1, &dummy_problem_shape, nullptr};
  // Scheduler TileShape comes from the same MoETileShape<Config> that
  // choose_tiled_mma() uses for the MMA WGTile, so the two can never drift.
  using TileShape = MoETileShape<Config>;
  using ClusterShape = Shape<_1, _1, _1>;
  auto scheduler_params =
      PersistentTileSchedulerXeMoE<ProblemShape>::to_underlying_arguments(
          dummy_group_problem_shape, TileShape{}, ClusterShape{}, hw_info,
          PersistentTileSchedulerXeMoE<ProblemShape>::Arguments{
              1, RasterOrderOptions::AlongN});
  auto group_distribution =
      PersistentTileSchedulerXeMoE<ProblemShape>::get_grid_shape(
          scheduler_params, dummy_group_problem_shape, TileShape{},
          ClusterShape{}, hw_info,
          PersistentTileSchedulerXeMoE<ProblemShape>::Arguments{
              1, RasterOrderOptions::AlongN});
  auto mma = choose_tiled_mma<Config>(activations, weights);
  auto MaxThreadsPerWorkgroup = size(mma);
  dim3 local_range{MaxThreadsPerWorkgroup, 1, 1};

  sycl::range<3> local = {local_range.z, local_range.y, local_range.x};
  sycl::range<3> groups = {group_distribution.z, group_distribution.y,
                           group_distribution.x};
  sycl::range<3> global = {local[0] * groups[0], local[1] * groups[1],
                           local[2] * groups[2]};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{syclex::sub_group_size<16>,
#if (defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))
                                  intelex::grf_size<512>
#else
                                  intelex::grf_size<256>
#endif
  };
  sycl::queue Q = compat::get_default_queue();

  GPU_Clock timer;
  timer.start();
  auto event = Q.parallel_for<
      GemmCuteName<ElementA, ElementB, ElementD, layoutA, layoutB, TileShape,
                   Config>>(
      sycl::nd_range<3>(global, local), kernel_props, [=](auto) {
        if constexpr (Config::scale_kind == ScaleKind::Plain) {
          MoE::MoEGEMM<void, void, void,
                       'R', 'R', 'R'>(activations, weights, scalesA, scalesB,
                                      outputs, mma, num_rows_per_expert_device,
                                      num_experts, gemm_n, gemm_k,
                                      scheduler_params);
        } else {
          MoE::MoEGEMM<void, void, void, 'R', 'R', 'R',
                       Config::group_n, Config::group_k>(
              activations, weights, scalesA, scalesB, outputs, mma,
              num_rows_per_expert_device, num_experts, gemm_n, gemm_k,
              scheduler_params, group_n, group_k);
        }
      });
  EventManager::getInstance().addEvent(event);
  Q.wait_and_throw();
  return double(timer.seconds() * 1000); // elapsed device time in ms
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Fills a scale buffer. The dequant range is bounded to [~0, 0.25] to ensure
// well-conditioned accumulation. For tensor
// (global) scale we use a single constant value across the whole buffer.
template <class Element>
void fill_scale(cutlass::DeviceAllocation<Element> &block, uint64_t seed,
                bool constant) {
  const float elt_max_f =
      float(cutlass::platform::numeric_limits<Element>::max());
  const float max_dequant_val = elt_max_f * 0.25f;
  const float min_dequant_val = 0.5f;
  const Element scale_max = Element(max_dequant_val / elt_max_f);
  const Element scale_min =
      constant ? scale_max : Element(min_dequant_val / elt_max_f);
#if defined(CUTLASS_TEST_FOR_CRI)
  cutlass::reference::device::BlockFillRandomUniformCopyFromHost(
      block.get(), block.size(), seed, scale_max, scale_min);
#else
  cutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, scale_max, scale_min);
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Per-variant configs. moe_kernel_launch.cpp selects one via -DMOE_BENCH_CONFIG.
//
//   Element      : packed input data type (A and B).
//   ElementScale : scale storage type (E8M0 for MX, FP32 for tensor scale).
//   group_k      : K block size for block scaling (ignored for tensor scale).
//   group_n      : N block size for block scaling (ignored for tensor scale).
//                  group_n = 1 is MX-exact (one scale per N column) loaded via
//                  the 2D block-scale path.
//   scale_kind   : Plain (BF16), Block (MX), or Tensor (single global scale).
///////////////////////////////////////////////////////////////////////////////////////////////////

// Workgroup tile (M, N, K) per config AND per hardware target. This is the #1
// perf knob for the scaled paths and is meant to be tuned, so every (dtype,
// target) tile is one named line below. Both the MMA atom tiling and the tile
// scheduler read MoETileShape<Config>, so they can never drift apart. K must
// be a multiple of the DPAS atom K (32 for 8-bit data, 64 for 4-bit).
//
// CRI (xe35): a deep K (2x atom-K) cuts the K-tile count and amortizes the
//   per-K-tile scale-load + dequant that was starving the scaled paths -- K is
//   the dominant lever for the MX dtypes. N widens to 256 (matches ex51); M
//   stays 256 (a wider M=512 hurt the small-N down-proj GEMM in A/B testing and
//   spilled the scalar tensor-scale path).
//     bf16        256x128x32   (plain XE_DPAS_TT path)
//     mxfp8       256x256x64   (K = 2x 8-bit atom-K)
//     mxfp4       512x256x128  (K = 2x 4-bit atom-K)
//     fp8-tensor  256x256x64   (K = 2x 8-bit atom-K)
// BMG (xe20): only the plain bf16 path runs here (the low-precision variants
//   are CRI-only), so bf16 is the only config that carries a BMG tile.
//     bf16        256x128x32
using Bf16TileCri = Shape<_256, _128, _32>;
using MxFp8TileCri = Shape<_256, _256, _64>;
using MxFp4TileCri = Shape<_512, _256, _128>;
using Fp8TensorTileCri = Shape<_256, _256, _64>;

using Bf16TileBmg = Shape<_256, _128, _32>;

// Output (D) element type is declared per Config (Config::ElementOutput) so the
// launcher reads it directly rather than inferring from scale_kind. All current
// paths (bf16, fp8-tensor, mxfp8, mxfp4) emit bf16.
// NOTE: fp8-tensor previously emitted fp16 (half_t); reverted to bf16 because the
// fp16 output triggered a wide-M D-store miscompute (kernel wrote ~3/16 of outputs
// as zero on the 352x256/128x896/320x512 tiles). See fp8 bf16-output fix.

// BF16: no scaling. ElementScale = void so the kernel takes the plain path.
struct Bf16Config {
  using Element = cutlass::bfloat16_t;
  using ElementScale = void;
  using ElementOutput = cutlass::bfloat16_t;
  using TileShapeCri = Bf16TileCri;
  using TileShapeBmg = Bf16TileBmg;
  static constexpr int group_k = 0;
  static constexpr int group_n = 0;
  static constexpr ScaleKind scale_kind = ScaleKind::Plain;
};

template <class TElement, class TScale, int GroupK, int GroupN, ScaleKind Kind,
          class TTileCri, class TOut = cutlass::bfloat16_t>
struct LowpConfig {
  using Element = TElement;
  using ElementScale = TScale;
  using ElementOutput = TOut;
  using TileShapeCri = TTileCri;
  static constexpr int group_k = GroupK;
  static constexpr int group_n = GroupN;
  static constexpr ScaleKind scale_kind = Kind;
};

// MXFP8 — 8-bit data, E8M0 (MX) block scale, K block = 32 (per the MX spec).
// group_n = 1: MX-exact per-N-row B scaling, loaded via the 2D block-scale path.
using MxFp8E4m3Config =
    LowpConfig<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1,
               ScaleKind::Block, MxFp8TileCri>;
using MxFp8E5m2Config =
    LowpConfig<cutlass::float_e5m2_t, cutlass::float_ue8m0_t, 32, 1,
               ScaleKind::Block, MxFp8TileCri>;

// MXFP4 — 4-bit data, E8M0 scale, MX-spec block size 32 (group_k = 32). The
// 4-bit DPAS atom-K is 64, so each MMA K-step spans two K-scale blocks; the
// hardware BDPAS path handles this via the scale K-offset (MMA_K / GroupK = 2)
// in make_scaled_offsets_k. group_n = 1 (per-N-row, MX-exact).
using MxFp4E2m1Config =
    LowpConfig<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1,
               ScaleKind::Block, MxFp4TileCri>;

// FP8 with a single global FP32 tensor scale (per operand / per expert weight).
// On CRI uses the narrower-M 256x256x64 tile: the tensor-scale broadcast path
// is scalar and register-heavy, so the wide 512-M MX tile spills.
using Fp8TensorE4m3Config =
    LowpConfig<cutlass::float_e4m3_t, float, 0, 0, ScaleKind::Tensor,
               Fp8TensorTileCri>;

// ===================== TILE-SWEEP:
using SG_8x4 = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
using SG_8x2 = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
using SG_4x8 = Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>;

template <class TElement, class TScale, int GroupK, int GroupN, ScaleKind Kind,
          class TTileCri, class TSG, class TOut = cutlass::bfloat16_t>
struct LowpConfigSG {
  using Element = TElement;
  using ElementScale = TScale;
  using ElementOutput = TOut;
  using TileShapeCri = TTileCri;
  using SGLayout = TSG;
  static constexpr int group_k = GroupK;
  static constexpr int group_n = GroupN;
  static constexpr ScaleKind scale_kind = Kind;
};
template <class TTileCri, class TSG>
struct Bf16ConfigSG {
  using Element = cutlass::bfloat16_t;
  using ElementScale = void;
  using ElementOutput = cutlass::bfloat16_t;
  using TileShapeCri = TTileCri;
  using TileShapeBmg = TTileCri;
  using SGLayout = TSG;
  static constexpr int group_k = 0;
  static constexpr int group_n = 0;
  static constexpr ScaleKind scale_kind = ScaleKind::Plain;
};

using MoeTile_64_1280_64 = Shape<_64, cute::Int<1280>, _64>;
using MoeTile_64_1792_64 = Shape<_64, cute::Int<1792>, _64>;
using MoeTile_96_640_128 = Shape<cute::Int<96>, cute::Int<640>, _128>;
using MoeTile_96_896_64 = Shape<cute::Int<96>, cute::Int<896>, _64>;
using MoeTile_64_896_64 = Shape<cute::Int<64>, cute::Int<896>, _64>;
using MoeTile_96_896_128 = Shape<cute::Int<96>, cute::Int<896>, _128>;
using MoeTile_96_1280_128 = Shape<cute::Int<96>, cute::Int<1280>, _128>;
using MoeTile_128_896_64 = Shape<_128, cute::Int<896>, _64>;
using MoeTile_128_896_128 = Shape<_128, cute::Int<896>, _128>;
using MoeTile_192_512_128 = Shape<cute::Int<192>, _512, _128>;
using MoeTile_192_640_64 = Shape<cute::Int<192>, cute::Int<640>, _64>;
using MoeTile_192_640_128 = Shape<cute::Int<192>, cute::Int<640>, _128>;
using MoeTile_256_128_32 = Shape<_256, _128, _32>;
using MoeTile_256_256_32 = Shape<_256, _256, _32>;
using MoeTile_256_256_64 = Shape<_256, _256, _64>;
using MoeTile_256_256_128 = Shape<_256, _256, _128>;
using MoeTile_320_512_64 = Shape<cute::Int<320>, _512, _64>;
using MoeTile_320_512_128 = Shape<cute::Int<320>, _512, _128>;
using MoeTile_352_256_64 = Shape<cute::Int<352>, _256, _64>;
using MoeTile_352_256_128 = Shape<cute::Int<352>, _256, _128>;
using MoeTile_448_256_64 = Shape<cute::Int<448>, _256, _64>;
using MoeTile_448_256_128 = Shape<cute::Int<448>, _256, _128>;
using MoeTile_448_320_64 = Shape<cute::Int<448>, cute::Int<320>, _64>;
using MoeTile_448_320_128 = Shape<cute::Int<448>, cute::Int<320>, _128>;
using MoeTile_512_256_32 = Shape<_512, _256, _32>;
using MoeTile_64_1536_32 = Shape<_64, cute::Int<1536>, _32>;
using MoeTile_608_128_64 = Shape<cute::Int<608>, _128, _64>;
using MoeTile_608_128_128 = Shape<cute::Int<608>, _128, _128>;
// Added for the src0 shapes.xlsx sweep (best per-shape tiles from the
// efficiency model). 8-bit dtypes use K-tile 64, 4-bit (mxfp4) uses K-tile 128.
using MoeTile_256_448_64 = Shape<_256, cute::Int<448>, _64>;
using MoeTile_256_448_128 = Shape<_256, cute::Int<448>, _128>;
using MoeTile_256_512_64 = Shape<_256, _512, _64>;
using MoeTile_256_512_128 = Shape<_256, _512, _128>;
using MoeTile_320_320_64 = Shape<cute::Int<320>, cute::Int<320>, _64>;
using MoeTile_320_320_128 = Shape<cute::Int<320>, cute::Int<320>, _128>;
using MoeTile_192_512_64 = Shape<cute::Int<192>, _512, _64>;

using Fp8Tensor_352_256_64 = LowpConfigSG<cutlass::float_e4m3_t, float, 0, 0, ScaleKind::Tensor, MoeTile_352_256_64, SG_4x8>;
using Fp8Tensor_128_896_64 = LowpConfigSG<cutlass::float_e4m3_t, float, 0, 0, ScaleKind::Tensor, MoeTile_128_896_64, SG_4x8>;
using Fp8Tensor_320_512_64 = LowpConfigSG<cutlass::float_e4m3_t, float, 0, 0, ScaleKind::Tensor, MoeTile_320_512_64, SG_4x8>;
using Fp8Tensor_64_1792_64 = LowpConfigSG<cutlass::float_e4m3_t, float, 0, 0, ScaleKind::Tensor, MoeTile_64_1792_64, SG_4x8>;
using Fp8Tensor_96_896_64 = LowpConfigSG<cutlass::float_e4m3_t, float, 0, 0, ScaleKind::Tensor, MoeTile_96_896_64, SG_4x8>;
using Fp8Tensor_64_896_64 = LowpConfigSG<cutlass::float_e4m3_t, float, 0, 0, ScaleKind::Tensor, MoeTile_64_896_64, SG_4x8>;
using Fp8Tensor_448_256_64 = LowpConfigSG<cutlass::float_e4m3_t, float, 0, 0, ScaleKind::Tensor, MoeTile_448_256_64, SG_8x4>;
using Fp8Tensor_608_128_64 = LowpConfigSG<cutlass::float_e4m3_t, float, 0, 0, ScaleKind::Tensor, MoeTile_608_128_64, SG_4x8>;
using Fp8Tensor_192_640_64 = LowpConfigSG<cutlass::float_e4m3_t, float, 0, 0, ScaleKind::Tensor, MoeTile_192_640_64, SG_4x8>;
using Fp8Tensor_448_320_64 = LowpConfigSG<cutlass::float_e4m3_t, float, 0, 0, ScaleKind::Tensor, MoeTile_448_320_64, SG_8x4>;
using Fp8Tensor_64_1280_64 = LowpConfigSG<cutlass::float_e4m3_t, float, 0, 0, ScaleKind::Tensor, MoeTile_64_1280_64, SG_4x8>;
using Fp8Tensor_256_256_64 = LowpConfigSG<cutlass::float_e4m3_t, float, 0, 0, ScaleKind::Tensor, MoeTile_256_256_64, SG_8x4>;
using MxFp8_352_256_64 = LowpConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_352_256_64, SG_4x8>;
using MxFp8_128_896_64 = LowpConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_128_896_64, SG_4x8>;
using MxFp8_320_512_64 = LowpConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_320_512_64, SG_4x8>;
using MxFp8_64_1792_64 = LowpConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_64_1792_64, SG_4x8>;
using MxFp8_96_896_64 = LowpConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_96_896_64, SG_4x8>;
using MxFp8_448_256_64 = LowpConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_448_256_64, SG_8x4>;
using MxFp8_608_128_64 = LowpConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_608_128_64, SG_4x8>;
using MxFp8_192_640_64 = LowpConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_192_640_64, SG_4x8>;
using MxFp8_448_320_64 = LowpConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_448_320_64, SG_8x4>;
using MxFp8_64_1280_64 = LowpConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_64_1280_64, SG_4x8>;
using MxFp8_256_256_64 = LowpConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_256_256_64, SG_8x4>;
using MxFp8E5m2_256_256_64 = LowpConfigSG<cutlass::float_e5m2_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_256_256_64, SG_8x4>;
// E5M2 tile-sweep set: mirrors the mxfp8 (e4m3) shapes/SG layouts above — same
// 8-bit MX block path, only the data element differs (e5m2 vs e4m3).
using MxFp8E5m2_352_256_64 = LowpConfigSG<cutlass::float_e5m2_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_352_256_64, SG_4x8>;
using MxFp8E5m2_128_896_64 = LowpConfigSG<cutlass::float_e5m2_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_128_896_64, SG_4x8>;
using MxFp8E5m2_320_512_64 = LowpConfigSG<cutlass::float_e5m2_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_320_512_64, SG_4x8>;
using MxFp8E5m2_64_1792_64 = LowpConfigSG<cutlass::float_e5m2_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_64_1792_64, SG_4x8>;
using MxFp8E5m2_96_896_64 = LowpConfigSG<cutlass::float_e5m2_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_96_896_64, SG_4x8>;
using MxFp8E5m2_448_256_64 = LowpConfigSG<cutlass::float_e5m2_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_448_256_64, SG_8x4>;
using MxFp8E5m2_608_128_64 = LowpConfigSG<cutlass::float_e5m2_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_608_128_64, SG_4x8>;
using MxFp8E5m2_192_640_64 = LowpConfigSG<cutlass::float_e5m2_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_192_640_64, SG_4x8>;
using MxFp8E5m2_448_320_64 = LowpConfigSG<cutlass::float_e5m2_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_448_320_64, SG_8x4>;
using MxFp8E5m2_64_1280_64 = LowpConfigSG<cutlass::float_e5m2_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_64_1280_64, SG_4x8>;
using MxFp4_352_256_128 = LowpConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_352_256_128, SG_4x8>;
using MxFp4_128_896_128 = LowpConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_128_896_128, SG_4x8>;
using MxFp4_320_512_128 = LowpConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_320_512_128, SG_4x8>;
using MxFp4_96_896_128 = LowpConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_96_896_128, SG_4x8>;
using MxFp4_192_512_128 = LowpConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_192_512_128, SG_4x8>;
using MxFp4_448_256_128 = LowpConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_448_256_128, SG_8x4>;
using MxFp4_96_640_128 = LowpConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_96_640_128, SG_4x8>;
using MxFp4_608_128_128 = LowpConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_608_128_128, SG_4x8>;
using MxFp4_192_640_128 = LowpConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_192_640_128, SG_4x8>;
using MxFp4_448_320_128 = LowpConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_448_320_128, SG_8x4>;
using MxFp4_96_1280_128 = LowpConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_96_1280_128, SG_4x8>;
using MxFp4_256_256_128 = LowpConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_256_256_128, SG_8x4>;

// ---- src0 shapes.xlsx best-per-shape tiles (added) ----
// SG layouts are the model's Best_sg_layout; all divisibility-verified
// (per-SG M %% 8 == 0, per-SG N %% 16 == 0). 256x448/320x320/448x256 use 8x4;
// 256x512/192x512/192x640 use 4x8.
using Fp8Tensor_256_448_64 = LowpConfigSG<cutlass::float_e4m3_t, float, 0, 0, ScaleKind::Tensor, MoeTile_256_448_64, SG_8x4>;
using Fp8Tensor_256_512_64 = LowpConfigSG<cutlass::float_e4m3_t, float, 0, 0, ScaleKind::Tensor, MoeTile_256_512_64, SG_4x8>;
using Fp8Tensor_320_320_64 = LowpConfigSG<cutlass::float_e4m3_t, float, 0, 0, ScaleKind::Tensor, MoeTile_320_320_64, SG_8x4>;
using Fp8Tensor_192_512_64 = LowpConfigSG<cutlass::float_e4m3_t, float, 0, 0, ScaleKind::Tensor, MoeTile_192_512_64, SG_4x8>;
using MxFp8_256_448_64 = LowpConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_256_448_64, SG_8x4>;
using MxFp8_256_512_64 = LowpConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_256_512_64, SG_4x8>;
using MxFp8_320_320_64 = LowpConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_320_320_64, SG_8x4>;
using MxFp8_192_512_64 = LowpConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_192_512_64, SG_4x8>;
using MxFp8E5m2_256_448_64 = LowpConfigSG<cutlass::float_e5m2_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_256_448_64, SG_8x4>;
using MxFp8E5m2_256_512_64 = LowpConfigSG<cutlass::float_e5m2_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_256_512_64, SG_4x8>;
using MxFp8E5m2_320_320_64 = LowpConfigSG<cutlass::float_e5m2_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_320_320_64, SG_8x4>;
using MxFp8E5m2_192_512_64 = LowpConfigSG<cutlass::float_e5m2_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_192_512_64, SG_4x8>;
using MxFp4_256_448_128 = LowpConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_256_448_128, SG_8x4>;
using MxFp4_256_512_128 = LowpConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_256_512_128, SG_4x8>;
using MxFp4_320_320_128 = LowpConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_320_320_128, SG_8x4>;
// (MxFp4_192_512_128 already defined above)

using Bf16_256_128_32 = Bf16ConfigSG<MoeTile_256_128_32, SG_8x2>;
using Bf16_256_256_32 = Bf16ConfigSG<MoeTile_256_256_32, SG_8x4>;
using Bf16_512_256_32 = Bf16ConfigSG<MoeTile_512_256_32, SG_8x4>;
using Bf16_64_1536_32 = Bf16ConfigSG<MoeTile_64_1536_32, SG_4x8>;


} // namespace cutlass::moe
