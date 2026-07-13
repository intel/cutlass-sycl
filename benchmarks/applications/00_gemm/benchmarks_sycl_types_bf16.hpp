/****************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ***************************************************************************************************/

#pragma once
#include "gemm_configuration_sycl.hpp"
#include "dual_gemm_benchmark_runner.hpp"
#include "cutlass/epilogue/thread/activation.h"

using Scheduler = cutlass::gemm::device::Scheduler;

template <
  typename ElementD,
  typename LayoutB,
  Scheduler Sched,
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA = void,
  typename GmemTiledCopyB = void>
using Gemm_Bench_BF16 = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, LayoutB,
    float, cutlass::layout::RowMajor,
    ElementD,
    TileShape, Sched, Tiler,
    GmemTiledCopyA, GmemTiledCopyB,
    cutlass::epilogue::fusion::LinearCombination<ElementD, float>>;

template <typename ElementD, int WG_M, int WG_N, int WG_K, int SG_M, int SG_N, int Splits = 0>
using BF16_RRR_GEMM_Base = Gemm_Bench_BF16<ElementD, cutlass::layout::RowMajor,
    (Splits > 0 ? Scheduler::GemmSplitK : Scheduler::Gemm),
    Shape<Int<WG_M>, Int<WG_N>, Int<WG_K>>,
    XeTiledMMA<WG_M, WG_N, WG_K, SG_M, SG_N, float, cutlass::bfloat16_t>>;

template <typename ElementD, int WG_M, int WG_N, int WG_K, int SG_M, int SG_N, int Splits = 0>
struct BF16_RRR_GEMM :
    BF16_RRR_GEMM_Base<ElementD, WG_M, WG_N, WG_K, SG_M, SG_N, Splits> {
  using Base = BF16_RRR_GEMM_Base<ElementD, WG_M, WG_N, WG_K, SG_M, SG_N, Splits>;
  using GemmKernel = typename Base::GemmKernel;
  static_assert(WG_M % SG_M == 0, "WG_M must be divisible by SG_M");
  static_assert(WG_N % SG_N == 0, "WG_N must be divisible by SG_N");

  constexpr static typename GemmKernel::Arguments defaultArguments() {
    if constexpr (Splits > 0) {
      using StreamKMode =
          cutlass::gemm::kernel::detail::PersistentTileSchedulerXeStreamKParams::DecompositionMode;
      typename GemmKernel::Arguments arguments{};
      arguments.scheduler = {Splits, StreamKMode::SplitK};
      return arguments;
    } else {
      return Base::defaultArguments();
    }
  }
};

// StreamK variant matching example 03_bmg_gemm_streamk (TileShape 256x256x32,
// KernelXeCooperative + StreamKScheduler).
template <int WG_M, int WG_N, int WG_K, int SG_M, int SG_N>
struct StreamKBF16BF16FP32FP32FP32_RRR_GEMM :
    Gemm_Bench_BF16<float, cutlass::layout::RowMajor, Scheduler::GemmStreamK,
        Shape<Int<WG_M>, Int<WG_N>, Int<WG_K>>,
        XeTiledMMA<WG_M, WG_N, WG_K, SG_M, SG_N, float, cutlass::bfloat16_t>> {
  static_assert(WG_M % SG_M == 0, "WG_M must be divisible by SG_M");
  static_assert(WG_N % SG_N == 0, "WG_N must be divisible by SG_N");
};

// Dual GEMM (sync from example 07_bmg_dual_gemm): one shared A matrix multiplied by two B
// matrices, fused through a SiLU activation epilogue. Uses MainloopIntelXeXMX16<2> + two
// linear-combination epilogues. TileShape <_128,_128,_64>, MMA XE_8x16x16_F32BF16BF16F32_TT.
using BmgDualGemm_BF16FP32_TileShape_128_128_64 = Shape<_128, _128, _64>;
using BmgDualGemm_BF16FP32_Tile_128_128_64 = typename TiledMMAHelper<
    MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
    Layout<BmgDualGemm_BF16FP32_TileShape_128_128_64>,
    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using DualGemm_BF16BF16FP32FP32_RRR_WG128x128x64_SG16x32x64 = cutlass::gemm::device::DualGemmConfiguration<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float,               cutlass::layout::RowMajor,
    BmgDualGemm_BF16FP32_TileShape_128_128_64,
    BmgDualGemm_BF16FP32_Tile_128_128_64,
    XE_2D_U16x16x32_LD_N, XE_2D_U16x32x32_LD_V, 2>;

// ---------------------------------------------------------------------------
// Activation-fused epilogue variants (sync from example 05_bmg_gemm_with_epilogues:
// 05_bmg_gemm_with_epilogue_{relu,silu,gelu}). Same BF16 GEMM as the baseline but
// with D = Act(alpha * A*B + beta * C). Uses MainloopXeL1Staged + IntelXeGeneric
// epilogue, TileShape <_256,_256,_32>, MMA XE_DPAS_TT<8,float,bf16>.
//
// NOTE: the generic benchmark verify() compares against a plain GEMM reference and
// does NOT apply the activation, so it only matches for the baseline. Verification
// is disabled under CUTLASS_TEST_FOR_CRI (the simulator path), so these cases are
// intended to be run on the CRI simulator (see input_files/cri/input_epilogue_gemm.in).
// ---------------------------------------------------------------------------
template <
  template <class> class ActivationFn,
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_BF16BF16FP32FP32FP32_RRR_EltAct = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB,
    cutlass::epilogue::fusion::LinCombEltAct<ActivationFn, float, float, float, float,
        cutlass::FloatRoundStyle::round_to_nearest>>;

// Compact benchmark declaration: bf16 in -> fp32 out, RRR, with fused activation.
template <template <class> class ActivationFn, int WG_M, int WG_N, int WG_K, int SG_M, int SG_N>
struct BF16BF16FP32FP32FP32_RRR_EltAct_GEMM :
    Gemm_Bench_BF16BF16FP32FP32FP32_RRR_EltAct<
        ActivationFn, Shape<Int<WG_M>, Int<WG_N>, Int<WG_K>>,
        XeTiledMMA<WG_M, WG_N, WG_K, SG_M, SG_N, float, cutlass::bfloat16_t>, void, void> {
  static_assert(WG_M % SG_M == 0, "WG_M must be divisible by SG_M");
  static_assert(WG_N % SG_N == 0, "WG_N must be divisible by SG_N");
};

using Gemm_BF16BF16FP32FP32FP32_RRR_WG512x256x32_SG64x64x32 = BF16_RRR_GEMM<float, 512, 256, 32, 64, 64>;
using Gemm_BF16BF16FP32FP32FP32_RRR_WG512x256x32_SG64x64x32_SplitK2 = BF16_RRR_GEMM<float, 512, 256, 32, 64, 64, 2>;
using Gemm_BF16BF16FP32FP32FP32_RRR_WG512x256x32_SG64x64x32_SplitK4 = BF16_RRR_GEMM<float, 512, 256, 32, 64, 64, 4>;
using Gemm_BF16BF16FP32FP32FP32_RRR_WG8x128x32_SG8x32x32 = BF16_RRR_GEMM<float, 8, 128, 32, 8, 32>;
using Gemm_BF16BF16FP32FP32FP32_RRR_WG16x64x32_SG8x16x32 = BF16_RRR_GEMM<float, 16, 64, 32, 8, 16>;
using Gemm_BF16BF16FP32FP32FP32_RRR_WG16x128x32_SG16x32x32 = BF16_RRR_GEMM<float, 16, 128, 32, 16, 32>;
using Gemm_BF16BF16FP32FP32FP32_RRR_WG64x128x32_SG64x32x32 = BF16_RRR_GEMM<float, 64, 128, 32, 64, 32>;
using Gemm_BF16BF16FP32FP32FP32_RRR_WG256x128x32_SG32x32x32 = BF16_RRR_GEMM<float, 256, 128, 32, 32, 32>;
using Gemm_BF16BF16FP32FP32FP32_RRR_WG4x128x32_SG4x32x32 = BF16_RRR_GEMM<float, 4, 128, 32, 4, 32>;
using Gemm_BF16BF16FP32FP32FP32_RRR_WG8x256x32_SG8x32x32 = BF16_RRR_GEMM<float, 8, 256, 32, 8, 32>;
using Gemm_BF16BF16FP32FP32FP32_RRR_WG8x128x32_SG8x16x32 = BF16_RRR_GEMM<float, 8, 128, 32, 8, 16>;

using Gemm_BF16BF16FP32BF16FP32_RRR_WG512x256x64_SG64x64x64 = BF16_RRR_GEMM<cutlass::bfloat16_t, 512, 256, 64, 64, 64>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG128x256x64_SG16x64x64 = BF16_RRR_GEMM<cutlass::bfloat16_t, 128, 256, 64, 16, 64>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG256x256x64_SG32x64x64 = BF16_RRR_GEMM<cutlass::bfloat16_t, 256, 256, 64, 32, 64>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG512x256x32_SG64x64x32_SplitK4 = BF16_RRR_GEMM<cutlass::bfloat16_t, 512, 256, 32, 64, 64, 4>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG512x256x32_SG64x64x32_SplitK2 = BF16_RRR_GEMM<cutlass::bfloat16_t, 512, 256, 32, 64, 64, 2>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG512x256x32_SG64x64x32 = BF16_RRR_GEMM<cutlass::bfloat16_t, 512, 256, 32, 64, 64>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG16x128x32_SG16x32x32 = BF16_RRR_GEMM<cutlass::bfloat16_t, 16, 128, 32, 16, 32>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG64x128x32_SG64x32x32 = BF16_RRR_GEMM<cutlass::bfloat16_t, 64, 128, 32, 64, 32>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG256x128x32_SG32x32x32 = BF16_RRR_GEMM<cutlass::bfloat16_t, 256, 128, 32, 32, 32>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG8x128x32_SG8x32x32 = BF16_RRR_GEMM<cutlass::bfloat16_t, 8, 128, 32, 8, 32>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG4x128x32_SG4x32x32 = BF16_RRR_GEMM<cutlass::bfloat16_t, 4, 128, 32, 4, 32>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG8x256x32_SG8x32x32 = BF16_RRR_GEMM<cutlass::bfloat16_t, 8, 256, 32, 8, 32>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG8x128x32_SG8x16x32 = BF16_RRR_GEMM<cutlass::bfloat16_t, 8, 128, 32, 8, 16>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG16x128x64_SG16x32x64 = BF16_RRR_GEMM<cutlass::bfloat16_t, 16, 128, 64, 16, 32>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG64x128x64_SG64x32x64 = BF16_RRR_GEMM<cutlass::bfloat16_t, 64, 128, 64, 64, 32>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG192x128x32_SG24x32x32 = BF16_RRR_GEMM<cutlass::bfloat16_t, 192, 128, 32, 24, 32>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG8x128x128_SG8x32x128 = BF16_RRR_GEMM<cutlass::bfloat16_t, 8, 128, 128, 8, 32>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG8x256x32_SG8x16x32 = BF16_RRR_GEMM<cutlass::bfloat16_t, 8, 256, 32, 8, 16>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG256x128x32_SG64x16x32 = BF16_RRR_GEMM<cutlass::bfloat16_t, 256, 128, 32, 64, 16>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG256x256x32_SG64x64x32 = BF16_RRR_GEMM<cutlass::bfloat16_t, 256, 256, 32, 64, 64>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG8x128x256_SG8x16x256 = BF16_RRR_GEMM<cutlass::bfloat16_t, 8, 128, 256, 8, 16>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG8x128x256_SG8x16x256_SplitK2 = BF16_RRR_GEMM<cutlass::bfloat16_t, 8, 128, 256, 8, 16, 2>;

using GemmStreamK_BF16BF16FP32FP32FP32_RRR_WG256x256x32_SG32x64x32 = StreamKBF16BF16FP32FP32FP32_RRR_GEMM<256, 256, 32, 32, 64>;

using GemmReLU_BF16BF16FP32FP32FP32_RRR_WG256x256x32_SG32x64x32 = BF16BF16FP32FP32FP32_RRR_EltAct_GEMM<cutlass::epilogue::thread::ReLu, 256, 256, 32, 32, 64>;
using GemmSiLU_BF16BF16FP32FP32FP32_RRR_WG256x256x32_SG32x64x32 = BF16BF16FP32FP32FP32_RRR_EltAct_GEMM<cutlass::epilogue::thread::SiLu, 256, 256, 32, 32, 64>;
using GemmGELU_BF16BF16FP32FP32FP32_RRR_WG256x256x32_SG32x64x32 = BF16BF16FP32FP32FP32_RRR_EltAct_GEMM<cutlass::epilogue::thread::GELU, 256, 256, 32, 32, 64>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG128x192x64_SG16x48x64 = BF16_RRR_GEMM<cutlass::bfloat16_t, 128, 192, 64, 16, 48>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG256x192x64_SG32x48x64 = BF16_RRR_GEMM<cutlass::bfloat16_t, 256, 192, 64, 32, 48>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG128x256x64_SG32x32x64 = BF16_RRR_GEMM<cutlass::bfloat16_t, 128, 256, 64, 32, 32>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG256x256x64_SG64x32x64 = BF16_RRR_GEMM<cutlass::bfloat16_t, 256, 256, 64, 64, 32>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG256x256x32_SG32x64x32 = BF16_RRR_GEMM<cutlass::bfloat16_t, 256, 256, 32, 32, 64>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG16x256x128_SG16x16x128 = BF16_RRR_GEMM<cutlass::bfloat16_t, 16, 256, 128, 16, 16>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG64x128x128_SG16x16x128 = BF16_RRR_GEMM<cutlass::bfloat16_t, 64, 128, 128, 16, 16>;
using Gemm_BF16BF16FP32BF16FP32_RRR_WG4x128x128_SG4x16x128 = BF16_RRR_GEMM<cutlass::bfloat16_t, 4, 128, 128, 4, 16>;

