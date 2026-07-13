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
#include "benchmark_runner.hpp"
#include "gemm_configuration_sycl.hpp"

using Scheduler = cutlass::gemm::device::Scheduler;

#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)

template <
  typename ElementC,
  typename ElementD,
  typename AccType,
  typename LayoutB,
  Scheduler Sched,
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA = void,
  typename GmemTiledCopyB = void,
  bool UseW8A8 = false>
using Gemm_Bench_E4M3 = cute::conditional_t<UseW8A8,
    cutlass::gemm::device::W8A8GemmConfiguration<
        cutlass::float_e4m3_t, cutlass::layout::RowMajor,
        cutlass::float_e4m3_t, cutlass::layout::RowMajor,
        ElementC, cutlass::layout::RowMajor,
        AccType,
        TileShape,
        Tiler,
        GmemTiledCopyA, GmemTiledCopyB>,
    cutlass::gemm::device::GemmConfiguration<
        cutlass::arch::IntelXe,
        cutlass::float_e4m3_t, cutlass::layout::RowMajor,
        cutlass::float_e4m3_t, LayoutB,
        ElementC, cutlass::layout::RowMajor,
        ElementD,
        TileShape, Sched, Tiler,
        GmemTiledCopyA, GmemTiledCopyB,
        cutlass::epilogue::fusion::LinearCombination<ElementD, AccType>>>;

template <typename ElementC, typename ElementD, typename AccType,
          int WG_M, int WG_N, int WG_K, int SG_M, int SG_N, int Splits = 0,
          bool UseW8A8 = false>
using E4M3_RRR_GEMM_Base = Gemm_Bench_E4M3<ElementC, ElementD, AccType, cutlass::layout::RowMajor,
    (Splits > 0 ? Scheduler::GemmSplitK : Scheduler::Gemm),
    Shape<Int<WG_M>, Int<WG_N>, Int<WG_K>>,
    cute::conditional_t<UseW8A8,
        XeW8A8TiledMMA<WG_M, WG_N, WG_K, SG_M, SG_N>,
        XeTiledMMA<WG_M, WG_N, WG_K, SG_M, SG_N, AccType, cutlass::float_e4m3_t>>,
    cute::conditional_t<UseW8A8, XE_2D_U8x32x32_LD_N, void>,
    cute::conditional_t<UseW8A8, XE_2D_U8x32x32_LD_V, void>,
    UseW8A8>;

template <typename ElementC, typename ElementD, typename AccType,
          int WG_M, int WG_N, int WG_K, int SG_M, int SG_N, int Splits = 0,
          bool UseW8A8 = false>
struct E4M3_RRR_GEMM :
    E4M3_RRR_GEMM_Base<ElementC, ElementD, AccType, WG_M, WG_N, WG_K, SG_M, SG_N, Splits, UseW8A8> {
  using Base = E4M3_RRR_GEMM_Base<ElementC, ElementD, AccType, WG_M, WG_N, WG_K, SG_M, SG_N, Splits, UseW8A8>;
  using GemmKernel = typename Base::GemmKernel;
  static_assert(WG_M % SG_M == 0, "WG_M must be divisible by SG_M");
  static_assert(WG_N % SG_N == 0, "WG_N must be divisible by SG_N");

  constexpr static typename GemmKernel::Arguments defaultArguments() {
    if constexpr (!UseW8A8 && Splits > 0) {
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

using Gemm_E4M3E4M3FP32FP32FP32_RRR_WG256x256x32_SG32x64x32 = E4M3_RRR_GEMM<float, float, float, 256, 256, 32, 32, 64>;
using Gemm_E4M3E4M3FP32FP32FP32_RRR_WG512x256x64_SG64x64x64 = E4M3_RRR_GEMM<float, float, float, 512, 256, 64, 64, 64>;
using Gemm_E4M3E4M3FP32FP32FP32_RRR_WG512x256x64_SG64x64x64_SplitK2 = E4M3_RRR_GEMM<float, float, float, 512, 256, 64, 64, 64, 2>;
using Gemm_E4M3E4M3FP32FP32FP32_RRR_WG512x256x64_SG64x64x64_SplitK4 = E4M3_RRR_GEMM<float, float, float, 512, 256, 64, 64, 64, 4>;
using Gemm_E4M3E4M3FP32FP32FP32_RRR_WG8x128x64_SG8x32x64 = E4M3_RRR_GEMM<float, float, float, 8, 128, 64, 8, 32>;
using Gemm_E4M3E4M3FP32FP32FP32_RRR_WG16x128x64_SG16x32x64 = E4M3_RRR_GEMM<float, float, float, 16, 128, 64, 16, 32>;
using Gemm_E4M3E4M3FP32FP32FP32_RRR_WG64x128x64_SG64x32x64 = E4M3_RRR_GEMM<float, float, float, 64, 128, 64, 64, 32>;
using Gemm_E4M3E4M3FP32FP32FP32_RRR_WG256x128x64_SG32x32x64 = E4M3_RRR_GEMM<float, float, float, 256, 128, 64, 32, 32>;
using Gemm_E4M3E4M3FP32FP32FP32_RRR_WG4x128x64_SG4x32x64 = E4M3_RRR_GEMM<float, float, float, 4, 128, 64, 4, 32>;
using Gemm_E4M3E4M3FP32FP32FP32_RRR_WG8x256x64_SG8x32x64 = E4M3_RRR_GEMM<float, float, float, 8, 256, 64, 8, 32>;
using Gemm_E4M3E4M3FP32FP32FP32_RRR_WG8x128x64_SG8x16x64 = E4M3_RRR_GEMM<float, float, float, 8, 128, 64, 8, 16>;

using Gemm_E4M3E4M3BF16BF16BF16_RRR_WG512x256x128_SG64x64x128 = E4M3_RRR_GEMM<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, 512, 256, 128, 64, 64>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG512x256x64_SG64x64x64_SplitK4 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 512, 256, 64, 64, 64, 4>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG512x256x64_SG64x64x64_SplitK2 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 512, 256, 64, 64, 64, 2>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG512x256x64_SG64x64x64 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 512, 256, 64, 64, 64>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG16x128x64_SG16x32x64 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 16, 128, 64, 16, 32>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG64x128x64_SG64x32x64 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 64, 128, 64, 64, 32>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG256x128x64_SG32x32x64 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 256, 128, 64, 32, 32>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG8x128x64_SG8x32x64 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 8, 128, 64, 8, 32>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG4x128x64_SG4x32x64 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 4, 128, 64, 4, 32>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG8x256x64_SG8x32x64 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 8, 256, 64, 8, 32>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG8x128x64_SG8x16x64 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 8, 128, 64, 8, 16>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG16x128x128_SG16x32x128 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 16, 128, 128, 16, 32>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG64x128x128_SG64x32x128 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 64, 128, 128, 64, 32>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG192x128x64_SG24x32x64 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 192, 128, 64, 24, 32>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG8x128x256_SG8x32x256 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 8, 128, 256, 8, 32>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG8x256x64_SG8x16x64 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 8, 256, 64, 8, 16>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG128x256x128_SG16x64x128 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 128, 256, 128, 16, 64>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG256x256x128_SG32x64x128 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 256, 256, 128, 32, 64>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG256x128x64_SG64x16x64 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 256, 128, 64, 64, 16>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG256x256x64_SG64x64x64 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 256, 256, 64, 64, 64>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG8x128x512_SG8x16x512 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 8, 128, 512, 8, 16>;
using Gemm_E4M3E4M3FP32E4M3FP32_RRR_WG8x128x512_SG8x16x512_SplitK2 = E4M3_RRR_GEMM<float, cutlass::float_e4m3_t, float, 8, 128, 512, 8, 16, 2>;

using GemmW8A8_E4M3E4M3FP32FP32FP32_RRR_WG256x256x32_SG32x64x32 = E4M3_RRR_GEMM<float, float, float, 256, 256, 32, 32, 64, 0, true>;
using Gemm_E4M3E4M3FP32BF16FP32_RRR_WG128x192x128_SG16x48x128 = E4M3_RRR_GEMM<float, cutlass::bfloat16_t, float, 128, 192, 128, 16, 48>;
using Gemm_E4M3E4M3FP32BF16FP32_RRR_WG256x192x128_SG32x48x128 = E4M3_RRR_GEMM<float, cutlass::bfloat16_t, float, 256, 192, 128, 32, 48>;
using Gemm_E4M3E4M3FP32BF16FP32_RRR_WG256x128x64_SG64x16x64 = E4M3_RRR_GEMM<float, cutlass::bfloat16_t, float, 256, 128, 64, 64, 16>;
using Gemm_E4M3E4M3FP32BF16FP32_RRR_WG128x256x128_SG32x32x128 = E4M3_RRR_GEMM<float, cutlass::bfloat16_t, float, 128, 256, 128, 32, 32>;
using Gemm_E4M3E4M3FP32BF16FP32_RRR_WG256x256x128_SG64x32x128 = E4M3_RRR_GEMM<float, cutlass::bfloat16_t, float, 256, 256, 128, 64, 32>;
using Gemm_E4M3E4M3FP32BF16FP32_RRR_WG256x256x64_SG32x64x64 = E4M3_RRR_GEMM<float, cutlass::bfloat16_t, float, 256, 256, 64, 32, 64>;
using Gemm_E4M3E4M3FP32BF16FP32_RRR_WG256x256x64_SG64x64x64 = E4M3_RRR_GEMM<float, cutlass::bfloat16_t, float, 256, 256, 64, 64, 64>;
using Gemm_E4M3E4M3FP32BF16FP32_RRR_WG128x256x128_SG16x64x128 = E4M3_RRR_GEMM<float, cutlass::bfloat16_t, float, 128, 256, 128, 16, 64>;
using Gemm_E4M3E4M3FP32BF16FP32_RRR_WG512x256x64_SG64x64x64 = E4M3_RRR_GEMM<float, cutlass::bfloat16_t, float, 512, 256, 64, 64, 64>;
using Gemm_E4M3E4M3FP32BF16FP32_RRR_WG16x256x256_SG16x16x256 = E4M3_RRR_GEMM<float, cutlass::bfloat16_t, float, 16, 256, 256, 16, 16>;
using Gemm_E4M3E4M3FP32BF16FP32_RRR_WG64x128x256_SG16x16x256 = E4M3_RRR_GEMM<float, cutlass::bfloat16_t, float, 64, 128, 256, 16, 16>;
using Gemm_E4M3E4M3FP32BF16FP32_RRR_WG192x128x64_SG24x32x64 = E4M3_RRR_GEMM<float, cutlass::bfloat16_t, float, 192, 128, 64, 24, 32>;
using Gemm_E4M3E4M3FP32BF16FP32_RRR_WG8x128x256_SG8x32x256 = E4M3_RRR_GEMM<float, cutlass::bfloat16_t, float, 8, 128, 256, 8, 32>;
using Gemm_E4M3E4M3FP32BF16FP32_RRR_WG4x128x64_SG4x32x64 = E4M3_RRR_GEMM<float, cutlass::bfloat16_t, float, 4, 128, 64, 4, 32>;
using Gemm_E4M3E4M3FP32BF16FP32_RRR_WG8x256x64_SG8x32x64 = E4M3_RRR_GEMM<float, cutlass::bfloat16_t, float, 8, 256, 64, 8, 32>;
using Gemm_E4M3E4M3FP32BF16FP32_RRR_WG8x256x64_SG8x16x64 = E4M3_RRR_GEMM<float, cutlass::bfloat16_t, float, 8, 256, 64, 8, 16>;
using Gemm_E4M3E4M3FP32BF16FP32_RRR_WG4x128x256_SG4x16x256 = E4M3_RRR_GEMM<float, cutlass::bfloat16_t, float, 4, 128, 256, 4, 16>;
using Gemm_E4M3E4M3FP32BF16FP32_RRR_WG8x128x64_SG8x16x64 = E4M3_RRR_GEMM<float, cutlass::bfloat16_t, float, 8, 128, 64, 8, 16>;
#endif
