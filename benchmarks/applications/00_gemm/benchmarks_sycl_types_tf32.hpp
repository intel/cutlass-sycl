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
using Gemm_Bench_TF32 = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::tfloat32_t, cutlass::layout::RowMajor,
    cutlass::tfloat32_t, LayoutB,
    float, cutlass::layout::RowMajor,
    ElementD,
    TileShape, Sched, Tiler,
    GmemTiledCopyA, GmemTiledCopyB,
    cutlass::epilogue::fusion::LinearCombination<ElementD, float>>;

template <typename ElementD, int WG_M, int WG_N, int WG_K, int SG_M, int SG_N, int Splits = 0>
using TF32_RRR_GEMM_Base = Gemm_Bench_TF32<ElementD, cutlass::layout::RowMajor,
    (Splits > 0 ? Scheduler::GemmSplitK : Scheduler::Gemm),
    Shape<Int<WG_M>, Int<WG_N>, Int<WG_K>>,
    XeTiledMMA<WG_M, WG_N, WG_K, SG_M, SG_N, float, cutlass::tfloat32_t>>;

template <typename ElementD, int WG_M, int WG_N, int WG_K, int SG_M, int SG_N, int Splits = 0>
struct TF32_RRR_GEMM :
    TF32_RRR_GEMM_Base<ElementD, WG_M, WG_N, WG_K, SG_M, SG_N, Splits> {
  using Base = TF32_RRR_GEMM_Base<ElementD, WG_M, WG_N, WG_K, SG_M, SG_N, Splits>;
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

using Gemm_TF32TF32FP32FP32FP32_RRR_WG512x256x16_SG64x64x16 = TF32_RRR_GEMM<float, 512, 256, 16, 64, 64>;
using Gemm_TF32TF32FP32FP32FP32_RRR_WG512x256x16_SG64x64x16_SplitK2 = TF32_RRR_GEMM<float, 512, 256, 16, 64, 64, 2>;
using Gemm_TF32TF32FP32FP32FP32_RRR_WG512x256x16_SG64x64x16_SplitK4 = TF32_RRR_GEMM<float, 512, 256, 16, 64, 64, 4>;
using Gemm_TF32TF32FP32FP32FP32_RRR_WG8x128x16_SG8x32x16 = TF32_RRR_GEMM<float, 8, 128, 16, 8, 32>;
using Gemm_TF32TF32FP32FP32FP32_RRR_WG16x128x16_SG16x32x16 = TF32_RRR_GEMM<float, 16, 128, 16, 16, 32>;
using Gemm_TF32TF32FP32FP32FP32_RRR_WG64x128x16_SG64x32x16 = TF32_RRR_GEMM<float, 64, 128, 16, 64, 32>;
using Gemm_TF32TF32FP32FP32FP32_RRR_WG256x128x16_SG32x32x16 = TF32_RRR_GEMM<float, 256, 128, 16, 32, 32>;
using Gemm_TF32TF32FP32FP32FP32_RRR_WG4x128x16_SG4x32x16 = TF32_RRR_GEMM<float, 4, 128, 16, 4, 32>;
using Gemm_TF32TF32FP32FP32FP32_RRR_WG8x256x16_SG8x32x16 = TF32_RRR_GEMM<float, 8, 256, 16, 8, 32>;
using Gemm_TF32TF32FP32FP32FP32_RRR_WG8x128x16_SG8x16x16 = TF32_RRR_GEMM<float, 8, 128, 16, 8, 16>;

using Gemm_TF32TF32FP32TF32FP32_RRR_WG512x256x16_SG64x64x16_SplitK4 = TF32_RRR_GEMM<cutlass::tfloat32_t, 512, 256, 16, 64, 64, 4>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG512x256x16_SG64x64x16_SplitK2 = TF32_RRR_GEMM<cutlass::tfloat32_t, 512, 256, 16, 64, 64, 2>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG512x256x16_SG64x64x16 = TF32_RRR_GEMM<cutlass::tfloat32_t, 512, 256, 16, 64, 64>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG16x128x16_SG16x32x16 = TF32_RRR_GEMM<cutlass::tfloat32_t, 16, 128, 16, 16, 32>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG64x128x16_SG64x32x16 = TF32_RRR_GEMM<cutlass::tfloat32_t, 64, 128, 16, 64, 32>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG256x128x16_SG32x32x16 = TF32_RRR_GEMM<cutlass::tfloat32_t, 256, 128, 16, 32, 32>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG8x128x16_SG8x32x16 = TF32_RRR_GEMM<cutlass::tfloat32_t, 8, 128, 16, 8, 32>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG4x128x16_SG4x32x16 = TF32_RRR_GEMM<cutlass::tfloat32_t, 4, 128, 16, 4, 32>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG8x256x16_SG8x32x16 = TF32_RRR_GEMM<cutlass::tfloat32_t, 8, 256, 16, 8, 32>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG8x128x16_SG8x16x16 = TF32_RRR_GEMM<cutlass::tfloat32_t, 8, 128, 16, 8, 16>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG16x128x32_SG16x32x32 = TF32_RRR_GEMM<cutlass::tfloat32_t, 16, 128, 32, 16, 32>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG64x128x32_SG64x32x32 = TF32_RRR_GEMM<cutlass::tfloat32_t, 64, 128, 32, 64, 32>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG192x128x16_SG24x32x16 = TF32_RRR_GEMM<cutlass::tfloat32_t, 192, 128, 16, 24, 32>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG8x128x64_SG8x32x64 = TF32_RRR_GEMM<cutlass::tfloat32_t, 8, 128, 64, 8, 32>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG8x256x16_SG8x16x16 = TF32_RRR_GEMM<cutlass::tfloat32_t, 8, 256, 16, 8, 16>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG128x256x32_SG16x64x32 = TF32_RRR_GEMM<cutlass::tfloat32_t, 128, 256, 32, 16, 64>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG256x256x32_SG32x64x32 = TF32_RRR_GEMM<cutlass::tfloat32_t, 256, 256, 32, 32, 64>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG256x128x16_SG64x16x16 = TF32_RRR_GEMM<cutlass::tfloat32_t, 256, 128, 16, 64, 16>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG256x256x16_SG64x64x16 = TF32_RRR_GEMM<cutlass::tfloat32_t, 256, 256, 16, 64, 64>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG8x128x128_SG8x16x128 = TF32_RRR_GEMM<cutlass::tfloat32_t, 8, 128, 128, 8, 16>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG8x128x128_SG8x16x128_SplitK2 = TF32_RRR_GEMM<cutlass::tfloat32_t, 8, 128, 128, 8, 16, 2>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG128x192x32_SG16x48x32 = TF32_RRR_GEMM<cutlass::tfloat32_t, 128, 192, 32, 16, 48>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG256x192x32_SG32x48x32 = TF32_RRR_GEMM<cutlass::tfloat32_t, 256, 192, 32, 32, 48>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG128x256x32_SG32x32x32 = TF32_RRR_GEMM<cutlass::tfloat32_t, 128, 256, 32, 32, 32>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG256x256x32_SG64x32x32 = TF32_RRR_GEMM<cutlass::tfloat32_t, 256, 256, 32, 64, 32>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG256x256x16_SG32x64x16 = TF32_RRR_GEMM<cutlass::tfloat32_t, 256, 256, 16, 32, 64>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG16x256x64_SG16x16x64 = TF32_RRR_GEMM<cutlass::tfloat32_t, 16, 256, 64, 16, 16>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG64x128x64_SG16x16x64 = TF32_RRR_GEMM<cutlass::tfloat32_t, 64, 128, 64, 16, 16>;
using Gemm_TF32TF32FP32TF32FP32_RRR_WG4x128x64_SG4x16x64 = TF32_RRR_GEMM<cutlass::tfloat32_t, 4, 128, 64, 4, 16>;
