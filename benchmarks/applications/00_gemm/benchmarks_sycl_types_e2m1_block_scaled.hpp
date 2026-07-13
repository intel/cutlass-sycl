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

using E2M1ElementType = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using E2M1ElementDataType = typename E2M1ElementType::DataType;
using E2M1ElementScale = typename E2M1ElementType::ScaleFactorType;

template <
  typename ElementC,
  typename ElementD,
  typename AccType,
  typename LayoutB,
  typename TileShape,
  typename Tiler,
  typename GroupSize = cute::_32>
using BLockScalingGemm_Bench_E2M1 = cutlass::gemm::device::BlockScalingGemmConfiguration<
    cutlass::arch::IntelXe,
    E2M1ElementDataType, cutlass::layout::RowMajor,
    E2M1ElementDataType, LayoutB,
    ElementC, cutlass::layout::RowMajor,
    E2M1ElementScale,
    cute::Stride<cute::_1, int64_t, int64_t>,
    ElementD,
    TileShape, Scheduler::Gemm, Tiler,
    void, void, void, void, GroupSize,
    cutlass::epilogue::fusion::LinearCombination<ElementD, AccType>>;

template <
  typename ElementC,
  typename ElementD,
  typename AccType,
  typename LayoutB,
  typename TileShape,
  typename Tiler>
using BLockScalingGemmNonNative_Bench_E2M1 = cutlass::gemm::device::BlockScalingGemmConfiguration<
    cutlass::arch::IntelXe,
    E2M1ElementDataType, cutlass::layout::RowMajor,
    E2M1ElementDataType, LayoutB,
    ElementC, cutlass::layout::RowMajor,
    E2M1ElementScale,
    cute::Stride<cute::_1, int64_t, int64_t>,
    ElementD,
    TileShape, Scheduler::Gemm, Tiler,
    void, void, void, void, cute::tuple<cute::_1, cute::_1, cute::_32>,
    cutlass::epilogue::fusion::LinearCombination<ElementD, AccType>>;

template <typename ElementC, typename ElementD, typename AccType,
          int WG_M, int WG_N, int WG_K, int SG_M, int SG_N,
          bool NonNative = false, typename GroupSize = cute::_32>
using E2M1_RCR_GEMM_BlockScaled_Base = cute::conditional_t<NonNative,
    BLockScalingGemmNonNative_Bench_E2M1<ElementC, ElementD, AccType, cutlass::layout::ColumnMajor,
        cute::Shape<cute::Int<WG_M>, cute::Int<WG_N>, cute::Int<WG_K>>,
        XeBlockScalingTiledMMA<WG_M, WG_N, WG_K, SG_M, SG_N, AccType, E2M1ElementDataType>>,
    BLockScalingGemm_Bench_E2M1<ElementC, ElementD, AccType, cutlass::layout::ColumnMajor,
        cute::Shape<cute::Int<WG_M>, cute::Int<WG_N>, cute::Int<WG_K>>,
        XeBlockScalingTiledMMA<WG_M, WG_N, WG_K, SG_M, SG_N, AccType, E2M1ElementDataType>, GroupSize>>;

template <typename ElementC, typename ElementD, typename AccType,
          int WG_M, int WG_N, int WG_K, int SG_M, int SG_N,
          bool NonNative = false, typename GroupSize = cute::_32>
struct E2M1_RCR_GEMM_BlockScaled :
    E2M1_RCR_GEMM_BlockScaled_Base<ElementC, ElementD, AccType, WG_M, WG_N, WG_K, SG_M, SG_N, NonNative, GroupSize> {
  static_assert(WG_M % SG_M == 0, "WG_M must be divisible by SG_M");
  static_assert(WG_N % SG_N == 0, "WG_N must be divisible by SG_N");
};

using BLockScalingGemm_E2M1E2M1FP32FP32FP32_RCR_WG256x256x64_SG32x64x64_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 256, 256, 64, 32, 64>;
using BLockScalingGemm_E2M1E2M1FP32FP32FP32_RCR_WG512x256x128_SG64x64x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 512, 256, 128, 64, 64>;
using BLockScalingGemm_E2M1E2M1FP32FP32FP32_RCR_WG8x128x128_SG8x32x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 8, 128, 128, 8, 32>;
using BLockScalingGemm_E2M1E2M1FP32FP32FP32_RCR_WG16x128x128_SG16x32x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 16, 128, 128, 16, 32>;
using BLockScalingGemm_E2M1E2M1FP32FP32FP32_RCR_WG64x128x128_SG64x32x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 64, 128, 128, 64, 32>;
using BLockScalingGemm_E2M1E2M1FP32FP32FP32_RCR_WG256x128x128_SG32x32x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 256, 128, 128, 32, 32>;
using BLockScalingGemm_E2M1E2M1FP32FP32FP32_RCR_WG4x128x128_SG4x32x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 4, 128, 128, 4, 32>;
using BLockScalingGemmNonNative_E2M1E2M1FP32FP32FP32_RCR_WG512x256x128_SG64x64x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 512, 256, 128, 64, 64, true>;
using BLockScalingGemmNonNative_E2M1E2M1FP32FP32FP32_RCR_WG8x256x128_SG8x32x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 8, 256, 128, 8, 32, true>;
using BLockScalingGemmNonNative_E2M1E2M1FP32FP32FP32_RCR_WG8x128x128_SG8x32x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 8, 128, 128, 8, 32, true>;
using BLockScalingGemmNonNative_E2M1E2M1FP32FP32FP32_RCR_WG8x128x128_SG8x16x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 8, 128, 128, 8, 16, true>;
using BLockScalingGemm_E2M1E2M1BF16BF16BF16_RCR_WG512x256x256_SG64x64x256_GS32 = E2M1_RCR_GEMM_BlockScaled<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, 512, 256, 256, 64, 64>;
using BLockScalingGemmNonNative_E2M1E2M1BF16BF16BF16_RCR_WG512x256x256_SG64x64x256_GS32 = E2M1_RCR_GEMM_BlockScaled<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, 512, 256, 256, 64, 64, true>;
using BLockScalingGemm_E2M1E2M1FP32FP32FP32_RCR_WG16x128x256_SG16x32x256_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 16, 128, 256, 16, 32>;
using BLockScalingGemm_E2M1E2M1FP32FP32FP32_RCR_WG64x128x256_SG64x32x256_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 64, 128, 256, 64, 32>;
using BLockScalingGemm_E2M1E2M1FP32FP32FP32_RCR_WG192x128x128_SG24x32x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 192, 128, 128, 24, 32>;
using BLockScalingGemmNonNative_E2M1E2M1FP32FP32FP32_RCR_WG8x128x512_SG8x32x512_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 8, 128, 512, 8, 32, true>;
using BLockScalingGemmNonNative_E2M1E2M1FP32FP32FP32_RCR_WG8x256x128_SG8x16x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 8, 256, 128, 8, 16, true>;
using BLockScalingGemm_E2M1E2M1FP32FP32FP32_RCR_WG128x256x256_SG16x64x256_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 128, 256, 256, 16, 64>;
using BLockScalingGemm_E2M1E2M1FP32FP32FP32_RCR_WG256x256x256_SG32x64x256_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 256, 256, 256, 32, 64>;
using BLockScalingGemm_E2M1E2M1FP32FP32FP32_RCR_WG256x128x128_SG64x16x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 256, 128, 128, 64, 16>;
using BLockScalingGemm_E2M1E2M1FP32FP32FP32_RCR_WG256x256x128_SG64x64x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 256, 256, 128, 64, 64>;
using BLockScalingGemmNonNative_E2M1E2M1FP32FP32FP32_RCR_WG8x128x1024_SG8x16x1024_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 8, 128, 1024, 8, 16, true>;
using BLockScalingGemm_E2M1E2M1FP32FP32FP32_RCR_WG8x128x128_SG8x16x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, float, float, 8, 128, 128, 8, 16>;
using BLockScalingGemm_E2M1E2M1FP32BF16FP32_RCR_WG128x192x256_SG16x48x256_GS32 = E2M1_RCR_GEMM_BlockScaled<float, cutlass::bfloat16_t, float, 128, 192, 256, 16, 48>;
using BLockScalingGemm_E2M1E2M1FP32BF16FP32_RCR_WG256x192x256_SG32x48x256_GS32 = E2M1_RCR_GEMM_BlockScaled<float, cutlass::bfloat16_t, float, 256, 192, 256, 32, 48>;
using BLockScalingGemm_E2M1E2M1FP32BF16FP32_RCR_WG256x128x128_SG64x16x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, cutlass::bfloat16_t, float, 256, 128, 128, 64, 16>;
using BLockScalingGemm_E2M1E2M1FP32BF16FP32_RCR_WG128x256x256_SG32x32x256_GS32 = E2M1_RCR_GEMM_BlockScaled<float, cutlass::bfloat16_t, float, 128, 256, 256, 32, 32>;
using BLockScalingGemm_E2M1E2M1FP32BF16FP32_RCR_WG256x256x256_SG64x32x256_GS32 = E2M1_RCR_GEMM_BlockScaled<float, cutlass::bfloat16_t, float, 256, 256, 256, 64, 32>;
using BLockScalingGemm_E2M1E2M1FP32BF16FP32_RCR_WG256x256x128_SG32x64x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, cutlass::bfloat16_t, float, 256, 256, 128, 32, 64>;
using BLockScalingGemm_E2M1E2M1FP32BF16FP32_RCR_WG256x256x128_SG64x64x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, cutlass::bfloat16_t, float, 256, 256, 128, 64, 64>;
using BLockScalingGemm_E2M1E2M1FP32BF16FP32_RCR_WG128x256x256_SG16x64x256_GS32 = E2M1_RCR_GEMM_BlockScaled<float, cutlass::bfloat16_t, float, 128, 256, 256, 16, 64>;
using BLockScalingGemm_E2M1E2M1FP32BF16FP32_RCR_WG512x256x128_SG64x64x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, cutlass::bfloat16_t, float, 512, 256, 128, 64, 64>;
using BLockScalingGemm_E2M1E2M1FP32BF16FP32_RCR_WG16x256x512_SG16x16x512_GS32 = E2M1_RCR_GEMM_BlockScaled<float, cutlass::bfloat16_t, float, 16, 256, 512, 16, 16>;
using BLockScalingGemm_E2M1E2M1FP32BF16FP32_RCR_WG64x128x512_SG16x16x512_GS32 = E2M1_RCR_GEMM_BlockScaled<float, cutlass::bfloat16_t, float, 64, 128, 512, 16, 16>;
using BLockScalingGemm_E2M1E2M1FP32BF16FP32_RCR_WG192x128x128_SG24x32x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, cutlass::bfloat16_t, float, 192, 128, 128, 24, 32>;
using BLockScalingGemmNonNative_E2M1E2M1FP32BF16FP32_RCR_WG8x128x512_SG8x32x512_GS32 = E2M1_RCR_GEMM_BlockScaled<float, cutlass::bfloat16_t, float, 8, 128, 512, 8, 32, true>;
using BLockScalingGemm_E2M1E2M1FP32BF16FP32_RCR_WG4x128x128_SG4x32x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, cutlass::bfloat16_t, float, 4, 128, 128, 4, 32>;
using BLockScalingGemmNonNative_E2M1E2M1FP32BF16FP32_RCR_WG8x256x128_SG8x32x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, cutlass::bfloat16_t, float, 8, 256, 128, 8, 32, true>;
using BLockScalingGemmNonNative_E2M1E2M1FP32BF16FP32_RCR_WG8x256x128_SG8x16x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, cutlass::bfloat16_t, float, 8, 256, 128, 8, 16, true>;
using BLockScalingGemmNonNative_E2M1E2M1FP32BF16FP32_RCR_WG4x128x512_SG4x16x512_GS32 = E2M1_RCR_GEMM_BlockScaled<float, cutlass::bfloat16_t, float, 4, 128, 512, 4, 16, true>;
using BLockScalingGemm_E2M1E2M1FP32BF16FP32_RCR_WG8x128x128_SG8x16x128_GS32 = E2M1_RCR_GEMM_BlockScaled<float, cutlass::bfloat16_t, float, 8, 128, 128, 8, 16>;
#endif
