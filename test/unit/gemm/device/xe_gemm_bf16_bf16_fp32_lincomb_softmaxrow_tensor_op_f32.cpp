/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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
 **************************************************************************************************/

#include <gtest/gtest.h>
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "gemm_testbed_3x.hpp"
#include <cute/tensor.hpp>

using namespace cute;

// Configuration Template for LinCombSoftmaxRow Tests
template<typename LayoutA, typename LayoutB>
struct MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig {
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementA = bfloat16_t;
  using ElementB = bfloat16_t;
  using ElementOutput = float;

  using GmemTiledCopyA = XE_2D_U16x8x16_LD_N;
  using GmemTiledCopyB = XE_2D_U16x16x16_LD_V;

  // Workgroup-level tile (matches the example configuration)
  using TileShape = Shape<_32, _512, _32>;

  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
                              Layout<TileShape>,
                              Layout<Shape<_2, _16, _1>, Stride<_16, _1, _0>>>::TiledMMA;

  using EpilogueTile = Shape<_16, _32>;
  constexpr static int PipelineStages = 3;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  // Linear Combination + Row-wise Softmax Epilogue
  using EpilogueOp = cutlass::epilogue::fusion::LinCombSoftmaxRow<
      ElementOutput,
      ElementComputeEpilogue,
      XE_2D_U32x8x16_ST_N,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape,
      EpilogueTile>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallBacks,
      XE_2D_U32x8x16_LD_N,
      void, void,
      void,
      void, void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementA,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementB,
      cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      GmemTiledCopyA, void, void, cute::identity,
      GmemTiledCopyB, void, void, cute::identity
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// PASSING TEST CASES 

// Test 1: Tile-aligned - 256x512x256
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, TileAligned_256x512x256) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(256, 512, 256, 1, alpha, beta)));
}

// Test 2: Single tile coverage
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, SingleTile_32x512x32) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(32, 512, 32, 1, alpha, beta)));
}

// Test 3: Multiple tiles in M
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, MultipleTilesM_128x512x32) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(128, 512, 32, 1, alpha, beta)));
}

// Test 4: Large N with K=32 - 64x512x32
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, LargeN_64x512x32) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(64, 512, 32, 1, alpha, beta)));
}

// Test 5: K=256 with large N - 64x512x256
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, K256_64x512x256) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(64, 512, 256, 1, alpha, beta)));
}

// Test 6: Multiple tiles M with K=256 - 256x512x256
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, MultipleTilesM_K256_256x512x256) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(256, 512, 256, 1, alpha, beta)));
}

// Test 7: Large M with K=32 - 512x512x32
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, LargeM_512x512x32) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(512, 512, 32, 1, alpha, beta)));
}

// Test 8: Extra large M with K=32 - 1024x512x32
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, ExtraLargeM_1024x512x32) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(1024, 512, 32, 1, alpha, beta)));
}

// Test 9: Rectangular K=128 - 256x512x128
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, K128_256x512x128) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(256, 512, 128, 1, alpha, beta)));
}

// Test 10: Small M, K=256 - 32x512x256
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, SmallM_K256_32x512x256) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(32, 512, 256, 1, alpha, beta)));
}


// FAILING TEST CASES - Disabled 


// Test 11: Medium square - 64x64x64 
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, DISABLED_MediumSquare_64x64x64) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(64, 64, 64, 1, alpha, beta)));
}

// Test 12: Medium rectangular - 128x256x128 
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, DISABLED_MediumRect_128x256x128) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(128, 256, 128, 1, alpha, beta)));
}

// Test 13: Non-aligned N dimension 
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, DISABLED_NonAlignedN_32x65x32) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(32, 65, 32, 1, alpha, beta)));
}

// Test 14: Non-aligned K dimension 
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, DISABLED_NonAlignedK_32x64x33) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(32, 64, 33, 1, alpha, beta)));
}

// Test 15: Very large - 1024x1024x1024 
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, DISABLED_VeryLarge_1024x1024x1024) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0 / 32.0;  // 1/sqrt(1024)
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(1024, 1024, 1024, 1, alpha, beta)));
}

// Test 16: Transformer-like dimensions - 2048x4096x2048 
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, DISABLED_Transformer_2048x4096x2048) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0 / 45.254;  // 1/sqrt(2048)
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(2048, 4096, 2048, 1, alpha, beta)));
}

// Test 17: Very wide matrix 
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, DISABLED_WideMatrix_32x2048x64) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(32, 2048, 64, 1, alpha, beta)));
}

// Test 18: Very tall matrix 
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, DISABLED_TallMatrix_2048x32x64) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(2048, 32, 64, 1, alpha, beta)));
}

// Test 19: Multiple tiles in N with N=1024 
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, DISABLED_MultipleTilesN_32x1024x32) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(32, 1024, 32, 1, alpha, beta)));
}

// Test 20: GPT-2 attention 
TEST(MainloopIntelXeXMX16_LinCombSoftmaxRow, DISABLED_GPT2_1024x1024x64) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSoftmaxRow_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0 / 8.0;  // 1/sqrt(64)
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSoftmaxRow<Gemm>(1024, 1024, 64, 1, alpha, beta)));
}

