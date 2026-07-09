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

// Configuration Template for LinCombSplitK Tests
template<typename LayoutA, typename LayoutB>
struct MainloopIntelXeXMX16_LinCombSplitK_GemmConfig {
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementA = bfloat16_t;
  using ElementB = bfloat16_t;
  using ElementOutput = float;

  using GmemTiledCopyA = XE_2D_U16x8x16_LD_N;
  using GmemTiledCopyB = XE_2D_U16x16x16_LD_V;

  // Workgroup-level tile
  using TileShape = Shape<_32, _512, _32>;

  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
                              Layout<TileShape>,
                              Layout<Shape<_2, _16, _1>, Stride<_16, _1, _0>>>::TiledMMA;

  using EpilogueTile = Shape<_16, _32>;
  constexpr static int PipelineStages = 3;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  // Linear Combination + Split-K Epilogue (splits output into NOPE and ROPE)
  using EpilogueOp = cutlass::epilogue::fusion::LinCombSplitK<
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

// PASSING TEST CASES - Basic Functionality

// Test 1: Single tile coverage - minimal valid configuration
TEST(MainloopIntelXeXMX16_LinCombSplitK, SingleTile_32x192x32_1head_128nope_64rope) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSplitK_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSplitK<Gemm>(
      32, 192, 32, 1,      // m, n, k, l
      1, 128, 64,          // num_head, nope_dim, rope_dim
      alpha, beta)));
}

// Test 2: Multiple heads - standard attention configuration
TEST(MainloopIntelXeXMX16_LinCombSplitK, MultiHead_64x384x64_2heads_128nope_64rope) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSplitK_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSplitK<Gemm>(
      64, 384, 64, 1,      // m, n, k, l
      2, 128, 64,          // num_head, nope_dim, rope_dim
      alpha, beta)));
}

// Test 3: Tile-aligned dimensions
TEST(MainloopIntelXeXMX16_LinCombSplitK, TileAligned_256x512x256_2heads_128nope_128rope) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSplitK_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSplitK<Gemm>(
      256, 512, 256, 1,    // m, n, k, l
      2, 128, 128,         // num_head, nope_dim, rope_dim
      alpha, beta)));
}

// Test 4: Equal NOPE and ROPE dimensions
TEST(MainloopIntelXeXMX16_LinCombSplitK, EqualSplit_128x512x128_4heads_64nope_64rope) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSplitK_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSplitK<Gemm>(
      128, 512, 128, 1,    // m, n, k, l
      4, 64, 64,           // num_head, nope_dim, rope_dim
      alpha, beta)));
}

// Test 5: Small ROPE dimension (positional embeddings smaller than content)
TEST(MainloopIntelXeXMX16_LinCombSplitK, SmallRope_64x576x64_3heads_160nope_32rope) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSplitK_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSplitK<Gemm>(
      64, 576, 64, 1,      // m, n, k, l
      3, 160, 32,          // num_head, nope_dim, rope_dim
      alpha, beta)));
}

// Test 6: Large number of heads
TEST(MainloopIntelXeXMX16_LinCombSplitK, ManyHeads_128x1536x128_8heads_96nope_96rope) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSplitK_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSplitK<Gemm>(
      128, 1536, 128, 1,   // m, n, k, l
      8, 96, 96,           // num_head, nope_dim, rope_dim
      alpha, beta)));
}

// Test 7: Large M dimension (many tokens)
TEST(MainloopIntelXeXMX16_LinCombSplitK, LargeM_512x384x64_2heads_128nope_64rope) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSplitK_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSplitK<Gemm>(
      512, 384, 64, 1,     // m, n, k, l
      2, 128, 64,          // num_head, nope_dim, rope_dim
      alpha, beta)));
}

// Test 8: Large K dimension (deep features)
TEST(MainloopIntelXeXMX16_LinCombSplitK, LargeK_128x512x512_2heads_128nope_128rope) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSplitK_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSplitK<Gemm>(
      128, 512, 512, 1,    // m, n, k, l
      2, 128, 128,         // num_head, nope_dim, rope_dim
      alpha, beta)));
}

// Test 9: Realistic LLM configuration (DeepSeek-like)
TEST(MainloopIntelXeXMX16_LinCombSplitK, DeepSeekLike_256x1536x256_8heads_128nope_64rope) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSplitK_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSplitK<Gemm>(
      256, 1536, 256, 1,   // m, n, k, l
      8, 128, 64,          // num_head, nope_dim, rope_dim
      alpha, beta)));
}

// Test 10: Minimum valid dimensions (32-aligned)
TEST(MainloopIntelXeXMX16_LinCombSplitK, MinimalDims_32x64x32_1head_32nope_32rope) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSplitK_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSplitK<Gemm>(
      32, 64, 32, 1,       // m, n, k, l
      1, 32, 32,           // num_head, nope_dim, rope_dim
      alpha, beta)));
}

// Test 11: Large NOPE, small ROPE (content-heavy)
TEST(MainloopIntelXeXMM16_LinCombSplitK, ContentHeavy_128x768x128_4heads_160nope_32rope) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSplitK_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSplitK<Gemm>(
      128, 768, 128, 1,    // m, n, k, l
      4, 160, 32,          // num_head, nope_dim, rope_dim
      alpha, beta)));
}

// Test 12: Small NOPE, large ROPE (position-heavy)
TEST(MainloopIntelXeXMX16_LinCombSplitK, PositionHeavy_128x768x128_4heads_32nope_160rope) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSplitK_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSplitK<Gemm>(
      128, 768, 128, 1,    // m, n, k, l
      4, 32, 160,          // num_head, nope_dim, rope_dim
      alpha, beta)));
}

// Test 13: Very large combined (stress test)
TEST(MainloopIntelXeXMX16_LinCombSplitK, StressTest_512x3072x256_16heads_128nope_64rope) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSplitK_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSplitK<Gemm>(
      512, 3072, 256, 1,   // m, n, k, l
      16, 128, 64,         // num_head, nope_dim, rope_dim
      alpha, beta)));
}

// Test 14: Very large combined (stress test)
TEST(MainloopIntelXeXMX16_LinCombSplitK, WithScaling_128x384x128_2heads_128nope_64rope) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSplitK_GemmConfig<
      cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 2.0;
  double beta = 0.5;
  EXPECT_TRUE((test::gemm::device::TestXeSplitK<Gemm>(
      128, 384, 128, 1,    // m, n, k, l
      2, 128, 64,          // num_head, nope_dim, rope_dim
      alpha, beta)));
}

// Test 15: Column major A 
// EXPECTED FAILURE TESTS - Dimension Validation
TEST(MainloopIntelXeXMX16_LinCombSplitK, DISABLED_ColumnMajorA_128x384x128_2heads_128nope_64rope) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombSplitK_GemmConfig<
      cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_TRUE((test::gemm::device::TestXeSplitK<Gemm>(
      128, 384, 128, 1,    // m, n, k, l
      2, 128, 64,          // num_head, nope_dim, rope_dim
      alpha, beta)));
}

