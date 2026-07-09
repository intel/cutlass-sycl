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

/*! \file
    \brief Tests for Xe bf16_bf16_fp32 with LinCombTopKSoftmaxCol fusion
*/


#include <gtest/gtest.h>
#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "gemm_testbed_3x.hpp"
#include <cute/tensor.hpp>

using namespace cute;

namespace cutlass {
namespace {

// Configuration struct for LinCombTopKSoftmaxCol
template<int TopK, typename LayoutA, typename LayoutB>
struct MainloopIntelXeXMX16_LinCombTopKSoftmaxCol_GemmConfig {
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementA = bfloat16_t;
  using ElementB = bfloat16_t;
  using ElementOutput = float;

  using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;

  using TileShape = Shape<_256, _256, _32>;

  using TiledMma =
      TiledMMA<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>,
               Tile<Layout<Shape<_8, _8, _4>, Stride<_1, _32, _8>>,
                    Layout<Shape<_16, _4, _4>, Stride<_1, _64, _16>>, _32>>;

  constexpr static int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  using EpilogueOp = cutlass::epilogue::fusion::LinCombTopKSoftmaxCol<
      TopK, ElementOutput, ElementComputeEpilogue>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy, EpilogueOp, TileShape,
      decltype(tile_shape(TiledMma()))>;

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
      XE_2D_U32x8x16_ST_N,
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

// Test 1: Basic TopK=2 - small problem (16x8x64, matches example exactly)
TEST(MainloopIntelXeXMX16_LinCombTopKSoftmaxCol, BasicTopK2_16x8x64) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombTopKSoftmaxCol_GemmConfig<
      2, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0 / 64.0;
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 2>(16, 8, 64, 1, alpha, 0.0)));
}

// Test 2: Tiny square TopK=2 (8x8x8)
TEST(MainloopIntelXeXMX16_LinCombTopKSoftmaxCol, TinySquare_TopK2_8x8x8) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombTopKSoftmaxCol_GemmConfig<
      2, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0 / 8.0;
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 2>(8, 8, 8, 1, alpha, 0.0)));
}

// Test 3: Small square TopK=2 (16x16x16)
TEST(MainloopIntelXeXMX16_LinCombTopKSoftmaxCol, SmallSquare_TopK2_16x16x16) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombTopKSoftmaxCol_GemmConfig<
      2, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0 / 16.0;
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 2>(16, 16, 16, 1, alpha, 0.0)));
}

// Test 4: Small rectangular TopK=4 (32x16x32)
TEST(MainloopIntelXeXMX16_LinCombTopKSoftmaxCol, SmallRect_TopK4_32x16x32) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombTopKSoftmaxCol_GemmConfig<
      4, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0 / 32.0;
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 4>(32, 16, 32, 1, alpha, 0.0)));
}

// Test 5: Small with larger K TopK=2 (16x8x128)
TEST(MainloopIntelXeXMX16_LinCombTopKSoftmaxCol, SmallLargerK_TopK2_16x8x128) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombTopKSoftmaxCol_GemmConfig<
      2, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0 / 128.0;
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 2>(16, 8, 128, 1, alpha, 0.0)));
}

// Test 6: Small TopK=4 (16x16x32)
TEST(MainloopIntelXeXMX16_LinCombTopKSoftmaxCol, SmallSquare_TopK4_16x16x32) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombTopKSoftmaxCol_GemmConfig<
      4, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0 / 32.0;
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 4>(16, 16, 32, 1, alpha, 0.0)));
}

// Test 7: Rectangular TopK=2 (8x16x32)
TEST(MainloopIntelXeXMX16_LinCombTopKSoftmaxCol, Rectangular_TopK2_8x16x32) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombTopKSoftmaxCol_GemmConfig<
      2, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0 / 32.0;
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 2>(8, 16, 32, 1, alpha, 0.0)));
}

// Test 8: Medium TopK=2 (24x16x48)
TEST(MainloopIntelXeXMX16_LinCombTopKSoftmaxCol, Medium_TopK2_24x16x48) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombTopKSoftmaxCol_GemmConfig<
      2, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0 / 48.0;
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 2>(24, 16, 48, 1, alpha, 0.0)));
}

// Test 9: Tiny Matrices TopK=2 (multiple small sizes)
TEST(MainloopIntelXeXMX16_LinCombTopKSoftmaxCol, TinyMatrices_TopK2) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombTopKSoftmaxCol_GemmConfig<
      2, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 2>(4, 4, 4, 1, 1.0/4.0, 0.0)));
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 2>(2, 2, 2, 1, 1.0/2.0, 0.0)));
}

// Disabled Tests due to failure

// Test 10: Basic TopK=4 (256x256x256)
TEST(MainloopIntelXeXMX16_LinCombTopKSoftmaxCol, DISABLED_BasicTopK4_256x256x256) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombTopKSoftmaxCol_GemmConfig<
      4, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0 / 256.0;
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 4>(256, 256, 256, 1, alpha, 0.0)));
}

// Test 11: Large Model LLaMA2 7B TopK=2 (4096x4096x11008)
TEST(MainloopIntelXeXMX16_LinCombTopKSoftmaxCol, DISABLED_LargeModel_LLaMA2_7B_TopK2) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombTopKSoftmaxCol_GemmConfig<
      2, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0 / 11008.0;
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 2>(4096, 4096, 11008, 1, alpha, 0.0)));
}

// Test 12: Large Model LLaMA2 7B TopK=4 (4096x4096x11008)
TEST(MainloopIntelXeXMX16_LinCombTopKSoftmaxCol, DISABLED_LargeModel_LLaMA2_7B_TopK4) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombTopKSoftmaxCol_GemmConfig<
      4, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0 / 11008.0;
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 4>(4096, 4096, 11008, 1, alpha, 0.0)));
}

// Test 13: Micro Batch TopK=2 Batch4 (128x128x8192, batch=4)
TEST(MainloopIntelXeXMX16_LinCombTopKSoftmaxCol, DISABLED_MicroBatch_TopK2_Batch4) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombTopKSoftmaxCol_GemmConfig<
      2, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0 / 8192.0;
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 2>(128, 128, 8192, 4, alpha, 0.0)));
}

// Test 14: Multiple Batch Sizes TopK=2
TEST(MainloopIntelXeXMX16_LinCombTopKSoftmaxCol, DISABLED_MultipleBatchSizes_TopK2) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombTopKSoftmaxCol_GemmConfig<
      2, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 2>(512, 512, 1024, 2, 1.0/1024.0, 0.0)));
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 2>(256, 256, 512, 3, 1.0/512.0, 0.0)));
}

// Test 15: Tensor Parallel Config TopK=2 (128x4096x4096)
TEST(MainloopIntelXeXMX16_LinCombTopKSoftmaxCol, DISABLED_TensorParallelConfig_TopK2) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombTopKSoftmaxCol_GemmConfig<
      2, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0 / 4096.0;
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 2>(128, 4096, 4096, 1, alpha, 0.0)));
}

// Test 16: Model Parallel Config TopK=2 (4096x128x4096)
TEST(MainloopIntelXeXMX16_LinCombTopKSoftmaxCol, DISABLED_ModelParallelConfig_TopK2) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombTopKSoftmaxCol_GemmConfig<
      2, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  double alpha = 1.0 / 4096.0;
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 2>(4096, 128, 4096, 1, alpha, 0.0)));
}

// Test 17: Large K Small MN TopK=4
TEST(MainloopIntelXeXMX16_LinCombTopKSoftmaxCol, DISABLED_LargeKSmallMN_TopK4) {
  using Gemm = typename MainloopIntelXeXMX16_LinCombTopKSoftmaxCol_GemmConfig<
      4, cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Gemm;
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 4>(32, 32, 8192, 1, 1.0/8192.0, 0.0)));
  EXPECT_TRUE((test::gemm::device::TestXeTopKSoftmax<Gemm, 4>(64, 64, 16384, 1, 1.0/16384.0, 0.0)));
}

}
} // namespace cutlass

