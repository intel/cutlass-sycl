/***************************************************************************************************
 * Copyright (c) 2025 - 2026 Intel Corporation, All rights reserved.
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
\brief CUTLASS Intel xe35 FP8 Block Scaled Grouped GEMM with E4M3 inputs and fp32 scale factors.

  This FP8 software-scaled grouped GEMM is consolidated under Example 51.

    To build & run this example (from your build dir):

      $ ninja 51_xe35_block_scaled_grouped_gemm_fp8_e4m3
      $ ./examples/51_xe35_block_scaled_grouped_gemm/51_xe35_block_scaled_grouped_gemm_fp8_e4m3

    Call with `--help` for information about available options
*/

#include "51_xe35_block_scaled_grouped_gemm_runner.hpp"

template <typename ElementInputA,
          typename ElementInputB,
          typename ElementScale,
          typename TileShape,
          typename GroupSizeMNK,
          typename ThreadLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>,
          typename LayoutA = cutlass::layout::RowMajor,
          typename LayoutB = cutlass::layout::RowMajor>
cutlass::Status run_fp8_blockscaled_case(Options & options){

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementOutput = float;

  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  // Scale strides — different for A (column-major) and B (row-major)
  using StrideScaleA = cute::Stride<_1, int64_t, int64_t>;
  using StrideScaleB = cute::Stride<int64_t, _1, int64_t>;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  using GmemTiledCopyScaleA = void;
  using GmemTiledCopyScaleB = void;

  // XE_BDPAS_TT — block-scaled DPAS. With fp32 scale factors and 2-element zip tensor,
  // the software scaling path is selected automatically in mma_unpack.
  using TiledMma = typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, float, ElementInputA>>, Layout<TileShape>, ThreadLayout>::TiledMMA;

  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16BlockScaledGroupImpl<PipelineStages, GroupSizeMNK>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGenericGroup;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementComputeEpilogue,
          ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          void,
          ElementAccumulator,
          cutlass::gemm::TagToStrideC_t<LayoutC*>,
          ElementOutput,
          cutlass::gemm::TagToStrideC_t<LayoutD*>,
          FusionCallBacks,
          void, void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          GEMMDispatchPolicy,
          TileShape,
          cute::tuple<ElementInputA, ElementScale>,
          cute::tuple<cutlass::gemm::TagToStrideA_t<LayoutA*>, StrideScaleA*>,
          cute::tuple<ElementInputB, ElementScale>,
          cute::tuple<cutlass::gemm::TagToStrideB_t<LayoutB*>, StrideScaleB*>,
          TiledMma,
          cute::tuple<GmemTiledCopyA, GmemTiledCopyScaleA>, void, void, cute::identity,  // A
          cute::tuple<GmemTiledCopyB, GmemTiledCopyScaleB>, void, void, cute::identity   // B
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue,
    GroupScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  CUTLASS_CHECK(ExampleRunner<Gemm>{}.run(options, hw_info));

  return cutlass::Status::kSuccess;
}

int main(int argc, const char** argv) {
  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  using GroupSizeMNK = cute::tuple<cute::_1, cute::Int<128>, cute::Int<128>>;

  CUTLASS_CHECK((run_fp8_blockscaled_case
    <float_e4m3_t, float_e4m3_t, float,
     Shape<_256, _256, _32>, GroupSizeMNK>(options)));

  return 0;
}
