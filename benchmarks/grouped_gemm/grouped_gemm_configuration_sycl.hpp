/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
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

#pragma once

#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/layout.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"

using namespace cute;

namespace cutlass::gemm::device {

/////////////////////////////////////////////////////////////////////////
// Grouped GEMM Configuration Templates
/////////////////////////////////////////////////////////////////////////

template<
  class ArchTag,
  class ElementA, class LayoutA,
  class ElementB, class LayoutB,
  class ElementC, class LayoutC,
  class ElementAccumulator,
  class TileShape, class TiledMma = void,
  class GmemTiledCopyA = void, class GmemTiledCopyB = void,
  class EpilogueOp = epilogue::fusion::LinearCombination<float, float, float, float, FloatRoundStyle::round_to_nearest>>
struct GroupedGemmConfiguration {
  static_assert(sizeof(ElementA) == 0, "No valid GroupedGemmConfiguration configuration exists.");
};

template<
  class ArchTag,
  class ElementA, class LayoutA,
  class ElementB, class LayoutB,
  class ElementC, class LayoutC,
  class ElementScale, class StrideScale,
  class ElementAccumulator,
  class TileShape, class TiledMma = void,
  class GmemTiledCopyA = void, class GmemTiledCopyB = void,
  class GmemTiledCopyScaleA = void, class GmemTiledCopyScaleB = void,
  class EpilogueOp = epilogue::fusion::LinearCombination<float, float, float, float, FloatRoundStyle::round_to_nearest>>
struct BlockScalingGroupedGemmConfiguration {
  static_assert(sizeof(ElementA) == 0, "No valid BlockScalingGroupedGemmConfiguration configuration exists.");
};

/////////////////////////////////////////////////////////////////////////

// bfloat16 Grouped GEMM

template<class ElementA, class LayoutA,
  class ElementB, class LayoutB, typename LayoutC,
  class TileShape,
  class TiledMma, class GmemTiledCopyA, class GmemTiledCopyB, class EpilogueOp>
struct GroupedGemmConfiguration<
      arch::IntelXe,
      ElementA, LayoutA,
      ElementB, LayoutB,
      float, LayoutC,
      float,
      TileShape, TiledMma,
      GmemTiledCopyA, GmemTiledCopyB, EpilogueOp>
{
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group

  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1StagedGroup<PipelineStages>;
  // Match example 04_bmg_grouped_gemm: use IntelXeGenericGroup epilogue policy
  // (replaces legacy IntelXeXMX16Group). Copy atoms are picked automatically.
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGenericGroup;

  // Configurations in benchmarks.hpp can pass either a layout tag (e.g. RowMajor) or a Stride directly
  using StrideA = std::conditional_t<cute::is_tuple_v<LayoutA>, LayoutA, TagToStrideA_t<LayoutA>>;
  using StrideB = std::conditional_t<cute::is_tuple_v<LayoutB>, LayoutB, TagToStrideB_t<LayoutB>>;

  // Mainloop
  using CollectiveMainloop =
      collective::CollectiveMma<
        GEMMDispatchPolicy, TileShape,
        ElementA, TagToStrideA_t<LayoutA*>,
        ElementB, TagToStrideB_t<LayoutB*>,
        TiledMma,
        GmemTiledCopyA, void, void, identity, // A
        GmemTiledCopyB, void, void, identity  // B
  >;

  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  using LayoutD = cutlass::layout::RowMajor;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          void,                                       // Epilogue tile (void = automatic)
          float,                                      // ElementAccumulator
          cutlass::gemm::TagToStrideC_t<LayoutC*>,    // Pointer syntax for grouped
          float,                                      // ElementOutput
          cutlass::gemm::TagToStrideC_t<LayoutD*>,    // Pointer syntax for grouped
          FusionCallbacks,
          void,                                       // CopyOp G2R (void = automatic)
          void>;                                      // CopyOp R2G (void = automatic)

  using GemmKernel = kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::GroupScheduler  // Add GroupScheduler
  >;

  using Gemm = GemmUniversalAdapter<GemmKernel>;

  constexpr static typename GemmKernel::Arguments defaultArguments() {
    using RasterOrderOptions = typename cutlass::gemm::kernel::detail::
        PersistentTileSchedulerXeGroup<ProblemShape>::RasterOrderOptions;
    
    typename GemmKernel::Arguments arguments{};
    arguments.scheduler = {1, RasterOrderOptions::AlongN};
    return arguments;
  }
};

#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)

// mxfp8/4 Grouped GEMM
template<class ElementA, class LayoutA,
  class ElementB, class LayoutB, typename LayoutC,
  class ElementScale, typename StrideScale,
  class TileShape,
  class TiledMma, class GmemTiledCopyA, class GmemTiledCopyB,  
  class GmemTiledCopyScaleA, class GmemTiledCopyScaleB,
  class EpilogueOp>
struct BlockScalingGroupedGemmConfiguration<
      arch::IntelXe,
      ElementA, LayoutA,
      ElementB, LayoutB,
      float, LayoutC,
      ElementScale, StrideScale,
      float,
      TileShape, TiledMma,
      GmemTiledCopyA, GmemTiledCopyB, 
      GmemTiledCopyScaleA, GmemTiledCopyScaleB,
      EpilogueOp>
{
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group

  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16BlockScaledGroup<PipelineStages>;
  // Match example 51_xe35_block_scaled_grouped_gemm: use IntelXeGenericGroup epilogue
  // policy (replaces legacy IntelXeXMX16Group). Copy atoms are picked automatically.
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGenericGroup;

  // Configurations in benchmarks.hpp can pass either a layout tag (e.g. RowMajor) or a Stride directly
  using StrideA = std::conditional_t<cute::is_tuple_v<LayoutA>, LayoutA, TagToStrideA_t<LayoutA>>;
  using StrideB = std::conditional_t<cute::is_tuple_v<LayoutB>, LayoutB, TagToStrideB_t<LayoutB>>;

  // Mainloop - with pointer syntax for both matrix and scale strides
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          GEMMDispatchPolicy,
          TileShape,
          cute::tuple<ElementA, ElementScale>,
          cute::tuple<TagToStrideA_t<LayoutA*>, StrideScale*>,  // Pointer syntax for both
          cute::tuple<ElementB, ElementScale>,
          cute::tuple<TagToStrideB_t<LayoutB*>, StrideScale*>,  // Pointer syntax for both
          TiledMma,
          cute::tuple<GmemTiledCopyA, GmemTiledCopyScaleA>, void, void, cute::identity,  // A
          cute::tuple<GmemTiledCopyB, GmemTiledCopyScaleB>, void, void, cute::identity   // B
  >;

  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  using LayoutD = cutlass::layout::RowMajor;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          void,                                       // Epilogue tile (void = automatic)
          float,                                      // ElementAccumulator
          cutlass::gemm::TagToStrideC_t<LayoutC*>,    // Pointer syntax for grouped
          float,                                      // ElementOutput
          cutlass::gemm::TagToStrideC_t<LayoutD*>,    // Pointer syntax for grouped
          FusionCallbacks,
          void,                                       // CopyOp G2R (void = automatic)
          void>;                                      // CopyOp R2G (void = automatic)

  using GemmKernel = kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::GroupScheduler  // Add GroupScheduler
  >;

  using Gemm = GemmUniversalAdapter<GemmKernel>;

  constexpr static typename GemmKernel::Arguments defaultArguments() {
    using RasterOrderOptions = typename cutlass::gemm::kernel::detail::
        PersistentTileSchedulerXeGroup<ProblemShape>::RasterOrderOptions;
    
    typename GemmKernel::Arguments arguments{};
    arguments.scheduler = {1, RasterOrderOptions::AlongN};
    return arguments;
  }
};

#endif

} // namespace cutlass::gemm::device
