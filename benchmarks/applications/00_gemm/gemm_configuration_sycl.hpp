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
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/xe_persistent_tile_scheduler_params_streamk.hpp"

using namespace cute;

// Compact WG/SG -> TiledMMA helper for benchmark declarations.
//
// Derives, from the work-group (WG) and sub-group (SG) tile dimensions:
//   * the sub-group (atom) layout    = <WG_M/SG_M, WG_N/SG_N, 1>
//   * the DPAS row-repeat (atom "M")  = min(SG_M, 8)
// so a benchmark only needs to spell out its WG_MxWG_NxWG_K / SG_MxSG_NxSG_K tile
// once instead of restating the Shape<> / TiledMMAHelper<> boilerplate.
template <int WG_M, int WG_N, int WG_K, int SG_M, int SG_N,
          typename AccType, typename InType,
          int DpasM = (SG_M < 8 ? SG_M : 8)>
using XeTiledMMA = typename cute::TiledMMAHelper<
    cute::MMA_Atom<cute::XE_DPAS_TT<DpasM, AccType, InType>>,
    cute::Layout<cute::Shape<cute::Int<WG_M>, cute::Int<WG_N>, cute::Int<WG_K>>>,
    cute::Layout<cute::Shape<cute::Int<WG_M / SG_M>, cute::Int<WG_N / SG_N>, cute::_1>,
                 cute::Stride<cute::Int<WG_N / SG_N>, cute::_1, cute::_0>>>::TiledMMA;

// Block-scaled (MX) variant of XeTiledMMA. Uses the block-scaling DPAS atom
// (XE_BDPAS_TT). The atom "M" repeat is fixed at 8 (unlike the plain-DPAS helper
// above): the block-scaled kernels use the depth-8 systolic atom for every tile,
// so DpasM is independent of SG_M here.
template <int WG_M, int WG_N, int WG_K, int SG_M, int SG_N,
          typename AccType, typename InType,
          int DpasM = 8>
using XeBlockScalingTiledMMA = typename cute::TiledMMAHelper<
    cute::MMA_Atom<cute::XE_BDPAS_TT<DpasM, AccType, InType>>,
    cute::Layout<cute::Shape<cute::Int<WG_M>, cute::Int<WG_N>, cute::Int<WG_K>>>,
    cute::Layout<cute::Shape<cute::Int<WG_M / SG_M>, cute::Int<WG_N / SG_N>, cute::_1>,
                 cute::Stride<cute::Int<WG_N / SG_N>, cute::_1, cute::_0>>>::TiledMMA;

// W8A8 (FP8 -> FP16-MMA fast path) variant of XeTiledMMA. There is only one
// supported MMA atom for this path (XE_8x16x16_F32F16F16F32_TT: FP16 inputs,
// FP32 accumulate), so — unlike XeTiledMMA/XeBlockScalingTiledMMA — the atom
// is not selected via AccType/InType template parameters.
template <int WG_M, int WG_N, int WG_K, int SG_M, int SG_N>
using XeW8A8TiledMMA = typename cute::TiledMMAHelper<
    cute::MMA_Atom<cute::XE_8x16x16_F32F16F16F32_TT>,
    cute::Layout<cute::Shape<cute::Int<WG_M>, cute::Int<WG_N>, cute::Int<WG_K>>>,
    cute::Layout<cute::Shape<cute::Int<WG_M / SG_M>, cute::Int<WG_N / SG_N>, cute::_1>,
                 cute::Stride<cute::Int<WG_N / SG_N>, cute::_1, cute::_0>>>::TiledMMA;

namespace cutlass::gemm::device {

enum class Scheduler { Gemm, GemmSplitK, GemmStreamK };

// Primary template (unimplemented)
template<
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC,
    class ElementD,
    class TileShape, Scheduler TileScheduler,
    class TiledMma = void,
    class GmemTiledCopyA = void,
    class GmemTiledCopyB = void,
    class EpilogueOp = epilogue::fusion::LinearCombination<
        float, float, float, float, FloatRoundStyle::round_to_nearest>>
struct GemmConfiguration {
  static_assert(sizeof(ElementA) == 0, "No valid GemmConfiguration configuration exists.");
};

// Primary template (unimplemented)
template<
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC,
    class ElementScale, class StrideScale,
    class ElementD,
    class TileShape, Scheduler TileScheduler,
    class TiledMma = void,
    class GmemTiledCopyA = void,
    class GmemTiledCopyB = void,
    class GmemTiledCopyScaleA = void,
    class GmemTiledCopyScaleB = void,
    class GroupSize = _32,
    class EpilogueOp = epilogue::fusion::LinearCombination<
        float, float, float, float, FloatRoundStyle::round_to_nearest>>
struct BlockScalingGemmConfiguration {
  static_assert(sizeof(ElementA) == 0, "No valid BlockScalingGemmConfiguration configuration exists.");
};

/////////////////////////////////////////////////////////////////////////
// GemmConfiguration — IntelXe specialization
/////////////////////////////////////////////////////////////////////////

template<
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC,
    class ElementD,
    class TileShape, Scheduler TileScheduler,
    class TiledMma, class GmemTiledCopyA, class GmemTiledCopyB,
    class EpilogueOp>
struct GemmConfiguration<
    arch::IntelXe,
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementD,
    TileShape, TileScheduler, TiledMma,
    GmemTiledCopyA, GmemTiledCopyB, EpilogueOp>
{
  static constexpr int PipelineStages = 2;

  // Use KernelXeCooperative + StreamKScheduler for StreamK/SplitK; default KernelXe for vanilla GEMM.
  static constexpr bool UseStreamK =
      (TileScheduler == Scheduler::GemmStreamK) || (TileScheduler == Scheduler::GemmSplitK);

  using KernelScheduleType = std::conditional_t<
      UseStreamK, cutlass::gemm::KernelXeCooperative, cutlass::gemm::KernelXe>;
  using GEMMDispatchPolicy    = cutlass::gemm::MainloopXeL1Staged<PipelineStages, KernelScheduleType>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

  // Accept either a layout tag (e.g. RowMajor) or a Stride directly
  using StrideA = std::conditional_t<cute::is_tuple_v<LayoutA>, LayoutA, TagToStrideA_t<LayoutA>>;
  using StrideB = std::conditional_t<cute::is_tuple_v<LayoutB>, LayoutB, TagToStrideB_t<LayoutB>>;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
      GEMMDispatchPolicy, TileShape,
      ElementA, StrideA,
      ElementB, StrideB,
      TiledMma,
      GmemTiledCopyA, void, void, identity,  // A
      GmemTiledCopyB, void, void, identity   // B
  >;

  // Epilogue
  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

  using LayoutD = cutlass::layout::RowMajor;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      Shape<_8, Int<64 / sizeof(ElementD)>>, // Epilogue tile (void = automatic)
      ElementC,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ElementD,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallbacks,
      void,
      void>;

  // Tile scheduler
  using TileSchedulerTag = std::conditional_t<UseStreamK, cutlass::gemm::StreamKScheduler, void>;

  using GemmKernel = kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileSchedulerTag>;

  using Gemm = GemmUniversalAdapter<GemmKernel>;

  constexpr static typename GemmKernel::Arguments defaultArguments() {
    using StreamKMode =
        cutlass::gemm::kernel::detail::PersistentTileSchedulerXeStreamKParams::DecompositionMode;
    if constexpr (TileScheduler == Scheduler::Gemm) {
      return {};
    } else if constexpr (TileScheduler == Scheduler::GemmStreamK) {
      typename GemmKernel::Arguments arguments{};
      arguments.scheduler = {1, StreamKMode::StreamK};
      return arguments;
    } else {
      static_assert(TileScheduler == Scheduler::GemmSplitK);
      typename GemmKernel::Arguments arguments{};
      arguments.scheduler = {2, StreamKMode::SplitK};
      return arguments;
    }
  }
};

#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)

/////////////////////////////////////////////////////////////////////////
// BlockScalingGemmConfiguration — IntelXe specialization (mxfp8/4)
/////////////////////////////////////////////////////////////////////////

template<
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC,
    class ElementScale, class StrideScale,
    class ElementD,
    class TileShape, Scheduler TileScheduler,
    class TiledMma,
    class GmemTiledCopyA, class GmemTiledCopyB,
    class GmemTiledCopyScaleA, class GmemTiledCopyScaleB,
    class GroupSize,
    class EpilogueOp>
struct BlockScalingGemmConfiguration<
    arch::IntelXe,
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementScale, StrideScale,
    ElementD,
    TileShape, TileScheduler, TiledMma,
    GmemTiledCopyA, GmemTiledCopyB,
    GmemTiledCopyScaleA, GmemTiledCopyScaleB,
    GroupSize,
    EpilogueOp>
{
  static constexpr int PipelineStages = 2;

  using GEMMDispatchPolicy     = cutlass::gemm::MainloopIntelXeXMX16BlockScaled<PipelineStages, GroupSize>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

  // Accept either a layout tag (e.g. RowMajor) or a Stride directly
  using StrideA = std::conditional_t<cute::is_tuple_v<LayoutA>, LayoutA, TagToStrideA_t<LayoutA>>;
  using StrideB = std::conditional_t<cute::is_tuple_v<LayoutB>, LayoutB, TagToStrideB_t<LayoutB>>;

  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      cute::tuple<ElementA, ElementScale>,
      cute::tuple<StrideA, StrideScale>,
      cute::tuple<ElementB, ElementScale>,
      cute::tuple<StrideB, StrideScale>,
      TiledMma,
      cute::tuple<GmemTiledCopyA, GmemTiledCopyScaleA>, void, void, cute::identity,  // A
      cute::tuple<GmemTiledCopyB, GmemTiledCopyScaleB>, void, void, cute::identity   // B
  >;

  // Epilogue
  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

  using LayoutD = cutlass::layout::RowMajor;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      Shape<_8, Int<64 / sizeof(ElementD)>>, // Epilogue tile (void = automatic)
      ElementC,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ElementD,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallbacks,
      void,
      void>;

  using GemmKernel = kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = GemmUniversalAdapter<GemmKernel>;

  constexpr static typename GemmKernel::Arguments defaultArguments() {
    using StreamKMode =
        cutlass::gemm::kernel::detail::PersistentTileSchedulerXeStreamKParams::DecompositionMode;
    if constexpr (TileScheduler == Scheduler::Gemm) {
      return {};
    } else if constexpr (TileScheduler == Scheduler::GemmStreamK) {
      typename GemmKernel::Arguments arguments{};
      arguments.scheduler = {1, StreamKMode::StreamK};
      return arguments;
    } else {
      static_assert(TileScheduler == Scheduler::GemmSplitK);
      typename GemmKernel::Arguments arguments{};
      arguments.scheduler = {2, StreamKMode::SplitK};
      return arguments;
    }
  }
};

#endif

/////////////////////////////////////////////////////////////////////////
// W8A8GemmConfiguration — FP8 -> FP16-MMA fast path
// Mirrors examples/08_bmg_gemm_f8. FP8 inputs are upcast to FP16 inside
// the W8A8 mainloop; MMA runs XE_8x16x16_F32F16F16F32_TT.
/////////////////////////////////////////////////////////////////////////

template<
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC,
    class ElementAccumulator,
    class TileShape,
    class TiledMma,
    class GmemTiledCopyA = XE_2D_U8x32x32_LD_N,
    class GmemTiledCopyB = XE_2D_U8x32x32_LD_V,
    class EpilogueOp = epilogue::fusion::LinearCombination<
        float, float, float, float, FloatRoundStyle::round_to_nearest>>
struct W8A8GemmConfiguration {
  static constexpr int PipelineStages = 2;

  using GEMMDispatchPolicy     = cutlass::gemm::MainloopIntelW8A8<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  using StrideA = std::conditional_t<cute::is_tuple_v<LayoutA>, LayoutA, TagToStrideA_t<LayoutA>>;
  using StrideB = std::conditional_t<cute::is_tuple_v<LayoutB>, LayoutB, TagToStrideB_t<LayoutB>>;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
      GEMMDispatchPolicy, TileShape,
      ElementA, StrideA,
      ElementB, StrideB,
      TiledMma,
      GmemTiledCopyA, void, void, cute::identity,
      GmemTiledCopyB, void, void, cute::identity>;

  // Epilogue
  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

  using LayoutD = cutlass::layout::RowMajor;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      ElementC,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      float,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallbacks,
      XE_2D_U32x8x16_LD_N,
      void, void,
      XE_2D_U32x8x16_ST_N,
      void, void>;

  using GemmKernel = kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = GemmUniversalAdapter<GemmKernel>;

  constexpr static typename GemmKernel::Arguments defaultArguments() { return {}; }
};

} // namespace cutlass::gemm::device
