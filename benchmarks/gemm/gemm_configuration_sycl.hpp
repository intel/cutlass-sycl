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

namespace cutlass::gemm::device {

enum class Scheduler { Gemm, GemmSplitK, GemmStreamK };

template<
  class ArchTag,
  class ElementA, class LayoutA,
  class ElementB, class LayoutB,
  class ElementC, class LayoutC,
  class ElementAccumulator,
  class TileShape, Scheduler TileScheduler, class TiledMma = void,
  class GmemTiledCopyA = void, class GmemTiledCopyB = void,
  class EpilogueOp = epilogue::fusion::LinearCombination<float, float, float, float, FloatRoundStyle::round_to_nearest>>
struct GemmConfiguration {
  static_assert(sizeof(ElementA) == 0, "No valid GemmConfiguration configuration exists.");
};

template<
  class ArchTag,
  class ElementA, class LayoutA,
  class ElementB, class LayoutB,
  class ElementC, class LayoutC,
  class ElementScale, class StrideScale,
  class ElementAccumulator,
  class TileShape, Scheduler TileScheduler, class TiledMma = void,
  class GmemTiledCopyA = void, class GmemTiledCopyB = void,
  class GmemTiledCopyScaleA = void, class GmemTiledCopyScaleB = void,
  class EpilogueOp = epilogue::fusion::LinearCombination<float, float, float, float, FloatRoundStyle::round_to_nearest>>
struct BlockScalingGemmConfiguration {
  static_assert(sizeof(ElementA) == 0, "No valid BlockScalingGemmConfiguration configuration exists.");
};

/////////////////////////////////////////////////////////////////////////

template<class ElementA, class LayoutA,
  class ElementB, class LayoutB, typename LayoutC,
  class TileShape, Scheduler TileScheduler,
  class TiledMma, class GmemTiledCopyA, class GmemTiledCopyB,  class EpilogueOp>
struct GemmConfiguration<
      arch::IntelXe,
      ElementA, LayoutA,
      ElementB, LayoutB,
      float, LayoutC,
      float,
      TileShape, TileScheduler, TiledMma,
      GmemTiledCopyA, GmemTiledCopyB, EpilogueOp>
{
  static constexpr int PipelineStages = 2;
  // Match example 03_bmg_gemm_streamk: use KernelXeCooperative schedule + StreamKScheduler
  // tag for StreamK/SplitK decompositions; default KernelXe for vanilla GEMM.
  static constexpr bool UseStreamK =
      (TileScheduler == Scheduler::GemmStreamK) || (TileScheduler == Scheduler::GemmSplitK);
  using KernelScheduleType = std::conditional_t<UseStreamK,
      cutlass::gemm::KernelXeCooperative, cutlass::gemm::KernelXe>;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<PipelineStages, KernelScheduleType>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

  // Configurations in benchmarks.hpp can pass either a layout tag (e.g. RowMajor) or a Stride directly
  using StrideA = std::conditional_t<cute::is_tuple_v<LayoutA>, LayoutA, TagToStrideA_t<LayoutA>>;
  using StrideB = std::conditional_t<cute::is_tuple_v<LayoutB>, LayoutB, TagToStrideB_t<LayoutB>>;

  // Mainloop
  using CollectiveMainloop =
      collective::CollectiveMma<
        GEMMDispatchPolicy, TileShape,
        ElementA, StrideA,
        ElementB, StrideB,
        TiledMma,
        GmemTiledCopyA, void, void, identity, // A
        GmemTiledCopyB, void, void, identity // B
  >;

  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  using LayoutD = cutlass::layout::RowMajor;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          void,                 // Epilogue tile (void = automatic)
          float,// ElementAccumulator
          cutlass::gemm::TagToStrideC_t<LayoutC>, // Converts CUTLASS 2.x to CUTLASS 3.x representation
          float,// ElementOutput
          cutlass::gemm::TagToStrideC_t<LayoutD>, // Converts CUTLASS 2.x to CUTLASS 3.x representation
          FusionCallbacks,
          void,                 // The copy atom used to load matrix C  (void = automatic)
          void>;                // The copy atom used to store matrix D (void = automatic)
  using TileSchedulerTag = std::conditional_t<UseStreamK,
      cutlass::gemm::StreamKScheduler, void>;
    using GemmKernel = kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileSchedulerTag
  >;

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

template<class ElementA, class LayoutA,
  class ElementB, class LayoutB, typename LayoutC,
  class TileShape, Scheduler TileScheduler,
  class TiledMma, class GmemTiledCopyA, class GmemTiledCopyB,  class EpilogueOp>
struct GemmConfiguration<
      arch::IntelXe,
      ElementA, LayoutA,
      ElementB, LayoutB,
      bfloat16_t, LayoutC,
      bfloat16_t,
      TileShape, TileScheduler, TiledMma,
      GmemTiledCopyA, GmemTiledCopyB, EpilogueOp>
{
  static constexpr int PipelineStages = 2;
  static constexpr bool UseStreamK =
      (TileScheduler == Scheduler::GemmStreamK) || (TileScheduler == Scheduler::GemmSplitK);
  using KernelScheduleType = std::conditional_t<UseStreamK,
      cutlass::gemm::KernelXeCooperative, cutlass::gemm::KernelXe>;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<PipelineStages, KernelScheduleType>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

  using StrideA = std::conditional_t<cute::is_tuple_v<LayoutA>, LayoutA, TagToStrideA_t<LayoutA>>;
  using StrideB = std::conditional_t<cute::is_tuple_v<LayoutB>, LayoutB, TagToStrideB_t<LayoutB>>;

  using CollectiveMainloop =
      collective::CollectiveMma<
        GEMMDispatchPolicy, TileShape,
        ElementA, StrideA,
        ElementB, StrideB,
        TiledMma,
        GmemTiledCopyA, void, void, identity,
        GmemTiledCopyB, void, void, identity
  >;

  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  using LayoutD = cutlass::layout::RowMajor;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          void,
          bfloat16_t,
          cutlass::gemm::TagToStrideC_t<LayoutC>,
          bfloat16_t,
          cutlass::gemm::TagToStrideC_t<LayoutD>,
          FusionCallbacks,
          void,
          void>;
  using TileSchedulerTag = std::conditional_t<UseStreamK,
      cutlass::gemm::StreamKScheduler, void>;
  using GemmKernel = kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileSchedulerTag
  >;

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

// mxfp8/4
template<class ElementA, class LayoutA,
  class ElementB, class LayoutB, typename LayoutC,
  class ElementScale, typename StrideScale,
  class TileShape, Scheduler TileScheduler,
  class TiledMma, class GmemTiledCopyA, class GmemTiledCopyB,  
  class GmemTiledCopyScaleA, class GmemTiledCopyScaleB,
  class EpilogueOp>
struct BlockScalingGemmConfiguration<
      arch::IntelXe,
      ElementA, LayoutA,
      ElementB, LayoutB,
      float, LayoutC,
      ElementScale, StrideScale,
      float,
      TileShape, TileScheduler, TiledMma,
      GmemTiledCopyA, GmemTiledCopyB, 
      GmemTiledCopyScaleA, GmemTiledCopyScaleB,
      EpilogueOp>
{
  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16BlockScaled<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

  // Configurations in benchmarks.hpp can pass either a layout tag (e.g. RowMajor) or a Stride directly
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

  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  using LayoutD = cutlass::layout::RowMajor;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          void,
          float,                // ElementAccumulator
          cutlass::gemm::TagToStrideC_t<LayoutC>,
          float,                // ElementOutput
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

template<class ElementA, class LayoutA,
  class ElementB, class LayoutB, typename LayoutC,
  class ElementScale, typename StrideScale,
  class TileShape, Scheduler TileScheduler,
  class TiledMma, class GmemTiledCopyA, class GmemTiledCopyB,
  class GmemTiledCopyScaleA, class GmemTiledCopyScaleB,
  class EpilogueOp>
struct BlockScalingGemmConfiguration<
      arch::IntelXe,
      ElementA, LayoutA,
      ElementB, LayoutB,
      bfloat16_t, LayoutC,
      ElementScale, StrideScale,
      bfloat16_t,
      TileShape, TileScheduler, TiledMma,
      GmemTiledCopyA, GmemTiledCopyB,
      GmemTiledCopyScaleA, GmemTiledCopyScaleB,
      EpilogueOp>
{
  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16BlockScaled<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

  using StrideA = std::conditional_t<cute::is_tuple_v<LayoutA>, LayoutA, TagToStrideA_t<LayoutA>>;
  using StrideB = std::conditional_t<cute::is_tuple_v<LayoutB>, LayoutB, TagToStrideB_t<LayoutB>>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          GEMMDispatchPolicy,
          TileShape,
          cute::tuple<ElementA, ElementScale>,
          cute::tuple<StrideA, StrideScale>,
          cute::tuple<ElementB, ElementScale>,
          cute::tuple<StrideB, StrideScale>,
          TiledMma,
          cute::tuple<GmemTiledCopyA, GmemTiledCopyScaleA>, void, void, cute::identity,
          cute::tuple<GmemTiledCopyB, GmemTiledCopyScaleB>, void, void, cute::identity
  >;

  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  using LayoutD = cutlass::layout::RowMajor;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          void,
          bfloat16_t,
          cutlass::gemm::TagToStrideC_t<LayoutC>,
          bfloat16_t,
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
// W8A8 FP8 -> FP16-MMA fast path, mirrors examples/08_bmg_gemm_f8.
// FP8 inputs (float_e4m3_t / float_e5m2_t) are upcast to FP16 inside the
// W8A8 mainloop; MMA runs on XE_8x16x16_F32F16F16F32_TT (FP16 inputs,
// FP32 accumulator). This is a distinct pipeline from the native-FP8
// XE_DPAS_TT<8, float, float_e4m3_t> path used by GemmConfiguration.
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
  class EpilogueOp = epilogue::fusion::LinearCombination<float, float, float, float, FloatRoundStyle::round_to_nearest>>
struct W8A8GemmConfiguration {
  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelW8A8<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  using StrideA = std::conditional_t<cute::is_tuple_v<LayoutA>, LayoutA, TagToStrideA_t<LayoutA>>;
  using StrideB = std::conditional_t<cute::is_tuple_v<LayoutB>, LayoutB, TagToStrideB_t<LayoutB>>;

  using CollectiveMainloop = collective::CollectiveMma<
        GEMMDispatchPolicy, TileShape,
        ElementA, StrideA,
        ElementB, StrideB,
        TiledMma,
        GmemTiledCopyA, void, void, cute::identity,
        GmemTiledCopyB, void, void, cute::identity>;

  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
        EpilogueDispatchPolicy, EpilogueOp, TileShape,
        decltype(tile_shape(TiledMma()))>;

  using LayoutD = cutlass::layout::RowMajor;
  // Note: the ElementAccumulator template parameter is retained for signature
  // symmetry with the rest of the file; the actual accumulator type is
  // deduced inside the kernel from TiledMma::ValTypeC. The slot below is the
  // C-matrix element type, so we pass ElementC (matches example 08).
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
