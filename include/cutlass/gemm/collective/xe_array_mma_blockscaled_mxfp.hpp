/***************************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/fp8_to_fp16.h"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cutlass/gemm/collective/xe_mma_blockscaled_mxfp.hpp"
/////////////////////////////////////////////////////////////////////////////////////////////////
namespace cutlass::gemm::collective {

template <
  int Stages,
  int GroupSize,
  class Schedule,
  class TileShape_,
  class ElementPairA_,
  class StridePairA_,
  class ElementPairB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyPairA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyPairB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
  MainloopIntelXeXMX16BlockScaledGroupImpl<Stages, cute::Int<GroupSize>, Schedule>,
    TileShape_,
    ElementPairA_,
    StridePairA_,
    ElementPairB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyPairA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyPairB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_> : 
    public CollectiveMma<MainloopIntelXeXMX16BlockScaledImpl<Stages, cute::Int<GroupSize>, KernelXe>,
                              TileShape_,
                              ElementPairA_,
                              StridePairA_,
                              ElementPairB_,
                              StridePairB_,
                              TiledMma_,
                              GmemTiledCopyPairA_,
                              SmemLayoutAtomA_,
                              SmemCopyAtomA_,
                              TransformA_,
                              GmemTiledCopyPairB_,
                              SmemLayoutAtomB_,
                              SmemCopyAtomB_,
                              TransformB_>
{
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopIntelXeXMX16BlockScaledGroupImpl<Stages, cute::Int<GroupSize>, Schedule>;
  using Base = CollectiveMma<MainloopIntelXeXMX16BlockScaledImpl<Stages, cute::Int<GroupSize>, KernelXe>,
                    TileShape_,
                    ElementPairA_,
                    StridePairA_,
                    ElementPairB_,
                    StridePairB_,
                    TiledMma_,
                    GmemTiledCopyPairA_,
                    SmemLayoutAtomA_,
                    SmemCopyAtomA_,
                    TransformA_,
                    GmemTiledCopyPairB_,
                    SmemLayoutAtomB_,
                    SmemCopyAtomB_,
                    TransformB_>;

    using BaseArguments = typename Base::Arguments;

    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using StrideA  = remove_cvref_t<decltype(get<0>(StridePairA_{}))>;
    using StrideB  = remove_cvref_t<decltype(get<0>(StridePairB_{}))>;
    using InternalStrideA = cute::remove_pointer_t<StrideA>;
    using InternalStrideB = cute::remove_pointer_t<StrideB>;

    using ElementScaleA = typename Base::ElementScaleA;
    using ElementScaleB = typename Base::ElementScaleB;
    using InternalStrideScaleA = typename Base::StrideScaleA;
    using InternalStrideScaleB = typename Base::StrideScaleB;
    using ElementSF = typename Base::ElementSF;

    using StrideScaleA = remove_cvref_t<decltype(get<1>(StridePairA_{}))>;
    using StrideScaleB = remove_cvref_t<decltype(get<1>(StridePairB_{}))>;

    using TensorMKL = decltype(make_tensor(make_gmem_ptr(static_cast<ElementA const*>(nullptr)), make_shape(0,0,0), InternalStrideA{}));   //(m, k)
    using TensorNKL = decltype(make_tensor(make_gmem_ptr(static_cast<ElementB const*>(nullptr)), make_shape(0,0,0), InternalStrideB{}));   //(n, k)
    using TensorScaleA = decltype(make_tensor(make_gmem_ptr(static_cast<ElementScaleA const*>(nullptr)), make_shape(0,0,0), InternalStrideScaleA{}));   //(m, scale_k)
    using TensorScaleB = decltype(make_tensor(make_gmem_ptr(static_cast<ElementScaleB const*>(nullptr)), make_shape(0,0,0), InternalStrideScaleB{}));   //(n, scale_k)

    using MainloopTensors = cute::tuple<TensorMKL, TensorNKL, TensorScaleA, TensorScaleB>;
  
  // Host side kernel arguments
  struct Arguments {
    ElementA const** ptr_A;
    StrideA dA;
    ElementB const** ptr_B;
    StrideB dB;
    ElementScaleA const** ptr_SA = nullptr;
    StrideScaleA dSA{};
    ElementScaleB const** ptr_SB = nullptr;
    StrideScaleB dSB{};
  };

  using Params = Arguments;

  //
  // Methods
  //

  CollectiveMma() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const &problem_shape,
                          Arguments const &args, void *workspace) {
    (void)problem_shape;
    (void)workspace;

    return Params{
      args
    };
  }

  CUTLASS_DEVICE static constexpr BaseArguments
  to_base_arguments(Arguments const &args, int idx) {
    return BaseArguments{ args.ptr_A[idx], args.dA[idx],
                          args.ptr_B[idx], args.dB[idx],
                          args.ptr_SA[idx], args.dSA[idx],
                          args.ptr_SB[idx], args.dSB[idx]};
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape problem_shapes,
      Arguments const& args) {
    constexpr int copy_alignment_bits = 128;
    constexpr int batch_alignment_bits = 512;
    auto problem_shape_MNKL = append<4>(problem_shapes, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    bool implementable = true;

    if constexpr (cute::is_same_v<ElementA, cutlass::float_e2m1_t> ||
                  cute::is_same_v<ElementB, cutlass::float_e2m1_t>) {
      if (GroupSize != 32) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Intel Xe blockscaled MMA only supports GroupSize=32 for e2m1 inputs.\n");
        implementable = false;
      }
    }

    constexpr int min_aligned_elements_A = copy_alignment_bits / sizeof_bits<ElementA>::value;
    constexpr int min_aligned_elements_B = copy_alignment_bits / sizeof_bits<ElementB>::value;
    constexpr int min_batch_aligned_elements_A = batch_alignment_bits / sizeof_bits<ElementA>::value;
    constexpr int min_batch_aligned_elements_B = batch_alignment_bits / sizeof_bits<ElementB>::value;
    for (int i = 0; i < problem_shapes.groups(); i++) {
      auto problem_shape_MNKL = append<4>(problem_shapes.get_host_problem_shape(i), 1);
      auto [M,N,K,L] = problem_shape_MNKL;

      implementable &= cutlass::detail::check_alignment<min_aligned_elements_A>(cute::make_shape(M,K,L), InternalStrideA{});
      implementable &= cutlass::detail::check_alignment<min_aligned_elements_B>(cute::make_shape(N,K,L), InternalStrideB{});

      if (L > 1) {
        implementable &= get<2>(InternalStrideA{}) % min_batch_aligned_elements_A == 0;
        implementable &= get<2>(InternalStrideB{}) % min_batch_aligned_elements_B == 0;
      }
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for XE 2D copy.\n");
    }

    return implementable;
  }

};

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
