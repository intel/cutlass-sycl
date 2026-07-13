/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/platform/platform.h"
#include "moe_grouped_gemm/kernel/xe_moe_tile_scheduler.hpp"
#include "moe_grouped_gemm/collective/xe_moe_gemm.hpp"
#include <cute/util/compat.hpp>

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE {
using namespace cute;

using ProblemShapeMNKL = Shape<int, int, int, int>;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using TileScheduler = typename MoE::PersistentTileSchedulerXeMoE<ProblemShape>;
using RasterOrderOptions = typename TileScheduler::RasterOrderOptions;

template <typename T, char LayoutKind>
CUTE_DEVICE auto make_moe_tensor(T *ptr, int r, int c) {
  auto shape = make_shape(r, c);
  if constexpr (cute::is_subbyte_v<T>) {
    // Sub-byte types (e.g. float_e2m1_t, 4-bit) must be accessed via
    // const pointers to avoid subbyte_iterator issues with 2D block loads.
    auto const_ptr = const_cast<T const*>(ptr);
    if constexpr (LayoutKind == 'C')
      return make_tensor(make_gmem_ptr(const_ptr),
                         make_layout(shape, make_stride(_1{}, r)));
    else
      return make_tensor(make_gmem_ptr(const_ptr),
                         make_layout(shape, make_stride(c, _1{})));
  } else {
    if constexpr (LayoutKind == 'C')
      return make_tensor(make_gmem_ptr<T>(ptr),
                         make_layout(shape, make_stride(_1{}, r)));
    else
      return make_tensor(make_gmem_ptr<T>(ptr),
                         make_layout(shape, make_stride(c, _1{})));
  }
}

template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD,
          char LayoutKindA, char LayoutKindB, char LayoutKindD,
          int CfgGroupN = 0, int CfgGroupK = 0, class TiledMMA = void,
          typename ElementA = void, typename ElementB = void,
          typename ElementS = void, typename ElementD = void>
CUTE_DEVICE void
MoEGEMM(const ElementA *Activations, const ElementB *Weights,
        const ElementS *ScalesA, const ElementS *ScalesB, ElementD *Outputs,
        TiledMMA const &mma, const int32_t *M_per_group,
        const int32_t num_experts, const int32_t N, const int32_t K,
        PersistentTileSchedulerSm90GroupParams<ProblemShape> scheduler_params,
        const int32_t GroupN = 0, const int32_t GroupK = 0) {
  // Scale-storage row alignment for the MX 2D block load (FILE 2 layout).
  // Physical scale extents (rows) must be a multiple of this for the
  // hardware block-2D scale load; only used on the Block (CfgGroupK>0) path.
  constexpr int kScaleAlign = 64;

  TileScheduler scheduler{scheduler_params, const_cast<int32_t *>(M_per_group),
                          N, K, num_experts};

  auto work_tile_info = scheduler.initial_work_tile_info(Shape<_1, _1, _1>{});
  constexpr char actual_layout_of_B = LayoutKindB ^ ('R' ^ 'C');
  bool did_group_change = true;
  int32_t curr_group = 0;
  int32_t prev_group = 0;
  int32_t cumulative_M = 0;
  // Padded (kScaleAlign-aligned) per-expert M prefix sum, for the MX block
  // scale storage layout. Only meaningful on the Block (CfgGroupK>0) path.
  int32_t padded_cumulative_M = 0;
  int32_t M = 0;

  if (work_tile_info.is_valid()) {
    // We don't really need this conditional outside the while loop.
    // It simply helps initialize tensors. If using nullptr would be
    // fine for their initialization, then we can remove this conditional.
    curr_group = work_tile_info.L_idx;
    M = M_per_group[curr_group];
  }

  auto A_tensor = make_moe_tensor<ElementA, LayoutKindA>(
      const_cast<ElementA *>(Activations), M, K);
  auto B_tensor = make_moe_tensor<ElementB, actual_layout_of_B>(
      const_cast<ElementB *>(Weights), N, K);
  auto D_tensor = make_moe_tensor<ElementD, LayoutKindD>(Outputs, M, N);

  while (work_tile_info.is_valid()) {
    auto m_coord = work_tile_info.M_idx;
    auto n_coord = work_tile_info.N_idx;
    auto tile_coord = make_coord(m_coord, n_coord, _, 0);

    if (did_group_change) {
      curr_group = work_tile_info.L_idx;
      M = M_per_group[curr_group];
      // recompute each time because the groups don't necessarily increment by 1
      for (int i = prev_group; i < curr_group; i++) {
        cumulative_M += M_per_group[i];
        padded_cumulative_M += cute::round_up(M_per_group[i], kScaleAlign);
      }
      prev_group = curr_group;

      // For sub-byte types, pointer arithmetic must account for packing
      // (e.g. 4-bit: 2 elements per byte). Advance via byte pointer.
      auto byte_offset_A = int64_t(cumulative_M) * K *
                           cute::sizeof_bits_v<ElementA> / 8;
      auto byte_offset_B = int64_t(curr_group) * K * N *
                           cute::sizeof_bits_v<ElementB> / 8;
      ElementA *ptr_A_curr_batch = reinterpret_cast<ElementA *>(
          reinterpret_cast<uint8_t *>(const_cast<ElementA *>(Activations)) +
          byte_offset_A);
      ElementB *ptr_B_curr_batch = reinterpret_cast<ElementB *>(
          reinterpret_cast<uint8_t *>(const_cast<ElementB *>(Weights)) +
          byte_offset_B);
      ElementD *ptr_D_curr_batch = Outputs + int64_t(cumulative_M) * N;

      A_tensor = make_moe_tensor<ElementA, LayoutKindA>(ptr_A_curr_batch, M, K);
      B_tensor =
          make_moe_tensor<ElementB, actual_layout_of_B>(ptr_B_curr_batch, N, K);
      D_tensor = make_moe_tensor<ElementD, LayoutKindD>(ptr_D_curr_batch, M, N);
      did_group_change = false;
    }

    // Dispatch: block-scaled (mixed-precision) path when a scale type is
    // provided, otherwise the plain 16-bit path. The branch is resolved at
    // compile time so 16-bit kernels stay byte-identical.
    if constexpr (!cute::is_void_v<ElementS>) {
      const int scale_k = ceil_div(int(K), int(GroupK));
      const int scale_n = ceil_div(int(N), int(GroupN));
      if constexpr (CfgGroupK == 0) {
        // TENSOR scale: legacy MN-major contiguous layout. (Byte-identical.)
        auto sA = make_tensor(
            make_gmem_ptr(const_cast<ElementS *>(ScalesA) +
                          int64_t(cumulative_M) * scale_k),
            make_layout(make_shape(M, scale_k), make_stride(_1{}, M)));
        auto sB = make_tensor(
            make_gmem_ptr(const_cast<ElementS *>(ScalesB) +
                          int64_t(curr_group) * scale_n * scale_k),
            make_layout(make_shape(scale_n, scale_k),
                        make_stride(scale_k, _1{})));
        moe_gemm_scaled<CfgGroupN, CfgGroupK, GmemTiledCopyA, GmemTiledCopyB,
                        GmemTiledCopyD>(A_tensor, B_tensor, sA, sB, D_tensor,
                                        tile_coord, mma, GroupN, GroupK);
      } else {
        // BLOCK (MX) scale: padded, MN-major, aligned 3D (MN, scale_k, 1)
        // layout for the hardware 2D block-scale load.
        // Workaround for an llvm-spirv getEntry "Id is not in map" crash (DPC++
        // build 2026-06-15/-16): cute::round_up does a signed remainder, and on
        // the loop-carried (optimizer-frozen) per-expert M that makes LLVM
        // insert a `freeze i32` inside the persistent-scheduler phi cycle, which
        // the SPIR-V binary writer mishandles. kScaleAlign is a power of two, so
        // use a division-free bitwise round-up and clamp M to non-negative here
        // (outside the phi cycle) to keep it provably defined.
        const int M_def = int(M) < 0 ? 0 : int(M);
        static_assert((kScaleAlign & (kScaleAlign - 1)) == 0,
                      "kScaleAlign must be a power of two");
        const int round_up_M = (M_def + (kScaleAlign - 1)) & ~(kScaleAlign - 1);
        const int padded_scale_n = cute::round_up(scale_n, kScaleAlign);
        auto sA = make_tensor(
            make_gmem_ptr(const_cast<ElementS *>(ScalesA) +
                          int64_t(padded_cumulative_M) * scale_k),
            make_layout(make_shape(M, scale_k, 1),
                        make_stride(_1{}, round_up_M,
                                    int64_t(round_up_M) * scale_k)));
        auto sB = make_tensor(
            make_gmem_ptr(const_cast<ElementS *>(ScalesB) +
                          int64_t(curr_group) * padded_scale_n * scale_k),
            make_layout(make_shape(scale_n, scale_k, 1),
                        make_stride(_1{}, padded_scale_n,
                                    int64_t(padded_scale_n) * scale_k)));
        moe_gemm_scaled<CfgGroupN, CfgGroupK, GmemTiledCopyA, GmemTiledCopyB,
                        GmemTiledCopyD>(A_tensor, B_tensor, sA, sB, D_tensor,
                                        tile_coord, mma, GroupN, GroupK);
      }
    } else {
      moe_gemm<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(
          A_tensor, B_tensor, D_tensor, tile_coord, mma);
    }

    // Get next work tile
    work_tile_info = scheduler.fetch_next_work(work_tile_info);
    did_group_change = curr_group != work_tile_info.L_idx;
  } // end while loop
}

} // namespace MoE
