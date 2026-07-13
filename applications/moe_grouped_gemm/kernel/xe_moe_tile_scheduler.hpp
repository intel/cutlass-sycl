/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation. All rights reserved.
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

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/gemm_coord.hpp"
#include "cutlass/kernel_hardware_info.hpp"

namespace MoE {
using namespace cutlass::gemm::kernel::detail;
using namespace cutlass;
using namespace cutlass::gemm;
using namespace cute;
///////////////////////////////////////////////////////////////////////////////
// Adapted from xe_tile_scheduler_group.hpp
// Persistent Thread Block (TB) scheduler for MoE GEMM
template <class GroupProblemShape>
class PersistentTileSchedulerXeMoE
    : public PersistentTileSchedulerXeGroup<GroupProblemShape> {
  //
  // Data members
  //

private:
  uint64_t current_work_linear_idx_ = 0;
  uint64_t total_grid_size_ = 0;
  int32_t *num_rows_per_expert_ = nullptr;
  int32_t K_ = 0;
  int32_t N_ = 0;
  int32_t num_experts_ = 0;

  // Tracking tiles - assumes uniform M across all experts
  uint64_t tiles_per_group_ = 0;
  uint64_t tiles_in_m_ = 0;
  uint64_t tiles_in_n_ = 0;
  uint64_t total_tiles_ = 0;

public:
  struct WorkTileInfo {
    int32_t M_idx = 0;
    int32_t N_idx = 0;
    int32_t L_idx = 0;
    bool is_valid_tile = false;

    CUTLASS_HOST_DEVICE
    bool is_valid() const { return is_valid_tile; }

    CUTLASS_HOST_DEVICE
    static WorkTileInfo invalid_work_tile() { return {-1, -1, -1, false}; }
  };

  using ProblemShape = typename GroupProblemShape::UnderlyingProblemShape;
  using Params = PersistentTileSchedulerSm90GroupParams<GroupProblemShape>;
  using RasterOrder = typename Params::RasterOrder;
  using RasterOrderOptions = typename Params::RasterOrderOptions;
  using BaseClass = PersistentTileSchedulerXeGroup<GroupProblemShape>;

  Params scheduler_params;

  //
  // Methods
  //

  // Given the inputs, computes the total number of output blocks this problem
  // will compute over Note that this is only the logical size of our grid, not
  // the physical grid we will actually launch.
  template <class ClusterShape>
  CUTLASS_HOST_DEVICE static dim3
  get_tiled_cta_shape_mnl(const KernelHardwareInfo &hw_info,
                          ClusterShape cluster_shape) {
    uint32_t total_ctas = 0;
    uint32_t cta_in_N_dim =
        1; // We linearize the blocks across all the problems here

    total_ctas = hw_info.sm_count;

    return Params::get_tiled_cta_shape_mnl(to_gemm_coord(cluster_shape),
                                           total_ctas, cta_in_N_dim);
  }

  template <class TileShape, class ClusterShape>
  static Params to_underlying_arguments(
      GroupProblemShape problem_shapes, TileShape tile_shape,
      ClusterShape cluster_shape, KernelHardwareInfo const &hw_info,
      typename BaseClass::Arguments const &arguments,
      [[maybe_unused]] void *workspace = nullptr,
      [[maybe_unused]] const uint32_t epilogue_subtile = 1,
      [[maybe_unused]] uint32_t ktile_start_alignment_count = 1u) {
    return BaseClass::to_underlying_arguments(problem_shapes, tile_shape,
                                              cluster_shape, hw_info, arguments,
                                              workspace);
  }

  // Given the inputs, computes the physical grid we should launch.
  template <class TileShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static dim3
  get_grid_shape([[maybe_unused]] Params const &params,
                 GroupProblemShape problem_shapes, TileShape tile_shape,
                 ClusterShape cluster_shape, KernelHardwareInfo hw_info,
                 typename BaseClass::Arguments arguments,
                 bool truncate_by_problem_size = true) {

    return BaseClass::get_grid_shape(params, problem_shapes, tile_shape,
                                     cluster_shape, hw_info, arguments,
                                     truncate_by_problem_size);
  }

  CUTLASS_DEVICE explicit PersistentTileSchedulerXeMoE(
      Params const &params_, int32_t *num_rows_per_expert, int32_t N, int32_t K,
      int32_t num_experts)
      : scheduler_params(params_) {
    num_rows_per_expert_ = num_rows_per_expert;
    N_ = N;
    K_ = K;
    num_experts_ = num_experts;
    if (scheduler_params.raster_order_ == RasterOrder::AlongN) {
      current_work_linear_idx_ =
          uint64_t(BlockIdxX()) + uint64_t(BlockIdxY()) * uint64_t(GridDimX());
    } else {
      current_work_linear_idx_ =
          uint64_t(BlockIdxX()) * uint64_t(GridDimY()) + uint64_t(BlockIdxY());
    }
    total_grid_size_ =
        uint64_t(GridDimX()) * uint64_t(GridDimY()) * uint64_t(GridDimZ());
      // Precompute tiles - assumes all experts have same num_rows (uniform MoE)
    int32_t tile_m = scheduler_params.cta_shape_.m();;
    int32_t tile_n = scheduler_params.cta_shape_.n();;
    uint64_t ctas_along_m = (num_rows_per_expert_[0] + tile_m - 1) / tile_m;
    uint64_t ctas_along_n = (N_ + tile_n - 1) / tile_n;

    // Use actual tile counts, not swizzle-rounded values
    // Swizzling is for hardware scheduling, not logical tile count
    tiles_in_m_ = ctas_along_m;
    tiles_in_n_ = ctas_along_n;
    tiles_per_group_ = tiles_in_m_ * tiles_in_n_;
    total_tiles_ = tiles_per_group_ * num_experts_;
    }

  CUTLASS_DEVICE
  WorkTileInfo
  get_work_idx_m_and_n(uint64_t linear_idx, RasterOrder raster_order) {
    if (linear_idx >= total_tiles_) {
      return WorkTileInfo::invalid_work_tile();
    }

    // Direct computation: which expert and tile within expert
    int32_t group_idx = static_cast<int32_t>(linear_idx / tiles_per_group_);
    uint64_t tile_idx_within_group = linear_idx % tiles_per_group_;

    // Map tile index to (m, n) coordinates based on raster order
    int32_t m_tile, n_tile;

    if (raster_order == RasterOrder::AlongN) {
      // AlongN: vary N fastest
      m_tile = static_cast<int32_t>(tile_idx_within_group / tiles_in_n_);
      n_tile = static_cast<int32_t>(tile_idx_within_group % tiles_in_n_);
    } else {
      // AlongM: vary M fastest
      n_tile = static_cast<int32_t>(tile_idx_within_group / tiles_in_m_);
      m_tile = static_cast<int32_t>(tile_idx_within_group % tiles_in_m_);
    }

    return {m_tile, n_tile, group_idx, true};
  }

  // Kernel helper function to get next work tile
  CUTLASS_DEVICE
  auto fetch_next_work(WorkTileInfo work_tile_info) {
    current_work_linear_idx_ += total_grid_size_;
    return get_work_idx_m_and_n(current_work_linear_idx_, scheduler_params.raster_order_);
  }

  // Returns the initial work tile info that will be computed over
  template <class ClusterShape>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(ClusterShape) {
    return get_work_idx_m_and_n(current_work_linear_idx_, scheduler_params.raster_order_);
  }
};

} // namespace MoE
