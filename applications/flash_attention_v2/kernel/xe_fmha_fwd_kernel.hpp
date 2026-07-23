/***************************************************************************************************
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
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp"
#include "cute/util/type_traits.hpp"
#include "cute/util/xe_split_barrier.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"

namespace cutlass::fmha::kernel {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
template <bool IsVarLen_ = false>
struct FMHAProblemShape {
  using SeqLenType = cute::conditional_t<IsVarLen_, cutlass::fmha::collective::VariableLength, int>;
  int batch;
  int num_heads_q, num_heads_kv;
  SeqLenType seq_len_qo, seq_len_kv, seq_len_kv_cache;
  int head_size_qk, head_size_vo;
};

///////////////////////////////////////////////////////////////////////////////

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class XeFMHAFwdKernel {

public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  using VariableLength = cutlass::fmha::collective::VariableLength;
  static constexpr bool is_var_len = cutlass::fmha::collective::is_variable_length_v<typename ProblemShape::SeqLenType>;
  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using TiledMMAQK = typename CollectiveMainloop::TiledMMAQK;
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;
  using SubgroupLayoutQK = typename CollectiveMainloop::SubgroupLayoutQK;
  using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
  using ElementK = typename CollectiveMainloop::TensorK::element_type;
  using ElementV = typename CollectiveMainloop::TensorV::element_type;
  using ElementScale = typename CollectiveMainloop::TensorScaleQ::element_type;
  using StrideScaleQ = decltype(stride(typename CollectiveMainloop::TensorScaleQ{}));
  using StrideScaleK = decltype(stride(typename CollectiveMainloop::TensorScaleK{}));
  using StrideScaleV = decltype(stride(typename CollectiveMainloop::TensorScaleV{}));
  using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
  using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));
  static constexpr bool BlockScale = CollectiveMainloop::BlockScale;

  using SGPerWG = typename CollectiveMainloop::SGPerWG;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using FragSPartialRow = typename CollectiveMainloop::FragSPartialRow;

  // Tile scheduler derived types
  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));

  // Kernel level shared memory storage
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0)
                                                                     : sizeof(SharedStorage);

  // Device side arguments
  struct KernelArguments {
    ProblemShape shape;
    const ElementQ *Q;
    StrideQ dQ;
    const ElementK *K;
    StrideK dK;
    const ElementV *V;
    StrideV dV;
    ElementO *O;
    StrideO dO;
    const ElementScale *scaleQ = nullptr;
    StrideScaleQ dScaleQ{};
    const ElementScale *scaleK = nullptr;
    StrideScaleK dScaleK{};
    const ElementScale *scaleV = nullptr;
    StrideScaleV dScaleV{};
    float scale_k = 1.f;
    float scale_v = 1.f;
    float scale_q = 1.f;
    int group_size = 32;
    const ElementK *K_cache;
    StrideK dK_cache{};
    const ElementV *V_cache;
    StrideV dV_cache{};
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
  };

  // Kernel entry point API
  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  //
  // Methods
  //

  static Params to_underlying_arguments(Arguments const &args, void *workspace) {
    return {args.kernel,
            CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
            TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShapeO{})};
  }

  static bool can_implement(Arguments const &args) {
    return CollectiveMainloop::can_implement(args.mainloop)
        && CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const &args) { return 0; }

  static cutlass::Status initialize_workspace(Arguments const &args, void *workspace = nullptr,
                                              cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const &params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }

  static dim3 get_block_shape() { return dim3(SGPerWG::value * intel::sg_size, 1, 1); }

  CUTLASS_DEVICE
  Shape<int, int, int> get_sequence_length_shape(ProblemShape const& problem_shape, int const& batch) {
    if constexpr (is_var_len) {
      return cutlass::fmha::collective::apply_variable_length(Shape<VariableLength, VariableLength, VariableLength>{problem_shape.seq_len_qo, problem_shape.seq_len_kv, problem_shape.seq_len_kv_cache}, batch);
    } else {
      return Shape<int, int, int>{problem_shape.seq_len_qo, problem_shape.seq_len_kv, problem_shape.seq_len_kv_cache};
    }
  }

  CUTLASS_DEVICE
  void operator()(Params const &params, char *smem_buf)
  {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);

    auto &p = params.kernel;
    ProblemShape const& s = p.shape;
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());

    TileScheduler tile_scheduler{params.scheduler};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head_q, idx_b] = tile_scheduler.get_block_coord(); // (Q,V,h,b)
      auto blk_qv = make_coord(blk_q, blk_v);
      int head = tile_scheduler.divide_head_group(head_q);

      auto sequence_length_shape = get_sequence_length_shape(s, idx_b);
      auto [seq_len_qo, seq_len_kv, seq_len_kv_cache] = sequence_length_shape;
      if (blk_q * get<0>(TileShapeQK{}) >= seq_len_qo) continue;

      int discard_seq_coord = 0;
      int full_tile_offset = 0;
      int seq_len_new = seq_len_kv;
      if constexpr (CollectiveMainloop::CausalMask) {
        int q_sg_tile = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));
        auto cS = make_identity_tensor(take<0,2>(TiledMMAQK{}.tile_mnk()));
        auto tScS = TiledMMAQK{}.get_slice(thr_id).partition_C(cS);
        auto q_offset_wi = get<0>(tScS(0));
        auto q_offset_sg = group_broadcast(
            sycl::ext::oneapi::this_work_item::get_sub_group(), q_offset_wi, 0);

        int offset = cute::min(seq_len_qo, seq_len_kv);
        discard_seq_coord = seq_len_qo - offset;
        full_tile_offset = seq_len_kv - offset;
        int seq_coord = cute::min(seq_len_qo, (blk_q * get<0>(TileShapeQK{}) + q_offset_sg));
        if (seq_coord < discard_seq_coord) continue;
        seq_len_new = full_tile_offset + cute::min(seq_len_kv, seq_coord - discard_seq_coord) + q_sg_tile;
      }
      const int seq_len = seq_len_new + seq_len_kv_cache;
      // Compute k_blocks as sum of cache tiles + new tiles to avoid losing new data
      // when seq_len_kv_cache is not a multiple of the tile size.
      int k_blocks;
      if constexpr (CollectiveMainloop::CausalMask || CollectiveMainloop::CachedKV) {
        const int kblocks_cache = CollectiveMainloop::CachedKV ? cute::ceil_div(seq_len_kv_cache, get<1>(TileShapeQK{})) : 0;
        const int kblocks_new = cute::ceil_div(seq_len_new, get<1>(TileShapeQK{}));
        k_blocks = kblocks_cache + kblocks_new;
      } else {
        k_blocks = static_cast<int>(static_cast<unsigned>(seq_len) / static_cast<unsigned>(get<1>(TileShapeQK{})));
      }

      int offset_q = 0, offset_k = 0, offset_v = 0, offset_o = 0;
      int offset_k_cache = 0, offset_v_cache = 0;
      int batch_dim = s.batch;
      int l_coord = idx_b;
      if constexpr (is_var_len) {
        batch_dim = 1;
        l_coord = 0;
        auto qo_cumulative = s.seq_len_qo.cumulative_length;
        auto kv_cumulative = s.seq_len_kv.cumulative_length;
        offset_q = s.num_heads_q * s.head_size_qk * qo_cumulative[idx_b];
        offset_k = s.num_heads_kv * s.head_size_qk * kv_cumulative[idx_b];
        offset_v = s.num_heads_kv * s.head_size_vo * kv_cumulative[idx_b];
        offset_o = s.num_heads_q * s.head_size_vo * qo_cumulative[idx_b];
        if (s.seq_len_kv_cache.cumulative_length) {
          auto kv_cumulative_cache = s.seq_len_kv_cache.cumulative_length;
          offset_k_cache = s.num_heads_kv * s.head_size_qk * kv_cumulative_cache[idx_b];
          offset_v_cache = s.num_heads_kv * s.head_size_vo * kv_cumulative_cache[idx_b];
        }
      }

      auto shape_Q = make_shape(seq_len_qo, s.head_size_qk, s.num_heads_q, batch_dim);
      auto shape_K = make_shape(seq_len_kv, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V = make_shape(s.head_size_vo, seq_len_kv, s.num_heads_kv, batch_dim);
      auto shape_O = make_shape(seq_len_qo, s.head_size_vo, s.num_heads_q, batch_dim);

      auto shape_K_cache = make_shape(seq_len_kv_cache, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V_cache = make_shape(s.head_size_vo, seq_len_kv_cache, s.num_heads_kv, batch_dim);

      auto dcQ = const_cast<ElementQ*>(p.Q + offset_q);
      auto dcK = const_cast<ElementK*>(p.K + offset_k);
      auto dcV = const_cast<ElementV*>(p.V + offset_v);
      auto dcK_cache = const_cast<ElementK*>(p.K_cache + offset_k_cache);
      auto dcV_cache = const_cast<ElementV*>(p.V_cache + offset_v_cache);
      auto ptrO = p.O + offset_o;

      StrideQ stride_q = p.dQ;
      StrideK stride_k = p.dK;
      StrideV stride_v = p.dV;
      StrideO stride_o = p.dO;
      StrideK stride_k_cache = p.dK_cache;
      StrideV stride_v_cache = p.dV_cache;
      if constexpr (is_var_len) {
        stride_q = cutlass::make_cute_packed_stride(StrideQ{}, shape_Q);
        stride_k = cutlass::make_cute_packed_stride(StrideK{}, shape_K);
        stride_v = cutlass::make_cute_packed_stride(StrideV{}, shape_V);
        stride_o = cutlass::make_cute_packed_stride(StrideO{}, shape_O);
        stride_k_cache = cutlass::make_cute_packed_stride(StrideK{}, shape_K_cache);
        stride_v_cache = cutlass::make_cute_packed_stride(StrideV{}, shape_V_cache);
      }

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), make_layout(shape_Q, stride_q));
      Tensor K = make_tensor(make_gmem_ptr(dcK), make_layout(shape_K, stride_k));
      Tensor V = make_tensor(make_gmem_ptr(dcV), make_layout(shape_V, stride_v));
      Tensor K_cache = make_tensor(make_gmem_ptr(dcK_cache), make_layout(shape_K_cache, stride_k_cache));
      Tensor V_cache = make_tensor(make_gmem_ptr(dcV_cache), make_layout(shape_V_cache, stride_v_cache));
      Tensor O = make_tensor(make_gmem_ptr(ptrO), make_layout(shape_O, stride_o));

      // O accumulator types
      FragA tArA;
      FragARow tA_max;
      FragSPartialRow tA_sum;

      // Main loop
      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
      if constexpr (BlockScale) {
        auto scale_q = cute::ceil_div(s.head_size_qk, p.group_size);
        auto scale_k = cute::ceil_div(s.head_size_qk, p.group_size);
        int scale_v = cute::ceil_div(seq_len_kv, p.group_size);

        auto shape_scale_Q = make_shape(seq_len_qo, scale_q, s.num_heads_q, batch_dim);
        auto shape_scale_K = make_shape(seq_len_kv, scale_k, s.num_heads_kv, batch_dim);
        auto shape_scale_V = make_shape(s.head_size_vo, scale_v, s.num_heads_kv, batch_dim);
        int offset_scaleQ = 0; int offset_scaleK = 0; int offset_scaleV = 0;
        StrideScaleQ stride_scaleQ = p.dScaleQ;
        StrideScaleK stride_scaleK = p.dScaleK;
        StrideScaleV stride_scaleV = p.dScaleV;
        if constexpr (is_var_len) {
          auto qo_cumulative = s.seq_len_qo.cumulative_length;
          auto kv_cumulative = s.seq_len_kv.cumulative_length;
          auto kv_scale_cumulative = s.seq_len_kv.cumulative_scale_length;
          offset_scaleQ = s.num_heads_q * scale_q * qo_cumulative[idx_b];
          offset_scaleK = s.num_heads_kv * scale_k * kv_cumulative[idx_b];
          offset_scaleV = s.num_heads_kv * kv_scale_cumulative[idx_b];
          stride_scaleQ = cutlass::make_cute_packed_stride(StrideScaleQ{}, shape_scale_Q);
          stride_scaleK = cutlass::make_cute_packed_stride(StrideScaleK{}, shape_scale_K);
          stride_scaleV = cutlass::make_cute_packed_stride(StrideScaleV{}, shape_scale_V);
        }

        auto dcScaleQ = const_cast<ElementScale*>(p.scaleQ + offset_scaleQ);
        auto dcScaleK = const_cast<ElementScale*>(p.scaleK + offset_scaleK);
        auto dcScaleV = const_cast<ElementScale*>(p.scaleV + offset_scaleV);

        Tensor ScaleQ = make_tensor(make_gmem_ptr(dcScaleQ), make_layout(shape_scale_Q, stride_scaleQ));
        Tensor ScaleK = make_tensor(make_gmem_ptr(dcScaleK), make_layout(shape_scale_K, stride_scaleK));
        Tensor ScaleV = make_tensor(make_gmem_ptr(dcScaleV), make_layout(shape_scale_V, stride_scaleV));

        auto ScaleQ_head = ScaleQ(_, _, head_q, l_coord);
        auto ScaleK_head = ScaleK(_, _, head, l_coord);
        auto ScaleV_head = ScaleV(_, _, head, l_coord);

        mainloop(Q(_,_,head_q,l_coord),
                 K(_,_,head,l_coord),
                 V(_,_,head,l_coord),
                 tArA, tA_max, tA_sum,
                 blk_qv, 0, k_blocks, k_blocks,
                 thr_id, seq_len, 0, l_coord,
                 full_tile_offset, discard_seq_coord,
                 K_cache(_,_,head,l_coord),
                 V_cache(_,_,head,l_coord),
                 p.scale_k, p.scale_v, p.scale_q,
                 ScaleQ_head,
                 ScaleK_head,
                 ScaleV_head);
      } else {
        mainloop(Q(_,_,head_q,l_coord),
                 K(_,_,head,l_coord),
                 V(_,_,head,l_coord),
                 tArA, tA_max, tA_sum,
                 blk_qv, 0, k_blocks, k_blocks,
                 thr_id, seq_len, seq_len_kv_cache, idx_b,
                 full_tile_offset, discard_seq_coord,
                 K_cache(_,_,head,l_coord),
                 V_cache(_,_,head,l_coord),
                 p.scale_k, p.scale_v, p.scale_q);
      }
      if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      // Epilogue
      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      epilogue(O(_,_,head_q,l_coord),
               tArA, tA_max, tA_sum,
               blk_qv, thr_id, p.scale_v);
    }
  }
};

// Compute the maximum number of partitions (WGs) that can contribute to a single
// batch_head. This depends on sm_count and num_batch_heads at runtime.
// Free function (template-independent) so it can be unit-tested without
// instantiating the full kernel type.
inline int compute_max_num_partitions(int sm_count, int num_batch_heads) {
  return cute::ceil_div(sm_count, cute::max(1, num_batch_heads)) + 1;
}

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class XeFMHAFwdDynamicSplitKernel {

public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using TiledMMAQK = typename CollectiveMainloop::TiledMMAQK;
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;

  using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
  using ElementK = typename CollectiveMainloop::TensorK::element_type;
  using ElementV = typename CollectiveMainloop::TensorV::element_type;

  using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
  using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));

  using SGPerWG = typename CollectiveMainloop::SGPerWG;

  using FragA = typename CollectiveMainloop::FragA;
  using SingleFragA = typename CollectiveMainloop::SingleFragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using FragSPartialRow = typename CollectiveMainloop::FragSPartialRow;
  // element dtype for MmaPV results
  using ElementA = typename CollectiveMainloop::ElementA;

  // Tile scheduler derived types
  static_assert(is_same_v<TileScheduler_, XeFHMAIndividualPersistentTileScheduler>);
  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));

  // Kernel level shared memory storage
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr bool BlockScale = false;

  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0)
                                                                     : sizeof(SharedStorage);

  // Important: make sure multiple of 16 element for each copy
  // this is for storing partial results from different KV partitions
  static constexpr int num_elem_per_thread = (size(FragA{}.shape()) + 2 * size(FragARow{}.shape()) + 15) / 16 * 16;

  // Device side arguments
  struct KernelArguments {
    ProblemShape shape;
    const ElementQ *Q;
    StrideQ dQ;
    const ElementK *K;
    StrideK dK;
    const ElementV *V;
    StrideV dV;
    ElementO *O;
    StrideO dO;
    const ElementK *K_cache = nullptr;
    StrideK dK_cache{};
    const ElementV *V_cache = nullptr;
    StrideV dV_cache{};
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
  };

  // Kernel entry point API
  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
    // workspace for storing partial results of different KV partitions
    ElementA *partial_results_ptr = nullptr;
    // for atomic add
    int32_t *atomic_reduce_cnt_ptr = nullptr;
    // max partitions per batch_head (computed from sm_count)
    int max_num_partitions = 0;
  };

  //
  // Methods
  //

  static Params to_underlying_arguments(Arguments const &args, void *workspace) {
    int num_batch_heads = args.kernel.shape.batch * args.kernel.shape.num_heads_q;
    int max_parts = compute_max_num_partitions(args.hw_info.sm_count, num_batch_heads);
    int32_t *atomic_reduce_cnt_ptr = reinterpret_cast<int32_t *>(workspace);
    ElementA *partial_results_ptr = reinterpret_cast<ElementA *>(atomic_reduce_cnt_ptr + num_batch_heads);
    return {args.kernel,
            CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
            TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShapeO{}),
            partial_results_ptr, atomic_reduce_cnt_ptr, max_parts
          };
  }

  static bool can_implement(Arguments const &args) {
    // current kernel only support decode
    if (args.kernel.shape.seq_len_qo > 1) {
      return false;
    }
    // current kernel only support num batch heads less than total XeCore count
    if (args.kernel.shape.batch * args.kernel.shape.num_heads_q > args.hw_info.sm_count) {
      return false;
    }
    return CollectiveMainloop::can_implement(args.mainloop)
        && CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const &args) {
    int ws_size = 0;
    int num_batch_heads = args.kernel.shape.batch * args.kernel.shape.num_heads_q;
    int max_parts = compute_max_num_partitions(args.hw_info.sm_count, num_batch_heads);
    const int wg_size = SGPerWG::value * intel::sg_size;

    // partial attn outputs, exp sum and max logits
    ws_size += (max_parts * num_batch_heads) * wg_size * num_elem_per_thread * sizeof(ElementA);
    // atomic counter
    ws_size += num_batch_heads * sizeof(int32_t);
    return ws_size;
  }

  static cutlass::Status initialize_workspace(Arguments const &args, void *workspace = nullptr,
                                              cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr) {
    int num_batch_heads = args.kernel.shape.batch * args.kernel.shape.num_heads_q;
    compat::fill(reinterpret_cast<int32_t*>(workspace), (int32_t)0, num_batch_heads);
    auto partial_ws_count = (get_workspace_size(args) - num_batch_heads * sizeof(int32_t)) / sizeof(ElementA);
    auto* partial_results_ptr = reinterpret_cast<ElementA*>(reinterpret_cast<int32_t*>(workspace) + num_batch_heads);
    compat::fill(partial_results_ptr, (ElementA)0, partial_ws_count);
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const &params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }

  static dim3 get_block_shape() { return dim3(SGPerWG::value * intel::sg_size, 1, 1); }

  CUTLASS_DEVICE
  int get_partition_id(const int cur_wg_id, const int batch_head_id, const int num_blocks_per_wg, const int local_k_blocks) {
    int partition_id = 0;
    if (batch_head_id == 0) {
      return cur_wg_id;
    }
    int start_wg_id = batch_head_id * local_k_blocks / num_blocks_per_wg;
    partition_id = cur_wg_id - start_wg_id;
    return partition_id;
  }

  CUTLASS_DEVICE
  int get_num_partitions(const int batch_head_id, const int num_blocks_per_wg, const int local_k_blocks) {
    int num_partitions = 1;
    int start_wg_id = batch_head_id * local_k_blocks / num_blocks_per_wg;
    int end_wg_id = (batch_head_id + 1) * local_k_blocks / num_blocks_per_wg;
    num_partitions = end_wg_id - start_wg_id + 1;
    // end_wg_id is the starting wg id of next batch head id
    if (((batch_head_id + 1) * local_k_blocks) % num_blocks_per_wg == 0) {
      num_partitions -= 1;
    }
    return num_partitions;
  }

  template <class Params, class FragA, class FragARow>
  CUTLASS_DEVICE
  void reduce_split2(const Params &params, FragA &out1, FragARow& max_val1, FragARow& exp_sum_val1, FragA &out2, FragARow& max_val2, FragARow& exp_sum_val2) {
    // global max value
    FragARow max_prev1 = max_val1;
    FragARow max_prev2 = max_val2;

    auto scale = params.mainloop.scale;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < max_val1.size(); i++) {
      max_val1(i) = sycl::max(max_val1(i), max_val2(i));
    }

    FragARow rescale1, rescale2;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < max_val1.size(); i++) {
      rescale1(i) = sycl::native::exp2(max_prev1(i) - max_val1(i));
      rescale2(i) = sycl::native::exp2(max_prev2(i) - max_val1(i));
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < exp_sum_val1.size(); i++) {
      exp_sum_val1(i) = exp_sum_val1(i) * rescale1(i) + exp_sum_val2(i) * rescale2(i);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < out1.size(); i++)
      out1(i) = out1(i) * broadcast<0>(rescale1, out1, i) + out2(i) * broadcast<0>(rescale2, out2, i);
  }

  CUTLASS_DEVICE
  void operator()(Params const &params, char *smem_buf)
  {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);

    auto &p = params.kernel;
    ProblemShape const& s = p.shape;
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());
    int wg_id = int(BlockIdxZ());

    int sg_id = thr_id / intel::sg_size;
    int tid_in_sg = thr_id % intel::sg_size;
    int num_batch_heads = s.batch * s.num_heads_q;

    int local_k_blocks = cute::ceil_div(s.seq_len_kv, get<1>(TileShapeQK{}));
    // total number of blocks need to be processed across all wgs
    int total_k_blocks = local_k_blocks * num_batch_heads;
    // to guarantee all wg process similar number of blocks of KV
    int num_blocks_per_wg = cute::ceil_div(total_k_blocks, GridDimZ());

    TileScheduler tile_scheduler{params.scheduler, get<1>(TileShapeQK{}), local_k_blocks, num_batch_heads};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, start_batch_head_id] = tile_scheduler.get_block_coord(); // (Q,V, batch_head_idx)
      auto blk_qv = make_coord(blk_q, blk_v);

      auto shape_Q = make_shape(s.seq_len_qo, s.head_size_qk, s.num_heads_q,  s.batch);
      auto shape_K = make_shape(s.seq_len_kv, s.head_size_qk, s.num_heads_kv, s.batch);
      auto shape_V = make_shape(s.head_size_vo, s.seq_len_kv, s.num_heads_kv, s.batch);
      auto shape_O = make_shape(s.seq_len_qo, s.head_size_vo, s.num_heads_q, s.batch);

      auto dcQ = const_cast<ElementQ*>(p.Q);  // de-const these for uniformity
      auto dcK = const_cast<ElementK*>(p.K);
      auto dcV = const_cast<ElementV*>(p.V);

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), make_layout(shape_Q, p.dQ));    // (q,d,h,b)
      Tensor K = make_tensor(make_gmem_ptr(dcK), make_layout(shape_K, p.dK));    // (k,d,h,b)
      Tensor V = make_tensor(make_gmem_ptr(dcV), make_layout(shape_V, p.dV));    // (v,k,h,b)
      Tensor O = make_tensor(make_gmem_ptr(p.O), make_layout(shape_O, p.dO));    // (q,v,h,b)

      auto shape_K_cache = make_shape(s.seq_len_kv_cache, s.head_size_qk, s.num_heads_kv, s.batch);
      auto shape_V_cache = make_shape(s.head_size_vo, s.seq_len_kv_cache, s.num_heads_kv, s.batch);
      auto dcK_cache = const_cast<ElementK*>(p.K_cache);
      auto dcV_cache = const_cast<ElementV*>(p.V_cache);
      Tensor K_cache = make_tensor(make_gmem_ptr(dcK_cache), make_layout(shape_K_cache, p.dK_cache));
      Tensor V_cache = make_tensor(make_gmem_ptr(dcV_cache), make_layout(shape_V_cache, p.dV_cache));
      
      // O accumulator types
      FragA tArA;
      FragARow tA_max, tA_sum;
      FragSPartialRow tA_sum_partial;

      // compute num computed blocks for start batch head id
      int num_computed_blocks = (start_batch_head_id == 0) ? (wg_id * num_blocks_per_wg) : (wg_id * num_blocks_per_wg - start_batch_head_id * local_k_blocks);
      int start_blk, end_blk, head_q, idx_b, head_kv;
      // leader wg is also responsible for reducing partial results, while other
      // worker wg only to compute partial results
      bool is_leader_wg = wg_id < num_batch_heads;

      if (thr_id == 0 && is_leader_wg) {
        // reset atomic counter before computation
        *(params.atomic_reduce_cnt_ptr + wg_id) = 0;
      }

      // Main loop
      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);

      // compute blocks budget remained for each wg
      int block_budget_remained = num_blocks_per_wg;
      int batch_head_id = start_batch_head_id;
      bool is_update_batch_head_id = false;
      // Skip excess WGs whose start_batch_head_id is already out of range.
      // This happens when total_k_blocks < GridDimZ (more WGs than work).
      if (batch_head_id >= num_batch_heads) {
        block_budget_remained = 0;
      }
      while (block_budget_remained > 0) {
        int num_new_blocks = local_k_blocks - num_computed_blocks;
        if (num_new_blocks <= block_budget_remained) {
          // finished current batch head id
          start_blk = num_computed_blocks;
          end_blk = start_blk + num_new_blocks;

          // update states
          num_computed_blocks = 0;
          block_budget_remained -= num_new_blocks;
          is_update_batch_head_id = true;
        } else {
          // budget cannot afford finishing current batch head id
          start_blk = num_computed_blocks;
          end_blk = start_blk + block_budget_remained;

          block_budget_remained = 0;
          is_update_batch_head_id = false;
        }

        head_q = batch_head_id % s.num_heads_q;
        idx_b = batch_head_id / s.num_heads_q;
        head_kv = head_q / head_group_q;
        // mainloop
        mainloop(Q(_,_,head_q,idx_b),
              K(_,_,head_kv,idx_b),
              V(_,_,head_kv,idx_b),
              tArA, tA_max, tA_sum_partial,
              blk_qv, start_blk, end_blk, local_k_blocks,
              thr_id, s.seq_len_kv, 0, 0, 0, 0);

        tA_sum = reduce<0, cute::ReduceMode::Horizontal>(tA_sum_partial, sycl::plus<void>{});

        // partition id of start batch head id in current wg
        int partition_id = get_partition_id(wg_id, batch_head_id, num_blocks_per_wg, local_k_blocks);

        // store partial result: tArA, tA_max and tA_sum
        int offset = batch_head_id * params.max_num_partitions * num_elem_per_thread * SGPerWG::value * intel::sg_size
                    + partition_id * num_elem_per_thread * SGPerWG::value * intel::sg_size
                    + sg_id * intel::sg_size * num_elem_per_thread
                    + tid_in_sg * num_elem_per_thread;
        Tensor tPartial = make_tensor(params.partial_results_ptr + offset, make_shape(Int<num_elem_per_thread>{}));
        Tensor merged_res = make_tensor<ElementA>(Int<num_elem_per_thread>{});

        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < size(FragA{}.shape()); ++i) {
          merged_res(i) = tArA(i);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(FragARow{}.shape()); ++i) {
          merged_res(2 * i + size(FragA{}.shape())) = tA_max(i);
          merged_res(2 * i + 1 + size(FragA{}.shape())) = tA_sum(i);
        }
        copy(merged_res, tPartial);

        // Split barrier: arrive (flush stores) then wait (ensure all WG threads arrived)
        cute::barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsCrossWGMemory);
        cute::barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsCrossWGMemory);

        // after store, set atomic cnt
        if (thr_id == 0) {
          atomicAdd(params.atomic_reduce_cnt_ptr + batch_head_id, 1);
        }

        // advance to next batch head id
        if (is_update_batch_head_id) {
          batch_head_id += 1;
          if (batch_head_id >= num_batch_heads) {
            break;
          }
        }
      }

      if (is_leader_wg) {
        int num_partitions = get_num_partitions(wg_id, num_blocks_per_wg, local_k_blocks);

        // check atomic to wait for partial results ready
        while(atomicLoad(params.atomic_reduce_cnt_ptr + wg_id) != num_partitions) {}

        clear(tArA);
        clear(tA_max);
        clear(tA_sum);

        for (int i = 0; i < num_partitions; ++i) {
          int offset = wg_id * params.max_num_partitions * SGPerWG::value * intel::sg_size * num_elem_per_thread
                     + i * SGPerWG::value * intel::sg_size * num_elem_per_thread
                     + sg_id * intel::sg_size * num_elem_per_thread
                     + tid_in_sg * num_elem_per_thread;
          Tensor tPartial = make_tensor(params.partial_results_ptr + offset, make_shape(Int<num_elem_per_thread>{}));
          Tensor merged_res = make_tensor<ElementA>(Int<num_elem_per_thread>{});
          copy(tPartial, merged_res);

          if (i == 0) {
            CUTLASS_PRAGMA_UNROLL
            for(int i = 0; i < size(FragA{}.shape()); ++i) {
              tArA(i) = merged_res(i);
            }

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(FragARow{}.shape()); ++i) {
              tA_max(i) = merged_res(2 * i + size(FragA{}.shape()));
              tA_sum(i) = merged_res(2 * i + 1 + size(FragA{}.shape()));
            }

            continue;
          }

          FragA tArA_2;
          FragARow tA_max_2, tA_sum_2;
          CUTLASS_PRAGMA_UNROLL
          for(int i = 0; i < size(FragA{}.shape()); ++i) {
            tArA_2(i) = merged_res(i);
          }

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(FragARow{}.shape()); ++i) {
            tA_max_2(i) = merged_res(2 * i + size(FragA{}.shape()));
            tA_sum_2(i) = merged_res(2 * i + 1 + size(FragA{}.shape()));
          }

          reduce_split2(params, tArA, tA_max, tA_sum, tArA_2, tA_max_2, tA_sum_2);
        }

        // require group barrier if using SLM
        if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
          sycl::group_barrier(get_work_group<3>());
        }

        head_q = wg_id % s.num_heads_q;
        idx_b = wg_id / s.num_heads_q;
        head_kv = head_q / head_group_q;

        // Epilogue
        CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
        epilogue.template operator()<true>(O(_,_,head_q,idx_b),
                tArA, tA_max, tA_sum,
                blk_qv, thr_id);
      }
    }
  }
};

} // namespace cutlass::fmha::kernel
