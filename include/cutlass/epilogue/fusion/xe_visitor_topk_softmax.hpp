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
#include "cutlass/array.h"
#include "cute/tensor.hpp"
#include "cute/util/xe_split_barrier.hpp"
// Reuse the hardware-agnostic Top-K + Softmax math helpers (cutlass::epilogue::fusion::detail)
// shared with the SM90 visitor: topk_logsumexp() and masked_softmax() have non-NVIDIA fallbacks.
#include "cutlass/epilogue/fusion/sm90_visitor_topk_softmax.hpp"

#include <sycl/sycl.hpp>

namespace cutlass::epilogue::fusion {

template <
  int TopK,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class CopyOpR2G,
  FloatRoundStyle RoundStyle
>
struct XeTopKSoftmaxColReduction
{
public:
  static_assert(TopK == 2 || TopK == 4,
    "Fused Top-K + Softmax reduction only allows K=2 and K=4.");

  static constexpr int Tile_M = get<0>(CtaTileShapeMNK{});
  static constexpr int Tile_N = get<1>(CtaTileShapeMNK{});
  static constexpr int Epi_M = get<0>(EpilogueTile{});
  static constexpr int Epi_N = get<1>(EpilogueTile{});
  static constexpr int Sg_M = Tile_M / Epi_M;
  static constexpr int Sg_N = Tile_N / Epi_N;
  static constexpr int Sg_Nums = Sg_M * Sg_N;
  static constexpr int SubgroupSize = IntelXeXMX16::SubgroupSize;

  using Trait_Output = Copy_Traits<CopyOpR2G>;
  using XE_Copy_output = decltype(make_tiled_copy(Copy_Atom<Trait_Output, ElementOutput>{}
                                             .with(static_cast<ElementOutput const*>(nullptr), int32_t(0), int32_t(0)),
                                             Layout<Shape<_1, Int<SubgroupSize>>>{},
                                             make_layout(make_shape(get<0>(typename Trait_Output::BlockShape{}),
                                                                    get<1>(typename Trait_Output::BlockShape{}) / Int<SubgroupSize>{}))));

  struct SharedStorage { };

  struct Arguments {
    ElementOutput* ptr_D = nullptr;
  };

  struct Params {
    XE_Copy_output xe_store_output;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void*) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    // The visitor owns the final D store. It builds an Xe 2D block store directly from
    // (ptr_D, M, N) with the row pitch defaulting to N, which assumes D is a packed
    // RowMajor tensor (leading dimension == N). Batched D is supported as long as it is
    // packed (batches stacked in M via a packed stride); strided / padded D is not
    // addressable through this path.
    XE_Copy_output output = make_tiled_copy(Copy_Atom<Copy_Traits<CopyOpR2G>, ElementOutput>{}.with(
                            args.ptr_D, M, N),
                            Layout<Shape<_1, Int<SubgroupSize>>>{},
                            make_layout(make_shape(get<0>(typename XE_Copy_output::BlockShape{}),
                                                   get<1>(typename XE_Copy_output::BlockShape{}) / Int<SubgroupSize>{})));
    return {output};
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const&) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    auto [tile_M, tile_N, tile_K] = CtaTileShapeMNK{};
    // N must fit in one CTA tile (the full row is reconstructed within a CTA) and have at
    // least TopK columns. D must be a packed RowMajor tensor (leading dimension == N);
    // batched D is supported as long as it is packed (batches stacked in M via a packed
    // stride) -- see to_underlying_arguments().
    return N <= tile_N && N >= TopK;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const&, Arguments const&, void*, cudaStream_t,
    CudaHostAdapter* = nullptr) { return Status::kSuccess; }

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  CUTLASS_HOST_DEVICE XeTopKSoftmaxColReduction() { }
  CUTLASS_HOST_DEVICE XeTopKSoftmaxColReduction(Params const& params, SharedStorage const&) : params(params) { }

  Params params;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const&) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class RTensor, class TopKTensor, class CoordTensor>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {

    CUTLASS_DEVICE
    ConsumerStoreCallbacks(RTensor&& res_tensor, TopKTensor&& topk_tensor,
                           CoordTensor&& coord, Params const& params, int valid_frags_n,
                           int active_sg_n, int sg_m_index, int sg_n_index)
      : res_tensor(cute::forward<RTensor>(res_tensor)),
        topk_tensor(cute::forward<TopKTensor>(topk_tensor)),
        coord(cute::forward<CoordTensor>(coord)),
        params(params),
        valid_frags_n(valid_frags_n),
        active_sg_n(active_sg_n),
        sg_m_index(sg_m_index),
        sg_n_index(sg_n_index) {}

    RTensor res_tensor;
    TopKTensor topk_tensor;
    CoordTensor coord;
    Params const& params;
    int valid_frags_n;
    int active_sg_n;
    int sg_m_index;
    int sg_n_index;

    template <typename ElementInput, typename ElementAccumulator, int FragSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragSize> const& frg_input) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragSize; ++i) {
        res_tensor(i, epi_m, epi_n) = static_cast<ElementCompute>(frg_input[i]);
      }

      if (epi_n >= valid_frags_n) {
        return frg_input;
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragSize; ++i) {
        ElementCompute val = static_cast<ElementCompute>(frg_input[i]);
        auto& row_topk = topk_tensor(i, epi_m);
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < TopK; ++k) {
          if (row_topk[k] < val) {
            CUTLASS_PRAGMA_UNROLL
            for (int l = TopK - 1; l > k; --l) {
              row_topk[l] = row_topk[l - 1];
            }
            row_topk[k] = val;
            break;
          }
        }
      }

      return frg_input;
    }

    template<class STensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void
    reduce(STensor&&, SyncFn const&, int epi_m, int epi_n, bool is_last_iteration, VTensor visit_results) {
      if (!is_last_iteration) {
        return;
      }

      auto sg = compat::get_nd_item<1>().get_sub_group();
      auto sg_local_id = sg.get_local_id()[0];

      constexpr int TotalFragsM = decltype(size<1>(res_tensor))::value;
      constexpr int TotalFragsN = decltype(size<2>(res_tensor))::value;
      // Fragment size is the per-thread count produced by the Xe MMA atom (size<0> of the
      // register tensors), not a fixed 8. Driving every fv loop from the tensor extent keeps
      // visit() (templated on the epilogue's actual FragSize) and reduce() consistent, so we
      // never touch uninitialized (-inf) lanes in the logsumexp path when the atom is e.g.
      // XE_4x16x16 (FragmentSize == 4).
      constexpr int FragmentSize = decltype(size<0>(res_tensor))::value;

      Tensor local_topk = make_tensor<ElementCompute>(Shape<Int<Epi_M>, Int<TopK>>{});

      CUTLASS_PRAGMA_UNROLL
      for (int fm = 0; fm < TotalFragsM; ++fm) {
        CUTLASS_PRAGMA_UNROLL
        for (int fv = 0; fv < FragmentSize; ++fv) {
          ElementCompute local_max = topk_tensor(fv, fm)[0];
          Array<ElementCompute, TopK> row_topk;
          row_topk.fill(-cutlass::platform::numeric_limits<ElementCompute>::infinity());
          row_topk[0] = reduce_over_group(sg, local_max, sycl::maximum<ElementCompute>());

          if constexpr (TopK >= 2) {
            int local_max_count = 0;
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < TopK; ++k) {
              local_max_count += (topk_tensor(fv, fm)[k] == row_topk[0]) ? 1 : 0;
            }
            int global_max_count = reduce_over_group(sg, local_max_count, sycl::plus<int>());
            if (global_max_count >= 2) {
              row_topk[1] = row_topk[0];
            }
            else {
              ElementCompute second_candidate = (local_max == row_topk[0])
                ? topk_tensor(fv, fm)[1]
                : local_max;
              row_topk[1] = reduce_over_group(sg, second_candidate, sycl::maximum<ElementCompute>());
            }
          }
          if constexpr (TopK >= 3) {
            ElementCompute third_candidate = (local_max >= row_topk[1])
              ? topk_tensor(fv, fm)[(local_max == row_topk[0]) ? 2 : 1]
              : local_max;
            row_topk[2] = reduce_over_group(sg, third_candidate, sycl::maximum<ElementCompute>());
          }
          if constexpr (TopK >= 4) {
            ElementCompute fourth_candidate = (local_max >= row_topk[2])
              ? topk_tensor(fv, fm)[(local_max == row_topk[0]) ? 3 : (local_max >= row_topk[1]) ? 2 : 1]
              : local_max;
            row_topk[3] = reduce_over_group(sg, fourth_candidate, sycl::maximum<ElementCompute>());
          }

          int row = fm * FragmentSize + fv;
          CUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < TopK; ++k) {
            local_topk(row, k) = row_topk[k];
          }
        }
      }

      auto smem = compat::local_mem<ElementCompute[Sg_Nums * Epi_M * TopK]>();
      Tensor stensor = make_tensor(make_smem_ptr(smem),
          make_shape(Int<Epi_M>{}, Int<TopK>{}, Int<Sg_N>{}, Int<Sg_M>{}));

      if (active_sg_n > 1) {
        if (sg_local_id == 0) {
          CUTLASS_PRAGMA_UNROLL
          for (int row = 0; row < Epi_M; ++row) {
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < TopK; ++k) {
              stensor(row, k, sg_n_index, sg_m_index) = local_topk(row, k);
            }
          }
        }

        cute::barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
        cute::barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
      }

      if (valid_frags_n == 0) {
        return;
      }

      CUTLASS_PRAGMA_UNROLL
      for (int fm = 0; fm < TotalFragsM; ++fm) {
        CUTLASS_PRAGMA_UNROLL
        for (int fv = 0; fv < FragmentSize; ++fv) {
          int row = fm * FragmentSize + fv;
          Array<ElementCompute, TopK> row_topk;
          row_topk.fill(-cutlass::platform::numeric_limits<ElementCompute>::infinity());
          int topk_sg_n = active_sg_n > 1 ? active_sg_n : 1;

          CUTLASS_PRAGMA_UNROLL
          for (int sn = 0; sn < Sg_N; ++sn) {
            if (sn >= topk_sg_n) {
              continue;
            }
            CUTLASS_PRAGMA_UNROLL
            for (int candidate_idx = 0; candidate_idx < TopK; ++candidate_idx) {
              ElementCompute candidate = active_sg_n > 1
                ? stensor(row, candidate_idx, sn, sg_m_index)
                : local_topk(row, candidate_idx);
              CUTLASS_PRAGMA_UNROLL
              for (int k = 0; k < TopK; ++k) {
                if (row_topk[k] <= candidate) {
                  CUTLASS_PRAGMA_UNROLL
                  for (int l = TopK - 1; l > k; --l) {
                    row_topk[l] = row_topk[l - 1];
                  }
                  row_topk[k] = candidate;
                  break;
                }
              }
            }
          }

          // Reuse the SM90 visitor's hardware-agnostic helpers. row_topk is descending-sorted,
          // so the smallest top-K value is the masking threshold and topk_logsumexp() folds the
          // set into a single logsumexp (m + log(1 + sum_{i>0} exp(x_i - m))).
          ElementCompute threshold = row_topk[TopK - 1];
          ElementCompute logsumexp = detail::topk_logsumexp(row_topk);

          CUTLASS_PRAGMA_UNROLL
          for (int fn = 0; fn < TotalFragsN; ++fn) {
            if (fn >= valid_frags_n) {
              res_tensor(fv, fm, fn) = ElementCompute(0.0);
              continue;
            }
            ElementCompute val = res_tensor(fv, fm, fn);
            res_tensor(fv, fm, fn) = detail::masked_softmax(val, threshold, logsumexp);
          }
        }
      }

      CUTLASS_PRAGMA_UNROLL
      for (int fn = 0; fn < TotalFragsN; ++fn) {
        if (fn >= valid_frags_n) {
          continue;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int fm = 0; fm < TotalFragsM; ++fm) {
          Tensor trD = make_tensor<ElementOutput>(Shape<Int<FragmentSize>>{});
          CUTLASS_PRAGMA_UNROLL
          for (int fv = 0; fv < FragmentSize; ++fv) {
            trD(fv) = static_cast<ElementOutput>(res_tensor(fv, fm, fn));
          }
          copy(params.xe_store_output, trD, coord(_, fm, fn));
        }
      }
    }

    CUTLASS_DEVICE void end_loop(int, int) { }
    CUTLASS_DEVICE void end() { }
  };

  template <
    bool ReferenceSrc,
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    using MmaAtomShape = typename decltype(args.tiled_mma)::AtomShape_MNK;
    static constexpr int FragmentSize = get<0>(MmaAtomShape());
    static constexpr int FragsM = get<0>(EpilogueTile{}) / get<0>(MmaAtomShape());
    static constexpr int FragsN = get<1>(EpilogueTile{}) / get<1>(MmaAtomShape());

    Tensor res = make_tensor<ElementCompute>(Shape<Int<FragmentSize>, Int<FragsM>, Int<FragsN>>{});
    Tensor topk = make_tensor<Array<ElementCompute, TopK>>(Shape<Int<FragmentSize>, Int<FragsM>>{});

    CUTLASS_PRAGMA_UNROLL
    for (int fm = 0; fm < FragsM; ++fm) {
      CUTLASS_PRAGMA_UNROLL
      for (int fv = 0; fv < FragmentSize; ++fv) {
        topk(fv, fm).fill(-cutlass::platform::numeric_limits<ElementCompute>::infinity());
      }
    }

    auto [sg_m_coord, sg_n_coord, k_coord, l_offset] = args.tile_coord_mnkl;
    auto [M, N, K, L] = args.problem_shape_mnkl;
    Tensor mD_mnl = cute::get_xe_tensor(make_shape(M, N, L));
    Tensor gD = local_tile(mD_mnl, select<0,1>(EpilogueTile{}), make_coord(sg_m_coord, sg_n_coord, l_offset));
    Tensor tCgD = params.xe_store_output.get_thread_slice(args.thread_idx).partition_D(gD);

    constexpr int MmaAtomN = get<1>(MmaAtomShape());
    int sg_start_col = static_cast<int>(sg_n_coord) * static_cast<int>(get<1>(EpilogueTile{}));
    int remaining_cols = static_cast<int>(N) - sg_start_col;
    int valid_frags_n = (remaining_cols > 0) ? (remaining_cols + MmaAtomN - 1) / MmaAtomN : 0;
    if (valid_frags_n > FragsN) {
      valid_frags_n = FragsN;
    }

    int active_sg_n = (static_cast<int>(N) + static_cast<int>(get<1>(EpilogueTile{})) - 1)
      / static_cast<int>(get<1>(EpilogueTile{}));
    if (active_sg_n > static_cast<int>(Sg_N)) {
      active_sg_n = static_cast<int>(Sg_N);
    }

    int sg_m_index = static_cast<int>(sg_m_coord) % static_cast<int>(Sg_M);
    int sg_n_index = static_cast<int>(sg_n_coord) % static_cast<int>(Sg_N);

    return ConsumerStoreCallbacks<decltype(res), decltype(topk), decltype(tCgD)>(
      cute::move(res),
      cute::move(topk),
      cute::move(tCgD),
      params,
      valid_frags_n,
      active_sg_n,
      sg_m_index,
      sg_n_index);
  }
};

} // namespace cutlass::epilogue::fusion
