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
#include "cutlass/fp8_to_fp16.h"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_zip.hpp"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Collective MMA for FP8 block-scaled GEMM with fp32 scale factors (base single-GEMM version).
///
/// Uses XE_BDPAS_TT with software-level scale application via zip-tensor MMA unpack.
/// The 2-element zip tensor triggers the software scaling path in MMA_Traits<XE_BDPAS_TT>::mma_unpack.
/// Computation: fp8 DPAS -> fp32 -> apply scaleA * scaleB per element in mma_unpack -> accumulate.
///
/// GroupSizeMNK is a cute::tuple specifying per-dimension block sizes, e.g. tuple<_1, _128, _128>.
///   M block=1: per-row scaling for A -> scaleA shape = (M, K/GroupK)
///   N/K block sizes define B scaling -> scaleB shape = (ceil(N/GroupN), ceil(K/GroupK))
template <
  int Stages,
  class GroupSizeM_,
  class GroupSizeN_,
  class GroupSizeK_,
  class KernelSchedule,
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
  MainloopIntelXeXMX16BlockScaledImpl<Stages, cute::tuple<GroupSizeM_, GroupSizeN_, GroupSizeK_>, KernelSchedule>,
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
  using DispatchPolicy = MainloopIntelXeXMX16BlockScaledImpl<Stages, cute::tuple<GroupSizeM_, GroupSizeN_, GroupSizeK_>, KernelSchedule>;
  using WorkgroupTileShape = TileShape_;

  using GmemTiledCopyPairA = GmemTiledCopyPairA_;
  using GmemTiledCopyPairB = GmemTiledCopyPairB_;

  using TiledMma = TiledMma_;
  using ElementPairA = ElementPairA_;
  using ElementPairB = ElementPairB_;
  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementBMma = typename TiledMma::ValTypeB;
  using StridePairA = StridePairA_;
  using StridePairB = StridePairB_;

  using ElementMMA = typename TiledMma_::ValTypeA;

  // A and B matrices
  using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
  using StrideA  = cute::remove_pointer_t<remove_cvref_t<decltype(get<0>(StridePairA{}))>>;

  using ElementB = remove_cvref_t<decltype(get<0>(ElementPairB{}))>;
  using StrideB  = cute::remove_pointer_t<remove_cvref_t<decltype(get<0>(StridePairB{}))>>;

  // Scale factors
  using ElementScaleA = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;
  using ElementScaleB = remove_cvref_t<decltype(get<1>(ElementPairB{}))>;
  using StrideScaleA = cute::remove_pointer_t<remove_cvref_t<decltype(get<1>(StridePairA{}))>>;
  using StrideScaleB = cute::remove_pointer_t<remove_cvref_t<decltype(get<1>(StridePairB{}))>>;

  using ElementAccumulator = typename TiledMma::ValTypeC;

  using GmemTiledCopyA = typename std::tuple_element<0, GmemTiledCopyPairA>::type;
  using GmemTiledCopyB = typename std::tuple_element<0, GmemTiledCopyPairB>::type;
  using GmemTiledCopyScaleA = typename std::tuple_element<1, GmemTiledCopyPairA>::type;
  using GmemTiledCopyScaleB = typename std::tuple_element<1, GmemTiledCopyPairB>::type;

  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;
  using MmaType = typename TiledMma::ValTypeA;

  // GroupSize per dimension from the tuple
  using GroupSizeMNK = cute::tuple<GroupSizeM_, GroupSizeN_, GroupSizeK_>;
  static constexpr int GroupM = get<0>(GroupSizeMNK{});
  static constexpr int GroupN = get<1>(GroupSizeMNK{});
  static constexpr int GroupK = get<2>(GroupSizeMNK{});

  static_assert(GroupM == 1, "FP8 block-scaled MMA only supports GroupM=1 (per-row scaling for A).");
  static_assert(GroupK > 0 && GroupN > 0, "GroupN and GroupK must be positive.");

  static_assert(std::is_same_v<TransformA, cute::identity>, "Transformation for A is not currently supported");
  static_assert(std::is_same_v<TransformB, cute::identity>, "Transformation for B is not currently supported");

public:
  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  using MmaAtomShape = typename TiledMma::AtomShape_MNK;
  static constexpr int ATOM_M = get<0>(MmaAtomShape{});
  static constexpr int ATOM_N = get<1>(MmaAtomShape{});
  static constexpr int ATOM_K = get<2>(MmaAtomShape{});

  static constexpr int BLK_M = get<0>(WorkgroupTileShape{});
  static constexpr int BLK_N = get<1>(WorkgroupTileShape{});
  static constexpr int BLK_K = get<2>(WorkgroupTileShape{});

  static constexpr int SG_NUMS_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr int SG_NUMS_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr int SG_NUMS_K = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());

  static constexpr int MMA_M = get<0>(typename TiledMma::Shape_MNK{});
  static constexpr int MMA_N = get<1>(typename TiledMma::Shape_MNK{});
  static constexpr int MMA_K = get<2>(typename TiledMma::Shape_MNK{});

  static constexpr int SG_M = ceil_div(BLK_M, SG_NUMS_M);
  static constexpr int SG_N = ceil_div(BLK_N, SG_NUMS_N);
  static constexpr int SG_K = ceil_div(BLK_K, SG_NUMS_K);
  using SubgroupTileShape = Shape<C<SG_M>, C<SG_N>, C<SG_K>>;

  // Compile-time path selection for scale loading strategy
  // Scale-B is indexed by N scale blocks (extent ceil(N / GroupN)).
  // Per-lane loading is valid only for true per-element N scaling.
  static constexpr bool kPerLaneScaleB = (GroupN == 1);
  static_assert(kPerLaneScaleB || SG_N <= GroupN,
                "Broadcast scale-B path requires each subgroup N tile to remain within a single "
                "GroupN block; use GroupN == 1 for per-lane scale-B loading or ensure SG_N <= GroupN.");
  // Use fine-grain K-scale loading whenever GroupK boundaries can fall within a BLK_K tile.
  static constexpr bool kFineGrainScaleK = (GroupK < BLK_K) || ((GroupK % BLK_K) != 0);

  // Deferred-scale path: applies to the broadcast-scale-B path (GroupN != 1;
  // the static_assert above guarantees each SG_N tile stays within one GroupN block)
  // and to aligned K-scale groups (GroupK >= BLK_K and GroupK is a multiple of BLK_K).
  // In this regime scaleA/scaleB stay constant over each BLK_K tile and GroupK boundaries
  // align with tile boundaries, so DPAS can accumulate raw results and apply the combined
  // scale once when draining each GroupK block.
  static constexpr bool kUseDeferredScale = (!kPerLaneScaleB) && (!kFineGrainScaleK);

  // Accumulator iterations
  static constexpr int M_ITERS = SG_M / ATOM_M;
  static constexpr int N_ITERS = SG_N / ATOM_N;
  static constexpr int ScaleAChunks = cute::ceil_div(SG_M, SubgroupSize);

  static constexpr auto Num_SGs = SG_NUMS_M * SG_NUMS_N * SG_NUMS_K;
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});

  using CopyThreadShape = Shape<_1, Int<SubgroupSize>>;

  template<class Element, class Stride>
  using TensorType = decltype(make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                                        make_layout(make_shape(int{}, int{}, int{}), Stride{})));

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    ElementScaleA const* ptr_SA = nullptr;
    StrideScaleA dSA{};
    ElementScaleB const* ptr_SB = nullptr;
    StrideScaleB dSB{};
  };

  struct Params {
    TensorType<ElementA, StrideA> mA_mkl;
    TensorType<ElementB, StrideB> mB_nkl;
    TensorType<ElementScaleA, StrideScaleA> mAscale;
    TensorType<ElementScaleB, StrideScaleB> mBscale;
  };

  CollectiveMma() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const &problem_shape,
                          Arguments const &args, void *workspace) {
    (void)workspace;

    auto [M, N, K, L] = problem_shape;

    auto mA_mkl =
        make_tensor(make_gmem_ptr(args.ptr_A), make_layout(make_shape(M, K, L), args.dA));
    auto mB_nkl =
        make_tensor(make_gmem_ptr(static_cast<ElementB const *>(args.ptr_B)),
                    make_layout(make_shape(N, K, L), args.dB));

    // Scale A: (M, ceil(K/GroupK), L) — per-row, per-K-block
    auto scale_k = cute::ceil_div(K, GroupK);
    auto mScaleA = make_tensor(make_gmem_ptr(static_cast<ElementScaleA const *>(args.ptr_SA)),
                               make_layout(make_shape(M, scale_k, L), args.dSA));
    // Scale B: (ceil(N/GroupN), ceil(K/GroupK), L) — per-N-block, per-K-block
    auto scale_n = cute::ceil_div(N, GroupN);
    auto mScaleB = make_tensor(make_gmem_ptr(static_cast<ElementScaleB const *>(args.ptr_SB)),
                               make_layout(make_shape(scale_n, scale_k, L), args.dSB));

    return Params{mA_mkl, mB_nkl, mScaleA, mScaleB};
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

    constexpr int min_aligned_elements_A = copy_alignment_bits / sizeof_bits<ElementA>::value;
    implementable &= cutlass::detail::check_alignment<min_aligned_elements_A>(cute::make_shape(M,K,L), args.dA);
    constexpr int min_aligned_elements_B = copy_alignment_bits / sizeof_bits<ElementB>::value;
    implementable &= cutlass::detail::check_alignment<min_aligned_elements_B>(cute::make_shape(N,K,L), args.dB);

    if (L > 1) {
      constexpr int min_batch_aligned_elements_A = batch_alignment_bits / sizeof_bits<ElementA>::value;
      implementable &= get<2>(args.dA) % min_batch_aligned_elements_A == 0;
      constexpr int min_batch_aligned_elements_B = batch_alignment_bits / sizeof_bits<ElementB>::value;
      implementable &= get<2>(args.dB) % min_batch_aligned_elements_B == 0;
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for XE 2D copy.\n");
    }

    return implementable;
  }

  /// Load a single scale-A value with bounds check.
  CUTLASS_DEVICE static ElementScaleA
  load_scale_a_value(Params const &mainloop, int m_abs, int k_scale_idx, int l_coord, int M_extent) {
    return (m_abs < M_extent) ? mainloop.mAscale(m_abs, k_scale_idx, l_coord) : ElementScaleA(0);
  }

  /// Load a single scale-B value with bounds check.
  CUTLASS_DEVICE static ElementScaleB
  load_scale_b_value(Params const &mainloop, int n_abs, int k_scale_idx, int l_coord, int N_extent) {
    return (n_abs < N_extent) ? mainloop.mBscale(n_abs, k_scale_idx, l_coord) : ElementScaleB(0);
  }

  /// Fill tScaleA(v, mi, ki) for a single ki with per-element scale values.
  template <class ScaleATensor>
  CUTLASS_DEVICE static void
  fill_scale_a_ki(ScaleATensor &tScaleA, int ki,
                  Params const &mainloop, int m_coord, int k_scale_idx, int l_coord, int M_extent) {
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < M_ITERS; mi++) {
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < ATOM_M; v++) {
        tScaleA(v, mi, ki) = load_scale_a_value(
            mainloop, m_coord + mi * ATOM_M + v, k_scale_idx, l_coord, M_extent);
      }
    }
  }

  /// Fill tScaleA for all ki with the same per-element value (broadcast across K).
  template <int KIters, class ScaleATensor>
  CUTLASS_DEVICE static void
  fill_scale_a_all_ki(ScaleATensor &tScaleA,
                      Params const &mainloop, int m_coord, int k_scale_idx, int l_coord, int M_extent) {
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < M_ITERS; mi++) {
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < ATOM_M; v++) {
        const ElementScaleA sa = load_scale_a_value(
            mainloop, m_coord + mi * ATOM_M + v, k_scale_idx, l_coord, M_extent);
        CUTLASS_PRAGMA_UNROLL
        for (int ki = 0; ki < KIters; ++ki) {
          tScaleA(v, mi, ki) = sa;
        }
      }
    }
  }

  /// Fill tScaleB per-lane for a single ki (GroupN=1 path).
  template <int ScaleBV, class ScaleBTensor>
  CUTLASS_DEVICE static void
  fill_scale_b_per_lane_ki(ScaleBTensor &tScaleB, int ki,
                           Params const &mainloop, int n_coord, int k_scale_idx, int l_coord,
                           int N_scale_extent, int lane_id) {
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < N_ITERS; ni++) {
      const auto sb = load_scale_b_value(
          mainloop, n_coord + ni * ATOM_N + lane_id, k_scale_idx, l_coord, N_scale_extent);
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < ScaleBV; v++) {
        tScaleB(v, ni, ki) = sb;
      }
    }
  }

  /// Fill tScaleB with broadcast value for a single ki.
  template <int ScaleBV, class ScaleBTensor>
  CUTLASS_DEVICE static void
  fill_scale_b_broadcast_ki(ScaleBTensor &tScaleB, int ki,
                            Params const &mainloop, int n_scale_coord, int k_scale_idx, int l_coord) {
    const ElementScaleB sb = static_cast<ElementScaleB>(
        mainloop.mBscale(n_scale_coord, k_scale_idx, l_coord));
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < N_ITERS; ni++) {
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < ScaleBV; v++) {
        tScaleB(v, ni, ki) = sb;
      }
    }
  }

  /// Fill tScaleB with broadcast value for all ki.
  template <int ScaleBV, int KIters, class ScaleBTensor>
  CUTLASS_DEVICE static void
  fill_scale_b_broadcast_all_ki(ScaleBTensor &tScaleB,
                                Params const &mainloop, int n_scale_coord, int k_scale_idx, int l_coord) {
    const ElementScaleB sb = static_cast<ElementScaleB>(
        mainloop.mBscale(n_scale_coord, k_scale_idx, l_coord));
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < N_ITERS; ni++) {
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < ScaleBV; v++) {
        CUTLASS_PRAGMA_UNROLL
        for (int ki = 0; ki < KIters; ++ki) {
          tScaleB(v, ni, ki) = sb;
        }
      }
    }
  }

  /// Load deferred scale-A into compact per-chunk fragment (for later broadcast).
  template <class ScaleAFragment>
  CUTLASS_DEVICE static void
  copy_deferred_scale_a(ScaleAFragment &fragment_scaleA,
                        Params const &mainloop,
                        int m_coord,
                        int k_scale_idx,
                        int l_coord,
                        int lane_id,
                        int M_extent) {
    CUTLASS_PRAGMA_UNROLL
    for (int load_idx = 0; load_idx < ScaleAChunks; load_idx++) {
      fragment_scaleA(0, load_idx, 0) = load_scale_a_value(
          mainloop, m_coord + load_idx * SubgroupSize + lane_id, k_scale_idx, l_coord, M_extent);
    }
  }

  /// Drain raw_accum by applying combined scaleA * scaleB at a GroupK boundary.
  template <class FrgTensorD, class RawAccum, class ScaleAFragment, class Subgroup>
  CUTLASS_DEVICE static void
  drain_deferred_scale_accum(FrgTensorD &accum,
                             RawAccum &raw_accum,
                             ScaleAFragment &fragment_scaleA,
                             Subgroup sg_handle,
                             Params const &mainloop,
                             int m_coord,
                             int n_scale_coord,
                             int k_scale_idx,
                             int l_coord,
                             int lane_id,
                             int M_extent) {
    const ElementAccumulator scale_b_val = static_cast<ElementAccumulator>(
        mainloop.mBscale(n_scale_coord, k_scale_idx, l_coord));

    copy_deferred_scale_a(
        fragment_scaleA, mainloop, m_coord, k_scale_idx, l_coord, lane_id, M_extent);

    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < M_ITERS; mi++) {
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < ATOM_M; v++) {
        const int m_local = mi * ATOM_M + v;
        const int load_idx = m_local / SubgroupSize;
        const int lane_idx = m_local % SubgroupSize;
        const ElementScaleA sa = group_broadcast(
            sg_handle, fragment_scaleA(0, load_idx, 0), lane_idx);
        const ElementAccumulator combined_scale = static_cast<ElementAccumulator>(sa) * scale_b_val;
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < N_ITERS; ni++) {
          accum(v, mi, ni) += raw_accum(v, mi, ni) * combined_scale;
          raw_accum(v, mi, ni) = ElementAccumulator(0);
        }
      }
    }
  }

  template <class FrgTensorD,
    class TensorA,
    class TensorB,
    class FrgTensorC,
    class KTileIterator,
    class BlkCoord
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      TensorA gA,
      TensorB gB,
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      BlkCoord const &blk_coord,
      int const &K_start,
      int thread_idx,
      Params const& mainloop) 
  {
    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    (void)src_accum;
    auto batch_idx = get<3>(blk_coord);
    auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(TiledMma{}, mainloop.mA_mkl(_,_,batch_idx));
    auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(TiledMma{}, mainloop.mB_nkl(_,_,batch_idx));

    auto thr_copy_a = copy_a.get_slice(thread_idx);
    auto thr_copy_b = copy_b.get_slice(thread_idx);

    // Instantiate the MMA object and get thread slice
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);

    /* Register fragments for MMA */
    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));
    /* Register fragments for copies */
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));

    /* Partition global tensor (proxies) for copies */
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    
    /* Create prefetch TiledCopy instances */
    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);
      
    auto thr_prefetch_A = prefetch_a.get_slice(thread_idx);
    auto thr_prefetch_B = prefetch_b.get_slice(thread_idx);

    /* Partition global tensor (proxies) for prefetch */
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    // Initialize the output accumulator
    clear(accum);

    // Compute subgroup coordinates
    auto [m_idx, n_idx, k_idx, l_idx] = blk_coord;
    const int m_coord = m_idx * BLK_M + (get_sub_group_id() / SG_NUMS_N) * SG_M;
    const int n_coord = n_idx * BLK_N + (get_sub_group_id() % SG_NUMS_N) * SG_N;
    const int l_coord = l_idx;

    // Scale tensor extents for bounds checking in partial tiles
    const int M_extent = get<0>(mainloop.mAscale.shape());
    const int N_scale_extent = get<0>(mainloop.mBscale.shape());

    // Scale B index for broadcast path (GroupN >= ATOM_N)
    [[maybe_unused]] const int n_scale_coord = [&]() {
      if constexpr (kPerLaneScaleB) { return 0; }
      else { return cute::min(n_coord / GroupN, N_scale_extent - 1); }
    }();
    [[maybe_unused]] const int lane_id = thread_idx % SubgroupSize;

    const int k_start_idx = crd2idx((*k_tile_iter), make_shape(K_start));

    [[maybe_unused]] auto sg_handle = sycl::ext::oneapi::this_work_item::get_sub_group();
    [[maybe_unused]] auto fragment_scaleA = make_tensor<ElementScaleA>(
      Layout<Shape<_1, Int<ScaleAChunks>, _1>>{});

    // Pre-prefetch data tiles
    constexpr auto barrier_scope = SPIRVScope::ScopeWorkgroup;
    int prefetch_k = k_start_idx;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DispatchPolicy::Stages; i++, prefetch_k++) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
    }

    // Scale fragments
    [[maybe_unused]] auto tScaleA = make_tensor<ElementScaleA>(tCrA.layout());
    [[maybe_unused]] auto tScaleB = make_tensor<ElementScaleB>(tCrB.layout());
    [[maybe_unused]] auto zippedA = make_zip_tensor(tCrA, tScaleA);
    [[maybe_unused]] auto zippedB = make_zip_tensor(tCrB, tScaleB);

    constexpr int GEMM_K_ITERS = decltype(size<2>(tScaleA.shape()))::value;
    constexpr int SCALE_B_V = decltype(size<0>(tScaleB.shape()))::value;

    [[maybe_unused]] auto raw_accum = [&] {
      if constexpr (kUseDeferredScale) {
        auto frag = make_fragment_like(accum);
        clear(frag);
        return frag;
      } else {
        return cute::make_tuple();
      }
    }();

    //
    // Mainloop
    //
    const int k_tile_end = k_tile_count + k_start_idx;
    int prev_k_scale_idx = -1;

    for (int k_tile = k_start_idx; k_tile < k_tile_end; k_tile++, prefetch_k++) {

      if constexpr (kPerLaneScaleB) {
        // Per-lane scale path (GroupN=1): reload both scales per ki.
        CUTLASS_PRAGMA_UNROLL
        for (int ki = 0; ki < GEMM_K_ITERS; ++ki) {
          const int k_scale_idx = (k_tile * BLK_K + ki * MMA_K) / GroupK;
          fill_scale_a_ki(tScaleA, ki, mainloop, m_coord, k_scale_idx, l_coord, M_extent);
          fill_scale_b_per_lane_ki<SCALE_B_V>(tScaleB, ki, mainloop, n_coord, k_scale_idx, l_coord, N_scale_extent, lane_id);
        }
      } else if constexpr (kFineGrainScaleK) {
        // Fine-grain K path: different ki may map to different scale groups.
        CUTLASS_PRAGMA_UNROLL
        for (int ki = 0; ki < GEMM_K_ITERS; ++ki) {
          const int k_scale_idx = (k_tile * BLK_K + ki * MMA_K) / GroupK;
          fill_scale_a_ki(tScaleA, ki, mainloop, m_coord, k_scale_idx, l_coord, M_extent);
          fill_scale_b_broadcast_ki<SCALE_B_V>(tScaleB, ki, mainloop, n_scale_coord, k_scale_idx, l_coord);
        }
      } else {
        // Per-tile scale: group doesn't change within a BLK_K tile.
        const int k_scale_idx = (k_tile * BLK_K) / GroupK;
        if (k_scale_idx != prev_k_scale_idx) {
          if constexpr (kUseDeferredScale) {
            // Drain raw_accum using the PREVIOUS GroupK block's scale before switching groups.
            if (prev_k_scale_idx >= 0) {
              drain_deferred_scale_accum(accum, raw_accum, fragment_scaleA, sg_handle,
                  mainloop, m_coord, n_scale_coord, prev_k_scale_idx, l_coord, lane_id, M_extent);
            }
            prev_k_scale_idx = k_scale_idx;
          } else {
            prev_k_scale_idx = k_scale_idx;
            fill_scale_a_all_ki<GEMM_K_ITERS>(tScaleA, mainloop, m_coord, k_scale_idx, l_coord, M_extent);
            fill_scale_b_broadcast_all_ki<SCALE_B_V, GEMM_K_ITERS>(tScaleB, mainloop, n_scale_coord, k_scale_idx, l_coord);
          }
        }
      }

      barrier_arrive(barrier_scope);

      copy(copy_a, tAgA(_,_,_,k_tile), tArA);
      copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

      if (prefetch_k < k_tile_end) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      }

      reorder(tArA, tCrA);
      reorder(tBrB, tCrB);

      if constexpr (kUseDeferredScale) {
        // Direct DPAS accumulation into raw_accum (no per-element scale multiply).
        // The 16 DPAS atoms become back-to-back, freeing float ALU bandwidth between tiles.
        cute::gemm(tiled_mma, tCrA, tCrB, raw_accum);
      } else {
        cute::gemm(tiled_mma, zippedA, zippedB, accum);
      }

      barrier_wait(barrier_scope);
    }

    // Drain the final GroupK block.
    if constexpr (kUseDeferredScale) {
      if (prev_k_scale_idx >= 0) {
        drain_deferred_scale_accum(accum, raw_accum, fragment_scaleA, sg_handle,
            mainloop, m_coord, n_scale_coord, prev_k_scale_idx, l_coord, lane_id, M_extent);
      }
    }
  }
};

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
