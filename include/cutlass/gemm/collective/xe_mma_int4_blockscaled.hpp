/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"

namespace cutlass::gemm::collective {
using namespace cute;

template <
  int Stages,
  int GroupK_,
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
  MainloopIntelXeXMX16IntBlockScaled<Stages, GroupK_, KernelSchedule>,
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
  using DispatchPolicy = MainloopIntelXeXMX16IntBlockScaled<Stages, GroupK_, KernelSchedule>;
  using WorkgroupTileShape = TileShape_;
  using TiledMma = TiledMma_;

  using ElementPairA = ElementPairA_;
  using ElementPairB = ElementPairB_;
  using StridePairA = StridePairA_;
  using StridePairB = StridePairB_;

  using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
  using ElementB = remove_cvref_t<decltype(get<0>(ElementPairB{}))>;
  using ElementScaleA = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;
  using ElementScaleB = remove_cvref_t<decltype(get<1>(ElementPairB{}))>;

  using StrideA = remove_pointer_t<remove_cvref_t<decltype(get<0>(StridePairA{}))>>;
  using StrideB = remove_pointer_t<remove_cvref_t<decltype(get<0>(StridePairB{}))>>;
  using StrideScaleA = remove_pointer_t<remove_cvref_t<decltype(get<1>(StridePairA{}))>>;
  using StrideScaleB = remove_pointer_t<remove_cvref_t<decltype(get<1>(StridePairB{}))>>;

  using GmemTiledCopyA = typename std::tuple_element<0, GmemTiledCopyPairA_>::type;
  using GmemTiledCopyB = typename std::tuple_element<0, GmemTiledCopyPairB_>::type;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;
  using ElementAccumulator = float;

  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;
  using MmaAtomShape = typename TiledMma::AtomShape_MNK;

  static constexpr int GroupK = GroupK_;
  static constexpr int BLK_M = get<0>(WorkgroupTileShape{});
  static constexpr int BLK_N = get<1>(WorkgroupTileShape{});
  static constexpr int BLK_K = get<2>(WorkgroupTileShape{});
  static constexpr int ATOM_M = get<0>(MmaAtomShape{});
  static constexpr int ATOM_N = get<1>(MmaAtomShape{});
  static constexpr int ATOM_K = get<2>(MmaAtomShape{});
  static constexpr int SG_NUMS_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr int SG_NUMS_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr int SG_M = ceil_div(BLK_M, SG_NUMS_M);
  static constexpr int SG_N = ceil_div(BLK_N, SG_NUMS_N);
  static constexpr int M_ITERS = SG_M / ATOM_M;
  static constexpr int N_ITERS = SG_N / ATOM_N;
  static constexpr int ScaleAChunks = cute::ceil_div(SG_M, SubgroupSize);
  using SubgroupTileShape = Shape<C<SG_M>, C<SG_N>, C<ceil_div(BLK_K, ATOM_K)>>;

  static constexpr auto Num_SGs = SG_NUMS_M * SG_NUMS_N;
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});

  static_assert(std::is_same_v<TransformA, cute::identity>, "Transformation for A is not supported.");
  static_assert(std::is_same_v<TransformB, cute::identity>, "Transformation for B is not supported.");
  static_assert(GroupK == 128 || GroupK == 256, "Integer block scaling supports K blocks of 128 or 256.");
  static_assert(GroupK % BLK_K == 0, "The block scale K size must be divisible by the workgroup K tile.");
  static_assert(BLK_K % ATOM_K == 0, "The workgroup K tile must be divisible by the DPAS K tile.");
  static_assert(
    std::is_same_v<typename TiledMma::ValTypeA, int4_t> ||
    std::is_same_v<typename TiledMma::ValTypeA, int8_t>,
    "A must use signed INT4 or INT8.");
  static_assert(
    std::is_same_v<typename TiledMma::ValTypeB, int4_t> ||
    std::is_same_v<typename TiledMma::ValTypeB, int8_t>,
    "B must use signed INT4 or INT8.");
  static_assert(std::is_same_v<typename TiledMma::ValTypeA, typename TiledMma::ValTypeB>,
                "A and B must use the same signed integer type.");

  template<class Element, class Stride>
  using TensorType = decltype(make_tensor(
      make_gmem_ptr(static_cast<Element const*>(nullptr)),
      make_layout(make_shape(int{}, int{}, int{}), Stride{})));

  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    ElementScaleA const* ptr_SA;
    StrideScaleA dSA;
    ElementScaleB const* ptr_SB;
    StrideScaleB dSB;
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
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void)workspace;
    auto [M, N, K, L] = problem_shape;
    auto scale_k = cute::ceil_div(K, GroupK);

    return {
      make_tensor(make_gmem_ptr(args.ptr_A), make_layout(make_shape(M, K, L), args.dA)),
      make_tensor(make_gmem_ptr(args.ptr_B), make_layout(make_shape(N, K, L), args.dB)),
      make_tensor(make_gmem_ptr(args.ptr_SA), make_layout(make_shape(M, scale_k, L), args.dSA)),
      make_tensor(make_gmem_ptr(args.ptr_SB), make_layout(make_shape(N, scale_k, L), args.dSB))
    };
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape problem_shapes, Arguments const& args) {
    constexpr int copy_alignment_bits = 128;
    constexpr int batch_alignment_bits = 512;
    auto problem_shape_MNKL = append<4>(problem_shapes, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    bool implementable = true;

    implementable &= (K % GroupK == 0);
    implementable &= (K % BLK_K == 0);

    constexpr int min_aligned_elements_A = copy_alignment_bits / sizeof_bits<ElementA>::value;
    constexpr int min_aligned_elements_B = copy_alignment_bits / sizeof_bits<ElementB>::value;
    implementable &= cutlass::detail::check_alignment<min_aligned_elements_A>(make_shape(1, K, L), args.dA);
    implementable &= cutlass::detail::check_alignment<min_aligned_elements_B>(make_shape(N, K, L), args.dB);

    if (L > 1) {
      constexpr int min_batch_aligned_elements_A = batch_alignment_bits / sizeof_bits<ElementA>::value;
      constexpr int min_batch_aligned_elements_B = batch_alignment_bits / sizeof_bits<ElementB>::value;
      implementable &= get<2>(args.dA) % min_batch_aligned_elements_A == 0;
      implementable &= get<2>(args.dB) % min_batch_aligned_elements_B == 0;
    }

    return implementable;
  }

  template <class ScaleAFragment>
  CUTLASS_DEVICE static void
  load_block_scales(ScaleAFragment& fragment_scaleA,
                    ElementAccumulator (&scale_b_cache)[N_ITERS],
                    Params const& mainloop,
                    int m_coord,
                    int n_coord,
                    int k_scale_idx,
                    int batch_idx) {
    const int M_extent = get<0>(mainloop.mAscale.shape());
    const int N_extent = get<0>(mainloop.mBscale.shape());
    const int lane_id = get_sub_group_local_id();

    CUTLASS_PRAGMA_UNROLL
    for (int load_idx = 0; load_idx < ScaleAChunks; ++load_idx) {
      const int m = m_coord + load_idx * SubgroupSize + lane_id;
      fragment_scaleA(0, load_idx, 0) = m < M_extent
          ? static_cast<ElementScaleA>(mainloop.mAscale(m, k_scale_idx, batch_idx))
          : ElementScaleA(0);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < N_ITERS; ++ni) {
      const int n = n_coord + ni * ATOM_N + lane_id;
      scale_b_cache[ni] = n < N_extent
          ? static_cast<float>(mainloop.mBscale(n, k_scale_idx, batch_idx))
          : 0.0f;
    }
  }

  template <class FrgTensorD, class RawAccum, class ScaleAFragment, class Subgroup>
  CUTLASS_DEVICE static void
  drain_block_cached(FrgTensorD& accum,
                     RawAccum& raw_accum,
                     ScaleAFragment& fragment_scaleA,
                     ElementAccumulator const (&scale_b_cache)[N_ITERS],
                     Subgroup sg_handle) {
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < M_ITERS; ++mi) {
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < ATOM_M; ++v) {
        const int m_local = mi * ATOM_M + v;
        const int load_idx = m_local / SubgroupSize;
        const int lane_idx = m_local % SubgroupSize;
        const float scale_a = static_cast<float>(group_broadcast(
          sg_handle, fragment_scaleA(0, load_idx, 0), lane_idx));
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < N_ITERS; ++ni) {
          accum(v, mi, ni) += static_cast<float>(raw_accum(v, mi, ni)) * scale_a * scale_b_cache[ni];
          raw_accum(v, mi, ni) = int32_t(0);
        }
      }
    }
  }

  template <class FrgTensorD, class RawAccum, class ScaleAFragment, class Subgroup>
  CUTLASS_DEVICE static void
  drain_block(FrgTensorD& accum,
              RawAccum& raw_accum,
              ScaleAFragment& fragment_scaleA,
              Subgroup sg_handle,
              Params const& mainloop,
              int m_coord,
              int n_coord,
              int k_scale_idx,
              int batch_idx) {
    const int M_extent = get<0>(mainloop.mAscale.shape());
    const int N_extent = get<0>(mainloop.mBscale.shape());
    const int lane_id = get_sub_group_local_id();

    CUTLASS_PRAGMA_UNROLL
    for (int load_idx = 0; load_idx < ScaleAChunks; ++load_idx) {
      const int m = m_coord + load_idx * SubgroupSize + lane_id;
      fragment_scaleA(0, load_idx, 0) = m < M_extent
          ? static_cast<ElementScaleA>(mainloop.mAscale(m, k_scale_idx, batch_idx))
          : ElementScaleA(0);
    }

    ElementAccumulator scale_b_cache[N_ITERS];
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < N_ITERS; ++ni) {
      const int n = n_coord + ni * ATOM_N + lane_id;
      scale_b_cache[ni] = n < N_extent
          ? static_cast<float>(mainloop.mBscale(n, k_scale_idx, batch_idx))
          : 0.0f;
    }

    drain_block_cached(accum, raw_accum, fragment_scaleA, scale_b_cache, sg_handle);
  }

  template <class FrgTensorD,
            class TensorA,
            class TensorB,
            class FrgTensorC,
            class KTileIterator,
            class BlkCoord>
  CUTLASS_DEVICE void
  operator()(FrgTensorD& accum,
             TensorA gA,
             TensorB gB,
             FrgTensorC const& src_accum,
             KTileIterator k_tile_iter,
             int k_tile_count,
             BlkCoord const& blk_coord,
             int const& K_start,
             int thread_idx,
             Params const& mainloop) {
    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    (void)src_accum;

    const int batch_idx = get<3>(blk_coord);
    auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(TiledMma{}, mainloop.mA_mkl(_, _, batch_idx));
    auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(TiledMma{}, mainloop.mB_nkl(_, _, batch_idx));
    auto thr_copy_a = copy_a.get_slice(thread_idx);
    auto thr_copy_b = copy_b.get_slice(thread_idx);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);
    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);

    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);
    auto thr_prefetch_A = prefetch_a.get_slice(thread_idx);
    auto thr_prefetch_B = prefetch_b.get_slice(thread_idx);
    Tensor pAgA = thr_prefetch_A.partition_S(gA);
    Tensor pBgB = thr_prefetch_B.partition_S(gB);

    auto raw_accum = make_fragment_like<int32_t>(accum);
    clear(accum);
    clear(raw_accum);
    auto sg_handle = sycl::ext::oneapi::this_work_item::get_sub_group();
    auto fragment_scaleA = make_tensor<ElementScaleA>(
      Layout<Shape<_1, Int<ScaleAChunks>, _1>>{});
    ElementAccumulator scale_b_cache[N_ITERS];

    auto [m_idx, n_idx, k_idx, l_idx] = blk_coord;
    const int m_coord = m_idx * BLK_M + (get_sub_group_id() / SG_NUMS_N) * SG_M;
    const int n_coord = n_idx * BLK_N + (get_sub_group_id() % SG_NUMS_N) * SG_N;
    int loaded_scale_idx = -1;

    const int k_start_idx = crd2idx((*k_tile_iter), make_shape(K_start));
    int prefetch_k = k_start_idx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DispatchPolicy::Stages; ++i, ++prefetch_k) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
    }

    const int k_tile_end = k_start_idx + k_tile_count;
    for (int k_tile = k_start_idx; k_tile < k_tile_end; ++k_tile, ++prefetch_k) {
      if constexpr (GroupK == BLK_K) {
        const int k_scale_idx = (k_tile * BLK_K) / GroupK;
        if (k_scale_idx != loaded_scale_idx) {
          load_block_scales(fragment_scaleA, scale_b_cache,
                            mainloop, m_coord, n_coord, k_scale_idx, batch_idx);
          loaded_scale_idx = k_scale_idx;
        }
      }
        if (prefetch_k < k_tile_end) {
          prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
          prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
        }


      copy(copy_a, tAgA(_, _, _, k_tile), tArA);
      copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

      reorder(tArA, tCrA);
      reorder(tBrB, tCrB);
      cute::gemm(tiled_mma, tCrA, tCrB, raw_accum);

      const int k_end = (k_tile + 1) * BLK_K;
      if (k_end % GroupK == 0) {
        if constexpr (GroupK == BLK_K) {
          drain_block_cached(accum, raw_accum, fragment_scaleA, scale_b_cache, sg_handle);
        } else {
          drain_block(accum, raw_accum, fragment_scaleA, sg_handle,
                      mainloop, m_coord, n_coord,
                      (k_end - 1) / GroupK, batch_idx);
        }
      }
    }
  }
};

} // namespace cutlass::gemm::collective